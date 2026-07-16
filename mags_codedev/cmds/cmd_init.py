"""Init command for MAGs-CodeDev: workspace initialization and project setup."""

import os
import json
import yaml
import typer
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from mags_codedev.utils.logger import setup_logger, logger
from mags_codedev.utils.config_parser import load_config, get_llm, get_log_level
from mags_codedev.utils.db import init_db, TokenLoggingCallbackHandler
from mags_codedev.utils.git_ops import ensure_git_repo
from mags_codedev.backends import get_backend
from mags_codedev.utils.cli_helpers import extract_content
from mags_codedev.utils.cli_common import (
    resolve_base_dir,
    _bootstrap_config,
    _DEFAULT_BASE_DIR,
    _CONFIG_HELP_TEXT,
)

console = Console()


def configure_command(app: typer.Typer) -> None:
    """Register the init command on the provided Typer app."""
    app.command()(init)


def init(
    manifest_path: Optional[Path] = typer.Option(
        None, "--manifest", "-m",
        help=(
            "Path to create the manifest JSON file. "
            "Default: <base_dir>/manifest.json "
            "(e.g., .mags-codedev/manifest.json)"
        ),
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--non-interactive",
        help=(
            "Use AI to interactively design the project structure "
            "and open editor for config."
        ),
    ),
    offline: bool = typer.Option(
        False, "--offline", "--no-llm",
        help="Skip LLM API calls (use stubs instead).",
    ),
):
    """Initialize the MAGs-CodeDev workspace and project structure."""
    console.print(Panel("[bold cyan]Initializing MAGs-CodeDev Workspace...[/bold cyan]"))
    logger.info("Starting workspace initialization.")

    # If --config was explicitly provided, use it; otherwise bootstrap from defaults
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
            raise typer.Exit(1)
        base_dir = resolve_base_dir(config_path)
        config_was_created = False
    else:
        # Bootstrap: copy user default or template to <base_dir>/config.yaml
        config_path, config_was_created = _bootstrap_config(_DEFAULT_BASE_DIR)
        base_dir = resolve_base_dir(config_path)

    # Default manifest to <base_dir>/manifest.json if not specified
    if manifest_path is None:
        manifest_path = Path(base_dir) / "manifest.json"
    # Ensure manifest path is a Path
    manifest_path = Path(manifest_path)

    # Open editor if config was just created and we're interactive
    if config_was_created and interactive:
        _open_in_editor(config_path)

    setup_logger(base_dir=base_dir, log_level=get_log_level(config_path))


    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Target manifest: {manifest_path}")
    console.print(f"[cyan]Using configuration: {config_path}[/cyan]")

    # 1. Gitignore — only the base_dir (config.yaml lives inside it)
    gitignore_marker = "# MAGS-CodeDev"
    gitignore_content = f"\n{gitignore_marker}\n{base_dir}/\n"
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            current_content = f.read()
        if gitignore_marker not in current_content:
            with open(".gitignore", "a") as f:
                f.write(gitignore_content)
            console.print("[green]Updated .gitignore[/green]")
            logger.info("Updated .gitignore with MAGS-CodeDev patterns.")
    else:
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        console.print("[green]Created .gitignore[/green]")
        logger.info("Created .gitignore file.")

    # Warn about legacy directory
    if os.path.exists(".MAGS-CodeDev"):
        console.print(
            "[yellow]Legacy '.MAGS-CodeDev/' directory found. "
            "Run 'mags-codedev clean' in the old directory to remove it.[/yellow]"
        )

    # 1b. Ensure git repo exists
    try:
        ensure_git_repo()
        console.print("[green]Initialized Git repository[/green]")
        logger.info("Initialized Git repository.")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not initialize Git repo: {e}[/yellow]")
        logger.warning(f"Failed to initialize Git repo: {e}")

    # 2. Database
    init_db(base_dir=base_dir)
    console.print("[green]Initialized SQLite Database[/green]")
    logger.info("Initialized SQLite database.")

    # 3. Create base_dir structure
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "worktrees"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "containers"), exist_ok=True)

    # 4. Dependencies file (backend-agnostic)
    init_backend = get_backend(config_path)
    deps_file = init_backend.deps_filename
    if not os.path.exists(deps_file):
        Path(deps_file).touch()
        console.print(f"[green]Created empty {deps_file} for project dependencies.[/green]")
        logger.info(f"Created empty {deps_file}.")

    # 5. Interactive Setup (Manifest & AGENT.md)
    manifest_created = False

    # Offline mode disables interactive AI
    if offline:
        interactive = False
        console.print("[yellow]Offline mode: skipping interactive AI setup.[/yellow]")

    if interactive:
        config = load_config(config_path)
        chat_config = config.get("models", {}).get("interactive_commands", {}).get(
            "chat",
            config.get("models", {}).get("chat", {}),
        )
        chat_provider = chat_config.get("provider", "openai")

        key_name_map = {
            "openai": "openai",
            "google": "gemini",
            "anthropic": "anthropic",
            "ollama": "ollama",
            "local": "ollama",
            "custom_openai": "ollama",
        }
        required_key_name = key_name_map.get(chat_provider, "openai")
        api_key = config.get("api_keys", {}).get(required_key_name)

        # Local providers with base_url configured don't need an API key prompt
        # (the server handles auth; client uses "dummy" or configured key)
        is_local_provider = chat_provider in ("ollama", "local", "custom_openai")
        has_base_url = chat_config.get("base_url")

        if not api_key or "..." in api_key or "YOUR_KEY" in api_key:
            if is_local_provider and has_base_url:
                console.print(
                    f"[green]Local provider '{chat_provider}' configured with "
                    f"base_url — skipping API key prompt.[/green]"
                )
            else:
                console.print(
                    f"[yellow]{required_key_name.capitalize()} API key missing or is a "
                    f"placeholder in config.yaml.[/yellow]"
                )
                user_key = typer.prompt(
                    f"Enter {required_key_name.capitalize()} API Key for AI setup "
                    f"(leave empty to skip AI)",
                    default="", show_default=False, hide_input=True,
                )
                if user_key:
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f) or {}
                    config_data.setdefault('api_keys', {})[required_key_name] = user_key
                    with open(config_path, 'w') as f:
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    console.print(f"[green]Updated {config_path} with API key.[/green]")
                    logger.info(
                        f"Updated {config_path} with provided {required_key_name} API key."
                    )
                else:
                    interactive = False

    if interactive:
        try:
            from mags_codedev.agents.chat_agent import start_chat_repl
            logger.info("Starting AI Architect tool-based session.")

            architect_system_prompt = (
                f"You are a Senior Software Architect. Your goal is to help the user "
                f"define and set up a new software project from scratch.\n\n"
                f"**Your Process:**\n"
                f"1.  **Discuss:** Talk with the user about their project idea.\n"
                f"2.  **Plan:** Propose a plan with file structure, modules, dependencies.\n"
                f"3.  **Execute:** Once the user approves, use tools to create files.\n\n"
                f"**Key Files:**\n"
                f"- `{manifest_path}`: JSON manifest for the build system.\n"
                f"- `{base_dir}/AGENT.md`: Instructions for AI agents.\n"
                f"- `{deps_file}`: Project dependencies.\n"
                f"- `README.md`: Project description.\n\n"
                f"**Important Rules:**\n"
                f"- **Do not write files until the user approves your plan.**\n"
                f"- Use tools to create/modify files directly.\n"
                f"- Workflow artifacts live in `{base_dir}/` (config, manifest, logs, worktrees).\n"
                f"Start by greeting the user and asking about their project idea."
            )

            console.print(
                Panel(
                    "[bold green]AI Architect Mode[/bold green]\n"
                    "Describe your project idea. The AI will ask questions and then "
                    "use its tools to write `AGENT.md` and `manifest.json` for you.\n\n"
                    "Type 'exit' or 'quit' to end the session."
                )
            )

            agent_graph = start_chat_repl(
                config_path=config_path,
                system_message_override=architect_system_prompt,
                command_name="init",
            )
            llm = get_llm("chat", config_path)
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            graph_config = {
                "configurable": {"thread_id": "architect-session"},
                "callbacks": [
                    TokenLoggingCallbackHandler(
                        role="command_init", model_name=model_name, base_dir=base_dir
                    )
                ],
            }

            while True:
                try:
                    user_input = console.input("[bold green]You>[/bold green] ")
                    if user_input.lower() in ['exit', 'quit']:
                        logger.info("User exited AI Architect mode.")
                        break

                    logger.debug(f"Architect Input: {user_input}")

                    with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                        response = agent_graph.invoke(
                            {"messages": [("user", user_input)]}, config=graph_config
                        )
                    last_message = response["messages"][-1]
                    final_answer = extract_content(last_message.content)
                    logger.debug(f"Architect Final Answer: {final_answer}")
                    console.print(f"\n[blue]Architect>[/blue] {final_answer}\n")

                except (KeyboardInterrupt, EOFError):
                    break
            console.print("\n[blue]Exiting AI Architect mode.[/blue]")

        except Exception as e:
            logger.exception("AI Architect Mode encountered an error")
            console.print(f"[red]AI Setup Error: {format_llm_error(e)}[/red]")
            console.print("[yellow]Falling back to manual setup.[/yellow]")

        manifest_created = manifest_path.exists()

    # Define paths for manual creation (used in both interactive and offline modes)
    agent_md_path = Path(base_dir) / "AGENT.md"
    # 6. Fallback / Manual Creation
    if not agent_md_path.exists():
        project_name = os.path.basename(os.getcwd())
        lang_name = init_backend.display_name
        agent_md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(agent_md_path, "w") as f:
            f.write(f"# Agent Instructions for {project_name}\n\n")
            f.write(f"Language: {lang_name}\n")
            f.write("Follow standard coding conventions and best practices.\n")
        console.print(f"[green]Created default {agent_md_path}[/green]")
        logger.info(f"Created default {agent_md_path}.")

    if not manifest_created and not manifest_path.exists():
        dummy_manifest = [
            {
                "location": "src/utils.py",
                "description": "A module for shared utility functions, like data validation or formatting.",
                "dependencies": [],
            },
            {
                "location": "src/main_logic.py",
                "description": "The main business logic module. It will import and use functions from src/utils.py.",
                "dependencies": ["src/utils.py"],
            },
        ]
        with open(manifest_path, "w") as f:
            json.dump(dummy_manifest, f, indent=4)
        console.print(
            f"[green]Created dummy {manifest_path}. Please edit this to define your modules.[/green]"
        )
        logger.info(f"Created dummy manifest file at {manifest_path}.")

    # Print workspace summary
    console.print("\n[bold cyan]✓ Workspace initialized![/bold cyan]")
    console.print(f"  [dim]Config:[/dim]      {config_path}")
    console.print(f"  [dim]Manifest:[/dim]    {manifest_path}")
    console.print(f"  [dim]AGENT.md:[/dim]    {agent_md_path}")
    console.print(f"  [dim]Logs:[/dim]        {Path(base_dir) / 'workflow.log'}")
    console.print(f"  [dim]Artifacts:[/dim]   {Path(base_dir)}/")
    console.print(f"  [dim]Worktrees:[/dim]   {Path(base_dir) / 'worktrees/'}")


def _open_in_editor(path: Path) -> None:
    """Open file in the user's editor. Used by the init command."""
    from mags_codedev.utils.cli_helpers import _open_in_editor as _open_editor_impl
    _open_editor_impl(str(path))


def format_llm_error(e: Exception) -> str:
    """Format LLM error for display in the init command."""
    from mags_codedev.utils.cli_helpers import format_llm_error as _fmt
    return _fmt(e)
