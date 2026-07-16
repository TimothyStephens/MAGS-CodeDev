"""Chat command: freely chat with the LLM about the codebase."""

import typer
from typing import Optional
from pathlib import Path

from rich.console import Console

from mags_codedev.utils.logger import setup_logger, logger
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.db import TokenLoggingCallbackHandler
from mags_codedev.utils.cli_helpers import extract_content, format_llm_error
from mags_codedev.utils.cli_common import (
    resolve_base_dir,
    find_default_config_path,
    _CONFIG_HELP_TEXT,
)

console = Console()


def chat(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True,
    ),
    offline: bool = typer.Option(
        False, "--offline", "--no-llm",
        help="Skip LLM API calls (use stubs instead).",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True,
        help="Verbosity level (0=info, 1=debug, 2=trace).",
    ),
):
    """Freely chat with the LLM about the codebase. Can read/write files."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    base_dir = resolve_base_dir(config_path)
    verbose_levels = {0: "info", 1: "debug", 2: "trace"}
    log_level = verbose_levels.get(verbose, "info")
    setup_logger(base_dir=base_dir, log_level=log_level)

    # Per-session log for debugging chat interactions

    logger.info(f"Using configuration: {config_path}")
    from mags_codedev.agents.chat_agent import start_chat_repl

    if offline:
        console.print("[yellow]Offline mode: chat uses stub LLM (no API calls).[/yellow]")
    else:
        console.print("[bold blue]Entering Chat Mode (Type 'exit' to quit)...[/bold blue]")

    agent_graph = start_chat_repl(config_path=config_path, command_name="chat", offline=offline)

    # Token tracking (only meaningful in online mode)
    if not offline:
        llm = get_llm("chat", config_path)
        model_name = getattr(llm, "model_name", getattr(llm, "model", "unknown"))
        callback = TokenLoggingCallbackHandler(
            role="command_chat", model_name=model_name, base_dir=base_dir
        )
    else:
        callback = None

    config = {
        "configurable": {"thread_id": "cli-session"},
        "callbacks": [callback] if callback else [],
    }

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ")
            if user_input.lower() in ['exit', 'quit']:
                break

            logger.debug(f"Chat Input: {user_input}")

            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response = agent_graph.invoke(
                    {"messages": [("user", user_input)]}, config=config
                )

            last_message = response["messages"][-1]
            final_answer = extract_content(last_message.content)

            logger.debug(f"Chat Final Answer: {final_answer}")
            console.print(f"\n[blue]Agent>[/blue] {final_answer}\n")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            logger.exception("An error occurred during chat session")
            console.print(f"[red]Error: {format_llm_error(e)}[/red]")

    console.print("\n[blue]Chat ended.[/blue]")


def configure_command(app: typer.Typer) -> None:
    """Register the chat command on the given typer app."""
    app.command()(chat)
