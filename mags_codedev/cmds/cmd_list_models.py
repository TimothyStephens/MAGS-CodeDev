"""List-models command: list available models from configured providers."""

import typer
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mags_codedev.utils.logger import logger
from mags_codedev.utils.config_parser import load_config
from mags_codedev.utils.cli_helpers import format_llm_error
from mags_codedev.utils.cli_common import (
    find_default_config_path,
    _CONFIG_HELP_TEXT,
)

console = Console()


def list_models(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True,
    ),
):
    """List available models from the configured providers (OpenAI, Google, etc.)."""
    if config_path is None:
        config_path = find_default_config_path()

    if not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    logger.info(f"Using configuration: {config_path}")
    config = load_config(config_path)
    api_keys = config.get("api_keys", {})

    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="green")
    table.add_column("Model ID", style="yellow")

    # 1. OpenAI
    if api_keys.get("openai"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_keys["openai"])
            models = client.models.list()
            gpt_models = sorted([m.id for m in models.data if "gpt" in m.id])
            for m in gpt_models:
                table.add_row("OpenAI", m)
        except Exception as e:
            table.add_row("OpenAI", f"[red]Error: {format_llm_error(e)}[/red]")

    # 2. Google (Gemini)
    if api_keys.get("gemini"):
        try:
            try:
                from google import genai
                client = genai.Client(api_key=api_keys["gemini"])
                for m in client.models.list():
                    name = m.name.replace("models/", "") if m.name else str(m.name)
                    table.add_row("Google", name)
            except ImportError:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        import google.generativeai as genai
                    genai.configure(api_key=api_keys["gemini"])  # type: ignore[attr-defined]
                    for m in genai.list_models():  # type: ignore[attr-defined]
                        if 'generateContent' in m.supported_generation_methods:
                            name = m.name.replace("models/", "")
                            table.add_row("Google", name)
                except ImportError:
                    table.add_row(
                        "Google",
                        "[yellow]Skipped: 'google.genai' or 'google-generativeai' not installed[/yellow]",
                    )
        except Exception as e:
            table.add_row("Google", f"[red]Error: {format_llm_error(e)}[/red]")

    # 3. Anthropic
    if api_keys.get("anthropic"):
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_keys["anthropic"])
            response = client.models.list()
            model_ids = sorted([model.id for model in response.data])
            for m in model_ids:
                table.add_row("Anthropic", m)
        except Exception as e:
            table.add_row("Anthropic", f"[red]Error: {format_llm_error(e)}[/red]")

    # 4. Mistral
    if api_keys.get("mistral"):
        try:
            try:
                from mistralai.client import MistralClient  # type: ignore[import-not-found]
                client = MistralClient(api_key=api_keys["mistral"])
                models = client.list_models()
                for m in models.data:
                    table.add_row("Mistral", m.id)
            except ImportError:
                table.add_row("Mistral", "[yellow]Skipped: 'mistralai' not installed[/yellow]")
        except Exception as e:
            table.add_row("Mistral", f"[red]Error: {format_llm_error(e)}[/red]")

    # 5. Cohere
    if api_keys.get("cohere"):
        known_models = ["command-r-plus", "command-r", "command", "command-light"]
        for m in known_models:
            table.add_row("Cohere (Static)", m)

    # 6. Custom / Local (e.g. Ollama)
    custom_urls = set()
    models_config = config.get("models", {})
    build_config = models_config.get("build_workflow", {})
    interactive_config = models_config.get("interactive_commands", {})

    all_agent_configs = {
        **models_config,
        **build_config,
        **interactive_config,
    }

    for role in ["coder", "tester", "log_checker", "chat"]:
        m_cfg = all_agent_configs.get(role, {})
        if m_cfg.get("provider") in ("custom_openai", "local") and m_cfg.get("base_url"):
            custom_urls.add(m_cfg.get("base_url"))

    reviewers_list = (
        build_config.get("reviewers", []) or models_config.get("reviewers", [])
    )
    for r_cfg in reviewers_list:
        if r_cfg.get("provider") in ("custom_openai", "local") and r_cfg.get("base_url"):
            custom_urls.add(r_cfg.get("base_url"))

    for url in custom_urls:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=url, api_key="dummy")
            models = client.models.list()
            for m in models.data:
                table.add_row(f"Custom ({url})", m.id)
        except Exception as e:
            table.add_row(
                f"Custom ({url})", f"[red]Error: {format_llm_error(e)}[/red]"
            )

    console.print(table)


def configure_command(app: typer.Typer) -> None:
    """Register the list-models command on the given typer app."""
    app.command(name="list-models")(list_models)
