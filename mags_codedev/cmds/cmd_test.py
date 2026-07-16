"""Test command: run project unit tests in the configured environment."""

import os
import typer
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from mags_codedev.utils.logger import setup_logger, logger
from mags_codedev.backends import get_backend
from mags_codedev.utils.docker_ops import run_command_in_project_env
from mags_codedev.utils.cli_common import (
    resolve_base_dir,
    find_default_config_path,
    _CONFIG_HELP_TEXT,
)

console = Console()


def test(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True,
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True,
        help="Verbosity level (0=info, 1=debug, 2=trace).",
    ),
):
    """Run all pytest unit tests for the project in the configured environment."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    base_dir = resolve_base_dir(config_path)
    verbose_levels = {0: "info", 1: "debug", 2: "trace"}
    log_level = verbose_levels.get(verbose, "info")
    setup_logger(base_dir=base_dir, log_level=log_level)

    console.print(Panel("[bold cyan]Running Project Unit Tests...[/bold cyan]"))
    logger.info("Starting project-wide test run.")

    test_backend = get_backend(config_path)
    command = test_backend.test_command_project()
    project_root = os.getcwd()

    with console.status("[bold green]Running tests...[/bold green]", spinner="dots"):
        results = run_command_in_project_env(
            command, config_path, project_root, logger, test_backend
        )

    console.print(Panel(results, title="Test Results", border_style="blue"))

    if "failed" in results.lower() or "error" in results.lower():
        console.print("[bold red]Some tests failed or errors occurred.[/bold red]")
        raise typer.Exit(code=1)
    else:
        console.print("[bold green]All tests passed![/bold green]")


def configure_command(app: typer.Typer) -> None:
    """Register the test command on the given typer app."""
    app.command()(test)
