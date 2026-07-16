"""Clean command: remove generated cache files, logs, and worktrees."""

import shutil
import typer
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from mags_codedev.utils.cli_common import (
    resolve_base_dir,
    find_default_config_path,
)

console = Console()


def clean(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force deletion without confirmation."
    ),
):
    """Remove all generated cache files, logs, and temporary git worktrees."""
    # Try to find base_dir from config, fallback to default
    try:
        config_path = find_default_config_path()
        base_dir = resolve_base_dir(config_path)
    except Exception:
        base_dir = ".mags-codedev"

    console.print(Panel("[bold yellow]Cleaning up MAGs-CodeDev artifacts...[/bold yellow]"))

    mags_dir = Path(base_dir)

    items_to_delete = []
    if mags_dir.exists():
        items_to_delete.append(mags_dir)

    if not items_to_delete:
        console.print("[green]✓ No artifacts to clean.[/green]")
        raise typer.Exit()

    console.print("The following items will be permanently deleted:")
    for item in items_to_delete:
        console.print(f"- [red]{item}[/red]")

    if not force:
        if not typer.confirm("\nAre you sure you want to proceed?"):
            console.print("[yellow]Clean operation cancelled.[/yellow]")
            raise typer.Exit()

    console.print("")

    if mags_dir.exists():
        shutil.rmtree(str(mags_dir))
        console.print(f"Removed directory: {mags_dir}")

    console.print("\n[bold green]✓ Cleanup complete.[/bold green]")


def configure_command(app: typer.Typer) -> None:
    """Register the clean command on the given typer app."""
    app.command()(clean)
