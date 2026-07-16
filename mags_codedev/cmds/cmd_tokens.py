"""Tokens command: display token usage and costs across all models."""

import typer
from rich.console import Console
from rich.table import Table

from mags_codedev.utils.db import init_db, get_token_summary
from mags_codedev.utils.cli_common import (
    resolve_base_dir,
    find_default_config_path,
)

console = Console()


def tokens():
    """Display a rich table of token usage and costs across all models and runs."""
    # Try to find config to get base_dir
    try:
        config_path = find_default_config_path()
        base_dir = resolve_base_dir(config_path)
    except Exception:
        base_dir = ".mags-codedev"

    init_db(base_dir=base_dir)
    per_role_summary, per_model_summary, total = get_token_summary(base_dir=base_dir)

    if not per_role_summary:
        console.print("[yellow]No token usage has been recorded yet.[/yellow]")
        return

    # Per Role Table
    role_table = Table(
        title="Token Usage by Agent/Role",
        show_header=True,
        header_style="bold cyan",
        show_footer=True,
        footer_style="bold",
    )
    role_table.add_column("Agent/Role", style="green", footer="Total")
    role_table.add_column("Model", style="yellow")
    role_table.add_column("Input Tokens", justify="right")
    role_table.add_column("Output Tokens", justify="right")
    role_table.add_column("Total Tokens", justify="right", footer=f"{total[0] + total[1]:,}")

    for role, model, in_tokens, out_tokens in per_role_summary:
        role_table.add_row(
            role,
            model,
            f"{in_tokens:,}",
            f"{out_tokens:,}",
            f"{in_tokens + out_tokens:,}",
        )
    console.print(role_table)

    # Per Model Table
    if per_model_summary:
        model_table = Table(
            title="Token Usage by Model",
            show_header=True,
            header_style="bold cyan",
            show_footer=True,
            footer_style="bold",
        )
        model_table.add_column("Model", style="yellow", footer="Total")
        model_table.add_column("Input Tokens", justify="right", footer=f"{total[0]:,}")
        model_table.add_column("Output Tokens", justify="right", footer=f"{total[1]:,}")
        model_table.add_column(
            "Total Tokens", justify="right", footer=f"{total[0] + total[1]:,}"
        )

        for model, in_tokens, out_tokens in per_model_summary:
            model_table.add_row(
                model,
                f"{in_tokens:,}",
                f"{out_tokens:,}",
                f"{in_tokens + out_tokens:,}",
            )

        console.print(model_table)


def configure_command(app: typer.Typer) -> None:
    """Register the tokens command on the given typer app."""
    app.command()(tokens)
