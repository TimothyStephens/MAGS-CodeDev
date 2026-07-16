"""MAGs-CodeDev CLI entry point.

Provides commands for workspace initialization, multi-agent builds,
testing, debugging, token usage reporting, and cleanup.
"""

import typer

from rich.console import Console

app = typer.Typer(
    help="MAGs-CodeDev: Multi-Agent Graph System for Code Development",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["--help", "-h"]},
)
console = Console()

# -------------------------------------------------------------------
# Register commands from cmd modules
# -------------------------------------------------------------------

from mags_codedev.cmds.cmd_init import configure_command as configure_init
from mags_codedev.cmds.cmd_build import configure_command as configure_build
from mags_codedev.cmds.cmd_test import configure_command as configure_test
from mags_codedev.cmds.cmd_debug import configure_command as configure_debug
from mags_codedev.cmds.cmd_chat import configure_command as configure_chat
from mags_codedev.cmds.cmd_tokens import configure_command as configure_tokens
from mags_codedev.cmds.cmd_list_models import configure_command as configure_list_models
from mags_codedev.cmds.cmd_clean import configure_command as configure_clean

configure_init(app)
configure_build(app)
configure_test(app)
configure_debug(app)
configure_chat(app)
configure_tokens(app)
configure_list_models(app)
configure_clean(app)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    """CLI entry point for MAGs-CodeDev."""
    app()


if __name__ == "__main__":
    main()
