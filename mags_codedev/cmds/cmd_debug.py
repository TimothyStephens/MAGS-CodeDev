"""Debug command: pass an error trace to the LLM for automatic fixing."""

import os
import json
import asyncio
import typer
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from langchain_core.prompts import ChatPromptTemplate

from mags_codedev.utils.logger import setup_logger, logger
from mags_codedev.utils.db import (
    hash_spec,
    TokenLoggingCallbackHandler,
)
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.cli_helpers import extract_content
from mags_codedev.cmds.cmd_build import process_module
from mags_codedev.utils.cli_common import (
    resolve_base_dir,
    find_default_config_path,
    _CONFIG_HELP_TEXT,
)

console = Console()


def debug(
    error_msg: str = typer.Argument(
        ...,
        help="The error trace to fix, or a path to the error trace logfile.",
    ),
    module_location: Optional[str] = typer.Option(
        None, "--module", "--mod",
        help="The module location in manifest.json to apply the fix to.",
    ),
    manifest_path: Optional[Path] = typer.Option(
        None, "--manifest", "-m",
        help=(
            "Path to the manifest JSON file. "
            "Default: <base_dir>/manifest.json"
        ),
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True,
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True,
        help="Verbosity level (0=info, 1=debug, 2=trace).",
    ),
):
    """Pass an error trace or bug description to the LLM for automatic fixing of a module."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    base_dir = resolve_base_dir(config_path)
    verbose_levels = {0: "info", 1: "debug", 2: "trace"}
    log_level = verbose_levels.get(verbose, "info")
    setup_logger(base_dir=base_dir, log_level=log_level)


    # Default manifest to <base_dir>/manifest.json if not specified
    if manifest_path is None:
        manifest_path = Path(base_dir) / "manifest.json"
    manifest_path = Path(manifest_path)
    is_log_file = False
    log_file_path = None

    if os.path.exists(error_msg) and os.path.isfile(error_msg):
        is_log_file = True
        log_file_path = error_msg
        try:
            with open(error_msg, "r") as f:
                console.print(f"[cyan]Reading error trace from file: {error_msg}[/cyan]")
                error_msg = f.read()
        except Exception as e:
            console.print(f"[red]Error reading file '{error_msg}': {e}[/red]")
            raise typer.Exit(1)
    else:
        # H8: Only match .log files or SHA-256 hash filenames
        words = error_msg.split()
        for word in words:
            clean_word = word.strip(".,;:'\"")
            if not os.path.exists(clean_word) or not os.path.isfile(clean_word):
                continue
            basename = os.path.basename(clean_word)
            if not (basename.endswith(".log") or
                    (len(basename) == 64 and all(c in '0123456789abcdef' for c in basename))):
                continue
            is_log_file = True
            log_file_path = clean_word
            try:
                with open(clean_word, "r") as f:
                    console.print(f"[cyan]Reading error trace from file: {clean_word}[/cyan]")
                    file_content = f.read()
                    error_msg = f"{error_msg}\n\n--- Log File Content ---\n{file_content}"
            except Exception as e:
                console.print(f"[red]Error reading file '{clean_word}': {e}[/red]")
                raise typer.Exit(1)
            break

    # Auto-detect function from log file if not provided
    if is_log_file and not module_location:
        try:
            log_hash = Path(log_file_path).stem
            if len(log_hash) == 64 and all(c in '0123456789abcdef' for c in log_hash):
                if not manifest_path.exists():
                    console.print(
                        f"[yellow]Manifest '{manifest_path}' not found. "
                        f"Cannot auto-detect module from log.[/yellow]"
                    )
                else:
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                    for spec in manifest:
                        if hash_spec(spec) == log_hash:
                            module_location = spec.get("location")
                            if module_location:
                                console.print(
                                    f"[cyan]Auto-detected module "
                                    f"'[bold]{module_location}[/bold]' from log file.[/cyan]"
                                )
                                break
        except Exception as e:
            logger.warning(f"Could not auto-detect module from log file: {e}")

    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Using manifest: {manifest_path}")
    console.print(Panel(f"[bold red]Debugging Error:[/bold red]\n{error_msg}"))

    if module_location:
        if not manifest_path.exists():
            console.print(f"[red]Manifest '{manifest_path}' not found.[/red]")
            raise typer.Exit(1)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        spec = next(
            (s for s in manifest if s.get("location") == module_location), None
        )

        if not spec:
            console.print(f"[red]Module '{module_location}' not found in manifest.[/red]")
            raise typer.Exit(1)

        console.print(
            f"[cyan]Attempting to fix {module_location} based on the error...[/cyan]"
        )

        async def run_fix(error_to_fix: str):
            status = {
                module_location: {
                    "status": "Starting Fix...",
                    "iterations": 0,
                    "hash": "debug",
                    "log_file": os.path.join(base_dir, "debug.log"),
                }
            }
            sem = asyncio.Semaphore(1)
            lock = asyncio.Lock()
            console.print("[yellow]Running fix workflow with provided error...[/yellow]")
            await process_module(
                module_location, spec, status, sem, lock, config_path,
                initial_error=error_to_fix,
            )
        asyncio.run(run_fix(error_msg))

    else:

        console.print("[cyan]Analyzing error trace...[/cyan]")
        llm = get_llm("chat", config_path)
        model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
        llm.callbacks = [
            TokenLoggingCallbackHandler(
                role="command_debug", model_name=model_name, base_dir=base_dir
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert debugger. Analyze the error provided. "
                "Explain the likely cause and suggest which file or module "
                "is likely responsible."
            )),
            ("human", "{input}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"input": error_msg})

        console.print(
            Panel(extract_content(response.content), title="Debug Analysis", border_style="green")
        )


def configure_command(app: typer.Typer) -> None:
    """Register the debug command on the given typer app."""
    app.command()(debug)
