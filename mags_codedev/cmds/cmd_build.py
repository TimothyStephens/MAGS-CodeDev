"""Build command: parallel multi-agent module building via LangGraph."""

import os
import json
import asyncio
import shutil
import fcntl
import typer
import git
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.live import Live

from mags_codedev.state import ModuleState
from mags_codedev.graph import build_function_graph
from mags_codedev.utils.db import (
    init_db, is_function_built, mark_function_built,
    hash_spec,
    add_iterations_to_module, get_total_iterations,
    save_artifact, load_artifact, load_dependency_codes, TokenCounter,
)
from mags_codedev.utils.cli_common import (
    resolve_base_dir,
    find_default_config_path,
    _CONFIG_HELP_TEXT,
)
from mags_codedev.utils.logger import setup_logger, logger, get_function_logger
from mags_codedev.utils.config_parser import load_config
from mags_codedev.utils.display import generate_status_table
from mags_codedev.utils.git_ops import (
    validate_git_repo, create_parallel_worktree, merge_and_cleanup_worktree,
)
from mags_codedev.backends import get_backend

console = Console()
async def process_module(
    module_location: str,
    spec: dict,
    status_dict: dict,
    semaphore: asyncio.Semaphore,
    git_lock: asyncio.Lock,
    config_path: Path,
    initial_error: str | None = None,
    force_fresh: bool = False,
):
    """Handles the full lifecycle of a single module generation in isolation."""
    worktree_path = None
    branch_name = f"feature/{module_location}"
    func_logger = None
    log_filepath = None

    try:
        func_hash = hash_spec(spec)
        base_dir = resolve_base_dir(config_path)
        log_filepath = os.path.abspath(
            os.path.join(base_dir, "logs", f"{func_hash}.log")
        )

        func_logger = get_function_logger(func_hash, base_dir=base_dir)

        async with semaphore:
            status_dict[module_location] = {
                "status": "Initializing...",
                "iterations": 0,
                "hash": func_hash,
                "log_file": log_filepath,
                "worktree": None,
                "step": "Init",
            }

            # Validate spec
            if 'location' not in spec:
                raise ValueError(
                    f"Manifest entry for '{module_location}' is missing required field: 'location'"
                )

            # 1. Create isolated Git worktree
            status_dict[module_location]["status"] = "Creating Git Worktree..."
            async with git_lock:
                worktree_path = await asyncio.to_thread(
                    create_parallel_worktree, branch_name, base_dir=base_dir, force_fresh=force_fresh
                )
            status_dict[module_location]["worktree"] = worktree_path

            # Copy deps file to worktree
            backend = get_backend(config_path)
            req_file = backend.deps_filename
            if os.path.exists(req_file):
                dest_req = os.path.join(worktree_path, req_file)
                if not os.path.exists(dest_req):
                    shutil.copy2(req_file, dest_req)

            # Load existing code from worktree
            existing_code = ""
            existing_tests = ""
            code_path = os.path.join(worktree_path, spec['location'])

            abs_worktree_path = os.path.abspath(worktree_path)
            abs_code_path = os.path.abspath(code_path)
            if not abs_code_path.startswith(abs_worktree_path):
                raise PermissionError(
                    f"Path traversal detected: Location '{spec['location']}' is outside the project worktree."
                )

            # Generate robust test path
            location_path = Path(spec['location'])
            test_filename = f"test_{location_path.name}"
            relative_test_path = os.path.join("tests", location_path.parent, test_filename)
            test_path = os.path.join(worktree_path, relative_test_path)

            abs_test_path = os.path.abspath(test_path)
            if not abs_test_path.startswith(abs_worktree_path):
                raise PermissionError(
                    f"Path traversal detected: Test location for '{spec['location']}' is outside the project worktree."
                )

            if os.path.exists(code_path):
                with open(code_path, 'r') as f:
                    existing_code = f.read()

            if os.path.exists(test_path):
                with open(test_path, 'r') as f:
                    existing_tests = f.read()

            # Load artifact cache as fallback
            artifact_data = load_artifact(module_location, base_dir=base_dir)
            artifact_code = artifact_data.get("code") if artifact_data else ""
            artifact_tests = artifact_data.get("tests") if artifact_data else ""
            artifact_code_hash = artifact_data.get("code_hash") if artifact_data else None
            artifact_test_hash = artifact_data.get("test_hash") if artifact_data else None

            if not existing_code and artifact_code:
                existing_code = artifact_code
            if not existing_tests and artifact_tests:
                existing_tests = artifact_tests
            # Load dependency source code for LLM context
            dependency_locations = spec.get("dependencies", [])
            dependency_code = load_dependency_codes(
                dependency_locations, worktree_path, base_dir=base_dir
            )

            # Load project-level instructions from AGENT.md
            agent_md_path = Path(base_dir) / "AGENT.md"
            project_instructions = ""
            if agent_md_path.exists():
                with open(agent_md_path, "r") as f:
                    project_instructions = f.read().strip()
            # Load config settings
            config = load_config(config_path)
            settings = config.get("settings", {})
            max_test_fix_iterations = settings.get("max_test_fix_iterations", 5)
            max_review_rounds = settings.get("max_review_rounds", 3)

            # Load iteration history from DB for continuity
            previous_iterations = get_total_iterations(module_location, base_dir=base_dir)
            # Session number: count how many times we've tried this module
            # We track this via a session counter file
            session_counter_file = os.path.join(
                base_dir, "sessions", f"{func_hash}.session"
            )
            session_dir = os.path.join(base_dir, "sessions")
            os.makedirs(session_dir, exist_ok=True)
            session_number = 1
            try:
                with open(session_counter_file, "r+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        content = f.read().strip()
                        if content:
                            try:
                                session_number = int(content) + 1
                            except ValueError:
                                session_number = 1
                        f.seek(0)
                        f.truncate()
                        f.write(str(session_number))
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except FileNotFoundError:
                with open(session_counter_file, "w") as f:
                    f.write("1")

            if previous_iterations:
                func_logger.info(
                    f"Continuing from {previous_iterations} previous iterations "
                    f"(session {session_number})"
                )

            # 2. Initialize LangGraph State
            initial_state: ModuleState = {
                "backend": backend,
                "spec": spec,
                "module_location": module_location,
                "log_filepath": log_filepath,
                "config_path": config_path,
                "worktree_path": worktree_path,
                "test_location": relative_test_path,
                "dependency_code": dependency_code,
                "project_instructions": project_instructions,

                "code": existing_code,
                "tests": existing_tests,

                "test_results": "",
                "lint_results": "",
                "test_error_summary": initial_error or "",
                "review_comments": [],
                "error_location": None,

                "previous_code_hash": artifact_code_hash,
                "previous_test_hash": artifact_test_hash,
                "iteration_count": 1 if initial_error else 0,
                "max_test_fix_iterations": max_test_fix_iterations,
                "max_review_rounds": max_review_rounds,
                "review_round_count": 0,

                "base_dir": base_dir,
                "status": "in_progress",

                # Session tracking (lifecycle logging)
                "_previous_iterations": previous_iterations or 0,
                "_session_number": session_number,
            }

            # 3. Compile and Run the Graph
            status_dict[module_location]["status"] = "Running Multi-Agent Graph..."
            graph = graph_cache

            # Token tracking for this module
            token_counter = TokenCounter()
            graph_config = {"callbacks": [token_counter]}

            final_state = initial_state.copy()
            async for event in graph.astream(initial_state, config=graph_config):
                for node_name, state_update in event.items():
                    status_dict[module_location]["step"] = node_name
                    if state_update and "iteration_count" in state_update:
                        status_dict[module_location]["iterations"] = state_update["iteration_count"]
                    if state_update:
                        final_state.update(state_update)

            # Record tokens used
            tokens = token_counter.total
            status_dict[module_location]["tokens_in"] = tokens["in"]
            status_dict[module_location]["tokens_out"] = tokens["out"]
            status_dict[module_location]["step"] = "Complete"

            # Persist iterations
            iterations_this_run = final_state.get("iteration_count", 0)
            if iterations_this_run > 0:
                func_logger.info(
                    f"Run finished with {iterations_this_run} iterations. Updating total in database."
                )
                await asyncio.to_thread(
                    add_iterations_to_module, module_location, iterations_this_run, base_dir=base_dir
                )

            # Save artifacts regardless of success/failure
            if final_state.get("code") or final_state.get("tests"):
                await asyncio.to_thread(
                    save_artifact,
                    location=module_location,
                    code=final_state.get("code", ""),
                    tests=final_state.get("tests", ""),
                    spec_hash=func_hash,
                    base_dir=base_dir,
                )

            # Determine success
            max_test_iters = final_state.get("max_test_fix_iterations", 5)
            max_test_reached = (
                max_test_iters > 0 and final_state.get("iteration_count", 0) >= max_test_iters
            )
            max_review_reached = (
                final_state.get("max_review_rounds", 3) > 0
                and final_state.get("review_round_count", 0) >= final_state.get("max_review_rounds", 3)
            )
            is_failed = final_state.get("status") == "failed"

            success = not (max_test_reached or max_review_reached or is_failed)

            if success:
                status_dict[module_location]["status"] = "Success: Tests & Reviews Passed"
            else:
                if is_failed:
                    status_dict[module_location]["status"] = "Failed: Coder Convergence"
                elif max_test_reached:
                    status_dict[module_location]["status"] = "Failed: Max Test Fix Iterations"
                else:
                    status_dict[module_location]["status"] = "Failed: Max Review Rounds"

                warn_msg = (
                    f"Workflow for '{module_location}' aborted: {status_dict[module_location]['status']}."
                )
                logger.warning(warn_msg)
                func_logger.warning(warn_msg)
                func_logger.warning(
                    f"Final Error Summary: {final_state.get('test_error_summary', 'None')}"
                )

            # Write code/tests to worktree only on success (needed for git commit)
            test_output_path = test_path
            if success and final_state.get('code'):
                output_path = os.path.join(worktree_path, spec['location'])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(final_state['code'])

                os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
                if final_state.get('tests'):
                    with open(test_output_path, "w") as f:
                        f.write(final_state.get('tests'))
            if success:
                status_dict[module_location]["status"] = "Committing to Branch..."
                repo = git.Repo(worktree_path)
                relative_test_path_rel = os.path.relpath(test_output_path, worktree_path)
                repo.index.add([spec['location'], relative_test_path_rel])
                repo.index.commit(
                    f"feat: Implement module '{module_location}' via MAGs-CodeDev"
                )

            # Merge and Cleanup
            status_dict[module_location]["status"] = (
                "Merging Branch..." if success else "Cleaning up Failed Branch..."
            )

            merge_succeeded = False
            async with git_lock:
                merge_succeeded = await asyncio.to_thread(
                    merge_and_cleanup_worktree,
                    branch_name, worktree_path, success, base_dir=base_dir
                )

            if merge_succeeded:
                await asyncio.to_thread(
                    mark_function_built, module_location, spec, base_dir=base_dir
                )
                status_dict[module_location]["status"] = "Success: Merged to Main"
            elif success:
                status_dict[module_location]["status"] = "Failed: Merge Conflict"

    except Exception as e:
        effective_logger = func_logger if func_logger else logger
        effective_logger.exception(f"Error processing {module_location}")

        logger.error(f"Error processing {module_location} (see {log_filepath}): {e}")
        status_dict[module_location]["status"] = f"Error: {str(e)}"

        if worktree_path:
            async with git_lock:
                await asyncio.to_thread(
                    merge_and_cleanup_worktree,
                    branch_name, worktree_path, False, base_dir=resolve_base_dir(config_path)
                )


def build(
    manifest_path: Optional[Path] = typer.Option(
        None, "--manifest", "-m",
        help=(
            "Path to the manifest JSON file. "
            "Default: <base_dir>/manifest.json "
            "(e.g., .mags-codedev/manifest.json)"
        ),
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True,
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation",
        help="Skip the pre-build connection check.",
    ),
    force_fresh: bool = typer.Option(
        False, "--force-fresh",
        help="Force a fresh build by deleting existing worktrees and branches.",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True,
        help="Verbosity level (0=info, 1=debug, 2=trace).",
    ),
):
    """Build all pending modules in the manifest using parallel multi-agent LangGraphs."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    # Resolve base dir and setup logging
    base_dir = resolve_base_dir(config_path)

    # Default manifest to <base_dir>/manifest.json if not specified
    if manifest_path is None:
        manifest_path = Path(base_dir) / "manifest.json"
    manifest_path = Path(manifest_path)
    # Map verbose level: 0=info, 1=debug, 2=trace
    verbose_levels = {0: "info", 1: "debug", 2: "trace"}
    log_level = verbose_levels.get(verbose, "info")
    setup_logger(base_dir=base_dir, log_level=log_level)

    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Using manifest: {manifest_path}")

    console.print(Panel("[bold magenta]Starting Multi-Agent Build Process...[/bold magenta]"))

    # Validate Git Repo
    try:
        validate_git_repo()
    except RuntimeError as e:
        console.print(f"[bold red]Git Error:[/bold red] {e}")
        raise typer.Exit(1)

    if not skip_validation:
        from mags_codedev.utils.cli_common import validate_config_connections
        if not validate_config_connections(config_path):
            console.print("[bold red]Validation failed. Aborting build.[/bold red]")
            console.print(
                "Use --skip-validation to force execution if you believe this is an error."
            )
            raise typer.Exit(1)

    if not manifest_path.exists():
        console.print(
            f"[red]Error: Manifest file not found at '{manifest_path}'. "
            f"Run `mags-codedev init` first.[/red]"
        )
        raise typer.Exit(1)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = load_config(config_path)
    max_parallel = config.get("settings", {}).get("max_parallel_modules", 4)

    init_db(base_dir=base_dir)

    module_map = {
        spec['location']: spec for spec in manifest if 'location' in spec
    }
    built_modules = (
        set()
        if force_fresh
        else {
            loc for loc, spec in module_map.items()
            if is_function_built(spec, base_dir=base_dir)
        }
    )

    if len(built_modules) == len(module_map):
        console.print(
            "[green]All modules in manifest are already built and verified![/green]"
        )
        raise typer.Exit(0)

    console.print(
        f"[cyan]Found {len(module_map)} total modules. {len(built_modules)} already built.[/cyan]"
    )

    async def run_builds():
        failed_modules: set = set()
        # Cache the graph to avoid rebuilding it for each module
        graph_cache = build_function_graph()

        status_dict = {}
        for loc, spec in module_map.items():
            # Check for artifact data (previous attempt)
            artifact_data = load_artifact(loc, base_dir=base_dir)
            status_dict[loc] = {
                "status": "Pending",
                "iterations": 0,
                "hash": hash_spec(spec),
                "step": "-",
                "tokens_in": 0,
                "tokens_out": 0,
                "has_artifact": artifact_data is not None,
                "artifact_hash": (
                    artifact_data.get("code_hash", "")
                    if artifact_data
                    else ""
                ),
            }

        def done_count() -> int:
            """Count modules that are finished, failed, or blocked."""
            return len(built_modules) + len(failed_modules)
        semaphore = asyncio.Semaphore(max_parallel)
        git_lock = asyncio.Lock()

        while done_count() < len(module_map):
            # Refresh completed module status
            for loc in built_modules:
                status_dict[loc]['status'] = "Completed"
                status_dict[loc]['step'] = "Done"
                status_dict[loc]['iterations'] = get_total_iterations(loc, base_dir=base_dir)

            # Classify unbuilt modules
            remaining = {
                loc: spec
                for loc, spec in module_map.items()
                if loc not in built_modules and loc not in failed_modules
            }

            # Modules that can build now (all deps satisfied)
            buildable_now = {
                loc: spec for loc, spec in remaining.items()
                if all(dep in built_modules for dep in spec.get("dependencies", []))
            }

            # Modules blocked by a failed dependency
            blocked = {
                loc: spec for loc, spec in remaining.items()
                if loc not in buildable_now
                and any(dep in failed_modules for dep in spec.get("dependencies", []))
            }

            # Mark blocked modules
            for loc in blocked:
                failing_deps = [
                    dep for dep in spec.get("dependencies", [])
                    if dep in failed_modules
                ]
                status_dict[loc]['status'] = f"Blocked: {', '.join(failing_deps)} failed"
                status_dict[loc]['step'] = "Blocked"

            # Modules waiting on not-yet-built deps
            for loc, spec in remaining.items():
                if loc not in buildable_now and loc not in blocked:
                    missing_deps = [
                        dep for dep in spec.get("dependencies", [])
                        if dep not in built_modules
                    ]
                    status_dict[loc]['status'] = f"Waiting: {', '.join(missing_deps)}"
                    status_dict[loc]['step'] = "Waiting"

            # If nothing is buildable and nothing is waiting, we're stuck
            if not buildable_now:
                console.print(generate_status_table(status_dict, module_map))
                if remaining and not blocked:
                    console.print(
                        "\n[bold red]Error: Circular dependency or missing dependency detected.[/bold red]"
                    )
                    for loc, spec in remaining.items():
                        missing_deps = [
                            dep for dep in spec.get("dependencies", [])
                            if dep not in built_modules
                        ]
                        if missing_deps:
                            console.print(
                                f"- [yellow]{loc}[/yellow] "
                                f"(missing: {', '.join(missing_deps)})"
                            )
                raise typer.Exit(1)


            with Live(
                generate_status_table(status_dict, module_map), refresh_per_second=4
            ) as live:
                async def update_ui_loop():
                    while True:
                        sorted_status = dict(sorted(status_dict.items()))
                        live.update(generate_status_table(sorted_status, module_map))
                        await asyncio.sleep(0.25)

                build_tasks = [
                    process_module(
                        loc, spec, status_dict, semaphore, git_lock, config_path,
                        force_fresh=force_fresh,
                    )
                    for loc, spec in buildable_now.items()
                ]
                ui_task = asyncio.create_task(update_ui_loop())
                await asyncio.gather(*build_tasks)
                ui_task.cancel()
                live.update(generate_status_table(status_dict, module_map))

            # Tally wave results
            wave_failures = []
            wave_successes = []
            for loc in buildable_now:
                info = status_dict[loc]
                if "Success" in info["status"]:
                    wave_successes.append(loc)
                else:
                    wave_failures.append(loc)

            built_modules.update(wave_successes)
            failed_modules.update(wave_failures)

            # Report failures but continue
            if wave_failures:
                console.print(
                    f"\n[bold red]Wave completed with {len(wave_failures)} failure(s):[/bold red]"
                )
                for loc in wave_failures:
                    info = status_dict[loc]
                    console.print(f"  - [red]{loc}[/red] — {info.get('status', 'Failed')}")
                    if info.get("worktree") and os.path.exists(info["worktree"]):
                        console.print(f"    Worktree: [blue]{info['worktree']}[/blue]")
                    console.print(f"    Log: [blue]{info.get('log_file', '')}[/blue]")
                    console.print(f"    Tokens: {info.get('tokens_in', 0) + info.get('tokens_out', 0):,}")

        # Final summary
        if failed_modules:
            console.print(
                f"\n[bold red]Build completed with {len(failed_modules)} failure(s).[/bold red]"
            )
            blocked_count = sum(
                1 for loc in module_map
                if any(dep in failed_modules for dep in module_map[loc].get("dependencies", []))
            )
            if blocked_count:
                console.print(
                    f"[bold yellow]{blocked_count} module(s) blocked by failed dependencies.[/bold yellow]"
                )
        else:
            console.print(
                "\n[bold green]Build cycle complete! All modules built successfully.[/bold green]"
            )

    asyncio.run(run_builds())


def configure_command(app: typer.Typer) -> None:
    """Register the build command on the given typer app."""
    app.command()(build)
