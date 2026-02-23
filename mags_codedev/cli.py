import os
import json
import asyncio
import git
import shutil
import subprocess
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from rich.live import Live
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Import our MAGs-CodeDev modules
from mags_codedev.state import FunctionState
from mags_codedev.graph import build_function_graph
from mags_codedev.utils.db import init_db, is_function_built, mark_function_built, get_token_summary
from mags_codedev.utils.git_ops import create_parallel_worktree, merge_and_cleanup_worktree
from mags_codedev.utils.config_parser import load_config, get_llm
from mags_codedev.utils.logger import logger

app = typer.Typer(
    help="MAGs-CodeDev: Multi-Agent Graph System for Code Development",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# -------------------------------------------------------------------
# HELPER: Dynamic Table Generation for the UI
# -------------------------------------------------------------------
def generate_status_table(status_dict: dict) -> Table:
    """Generates a rich table tracking the live status of parallel builds."""
    table = Table(title="Parallel Function Builder", show_header=True, header_style="bold cyan")
    table.add_column("Function", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Iterations", justify="center")

    for func_name, info in status_dict.items():
        state_color = "yellow"
        if "Success" in info['status']:
            state_color = "green"
        elif "Fail" in info['status'] or "Error" in info['status']:
            state_color = "red"
            
        table.add_row(
            func_name, 
            f"[{state_color}]{info['status']}[/{state_color}]", 
            str(info['iterations'])
        )
    return table

# -------------------------------------------------------------------
# ASYNC WORKER: Processes a single function through LangGraph
# -------------------------------------------------------------------
async def process_function(func_name: str, spec: dict, status_dict: dict, semaphore: asyncio.Semaphore, git_lock: asyncio.Lock, config_path: Path, initial_error: str = None):
    """Handles the full lifecycle of a single function generation in isolation."""
    async with semaphore:
        status_dict[func_name] = {"status": "Initializing...", "iterations": 0}
        branch_name = f"feature/{func_name}"
        worktree_path = None # Initialize for error handling scope
        
        try:
            status_dict[func_name]["status"] = "Creating Git Worktree..."
            # 1. Create isolated Git worktree (serialized to prevent index.lock contention)
            async with git_lock:
                worktree_path = await asyncio.to_thread(create_parallel_worktree, branch_name)
            
            # Copy requirements.txt to worktree if it exists in root but not in worktree
            # This ensures dependencies are available for Docker tests even if not committed to main
            req_file = "requirements.txt"
            if os.path.exists(req_file):
                dest_req = os.path.join(worktree_path, req_file)
                if not os.path.exists(dest_req):
                    shutil.copy2(req_file, dest_req)

            # Load existing code from the worktree if it exists, to allow for updates.
            existing_code = ""
            existing_tests = ""
            code_path = os.path.join(worktree_path, spec['location'])

            # Security check to prevent path traversal attacks from manifest.json
            abs_worktree_path = os.path.abspath(worktree_path)
            abs_code_path = os.path.abspath(code_path)
            if not abs_code_path.startswith(abs_worktree_path):
                raise PermissionError(f"Path traversal detected: Location '{spec['location']}' is outside the project worktree.")

            # Generate a robust test path mirroring source structure to avoid collisions
            # e.g., src/utils/pricing.py -> tests/src/utils/test_pricing.py
            location_path = Path(spec['location'])
            test_filename = f"test_{location_path.name}"
            # Calculate relative path for state and absolute path for file ops
            relative_test_path = os.path.join("tests", location_path.parent, test_filename)
            test_path = os.path.join(worktree_path, relative_test_path)

            abs_test_path = os.path.abspath(test_path)
            if not abs_test_path.startswith(abs_worktree_path):
                raise PermissionError(f"Path traversal detected: Test location for '{spec['location']}' is outside the project worktree.")

            if os.path.exists(code_path):
                with open(code_path, 'r') as f:
                    existing_code = f.read()
            
            if os.path.exists(test_path):
                with open(test_path, 'r') as f:
                    existing_tests = f.read()

            # 2. Initialize LangGraph State
            initial_state: FunctionState = {
                "function_name": func_name,
                "spec": spec,
                "config_path": config_path,
                "worktree_path": worktree_path,
                "test_location": relative_test_path,
                "code": existing_code,
                "tests": existing_tests,
                "test_results": "",
                "lint_results": "",
                "review_comments": [],
                "error_summary": initial_error or "",
                "iteration_count": 1 if initial_error else 0,
                "max_iterations": 5, # Prevent infinite loops
                "status": "in_progress"
            }
            
            # 3. Compile and Run the Graph
            status_dict[func_name]["status"] = "Running Multi-Agent Graph..."
            graph = build_function_graph()
            
            # Execute the state machine asynchronously
            final_state = await graph.ainvoke(initial_state)
            
            # Update UI with final iteration count
            status_dict[func_name]["iterations"] = final_state.get("iteration_count", 0)
            
            # Determine success based on the state
            if final_state.get("status") == "success":
                status_dict[func_name]["status"] = "Success: Tests & Reviews Passed"
                success = True
            else:
                status_dict[func_name]["status"] = "Failed: Max Iterations Reached"
                success = False

            # If successful, write the code to file and commit it in the worktree
            if success:
                status_dict[func_name]["status"] = "Committing to Branch..."
                output_path = os.path.join(worktree_path, spec['location'])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(final_state['code'])
                
                # Use the robust test path generated earlier
                test_output_path = test_path
                os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
                with open(test_output_path, "w") as f:
                    f.write(final_state['tests'])

                # Commit the changes in the worktree's branch
                repo = git.Repo(worktree_path)
                relative_test_path = os.path.relpath(test_output_path, worktree_path)
                repo.index.add([spec['location'], relative_test_path])
                repo.index.commit(f"feat: Implement function '{func_name}' via MAGs-CodeDev")
            # 4. Merge and Cleanup
            status_dict[func_name]["status"] = "Merging Branch..." if success else "Cleaning up Failed Branch..."
            
            merge_succeeded = False
            # Use lock to ensure only one thread performs git merge operations on the main repo at a time
            async with git_lock:
                merge_succeeded = await asyncio.to_thread(merge_and_cleanup_worktree, branch_name, worktree_path, success)
            
            # 5. Mark as complete in DB only if the merge was successful
            if merge_succeeded:
                await asyncio.to_thread(mark_function_built, func_name, spec)
                status_dict[func_name]["status"] = "Success: Merged to Main"
            elif success: # This means the graph succeeded but the merge failed
                status_dict[func_name]["status"] = "Failed: Merge Conflict"

        except Exception as e:
            logger.error(f"Error processing {func_name}: {str(e)}")
            status_dict[func_name]["status"] = f"Error: {str(e)[:20]}..."
            if worktree_path:
                async with git_lock:
                    await asyncio.to_thread(merge_and_cleanup_worktree, branch_name, worktree_path, False)

# -------------------------------------------------------------------
# COMMANDS
# -------------------------------------------------------------------

@app.command()
def init(
    manifest_path: Path = typer.Option(
        "manifest.json", "--manifest", "-m", help="Path to create the manifest JSON file."
    ),
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to create the configuration YAML file."
    )
):
    """Initialize the workspace, create AGENT.md, and intelligently build manifest.json."""
    console.print(Panel("[bold cyan]Initializing MAGs-CodeDev Workspace...[/bold cyan]"))
    
    # 1. Interactive Setup
    project_name = typer.prompt("Project Name", default=os.path.basename(os.getcwd()))
    language = typer.prompt("Programming Language", default="Python")

    # 2. Gitignore
    gitignore_content = "\n# MAGS-CodeDev\n*.log\n*.sqlite3\nmags_cache.db\nconfig.yaml\n.worktree_*/\n"
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            current_content = f.read()
        if "mags_cache.db" not in current_content:
            with open(".gitignore", "a") as f:
                f.write(gitignore_content)
            console.print("[green]Updated .gitignore[/green]")
    else:
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        console.print("[green]Created .gitignore[/green]")

    # 3. Database
    init_db()
    console.print("[green]Initialized SQLite Database[/green]")
    
    # 4. AGENT.md
    if not os.path.exists("AGENT.md"):
        with open("AGENT.md", "w") as f:
            f.write(f"# Agent Instructions for {project_name}\n\n")
            f.write(f"Language: {language}\n")
            f.write("Follow standard coding conventions and best practices.\n")
            if language.lower() == "python":
                f.write("Use PEP 8 standards. Include type hints.\n")
        console.print("[green]Created AGENT.md[/green]")

    # 5. requirements.txt
    if not os.path.exists("requirements.txt"):
        Path("requirements.txt").touch()
        console.print("[green]Created empty requirements.txt for project dependencies.[/green]")

    # 6. Manifest
    # Prompt the user to set up their manifest manually for now
    if not manifest_path.exists():
        dummy_manifest = {
            "calculate_discount": {
                "purpose": "Calculates a final price given a base price and a discount percentage.",
                "input": "price: float, discount: float",
                "output": "float",
                "location": "src/pricing.py"
            }
        }
        with open(manifest_path, "w") as f:
            json.dump(dummy_manifest, f, indent=4)
        console.print(f"[green]Created dummy {manifest_path}. Please edit this to define your functions.[/green]")

    # 7. Config
    if not config_path.exists():
        default_config = """api_keys:
  openai: "YOUR_KEY_HERE"
models:
  coder:
    provider: "openai"
    model: "gpt-4o"
  tester:
    provider: "openai"
    model: "gpt-4o"
  log_checker:
    provider: "openai"
    model: "gpt-4o"
  chat:
    provider: "openai"
    model: "gpt-4o"
  reviewers:
    - provider: "openai"
      model: "gpt-4o"
"""
        with open(config_path, "w") as f:
            f.write(default_config)
        console.print(f"[green]Created default {config_path}. Please update your API keys.[/green]")

    console.print("[bold green]✓ Initialization complete. Edit manifest.json, then run `mags-codedev build`.[/bold green]")


@app.command()
def build(
    manifest_path: Path = typer.Option(
        "manifest.json", "--manifest", "-m", help="Path to the manifest JSON file.", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to the configuration YAML file.", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    )
):
    """Build all pending functions in manifest.json using parallel multi-agent LangGraphs."""
    console.print(Panel("[bold magenta]Starting Multi-Agent Build Process...[/bold magenta]"))
    
    if not manifest_path.exists():
        console.print("[red]Error: manifest.json not found. Run `mags-codedev init` first.[/red]")
        raise typer.Exit(1)
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = load_config(config_path)
    max_parallel = config.get("settings", {}).get("max_parallel_functions", 4)
    
    # Check DB for already completed functions
    init_db()
    pending_functions = {}
    for name, spec in manifest.items():
        if not is_function_built(spec):
            pending_functions[name] = spec
            
    if not pending_functions:
        console.print("[green]All functions in manifest are already built and verified![/green]")
        raise typer.Exit(0)

    console.print(f"[cyan]Found {len(pending_functions)} pending functions. Building up to {max_parallel} concurrently...[/cyan]")

    # Run the async loop
    async def run_builds():
        status_dict = {name: {"status": "Pending...", "iterations": 0} for name in pending_functions}
        semaphore = asyncio.Semaphore(max_parallel)
        git_lock = asyncio.Lock() # Prevents race conditions during merge
        
        with Live(generate_table=lambda: generate_status_table(status_dict), refresh_per_second=4) as live:
            tasks = [
                process_function(name, spec, status_dict, semaphore, git_lock, config_path) 
                for name, spec in pending_functions.items()
            ]
            await asyncio.gather(*tasks)
            
    asyncio.run(run_builds())
    console.print("[bold green]Build cycle complete![/bold green]")


@app.command()
def debug(
    error_msg: str = typer.Argument(..., help="The error trace to fix"),
    function_name: str = typer.Option(None, "--function", "-f", help="The function name in manifest.json to apply the fix to."),
    manifest_path: Path = typer.Option(
        "manifest.json", "--manifest", "-m", help="Path to the manifest JSON file."
    ),
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to the configuration YAML file.", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    )
):
    """Pass an error trace or bug description to the LLM for automatic fixing."""
    console.print(Panel(f"[bold red]Debugging Error:[/bold red]\n{error_msg}"))
    
    # Scenario A: Function name provided -> Run the Graph to fix it
    if function_name:
        if not manifest_path.exists():
            console.print("[red]Manifest not found.[/red]")
            raise typer.Exit(1)
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        if function_name not in manifest:
            console.print(f"[red]Function '{function_name}' not found in manifest.[/red]")
            raise typer.Exit(1)
            
        spec = manifest[function_name]
        console.print(f"[cyan]Attempting to fix {function_name} based on the error...[/cyan]")
        
        async def run_fix(error_to_fix: str):
            # Create a dummy status dict for the helper
            status = {function_name: {"status": "Starting Fix...", "iterations": 0}}
            sem = asyncio.Semaphore(1)
            lock = asyncio.Lock()
            console.print("[yellow]Running fix workflow with provided error...[/yellow]")
            await process_function(function_name, spec, status, sem, lock, config_path, initial_error=error_to_fix)
            
        asyncio.run(run_fix(error_msg))
        
    # Scenario B: No function name -> Analyze error and suggest strategy
    else:
        console.print("[cyan]Analyzing error trace...[/cyan]")
        llm = get_llm("chat", config_path)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert debugger. Analyze the error provided. Explain the likely cause and suggest which file or function is likely responsible."),
            ("human", "{input}")
        ])
        chain = prompt | llm
        response = chain.invoke({"input": error_msg})
        console.print(Panel(response.content, title="Debug Analysis", border_style="green"))


@app.command()
def chat(
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to the configuration YAML file.", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    )
):
    """Freely chat with the LLM about the codebase. Can read/write files."""
    from mags_codedev.agents.chat_agent import start_chat_repl
    
    console.print("[bold blue]Entering Chat Mode (Type 'exit' to quit)...[/bold blue]")
    agent_executor = start_chat_repl(config_path=config_path)
    
    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            logger.info(f"Chat Input: {user_input}")
            
            # Use invoke to execute the agent tool loop
            response = agent_executor.invoke({"input": user_input})
            
            logger.info(f"Chat Final Answer: {response['output']}")
            console.print(f"\n[blue]Agent>[/blue] {response['output']}\n")
            
        except (KeyboardInterrupt, EOFError):
            break
            
    console.print("\n[blue]Chat ended.[/blue]")


@app.command()
def tokens():
    """Display a rich table of token usage and costs across all models and runs."""
    init_db() # Ensure DB and table exist
    summary, total = get_token_summary()

    table = Table(
        title="MAGs-CodeDev Token Usage Summary", 
        show_header=True, 
        header_style="bold cyan",
        show_footer=True,
        footer_style="bold"
    )
    table.add_column("Agent/Role", style="green", footer="Total")
    table.add_column("Model", style="yellow")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Total Tokens", justify="right", footer=f"{total[0] + total[1]:,}")
    
    if not summary:
        console.print("[yellow]No token usage has been recorded yet.[/yellow]")
        return

    for role, model, in_tokens, out_tokens in summary:
        table.add_row(
            role,
            model,
            f"{in_tokens:,}",
            f"{out_tokens:,}",
            f"{in_tokens + out_tokens:,}"
        )
    
    console.print(table)


@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation.")
):
    """Remove all generated cache files, logs, and temporary git worktrees."""
    console.print(Panel("[bold yellow]Cleaning up MAGs-CodeDev artifacts...[/bold yellow]"))

    # Find artifacts to delete
    db_file = Path("mags_cache.db")
    log_file = Path("mags-codedev_workflow.log")
    worktree_dirs = [d for d in os.listdir('.') if d.startswith(".worktree_") and os.path.isdir(d)]

    items_to_delete = []
    if db_file.exists():
        items_to_delete.append(db_file)
    if log_file.exists():
        items_to_delete.append(log_file)
    items_to_delete.extend([Path(d) for d in worktree_dirs])

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

    console.print("") # for a newline
    for d in worktree_dirs:
        # This command tells git to forget about the worktree and removes the directory
        subprocess.run(["git", "worktree", "remove", "--force", d], check=False, capture_output=True)
        console.print(f"Removed worktree: {d}")

    if db_file.exists(): os.remove(db_file); console.print(f"Deleted database: {db_file}")
    if log_file.exists(): os.remove(log_file); console.print(f"Deleted log file: {log_file}")
    
    console.print("\n[bold green]✓ Cleanup complete.[/bold green]")

if __name__ == "__main__":
    app()