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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import our MAGs-CodeDev modules
from mags_codedev.state import FunctionState
from mags_codedev.graph import build_function_graph
from mags_codedev.utils.db import init_db, is_function_built, mark_function_built, get_token_summary
from mags_codedev.utils.git_ops import create_parallel_worktree, merge_and_cleanup_worktree
from mags_codedev.utils.config_parser import load_config, get_llm, get_reviewer_llms
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

def resolve_config_path(config_path: Path) -> Path:
    """Resolves the configuration path, falling back to the package default if local is missing."""
    if config_path.exists():
        return config_path
    
    # Check if we are looking for the default config.yaml
    if config_path.name == "config.yaml":
        package_config = Path(__file__).parent / "config.yaml"
        if package_config.exists():
            logger.debug(f"Local config not found. Using package config: {package_config}")
            return package_config
            
    return config_path

def validate_config_connections(config_path: Path) -> bool:
    """Verifies that the API keys and Models defined in config.yaml are valid and accessible."""
    console.print(Panel("[bold cyan]Validating LLM Connections...[/bold cyan]"))
    logger.debug("Starting validation of LLM connections.")
    
    all_passed = True
    roles = ["coder", "tester", "log_checker"]
    
    # Check primary agents
    for role in roles:
        try:
            llm = get_llm(role, config_path)
            console.print(f"Checking [bold]{role}[/bold] ({llm.model_name})...", end=" ")
            # Send a minimal token request to verify access
            llm.invoke([HumanMessage(content="Test")])
            console.print("[green]OK[/green]")
            logger.debug(f"Connection verified for {role} ({llm.model_name}).")
        except Exception as e:
            console.print("[red]FAILED[/red]")
            console.print(f"  [red]Error: {e}[/red]")
            logger.error(f"Connection failed for {role}: {e}")
            all_passed = False

    # Check reviewers
    try:
        reviewers = get_reviewer_llms(config_path)
        for i, llm in enumerate(reviewers):
            try:
                console.print(f"Checking [bold]Reviewer {i+1}[/bold] ({llm.model_name})...", end=" ")
                llm.invoke([HumanMessage(content="Test")])
                console.print("[green]OK[/green]")
                logger.debug(f"Connection verified for Reviewer {i+1} ({llm.model_name}).")
            except Exception as e:
                console.print("[red]FAILED[/red]")
                console.print(f"  [red]Error: {e}[/red]")
                logger.error(f"Connection failed for Reviewer {i+1}: {e}")
                all_passed = False
    except Exception as e:
        console.print(f"[red]Error loading reviewers: {e}[/red]")
        logger.error(f"Error loading reviewers: {e}")
        all_passed = False

    return all_passed

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
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Use AI to interactively design the project structure."
    )
):
    """Initialize the workspace, create AGENT.md, and intelligently build manifest.json."""
    console.print(Panel("[bold cyan]Initializing MAGs-CodeDev Workspace...[/bold cyan]"))
    logger.info("Starting workspace initialization.")
    
    # 1. Config Resolution
    # We no longer automatically create a local config.yaml.
    # We resolve to the package config unless the user explicitly provided a path that exists,
    # or if they want to override it (which they can do by manually creating one later).
    
    # If the user provided a specific path that doesn't exist, resolve_config_path will handle the fallback logic
    # or we can just let the subsequent commands use the resolved path.
    resolved_config_path = resolve_config_path(config_path)
    console.print(f"[cyan]Using configuration: {resolved_config_path}[/cyan]")

    # 2. Gitignore
    gitignore_content = "\n# MAGS-CodeDev\n*.log\n*.sqlite3\nmags_cache.db\nconfig.yaml\n.worktree_*/\n"
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            current_content = f.read()
        if "mags_cache.db" not in current_content:
            with open(".gitignore", "a") as f:
                f.write(gitignore_content)
            console.print("[green]Updated .gitignore[/green]")
            logger.info("Updated .gitignore with MAGS-CodeDev patterns.")
    else:
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        console.print("[green]Created .gitignore[/green]")
        logger.info("Created .gitignore file.")

    # 3. Database
    init_db()
    console.print("[green]Initialized SQLite Database[/green]")
    logger.info("Initialized SQLite database.")
    
    # 4. requirements.txt
    if not os.path.exists("requirements.txt"):
        Path("requirements.txt").touch()
        console.print("[green]Created empty requirements.txt for project dependencies.[/green]")
        logger.info("Created empty requirements.txt.")

    # 5. Interactive Setup (Manifest & AGENT.md)
    manifest_created = False
    agent_md_created = False

    if interactive:
        # Check for API Key
        config = load_config(resolved_config_path)
        chat_config = config.get("models", {}).get("chat", {})
        chat_provider = chat_config.get("provider", "openai")
        
        key_name_map = {
            "openai": "openai",
            "google": "gemini",
            "anthropic": "anthropic"
        }
        required_key_name = key_name_map.get(chat_provider, "openai")
        api_key = config.get("api_keys", {}).get(required_key_name)
        
        # A simple check for placeholder keys
        if not api_key or "..." in api_key or "YOUR_KEY" in api_key:
            console.print(f"[yellow]{required_key_name.capitalize()} API key missing or is a placeholder in config.yaml.[/yellow]")
            user_key = typer.prompt(f"Enter {required_key_name.capitalize()} API Key for AI setup (leave empty to skip AI)", default="", show_default=False, hide_input=True)
            if user_key:
                # If we are using the package config, we should NOT modify it directly.
                # Instead, we should copy it to the local directory and modify that.
                if resolved_config_path.parent == Path(__file__).parent:
                    console.print("[yellow]Using package configuration. Creating a local copy to save API key...[/yellow]")
                    shutil.copy2(resolved_config_path, "config.yaml")
                    resolved_config_path = Path("config.yaml")
                    logger.info("Created local config.yaml from package template to save API key.")

                # Safely update the (now potentially local) YAML file
                with open(resolved_config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                
                config_data.setdefault('api_keys', {})[required_key_name] = user_key

                with open(resolved_config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                console.print(f"[green]Updated {resolved_config_path} with API key.[/green]")
                logger.info(f"Updated {resolved_config_path} with provided {required_key_name} API key.")
            else:
                interactive = False

    if interactive:
        try:
            logger.info("Starting AI Architect Mode initialization.")
            llm = get_llm("chat", resolved_config_path)
            
            console.print(Panel("[bold green]AI Architect Mode[/bold green]\nDescribe your project idea, and I will generate the manifest and agent instructions.\n\n[bold]Commands:[/bold]\n- [bold]undo[/bold]: Remove the last interaction.\n- [bold]restart[/bold]: Clear history and start over.\n- [bold]exit[/bold]: Quit AI mode."))
            
            system_message = SystemMessage(content="""You are a Senior Software Architect. 
Your goal is to help the user define a software project.
1. Ask clarifying questions if the user's request is vague.
2. Propose a file structure and a list of functions.
3. Once the user agrees, output the final configuration in a specific JSON block.

The JSON block must look EXACTLY like this:
```json_output
{
    "agent_md": "The content for AGENT.md...",
    "manifest": {
        "function_name": {
            "purpose": "...",
            "input": "...",
            "output": "...",
            "location": "src/..."
        }
    }
}
```
Do not output the JSON block until the user explicitly confirms the plan.
""")
            messages = [system_message]
            
            while True:
                user_input = typer.prompt("You")
                cmd = user_input.lower().strip()
                
                if cmd in ["exit", "quit", "skip"]:
                    logger.info("User exited AI Architect mode manually.")
                    console.print("[yellow]Exiting AI Architect mode.[/yellow]")
                    break
                
                if cmd == "undo":
                    if len(messages) >= 3:
                        messages.pop() # Remove AI response
                        messages.pop() # Remove User input
                        console.print("[yellow]Undid last interaction.[/yellow]")
                        logger.info("User performed undo in AI Architect mode.")
                        if len(messages) > 1 and isinstance(messages[-1], AIMessage):
                            console.print(f"[blue]Architect (Previous):[/blue] {messages[-1].content}")
                    else:
                        console.print("[red]Nothing to undo.[/red]")
                        logger.info("User attempted undo with empty history.")
                    continue
                
                if cmd == "restart":
                    messages = [system_message]
                    console.print("[yellow]Restarting AI Architect session...[/yellow]")
                    logger.info("User restarted AI Architect session.")
                    continue
                
                logger.info(f"AI Architect User Input: {user_input}")
                messages.append(HumanMessage(content=user_input))
                
                with console.status("Architect is thinking...", spinner="dots"):
                    response = llm.invoke(messages)
                
                content = response.content
                messages.append(AIMessage(content=content))
                logger.debug(f"AI Architect Response Content: {content}")
                
                if "```json_output" in content:
                    # Extract JSON
                    try:
                        json_str = content.split("```json_output")[1].split("```")[0].strip()
                        data = json.loads(json_str)
                        
                        # Write AGENT.md
                        with open("AGENT.md", "w") as f:
                            f.write(data.get("agent_md", "# Agent Instructions"))
                        console.print("[green]Generated AGENT.md[/green]")
                        logger.info("Generated AGENT.md from AI response.")
                        agent_md_created = True
                        
                        # Write manifest.json
                        with open(manifest_path, "w") as f:
                            json.dump(data.get("manifest", {}), f, indent=4)
                        console.print(f"[green]Generated {manifest_path}[/green]")
                        logger.info(f"Generated {manifest_path} from AI response.")
                        manifest_created = True
                        
                        logger.info("AI Architect successfully created project configuration.")
                        console.print("[bold green]Project structure defined![/bold green]")
                        break
                    except Exception as e:
                        logger.error(f"Failed to parse AI Architect JSON output: {e}")
                        console.print(f"[red]Failed to parse AI output: {e}[/red]")
                        console.print("[blue]Architect:[/blue] " + content.replace("```json_output", ""))
                else:
                    console.print(f"[blue]Architect:[/blue] {content}")

        except Exception as e:
            logger.error(f"AI Architect Mode encountered an error: {e}")
            console.print(f"[red]AI Setup Error: {e}[/red]")
            console.print("[yellow]Falling back to manual setup.[/yellow]")

    # 6. Fallback / Manual Creation
    if not agent_md_created and not os.path.exists("AGENT.md"):
        project_name = os.path.basename(os.getcwd())
        with open("AGENT.md", "w") as f:
            f.write(f"# Agent Instructions for {project_name}\n\n")
            f.write("Language: Python\n")
            f.write("Follow standard coding conventions and best practices.\n")
            f.write("Use PEP 8 standards. Include type hints.\n")
        console.print("[green]Created default AGENT.md[/green]")
        logger.info("Created default AGENT.md.")

    if not manifest_created and not manifest_path.exists():
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
        logger.info(f"Created dummy manifest file at {manifest_path}.")

    console.print("[bold green]✓ Initialization complete. Run `mags-codedev build` to start coding.[/bold green]")
    logger.info("Initialization complete.")


@app.command()
def build(
    manifest_path: Path = typer.Option(
        "manifest.json", "--manifest", "-m", help="Path to the manifest JSON file.", exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to the configuration YAML file.", exists=False, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip the pre-build connection check."
    )
):
    """Build all pending functions in manifest.json using parallel multi-agent LangGraphs."""
    config_path = resolve_config_path(config_path)
    console.print(Panel("[bold magenta]Starting Multi-Agent Build Process...[/bold magenta]"))
    
    if not skip_validation:
        if not validate_config_connections(config_path):
            console.print("[bold red]Validation failed. Aborting build.[/bold red]")
            console.print("Use --skip-validation to force execution if you believe this is an error.")
            raise typer.Exit(1)

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
        "config.yaml", "--config", "-c", help="Path to the configuration YAML file.", exists=False, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    )
):
    """Pass an error trace or bug description to the LLM for automatic fixing."""
    config_path = resolve_config_path(config_path)
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
        "config.yaml", "--config", "-c", help="Path to the configuration YAML file.", exists=False, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    )
):
    """Freely chat with the LLM about the codebase. Can read/write files."""
    config_path = resolve_config_path(config_path)
    from mags_codedev.agents.chat_agent import start_chat_repl
    
    console.print("[bold blue]Entering Chat Mode (Type 'exit' to quit)...[/bold blue]")
    agent_graph = start_chat_repl(config_path=config_path)
    
    # Thread ID is required by LangGraph checkpointers to maintain conversation history
    config = {"configurable": {"thread_id": "cli-session"}}
    
    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            logger.debug(f"Chat Input: {user_input}")
            
            # Use invoke to execute the agent graph. Input is a list of messages.
            response = agent_graph.invoke({"messages": [("user", user_input)]}, config=config)
            
            # The response is the final state. We extract the content of the last message (the AI's reply).
            final_answer = response["messages"][-1].content
            
            logger.debug(f"Chat Final Answer: {final_answer}")
            console.print(f"\n[blue]Agent>[/blue] {final_answer}\n")
            
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


@app.command(name="list-models")
def list_models(
    config_path: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to the configuration YAML file.", exists=False, file_okay=True, dir_okay=False, readable=True, resolve_path=True
    )
):
    """List available models from the configured providers (OpenAI, Google, etc.)."""
    config_path = resolve_config_path(config_path)
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
            # List models
            models = client.models.list()
            # Filter for chat models to reduce noise
            gpt_models = [m.id for m in models.data if "gpt" in m.id]
            gpt_models.sort()
            for m in gpt_models:
                table.add_row("OpenAI", m)
        except Exception as e:
            table.add_row("OpenAI", f"[red]Error: {e}[/red]")

    # 2. Google (Gemini)
    if api_keys.get("gemini"):
        try:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_keys["gemini"])
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        name = m.name.replace("models/", "")
                        table.add_row("Google", name)
            except ImportError:
                table.add_row("Google", "[yellow]Skipped: 'google-generativeai' not installed[/yellow]")
        except Exception as e:
            table.add_row("Google", f"[red]Error: {e}[/red]")

    # 3. Anthropic
    if api_keys.get("anthropic"):
        # Anthropic does not support listing models via API yet.
        # We provide a static list of known recent models.
        known_models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        for m in known_models:
            table.add_row("Anthropic (Static)", m)

    # 4. Custom / Local (e.g. Ollama)
    # Scan config for custom_openai providers to find base_urls
    custom_urls = set()
    models_config = config.get("models", {})
    
    # Check roles
    for role in ["coder", "tester", "log_checker", "chat"]:
        m_cfg = models_config.get(role, {})
        if m_cfg.get("provider") == "custom_openai" and m_cfg.get("base_url"):
            custom_urls.add(m_cfg.get("base_url"))
            
    # Check reviewers list
    for r_cfg in models_config.get("reviewers", []):
        if r_cfg.get("provider") == "custom_openai" and r_cfg.get("base_url"):
            custom_urls.add(r_cfg.get("base_url"))

    for url in custom_urls:
        try:
            from openai import OpenAI
            # Use dummy key for local
            client = OpenAI(base_url=url, api_key="dummy")
            models = client.models.list()
            for m in models.data:
                table.add_row(f"Custom ({url})", m.id)
        except Exception as e:
            table.add_row(f"Custom ({url})", f"[red]Error: {e}[/red]")

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