import os
import re
import json
import asyncio
import yaml
import git
import shutil
import subprocess
import logging
import typer
from typing import Optional
from typing import Optional, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
from rich.live import Live
from rich.tree import Tree
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Import our MAGs-CodeDev modules
from mags_codedev.state import ModuleState
from mags_codedev.graph import build_function_graph
from mags_codedev.utils.db import init_db, is_function_built, mark_function_built, get_token_summary, TokenLoggingCallbackHandler, hash_spec
from mags_codedev.utils.git_ops import create_parallel_worktree, merge_and_cleanup_worktree, validate_git_repo
from mags_codedev.utils.config_parser import load_config, get_llm, get_reviewer_llms
from mags_codedev.utils.logger import logger

app = typer.Typer(
    help="MAGs-CodeDev: Multi-Agent Graph System for Code Development",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["--help", "-h"]},
)
console = Console()

def _find_package_config_path() -> Optional[Path]:
    """Helper to find the location of the config file shipped with the package."""
    here = Path(__file__).resolve().parent
    for candidate in [here / "config.template.yaml", here.parent / "config.template.yaml"]:
        if candidate.exists():
            return candidate
    return None

_PACKAGE_CONFIG_PATH = _find_package_config_path()
_CONFIG_HELP_TEXT = (
    "Path to the configuration YAML file. If 'config.yaml' is not found, it will be created from 'config.template.yaml'. "
    f"The package provides a template at '{_PACKAGE_CONFIG_PATH}'."
) if _PACKAGE_CONFIG_PATH else "Path to the configuration YAML file. If 'config.yaml' is not found, it will be created from 'config.template.yaml'."

# -------------------------------------------------------------------
# HELPER: Dynamic Table Generation for the UI
# -------------------------------------------------------------------
def extract_content(message_content) -> str:
    """Extracts string content from LangChain message content which might be a list."""
    if message_content is None:
        return ""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return "".join(
            block if isinstance(block, str) else 
            block.get("text", "") if isinstance(block, dict) else 
            getattr(block, "text", str(block))
            for block in message_content
        )
    return str(message_content)

def format_llm_error(e: Exception) -> str:
    """Formats common LLM exception messages to be more user-friendly."""
    e_str = str(e)

    # General pattern: Many API errors include a 'message' key in their string representation.
    # This works for Google, and often for others when wrapped in HTTP errors.
    # Example: {'error': {'code': 429, 'message': 'You exceeded your current quota...', ...}}
    match = re.search(r"'message':\s*'((?:[^'\\]|\\.)*)'", e_str)
    if match:
        msg = match.group(1).replace("\\n", "\n").strip()
        
        # Add a provider-specific prefix for context
        provider = "LLM"
        if "google" in e_str.lower() or "gemini" in e_str.lower():
            provider = "Google Gemini"
        elif "openai" in e_str.lower():
            provider = "OpenAI"
        elif "anthropic" in e_str.lower():
            provider = "Anthropic"
            
        return f"{provider} API Error:\n{msg}"

    # Fallback for common errors that might not have the 'message' key in this format.
    if "rate limit" in e_str.lower() or "resource_exhausted" in e_str.upper() or "429" in e_str:
        return "LLM rate limit or quota exceeded. Please check your plan and billing details or wait and try again."

    if "authenticationerror" in e_str.lower() or "api key" in e_str.lower() or "401" in e_str:
        return "LLM authentication error. Please check your API key."

    # If no specific pattern is found, return the original error string.
    return e_str

def generate_status_table(status_dict: dict) -> Table:
    """Generates a rich table tracking the live status of parallel builds."""
    table = Table(title="Parallel Function Builder", show_header=True, header_style="bold cyan")
    table.add_column("Hash", style="dim", width=8)
    table.add_column("Module", style="white")
    table.add_column("Step", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Iterations", justify="center")
def generate_status_table(status_dict: dict, module_map: Optional[dict] = None) -> Union[Table, Tree]:
    """Generates a rich table or tree tracking the live status of parallel builds."""
    if not module_map:
        # Legacy/Flat Table View
        table = Table(title="Parallel Function Builder", show_header=True, header_style="bold cyan")
        table.add_column("Hash", style="dim", width=8)
        table.add_column("Module", style="white")
        table.add_column("Step", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Iterations", justify="center")

    for func_name, info in status_dict.items():
        for func_name, info in status_dict.items():
            state_color = "yellow"
            if "Success" in info['status'] or "Completed" in info['status']:
                state_color = "green"
            elif "Fail" in info['status'] or "Error" in info['status']:
                state_color = "red"
                
            status_text = info['status']
            if len(status_text) > 60:
                status_text = status_text[:57] + "..."
                
            table.add_row(
                info.get('hash', '')[:8],
                func_name, 
                info.get('step', '-'),
                f"[{state_color}]{status_text}[/{state_color}]", 
                str(info['iterations'])
            )
        return table
    
    # Hierarchical Tree View
    tree = Tree("[bold cyan]Build Dependency Tree[/bold cyan]")
    
    # Identify Roots (Modules that are not dependencies of others)
    all_deps = set()
    for spec in module_map.values():
        for dep in spec.get("dependencies", []):
            all_deps.add(dep)
            
    roots = [loc for loc in module_map if loc not in all_deps]
    # If no roots found (e.g. circular or single node loop?), fallback to all modules
    if not roots and module_map:
        roots = sorted(list(module_map.keys()))
    else:
        roots.sort()

    def add_children(node, loc, path):
        if loc in path:
            node.add(f"[bold red]{loc} (Cycle Detected)[/bold red]")
            return

        info = status_dict.get(loc, {})
        status = info.get("status", "Pending")
        step = info.get("step", "-")
        iterations = info.get("iterations", 0)
        
        state_color = "yellow"
        if "Success" in info['status'] or "Completed" in info['status']:
        if "Success" in status or "Completed" in status:
            state_color = "green"
        elif "Fail" in info['status'] or "Error" in info['status']:
        elif "Fail" in status or "Error" in status:
            state_color = "red"
        elif "Waiting" in status:
            state_color = "dim yellow"
            
        status_text = info['status']
        if len(status_text) > 60:
            status_text = status_text[:57] + "..."
            
        table.add_row(
            info.get('hash', '')[:8],
            func_name, 
            info.get('step', '-'),
            f"[{state_color}]{status_text}[/{state_color}]", 
            str(info['iterations'])
        )
    return table
        # Format: Module - Status (Step, Iter)
        label = f"[bold white]{loc}[/bold white] : [{state_color}]{status}[/{state_color}] [dim](Step: {step}, Iter: {iterations})[/dim]"
        
        branch = node.add(label)
        
        spec = module_map.get(loc, {})
        deps = spec.get("dependencies", [])
        
        new_path = path | {loc}
        for dep in sorted(deps):
            if dep in module_map:
                add_children(branch, dep, new_path)
            else:
                branch.add(f"[dim red]{dep} (Missing/External)[/dim red]")

    for root in roots:
        add_children(tree, root, set())
        
    return tree

def find_default_config_path() -> Path:
    """Finds the default config path, prioritizing the program directory."""
    
    # 1. Try to find/create config in the program directory (where the template is)
    package_template = _find_package_config_path()
    
    if package_template:
        program_config = package_template.with_name("config.yaml")
        
        if program_config.exists():
            logger.debug(f"Found program-level config: {program_config.resolve()}")
            return program_config
            
        # Attempt to create it in the program directory
        try:
            logger.info(f"'{program_config}' not found. Copying from template: {package_template}")
            shutil.copy2(package_template, program_config)
            console.print(f"[yellow]Created 'config.yaml' at {program_config}.[/yellow]")
            console.print(f"[bold yellow]ACTION REQUIRED: Please edit '{program_config}' to add your API keys before proceeding.[/bold yellow]")
            return program_config
        except OSError:
            # If we can't write to program dir, we continue to check CWD
            pass

    # 2. Fallback: Check if config exists in CWD (user might have manually created it)
    cwd_config = Path("config.yaml")
    if cwd_config.exists():
        logger.debug(f"Found project-level config: {cwd_config.resolve()}")
        return cwd_config

    # 3. If we couldn't create in program dir and none in CWD, return program_config path 
    # if we found the template location (so error messages point there), otherwise CWD.
    if package_template:
        return package_template.with_name("config.yaml")
        
    return cwd_config

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
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            console.print(f"Checking [bold]{role}[/bold] ({model_name})...", end=" ")
            # Send a minimal token request to verify access
            response = llm.invoke([HumanMessage(content="Test")])
            console.print("[green]OK[/green]")
            logger.debug(f"Connection verified for {role} ({model_name}).")
        except Exception as e:
            console.print("[red]FAILED[/red]")
            console.print(f"  [red]Error: {format_llm_error(e)}[/red]")
            logger.exception(f"Connection failed for {role}")
            all_passed = False

    # Check reviewers
    try:
        reviewers = get_reviewer_llms(config_path)
        for i, llm in enumerate(reviewers):
            try:
                model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
                console.print(f"Checking [bold]Reviewer {i+1}[/bold] ({model_name})...", end=" ")
                response = llm.invoke([HumanMessage(content="Test")])
                console.print("[green]OK[/green]")
                logger.debug(f"Connection verified for Reviewer {i+1} ({model_name}).")
            except Exception as e:
                console.print("[red]FAILED[/red]")
                console.print(f"  [red]Error: {format_llm_error(e)}[/red]")
                model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
                logger.exception(f"Connection failed for Reviewer {i+1} ({model_name})")
                all_passed = False
    except Exception as e:
        console.print(f"[red]Error loading reviewers: {e}[/red]")
        logger.exception("Error loading reviewers")
        all_passed = False

    return all_passed

# -------------------------------------------------------------------
# ASYNC WORKER: Processes a single function through LangGraph
# -------------------------------------------------------------------
async def process_module(module_location: str, spec: dict, status_dict: dict, semaphore: asyncio.Semaphore, git_lock: asyncio.Lock, config_path: Path, initial_error: str = None, force_fresh: bool = False):
    """Handles the full lifecycle of a single module generation in isolation."""
    
    # Pre-initialize for the broader try/except/finally scope
    worktree_path = None
    branch_name = f"feature/{module_location}"
    func_logger = None
    log_filename = "main log" # Default for error messages

    try:
        # Calculate hash for logging and DB
        func_hash = hash_spec(spec)
        log_dir = ".MAGS-CodeDev"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"{func_hash}.log")
        log_filepath = os.path.abspath(log_filename)
        
        func_logger = logging.getLogger(f"mags.func.{func_hash}")
        func_logger.setLevel(logging.DEBUG)
        # Avoid adding handlers multiple times if this function is re-run in the same process
        if not func_logger.handlers:
            file_handler = logging.FileHandler(log_filepath, mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            func_logger.addHandler(file_handler)

        async with semaphore:
            status_dict[module_location] = {"status": "Initializing...", "iterations": 0, "hash": func_hash, "log_file": log_filepath, "worktree": None, "step": "Init"}
            # The original try block was here. We've moved it up to encompass the whole function.
            # The original except and finally are now replaced by the top-level ones.

            # Validate spec requirements
            if 'location' not in spec:
                raise ValueError(f"Manifest entry for '{module_location}' is missing required field: 'location'")

            status_dict[module_location]["status"] = "Creating Git Worktree..."
            # 1. Create isolated Git worktree (serialized to prevent index.lock contention)
            async with git_lock:
                worktree_path = await asyncio.to_thread(create_parallel_worktree, branch_name, force_fresh=force_fresh)
            status_dict[module_location]["worktree"] = worktree_path
            
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

            # Load config to get max_iterations
            config = load_config(config_path)
            max_iterations = config.get("settings", {}).get("max_iterations", 5)

            # 2. Initialize LangGraph State
            initial_state: ModuleState = {
                "spec": spec,
                "module_location": module_location,
                "log_filepath": log_filepath,
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
                "max_iterations": max_iterations, # Prevent infinite loops
                "status": "in_progress"
            }
            
            # 3. Compile and Run the Graph
            status_dict[module_location]["status"] = "Running Multi-Agent Graph..."
            graph = build_function_graph()
            
            # Execute the state machine asynchronously using streaming for live updates
            final_state = initial_state.copy()
            async for event in graph.astream(initial_state):
                for node_name, state_update in event.items():
                    status_dict[module_location]["step"] = node_name
                    if "iteration_count" in state_update:
                        status_dict[module_location]["iterations"] = state_update["iteration_count"]
                    final_state.update(state_update)
            
            status_dict[module_location]["step"] = "Complete"
            
            # Determine success based on the state
            max_iter_reached = final_state.get("iteration_count", 0) >= final_state.get("max_iterations", 5)

            # The graph successfully finishes if it reaches the END node *without* hitting the max iteration limit.
            # All failure paths in the graph that lead to END do so because of max_iterations.
            success = not max_iter_reached
            
            if success:
                status_dict[module_location]["status"] = "Success: Tests & Reviews Passed"
            else:
                status_dict[module_location]["status"] = "Failed: Max Iterations Reached"
                warn_msg = f"Workflow for '{module_location}' aborted: Reached maximum iterations ({final_state.get('max_iterations')})."
                logger.warning(warn_msg)
                func_logger.warning(warn_msg)
                func_logger.warning(f"Final Error Summary: {final_state.get('error_summary', 'None')}")

            # Always write the latest code/tests to file so the user can inspect them in the worktree
            # regardless of success/failure.
            output_path = os.path.join(worktree_path, spec['location'])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if final_state.get('code'):
                with open(output_path, "w") as f:
                    f.write(final_state['code'])
            
            # Use the robust test path generated earlier
            test_output_path = test_path
            os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
            if final_state.get('tests'):
                with open(test_output_path, "w") as f:
                    f.write(final_state['tests'])

            # If successful, commit it in the worktree
            if success:
                status_dict[module_location]["status"] = "Committing to Branch..."
                # Commit the changes in the worktree's branch
                repo = git.Repo(worktree_path)
                relative_test_path = os.path.relpath(test_output_path, worktree_path)
                repo.index.add([spec['location'], relative_test_path])
                repo.index.commit(f"feat: Implement module '{module_location}' via MAGs-CodeDev")
            # 4. Merge and Cleanup
            status_dict[module_location]["status"] = "Merging Branch..." if success else "Cleaning up Failed Branch..."
            
            merge_succeeded = False
            # Use lock to ensure only one thread performs git merge operations on the main repo at a time
            async with git_lock:
                merge_succeeded = await asyncio.to_thread(merge_and_cleanup_worktree, branch_name, worktree_path, success)
            
            # 5. Mark as complete in DB only if the merge was successful
            if merge_succeeded:
                await asyncio.to_thread(mark_function_built, module_location, spec)
                status_dict[module_location]["status"] = "Success: Merged to Main"
            elif success: # This means the graph succeeded but the merge failed
                status_dict[module_location]["status"] = "Failed: Merge Conflict"

    except Exception as e:
        # This is the new top-level exception handler.
        effective_logger = func_logger if func_logger else logger
        effective_logger.exception(f"Error processing {module_location}")
        
        logger.error(f"Error processing {module_location} (see {log_filename}): {e}")
        status_dict[module_location]["status"] = f"Error: {str(e)}"
        
        if worktree_path:
            async with git_lock:
                await asyncio.to_thread(merge_and_cleanup_worktree, branch_name, worktree_path, False)
    finally:
        if func_logger:
            for handler in list(func_logger.handlers): # Use list to avoid modifying during iteration
                handler.close()
                func_logger.removeHandler(handler)

# -------------------------------------------------------------------
# COMMANDS
# -------------------------------------------------------------------

@app.command()
def init(
    manifest_path: Path = typer.Option(
        "manifest.json", "--manifest", "-m", help="Path to create the manifest JSON file."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT
    ),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Use AI to interactively design the project structure."
    )
):
    """Initialize the workspace, create AGENT.md, and intelligently build the module manifest."""
    if config_path is None:
        config_path = find_default_config_path()

    console.print(Panel("[bold cyan]Initializing MAGs-CodeDev Workspace...[/bold cyan]"))
    logger.info("Starting workspace initialization.")
    
    # 1. Config Resolution
    # `find_default_config_path` will locate `config.yaml` or create it from a
    # template if it doesn't exist.
    console.print(f"[cyan]Using configuration: {config_path}[/cyan]")
    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Target manifest: {manifest_path}")

    # 2. Gitignore
    gitignore_content = "\n# MAGS-CodeDev\n.MAGS-CodeDev/\nconfig.yaml\n.worktree_*/\nDockerfile.mags-codedev\n*.sif.def\n"
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            current_content = f.read()
        if ".MAGS-CodeDev/" not in current_content:
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
        config = load_config(config_path)
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
                # Safely update the (now potentially local) YAML file
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                
                config_data.setdefault('api_keys', {})[required_key_name] = user_key

                with open(config_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                console.print(f"[green]Updated {config_path} with API key.[/green]")
                logger.info(f"Updated {config_path} with provided {required_key_name} API key.")
            else:
                interactive = False

    if interactive:
        try:
            from mags_codedev.agents.chat_agent import start_chat_repl
            logger.info("Starting AI Architect tool-based session.")

            architect_system_prompt = f"""You are a Senior Software Architect. Your goal is to help the user define and set up a new software project from scratch.

**Your Process:**
1.  **Discuss:** Talk with the user about their project idea. Ask clarifying questions to understand the requirements, scope, and desired technologies.
2.  **Plan:** Propose a plan that includes:
    *   A file and directory structure.
    *   A list of core modules for the manifest, including dependencies between them.
    *   Any necessary Python dependencies for `requirements.txt`.
    *   A brief project description for `README.md`.
    *   Any necessary entries for `.gitignore`.
3.  **Execute:** Once the user approves your plan, use your tools to create or modify the project files. You have the `read_file` and `write_file` tools. If a file already exists, read it first to decide if you should append or overwrite.

**Key Files to Create/Modify:**
*   `{manifest_path}`: A JSON file defining the modules to be built. This is your primary output for the build system. Each element in the JSON array represents one Python module file to be created.
*   `AGENT.md`: A markdown file with high-level instructions for the other AI agents (e.g., language, coding standards).
*   `requirements.txt`: Add the Python dependencies needed for the project.
*   `README.md`: Create a basic README file for the project.
*   `.gitignore`: Add any necessary entries.

**Example manifest.json file:**
[
            {{
                "location": "src/utils.py",
                "description": "Utility functions for data processing and validation.",
                "dependencies": []
            }},
            {{
                "location": "src/pricing.py",
                "description": "Core pricing logic. Contains functions to calculate discounts and apply them to products. Depends on src/utils.py for input validation.",
                "dependencies": ["src/utils.py"]
            }}
        ]

**Important Rules:**
*   **Do not write any files until the user has approved your plan.**
*   **Use the `write_file` tool to create/modify files directly.** Do not output file content in the chat.
*   **Create all project configuration files in the root directory.** Files like `{manifest_path}`, `AGENT.md`, `README.md`, `requirements.txt`, and `Dockerfile.dev` must be created in the current directory, not in a subdirectory. Source code itself can be in subdirectories (e.g., `src/my_code.py`).
*   Start by greeting the user and asking about their project idea."""

            console.print(Panel("[bold green]AI Architect Mode[/bold green]\nDescribe your project idea. The AI will ask questions and then use its tools to write `AGENT.md` and `manifest.json` for you.\n\nType 'exit' or 'quit' to end the session."))

            agent_graph = start_chat_repl(config_path=config_path, system_message_override=architect_system_prompt, command_name="init")
            llm = get_llm("chat", config_path) # The architect uses the 'chat' model
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            config = {
                "configurable": {"thread_id": "architect-session"},
                "callbacks": [TokenLoggingCallbackHandler(role="command_init", model_name=model_name)]
            }
            
            while True:
                try:
                    user_input = console.input("[bold green]You>[/bold green] ")
                    if user_input.lower() in ['exit', 'quit']:
                        logger.info("User exited AI Architect mode.")
                        break
                    
                    logger.debug(f"Architect Input: {user_input}")
                    
                    with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                        response = agent_graph.invoke({"messages": [("user", user_input)]}, config=config)
                    last_message = response["messages"][-1]
                    final_answer = extract_content(last_message.content)
                    logger.debug(f"Architect Final Answer: {final_answer}")
                    console.print(f"\n[blue]Architect>[/blue] {final_answer}\n")

                except (KeyboardInterrupt, EOFError):
                    break
            console.print("\n[blue]Exiting AI Architect mode.[/blue]")

        except Exception as e:
            logger.exception("AI Architect Mode encountered an error")
            console.print(f"[red]AI Setup Error: {format_llm_error(e)}[/red]")
            console.print("[yellow]Falling back to manual setup.[/yellow]")
        
        # After the interactive session, check if the files were created by the agent
        manifest_created = manifest_path.exists()
        agent_md_created = Path("AGENT.md").exists()

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
        dummy_manifest = [
            {
                "location": "src/utils.py",
                "description": "A module for shared utility functions, like data validation or formatting.",
                "dependencies": []
            },
            {
                "location": "src/main_logic.py",
                "description": "The main business logic module. It will import and use functions from src/utils.py.",
                "dependencies": ["src/utils.py"]
            }
        ]
        with open(manifest_path, "w") as f:
            json.dump(dummy_manifest, f, indent=4)
        console.print(f"[green]Created dummy {manifest_path}. Please edit this to define your modules.[/green]")
        logger.info(f"Created dummy manifest file at {manifest_path}.")

    console.print("[bold green]✓ Initialization complete. Run `mags-codedev build` to start coding.[/bold green]")
    logger.info("Initialization complete.")


@app.command()
def build(
    manifest_path: Path = typer.Option(
        "manifest.json", "--manifest", "-m", help="Path to the manifest JSON file.", file_okay=True, dir_okay=False, readable=True, resolve_path=True
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip the pre-build connection check."
    ),
    force_fresh: bool = typer.Option(
        False, "--force-fresh", help="Force a fresh build by deleting existing worktrees and branches for pending functions."
    ),
):
    """Build all pending modules in the manifest using parallel multi-agent LangGraphs."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Using manifest: {manifest_path}")
    console.print(Panel("[bold magenta]Starting Multi-Agent Build Process...[/bold magenta]"))
    
    # Validate Git Repo State before starting
    try:
        validate_git_repo()
    except RuntimeError as e:
        console.print(f"[bold red]Git Error:[/bold red] {e}")
        raise typer.Exit(1)
    
    if not skip_validation:
        if not validate_config_connections(config_path):
            console.print("[bold red]Validation failed. Aborting build.[/bold red]")
            console.print("Use --skip-validation to force execution if you believe this is an error.")
            raise typer.Exit(1)

    if not manifest_path.exists():
        console.print(f"[red]Error: Manifest file not found at '{manifest_path}'. Run `mags-codedev init` first.[/red]")
        raise typer.Exit(1)
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    config = load_config(config_path)
    max_parallel = config.get("settings", {}).get("max_parallel_modules", 4)
    
    init_db()
    
    # Create a map for easy lookup and get all initially built functions
    module_map = {spec['location']: spec for spec in manifest if 'location' in spec}
    built_modules = {loc for loc, spec in module_map.items() if is_function_built(spec)}

    if len(built_modules) == len(module_map):
        console.print("[green]All modules in manifest are already built and verified![/green]")
        raise typer.Exit(0)

    console.print(f"[cyan]Found {len(module_map)} total modules. {len(built_modules)} already built.[/cyan]")

    # Run the async loop
    async def run_builds():
        nonlocal built_modules # We will modify this set
        
        # This dict will persist across build waves, holding status for all modules.
        status_dict = {
            loc: {
                "status": "Pending",
                "iterations": 0,
                "hash": hash_spec(spec),
                "step": "-"
            }
            for loc, spec in module_map.items()
        }

        # Main build loop to handle dependencies
        while len(built_modules) < len(module_map):
            # 1. Update statuses based on what's already built
            for loc in built_modules:
                status_dict[loc]['status'] = "Completed"
                status_dict[loc]['step'] = "Done"

            # Find functions that are not built yet
            unbuilt_modules = {loc: spec for loc, spec in module_map.items() if loc not in built_modules}
            
            # Determine which of the unbuilt functions are ready to be built
            buildable_now = {
                loc: spec for loc, spec in unbuilt_modules.items()
                if all(dep in built_modules for dep in spec.get("dependencies", []))
            }
            
            # Update statuses for waiting modules
            for loc, spec in unbuilt_modules.items():
                if loc not in buildable_now:
                    missing_deps = [dep for dep in spec.get("dependencies", []) if dep not in built_modules]
                    status_dict[loc]['status'] = f"Waiting for: {', '.join(missing_deps)}"
                    status_dict[loc]['step'] = "Waiting"

            if not buildable_now:
                # Render final table before erroring out
                console.print(generate_status_table(status_dict))
                console.print("\n[bold red]Error: Circular dependency or missing dependency detected.[/bold red]")
                console.print("The following modules could not be built because their dependencies are not met:")
                for loc, spec in unbuilt_modules.items():
                    missing_deps = [dep for dep in spec.get("dependencies", []) if dep not in built_modules]
                    if missing_deps:
                        console.print(f"- [yellow]{loc}[/yellow] (missing: {', '.join(missing_deps)})")
                raise typer.Exit(1)

            console.print(f"\n[bold magenta]Starting build wave: {len(buildable_now)} module(s) ready.[/bold magenta]")

            semaphore = asyncio.Semaphore(max_parallel)
            git_lock = asyncio.Lock()

            with Live(generate_status_table(status_dict), refresh_per_second=4) as live:
            with Live(generate_status_table(status_dict, module_map), refresh_per_second=4) as live:
                async def update_ui_loop():
                    while True:
                        # Re-sort the dictionary for consistent display order
                        sorted_status = dict(sorted(status_dict.items()))
                        live.update(generate_status_table(sorted_status))
                        live.update(generate_status_table(sorted_status, module_map))
                        await asyncio.sleep(0.25)

                build_tasks = [
                    process_module(loc, spec, status_dict, semaphore, git_lock, config_path, force_fresh=force_fresh) 
                    for loc, spec in buildable_now.items()
                ]
                ui_task = asyncio.create_task(update_ui_loop())
                await asyncio.gather(*build_tasks)
                ui_task.cancel()
                live.update(generate_status_table(status_dict))
                live.update(generate_status_table(status_dict, module_map))
            
            # After the wave, check for successes and failures for the modules in this wave
            wave_failures = []
            wave_successes = []
            for loc in buildable_now:
                info = status_dict[loc]
                if "Success" in info["status"]:
                    wave_successes.append(loc)
                else:
                    wave_failures.append(loc)
            
            built_modules.update(wave_successes)
            
            if wave_failures:
                console.print(f"\n[bold red]Build wave completed with {len(wave_failures)} failure(s):[/bold red]")
                for loc in wave_failures:
                    info = status_dict[loc]
                    console.print(f"  - [red]{loc}[/red]")
                    if info.get("worktree") and os.path.exists(info["worktree"]):
                        console.print(f"    Worktree: [blue]{info['worktree']}[/blue]")
                    console.print(f"    Log: [blue]{info.get('log_file')}[/blue]")
                    console.print(f"    Status: {info['status']}")
                console.print("\n[bold red]Aborting build due to failures in the current wave.[/bold red]")
                raise typer.Exit(1)
        
        # This part is outside the while loop
        console.print("\n[bold green]Build cycle complete! All modules built successfully.[/bold green]")

    asyncio.run(run_builds())


@app.command()
def test(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True
    ),
):
    """Run all pytest unit tests for the project in the configured environment."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    from mags_codedev.utils.docker_ops import run_command_in_project_env

    console.print(Panel("[bold cyan]Running Project Unit Tests...[/bold cyan]"))
    logger.info("Starting project-wide test run.")

    # The command to run pytest. Pytest will discover tests automatically.
    command = "pytest -v"
    
    # The project root is the current working directory
    project_root = os.getcwd()

    # Use a spinner while tests are running
    with console.status("[bold green]Running tests...[/bold green]", spinner="dots"):
        results = run_command_in_project_env(command, config_path, project_root, logger)
    
    console.print(Panel(results, title="Test Results", border_style="blue"))
    
    if "failed" in results.lower() or "error" in results.lower():
        console.print("[bold red]Some tests failed or errors occurred.[/bold red]")
        raise typer.Exit(code=1)
    else:
        console.print("[bold green]All tests passed![/bold green]")


@app.command()
def debug(
    error_msg: str = typer.Argument(..., help="The error trace to fix, or a path to the error trace logfile."),
    module_location: Optional[str] = typer.Option(None, "--module", "-m", help="The module location in manifest.json to apply the fix to. (Optional if providing a log file)"),
    manifest_path: Path = typer.Option(
        "manifest.json", "--manifest", "-m", help="Path to the manifest JSON file."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True
    ),
):
    """Pass an error trace or bug description to the LLM for automatic fixing of a module."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    is_log_file = False
    log_file_path = None
    
    # Check if the argument is exactly a file
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
        # Check if the argument contains a file path (e.g. "Check file.log")
        words = error_msg.split()
        for word in words:
            # Remove common punctuation that might trail a path in a sentence
            clean_word = word.strip(".,;:'\"")
            if os.path.exists(clean_word) and os.path.isfile(clean_word):
                is_log_file = True
                log_file_path = clean_word
                try:
                    with open(clean_word, "r") as f:
                        console.print(f"[cyan]Reading error trace from file: {clean_word}[/cyan]")
                        file_content = f.read()
                        # Append content to the prompt
                        error_msg = f"{error_msg}\n\n--- Log File Content ---\n{file_content}"
                except Exception as e:
                    console.print(f"[red]Error reading file '{clean_word}': {e}[/red]")
                    raise typer.Exit(1)
                break

    # Auto-detect function from log file if not provided
    if is_log_file and not module_location:
        try:
            log_hash = Path(log_file_path).stem
            # Check if the stem looks like a sha256 hash
            if len(log_hash) == 64 and all(c in '0123456789abcdef' for c in log_hash):
                if not manifest_path.exists():
                    console.print(f"[yellow]Manifest '{manifest_path}' not found. Cannot auto-detect module from log.[/yellow]")
                else:
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                    for spec in manifest:
                        if hash_spec(spec) == log_hash:
                            module_location = spec.get("location")
                            if module_location:
                                console.print(f"[cyan]Auto-detected module '[bold]{module_location}[/bold]' from log file.[/cyan]")
                                break
        except Exception as e:
            # This is a convenience feature, so don't crash if it fails.
            logger.warning(f"Could not auto-detect module from log file: {e}")

    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Using manifest: {manifest_path}")
    console.print(Panel(f"[bold red]Debugging Error:[/bold red]\n{error_msg}"))
    
    # Scenario A: Function name is now known (either provided or detected)
    if module_location:
        if not manifest_path.exists():
            console.print(f"[red]Manifest '{manifest_path}' not found.[/red]")
            raise typer.Exit(1)
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        spec = next((s for s in manifest if s.get("location") == module_location), None)
            
        if not spec:
            console.print(f"[red]Module '{module_location}' not found in manifest.[/red]")
            raise typer.Exit(1)
            
        console.print(f"[cyan]Attempting to fix {module_location} based on the error...[/cyan]")
        
        async def run_fix(error_to_fix: str):
            # Create a dummy status dict for the helper
            status = {module_location: {"status": "Starting Fix...", "iterations": 0, "hash": "debug", "log_file": ".MAGS-CodeDev/debug.log"}}
            sem = asyncio.Semaphore(1)
            lock = asyncio.Lock()
            console.print("[yellow]Running fix workflow with provided error...[/yellow]")
            await process_module(module_location, spec, status, sem, lock, config_path, initial_error=error_to_fix)
            
        asyncio.run(run_fix(error_msg))
        
    # Scenario B: No function name -> Analyze error and suggest strategy
    else:
        console.print("[cyan]Analyzing error trace...[/cyan]")
        # Get the llm configured for 'chat', but overwrite the callback for specific logging.
        llm = get_llm("chat", config_path) 
        model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
        llm.callbacks = [TokenLoggingCallbackHandler(role="command_debug", model_name=model_name)]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert debugger. Analyze the error provided. Explain the likely cause and suggest which file or module is likely responsible."),
            ("human", "{input}")
        ])
        chain = prompt | llm
        response = chain.invoke({"input": error_msg})

        console.print(Panel(extract_content(response.content), title="Debug Analysis", border_style="green"))


@app.command()
def chat(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True
    ),
):
    """Freely chat with the LLM about the codebase. Can read/write files."""
    if config_path is None:
        config_path = find_default_config_path()
    elif not config_path.exists():
        console.print(f"[red]Error: Specified config file not found at '{config_path}'[/red]")
        raise typer.Exit(1)

    logger.info(f"Using configuration: {config_path}")
    from mags_codedev.agents.chat_agent import start_chat_repl
    
    console.print("[bold blue]Entering Chat Mode (Type 'exit' to quit)...[/bold blue]")
    agent_graph = start_chat_repl(config_path=config_path, command_name="chat")
    llm = get_llm("chat", config_path) # Get LLM to access model name for logging
    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
    
    # Thread ID is required by LangGraph checkpointers to maintain conversation history
    config = {
        "configurable": {"thread_id": "cli-session"},
        "callbacks": [TokenLoggingCallbackHandler(role="command_chat", model_name=model_name)]
    }
    
    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            logger.debug(f"Chat Input: {user_input}")
            
            # Use invoke to execute the agent graph. Input is a list of messages.
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response = agent_graph.invoke({"messages": [("user", user_input)]}, config=config)
            
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
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help=_CONFIG_HELP_TEXT, resolve_path=True
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
            # List models
            models = client.models.list()
            # Extract and sort just the model IDs (e.g., 'gpt-4o', 'gpt-4-turbo')
            gpt_models = sorted([m.id for m in models.data if "gpt" in m.id])
            for m in gpt_models:
                table.add_row("OpenAI", m)
        except Exception as e:
            table.add_row("OpenAI", f"[red]Error: {format_llm_error(e)}[/red]")

    # 2. Google (Gemini)
    if api_keys.get("gemini"):
        try:
            try:
                # Try the new Google GenAI SDK first
                from google import genai
                client = genai.Client(api_key=api_keys["gemini"])
                for m in client.models.list():
                    name = m.name.replace("models/", "")
                    table.add_row("Google", name)
            except ImportError:
                # Fallback to the deprecated SDK, suppressing the FutureWarning
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        import google.generativeai as genai
                    genai.configure(api_key=api_keys["gemini"])
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            name = m.name.replace("models/", "")
                            table.add_row("Google", name)
                except ImportError:
                    table.add_row("Google", "[yellow]Skipped: 'google.genai' or 'google-generativeai' not installed[/yellow]")
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
                from mistralai.client import MistralClient
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
        # Cohere does not have a simple model listing API like others.
        # We provide a static list of known recent models.
        known_models = ["command-r-plus", "command-r", "command", "command-light"]
        for m in known_models:
            table.add_row("Cohere (Static)", m)


    # 4. Custom / Local (e.g. Ollama)
    # Scan config for custom_openai providers to find base_urls
    custom_urls = set()
    models_config = config.get("models", {})    
    build_config = models_config.get("build_workflow", {})
    interactive_config = models_config.get("interactive_commands", {})

    # Combine all agent configs from new and old structures for scanning
    all_agent_configs = {
        **models_config, 
        **build_config, 
        **interactive_config
    }
    
    # Check roles
    for role in ["coder", "tester", "log_checker", "chat"]:
        m_cfg = all_agent_configs.get(role, {})
        if m_cfg.get("provider") == "custom_openai" and m_cfg.get("base_url"):
            custom_urls.add(m_cfg.get("base_url"))
            
    # Check reviewers list
    reviewers_list = build_config.get("reviewers", []) or models_config.get("reviewers", [])
    for r_cfg in reviewers_list:
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
            table.add_row(f"Custom ({url})", f"[red]Error: {format_llm_error(e)}[/red]")

    console.print(table)

@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation.")
):
    """Remove all generated cache files, logs, and temporary git worktrees."""
    console.print(Panel("[bold yellow]Cleaning up MAGs-CodeDev artifacts...[/bold yellow]"))

    # Find artifacts to delete
    mags_dir = Path(".MAGS-CodeDev")
    db_file = mags_dir / "mags_cache.db"
    log_file = mags_dir / "mags-codedev_workflow.log"
    worktree_dirs = [d for d in os.listdir('.') if d.startswith(".worktree_") and os.path.isdir(d)]
    hash_logs = [f for f in mags_dir.glob("*.log") if len(f.name) == 68] if mags_dir.exists() else [] # 64 chars hash + .log

    items_to_delete = []
    if db_file.exists():
        items_to_delete.append(db_file)
    if log_file.exists():
        items_to_delete.append(log_file)
    items_to_delete.extend([Path(d) for d in worktree_dirs])
    items_to_delete.extend([Path(f) for f in hash_logs])

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
    for f in hash_logs:
        if os.path.exists(f): os.remove(f); console.print(f"Deleted log file: {f}")
        
    # Try to remove the directory if empty
    if mags_dir.exists() and not any(mags_dir.iterdir()):
        os.rmdir(mags_dir)
        console.print(f"Removed directory: {mags_dir}")
    
    console.print("\n[bold green]✓ Cleanup complete.[/bold green]")

if __name__ == "__main__":
    app()