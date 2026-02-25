import docker
import logging
import os
from pathlib import Path
from mags_codedev.state import FunctionState
from mags_codedev.utils.config_parser import load_config
from mags_codedev.utils.logger import logger

def _run_in_container(state: FunctionState, command: str) -> str:
    """Helper to spin up a container, mount files, run a command, and return logs."""
    code, tests = state["code"], state["tests"]
    config_path, worktree_path = state["config_path"], state["worktree_path"]
    
    if state.get("log_filepath"):
        import os
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
    
    try:
        client = docker.from_env()
        # Ping to verify connection
        client.ping()
    except (docker.errors.DockerException, Exception) as e:
        err_msg = f"Docker connection failed: {e}. Ensure Docker is running and you have permissions (e.g., 'sudo usermod -aG docker $USER')."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg
    
    # Write the current iteration of code/tests to the worktree so they can be mounted.
    # This allows imports to work correctly.
    code_abs_path = os.path.join(worktree_path, state["spec"]["location"])
    test_abs_path = os.path.join(worktree_path, state["test_location"])
    
    os.makedirs(os.path.dirname(code_abs_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_abs_path), exist_ok=True)
    
    with open(code_abs_path, "w") as f: f.write(code)
    with open(test_abs_path, "w") as f: f.write(tests)

    config = load_config(config_path)
    image_name = config.get("settings", {}).get("docker_test_image", "python:3.11-slim")
    timeout_mins = config.get("settings", {}).get("timeout_per_function_mins", 15)
    timeout_seconds = timeout_mins * 60

    container = None

    # Construct the shell command to run inside the container
    install_deps_cmd = "pip install -r requirements.txt && " if os.path.exists(os.path.join(worktree_path, "requirements.txt")) else ""
    full_command = f"{install_deps_cmd}{command}"

    func_logger.debug(f"Docker: Running command in {image_name}:\n{full_command}")

    try:
        container = client.containers.run(
            image=image_name,
            command=["sh", "-c", full_command],
            # Mount the entire worktree to /app so imports resolve
            volumes={worktree_path: {'bind': '/app', 'mode': 'rw'}},
            working_dir="/app",
            detach=True, # Detach to manage timeout manually
            stderr=True,
            stdout=True
        )
        
        # Wait for the container to finish, with a timeout
        result = container.wait(timeout=timeout_seconds)
        
        # Get logs regardless of the exit code
        logs = container.logs(stdout=True, stderr=True).decode("utf-8")
        
        # A non-zero exit code is the expected path for test/lint failures.
        # We return the logs for the Log Checker agent to analyze.
        return logs
        
    except docker.errors.NotFound:
        err_msg = f"Docker image '{image_name}' not found. Please build it or check config.yaml."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg
    except docker.errors.APIError as e:
        err_msg = f"Docker API error: {str(e)}"
        func_logger.exception("Docker API error")
        return err_msg
    except Exception as e:
        # This catches timeout from container.wait() and other exceptions
        if container:
            container.stop()
        err_msg = f"Docker execution error: {str(e)}"
        func_logger.exception("Docker execution error")
        return err_msg
    finally:
        if container:
            container.remove(force=True)

def docker_test_node(state: FunctionState) -> dict:
    """LangGraph node: Executes pytest in an isolated Docker container."""
    # Use Path(...).as_posix() to ensure forward slashes for Linux container commands
    test_file = Path(state["test_location"]).as_posix()
    logs = _run_in_container(state, f"pytest {test_file} -v")
    return {"test_results": logs}

def linter_node(state: FunctionState) -> dict:
    """LangGraph node: Executes flake8 and mypy in an isolated Docker container."""
    target_file = Path(state["spec"]["location"]).as_posix()
    # The command is structured to run mypy even if flake8 fails.
    logs = _run_in_container(state, f"flake8 {target_file}; mypy {target_file}")

    # If the output is empty (flake8 success) or just the mypy success message,
    # return an empty string to prevent the log_checker from running on clean code.
    if not logs.strip() or logs.strip().startswith("Success: no issues found"):
        return {"lint_results": ""}
    return {"lint_results": logs}