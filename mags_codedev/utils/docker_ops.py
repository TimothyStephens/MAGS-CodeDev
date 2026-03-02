import docker
import logging
import os
import subprocess
from pathlib import Path
from mags_codedev.state import FunctionState
from mags_codedev.utils.config_parser import load_config
from mags_codedev.utils.logger import logger

def _run_with_docker(state: FunctionState, command: str, config: dict, func_logger: logging.Logger) -> str:
    """Runs a command inside a Docker container."""
    try:
        client = docker.from_env()
        client.ping()
    except (docker.errors.DockerException, Exception) as e:
        err_msg = f"Docker connection failed: {e}. Ensure Docker is running and you have permissions (e.g., 'sudo usermod -aG docker $USER')."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg

    image_name = config.get("settings", {}).get("docker_test_image", "python:3.11-slim")
    timeout_mins = config.get("settings", {}).get("timeout_per_function_mins", 15)
    timeout_seconds = timeout_mins * 60
    worktree_path = state["worktree_path"]
    container = None

    func_logger.debug(f"Docker: Running command in {image_name}:\n{command}")

    try:
        container = client.containers.run(
            image=image_name,
            command=["sh", "-c", command],
            volumes={worktree_path: {'bind': '/app', 'mode': 'rw'}},
            working_dir="/app",
            detach=True,
            stderr=True,
            stdout=True
        )
        
        container.wait(timeout=timeout_seconds)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8")
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
        if container:
            container.stop()
        err_msg = f"Docker execution error: {str(e)}"
        func_logger.exception("Docker execution error")
        return err_msg
    finally:
        if container:
            container.remove(force=True)

def _run_with_apptainer(state: FunctionState, command: str, config: dict, func_logger: logging.Logger) -> str:
    """Runs a command inside an Apptainer container."""
    image_name = config.get("settings", {}).get("apptainer_test_image", "mags-dev-env.sif")
    dockerfile_dev = "Dockerfile.dev"
    timeout_mins = config.get("settings", {}).get("timeout_per_function_mins", 15)
    timeout_seconds = timeout_mins * 60
    worktree_path = state["worktree_path"]
    
    image_path = Path(image_name)
    if not image_path.exists():
        func_logger.info(f"Apptainer image '{image_name}' not found. Attempting to build from '{dockerfile_dev}'...")
        
        if not Path(dockerfile_dev).exists():
            err_msg = f"Apptainer image '{image_name}' not found and '{dockerfile_dev}' is also missing. Cannot build the image."
            logger.error(err_msg)
            func_logger.error(err_msg)
            return err_msg
            
        build_command = ["apptainer", "build", "--force", image_name, dockerfile_dev]
        func_logger.debug(f"Running Apptainer build: {' '.join(build_command)}")
        
        try:
            # Use a longer timeout for the build process itself.
            build_timeout = 30 * 60 # 30 minutes for build
            subprocess.run(
                build_command,
                capture_output=True, text=True, timeout=build_timeout, check=True
            )
            func_logger.info(f"Successfully built Apptainer image '{image_name}'.")
        except FileNotFoundError:
            err_msg = "Apptainer command not found. Is Apptainer installed and in your PATH?"
            logger.error(err_msg)
            func_logger.error(err_msg)
            return err_msg
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            output = (e.stdout or "") + (e.stderr or "")
            err_msg = f"Failed to build Apptainer image '{image_name}'.\nError: {e}\nOutput:\n{output}"
            logger.error(err_msg)
            func_logger.error(err_msg)
            return err_msg

    apptainer_command = [
        "apptainer", "exec",
        "--bind", f"{worktree_path}:/app",
        "--pwd", "/app",
        str(image_path),
        "sh", "-c", command
    ]

    func_logger.debug(f"Apptainer: Running command: {' '.join(apptainer_command)}")

    try:
        result = subprocess.run(
            apptainer_command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False
        )
        return result.stdout + result.stderr
    except FileNotFoundError:
        err_msg = "Apptainer command not found. Is Apptainer installed and in your PATH?"
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg
    except subprocess.TimeoutExpired:
        err_msg = f"Apptainer command timed out after {timeout_seconds} seconds."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg
    except Exception as e:
        err_msg = f"Apptainer execution error: {e}"
        func_logger.exception("Apptainer execution error")
        return err_msg

def _run_locally(state: FunctionState, command: str, config: dict, func_logger: logging.Logger) -> str:
    """Runs a command in the local environment."""
    timeout_mins = config.get("settings", {}).get("timeout_per_function_mins", 15)
    timeout_seconds = timeout_mins * 60
    worktree_path = state["worktree_path"]

    func_logger.debug(f"Local: Running command in {worktree_path}:\n{command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=worktree_path,
            check=False
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        err_msg = f"Local command timed out after {timeout_seconds} seconds."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg
    except Exception as e:
        err_msg = f"Local execution error: {e}"
        func_logger.exception("Local execution error")
        return err_msg

def _run_in_environment(state: FunctionState, command: str) -> str:
    """Helper to run a command in the configured environment (docker, apptainer, or local)."""
    code, tests = state["code"], state["tests"]
    config_path, worktree_path = state["config_path"], state["worktree_path"]
    
    if state.get("log_filepath"):
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
    
    # Write code/tests to the worktree so they can be mounted/used.
    code_abs_path = os.path.join(worktree_path, state["spec"]["location"])
    test_abs_path = os.path.join(worktree_path, state["test_location"])
    
    os.makedirs(os.path.dirname(code_abs_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_abs_path), exist_ok=True)
    
    with open(code_abs_path, "w") as f: f.write(code)
    with open(test_abs_path, "w") as f: f.write(tests)

    config = load_config(config_path)
    runner = config.get("settings", {}).get("test_runner", "docker").lower()

    # Construct the shell command to run inside the container
    install_deps_cmd = "pip install -r requirements.txt && " if os.path.exists(os.path.join(worktree_path, "requirements.txt")) else ""
    full_command = f"{install_deps_cmd}{command}"

    if runner == "docker":
        return _run_with_docker(state, full_command, config, func_logger)
    elif runner == "apptainer":
        return _run_with_apptainer(state, full_command, config, func_logger)
    elif runner == "local":
        return _run_locally(state, full_command, config, func_logger)
    else:
        err_msg = f"Invalid test_runner '{runner}' in config.yaml. Must be 'docker', 'apptainer', or 'local'."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg

def test_node(state: FunctionState) -> dict:
    """LangGraph node: Executes pytest in the configured environment."""
    # Use Path(...).as_posix() to ensure forward slashes for Linux container commands
    test_file = Path(state["test_location"]).as_posix()
    logs = _run_in_environment(state, f"pytest {test_file} -v")
    return {"test_results": logs}

def linter_node(state: FunctionState) -> dict:
    """LangGraph node: Executes flake8 and mypy in the configured environment."""
    target_file = Path(state["spec"]["location"]).as_posix()
    # The command is structured to run mypy even if flake8 fails.
    logs = _run_in_environment(state, f"flake8 {target_file}; mypy {target_file}")

    # If the output is empty (flake8 success) or just the mypy success message,
    # return an empty string to prevent the log_checker from running on clean code.
    if not logs.strip() or logs.strip().startswith("Success: no issues found"):
        return {"lint_results": ""}
    return {"lint_results": logs}