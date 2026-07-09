import docker
import logging
import os
import json
import subprocess
from pathlib import Path
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import load_config
from mags_codedev.utils.logger import logger


def _generate_dockerfile_content(config: dict, backend) -> str:
    """Generates Dockerfile content based on project dependencies and backend."""
    base_image = config.get("settings", {}).get("python_base_image",
                                                backend.default_base_image)
    system_deps = config.get("settings", {}).get("system_dependencies", [])

    dockerfile_parts = [f"FROM {base_image}"]

    if system_deps:
        deps_str = " ".join(system_deps)
        dockerfile_parts.append(
            "RUN apt-get update && apt-get install -y --no-install-recommends "
            f"{deps_str} && rm -rf /var/lib/apt/lists/*"
        )

    # Core test/lint deps from backend
    dockerfile_parts.append(f"RUN {backend.install_core_deps_command()}")

    # Project deps from backend
    deps_file = backend.deps_filename
    if Path(deps_file).exists():
        dockerfile_parts.extend([
            f"COPY {deps_file} /app/{deps_file}",
            backend.install_project_deps_command(Path(deps_file))
        ])

    dockerfile_parts.append("WORKDIR /app")
    return "\n".join(dockerfile_parts)


def _generate_apptainer_def_content(config: dict, backend) -> str:
    """Generates Apptainer definition file content based on backend."""
    base_image = config.get("settings", {}).get("python_base_image",
                                                backend.default_base_image)
    system_deps = config.get("settings", {}).get("system_dependencies", [])

    post_section = ["    export DEBIAN_FRONTEND=noninteractive"]

    if system_deps:
        deps_str = " ".join(system_deps)
        post_section.extend([
            "    apt-get update",
            f"    apt-get install -y --no-install-recommends {deps_str}",
            "    rm -rf /var/lib/apt/lists/*"
        ])

    post_section.append(f"    {backend.install_core_deps_command()}")

    deps_file = backend.deps_filename
    files_section = ""
    if Path(deps_file).exists():
        files_section = f"\n%files\n    {deps_file} /app/{deps_file}\n"
        post_section.append(f"    {backend.install_project_deps_command(Path(deps_file))}")

    return f"""
Bootstrap: docker
From: {base_image}
{files_section}
%post
{chr(10).join(post_section)}
""".strip()


def _run_with_docker(state: dict, command: str, config: dict, func_logger: logging.Logger, backend=None) -> str:
    """Runs a command inside a Docker container, building the image if necessary."""
    try:
        client = docker.from_env()
        client.ping()
    except (docker.errors.DockerException, Exception) as e:
        err_msg = f"Docker connection failed: {e}. Ensure Docker is running and you have permissions (e.g., 'sudo usermod -aG docker $USER')."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg

    image_name = config.get("settings", {}).get("docker_test_image", "mags-dev-env:latest")

    # Check if image exists and build if not
    try:
        client.images.get(image_name)
        func_logger.debug(f"Docker image '{image_name}' found locally.")
    except docker.errors.ImageNotFound:
        func_logger.info(f"Docker image '{image_name}' not found. Attempting to build...")

        dockerfile_content = _generate_dockerfile_content(config, backend)
        temp_dockerfile_path = Path("Dockerfile.mags-codedev")

        try:
            with open(temp_dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            func_logger.debug(f"Building Docker image '{image_name}' using temporary Dockerfile...")
            _, build_log = client.images.build(
                path=".",
                dockerfile=str(temp_dockerfile_path),
                tag=image_name,
                rm=True
            )
            for chunk in build_log:
                if 'stream' in chunk:
                    func_logger.debug(f"Docker build: {chunk['stream'].strip()}")

            func_logger.info(f"Successfully built Docker image '{image_name}'.")

        except docker.errors.BuildError as e:
            err_msg = f"Failed to build Docker image '{image_name}'.\nError: {e}"
            logger.error(err_msg)
            func_logger.error(err_msg)
            return err_msg
        except Exception as e:
            err_msg = f"An error occurred during Docker image build: {e}"
            logger.error(err_msg)
            func_logger.error(err_msg)
            return err_msg
        finally:
            if temp_dockerfile_path.exists():
                temp_dockerfile_path.unlink()

    timeout_mins = config.get("settings", {}).get("timeout_per_module_mins", 15)
    timeout_seconds = timeout_mins * 60
    worktree_path = state["worktree_path"]
    container = None

    # Build environment vars from backend (container uses /app mount)
    env_vars = {}
    if backend:
        env_vars = backend.container_env_vars()

    func_logger.debug(f"Docker: Running command in {image_name}:\n{command}")

    try:
        container = client.containers.run(
            image=image_name,
            command=["sh", "-c", command],
            volumes={worktree_path: {'bind': '/app', 'mode': 'rw'}},
            working_dir="/app",
            detach=True,
            stderr=True,
            stdout=True,
            environment=env_vars
        )

        container.wait(timeout=timeout_seconds)
        logs = container.logs(stdout=True, stderr=True).decode("utf-8")
        return logs

    except docker.errors.NotFound:
        err_msg = f"Docker image '{image_name}' not found. Build failed or was interrupted."
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


def _run_with_apptainer(state: dict, command: str, config: dict, func_logger: logging.Logger, backend=None) -> str:
    """Runs a command inside an Apptainer container, building the image if necessary."""
    image_name = config.get("settings", {}).get("apptainer_test_image", "mags-dev-env.sif")
    timeout_mins = config.get("settings", {}).get("timeout_per_module_mins", 15)
    timeout_seconds = timeout_mins * 60
    worktree_path = state["worktree_path"]

    image_path = Path(image_name)
    if not image_path.exists():
        func_logger.info(f"Apptainer image '{image_name}' not found. Attempting to build...")

        def_content = _generate_apptainer_def_content(config, backend)
        def_file = f"{image_name}.def"

        try:
            with open(def_file, "w") as f:
                f.write(def_content)

            build_command = ["apptainer", "build", "--force", image_name, def_file]
            func_logger.debug(f"Running Apptainer build: {' '.join(build_command)}")

            build_timeout = 30 * 60  # 30 minutes for build
            process = subprocess.run(
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
        except Exception as e:
            err_msg = f"Error preparing Apptainer build: {e}"
            logger.error(err_msg)
            func_logger.error(err_msg)
            return err_msg
        finally:
            if os.path.exists(def_file):
                os.remove(def_file)

    # Build env string from backend (container uses /app mount)
    env_str = "PYTHONPATH=/app"  # fallback
    if backend:
        env_vars = backend.container_env_vars()
        env_str = ":".join(f"{k}={v}" for k, v in env_vars.items())

    apptainer_command = [
        "apptainer", "exec",
        "--bind", f"{worktree_path}:/app",
        "--env", env_str,
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


def _run_locally(state: dict, command: str, config: dict, func_logger: logging.Logger, backend=None) -> str:
    """Runs a command in the local environment."""
    timeout_mins = config.get("settings", {}).get("timeout_per_module_mins", 15)
    timeout_seconds = timeout_mins * 60
    worktree_path = state["worktree_path"]

    func_logger.debug(f"Local: Running command in {worktree_path}:\n{command}")

    # Build environment from backend
    env = os.environ.copy()
    if backend:
        for k, v in backend.env_vars(worktree_path).items():
            env[k] = v
    else:
        env["PYTHONPATH"] = f"{worktree_path}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=worktree_path,
            check=False,
            env=env
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


def _run_in_environment(state: ModuleState, command: str) -> str:
    """Helper to run a command in the configured environment (docker, apptainer, or local)."""
    code, tests = state["code"], state["tests"]
    config_path, worktree_path = state["config_path"], state["worktree_path"]
    backend = state.get("backend")

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

    with open(code_abs_path, "w") as f:
        f.write(code)
    with open(test_abs_path, "w") as f:
        f.write(tests)

    # Create init files based on backend patterns.
    source_dir = Path(os.path.dirname(code_abs_path))
    worktree_root = Path(worktree_path)

    current_dir = source_dir
    while worktree_root in current_dir.parents:
        if backend:
            for pattern in backend.init_file_patterns():
                init_file = current_dir / pattern
                if not init_file.exists():
                    func_logger.debug(f"Creating missing {pattern} at {init_file}")
                    init_file.touch()
        else:
            # Fallback: __init__.py
            init_py = current_dir / "__init__.py"
            if not init_py.exists():
                func_logger.debug(f"Creating missing __init__.py at {init_py}")
                init_py.touch()
        current_dir = current_dir.parent

    config = load_config(config_path)
    runner = config.get("settings", {}).get("test_runner", "docker").lower()

    # The command to run inside the environment.
    # Dependencies are baked into the image for container runners.
    full_command = command

    if runner == "docker":
        return _run_with_docker(state, full_command, config, func_logger, backend)
    elif runner == "apptainer":
        return _run_with_apptainer(state, full_command, config, func_logger, backend)
    elif runner == "local":
        # For local, install deps before running the command.
        local_prefix = ""
        if backend:
            local_prefix = f"{backend.local_install_command} && {backend.local_project_install_command} && "
        else:
            # Fallback for no backend
            deps_file = "requirements.txt"
            install_deps_cmd = f"pip install -r {deps_file} && " if os.path.exists(os.path.join(worktree_path, deps_file)) else ""
            local_prefix = f"pip install pytest pytest-cov flake8 mypy bandit && {install_deps_cmd}"
        return _run_locally(state, f"{local_prefix}{command}", config, func_logger, backend)
    else:
        err_msg = f"Invalid test_runner '{runner}' in config.yaml. Must be 'docker', 'apptainer', or 'local'."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg


def run_command_in_project_env(command: str, config_path: Path, project_root: str, func_logger: logging.Logger, backend=None) -> str:
    """Helper to run a command in the configured environment against the whole project."""
    config = load_config(config_path)
    runner = config.get("settings", {}).get("test_runner", "docker").lower()

    # Create a mock state-like object for the runner functions
    mock_state = {"worktree_path": project_root}

    full_command = command

    if runner == "docker":
        return _run_with_docker(mock_state, full_command, config, func_logger, backend)
    elif runner == "apptainer":
        return _run_with_apptainer(mock_state, full_command, config, func_logger, backend)
    elif runner == "local":
        local_prefix = ""
        if backend:
            local_prefix = f"{backend.local_install_command} && {backend.local_project_install_command} && "
        else:
            deps_file = "requirements.txt"
            install_deps_cmd = f"pip install -r {deps_file} && " if os.path.exists(os.path.join(project_root, deps_file)) else ""
            local_prefix = f"pip install pytest pytest-cov flake8 mypy bandit && {install_deps_cmd}"
        return _run_locally(mock_state, f"{local_prefix}{command}", config, func_logger, backend)
    else:
        err_msg = f"Invalid test_runner '{runner}' in config.yaml. Must be 'docker', 'apptainer', or 'local'."
        logger.error(err_msg)
        func_logger.error(err_msg)
        return err_msg


def test_node(state: ModuleState) -> dict:
    """LangGraph node: Executes tests in the configured environment."""
    backend = state.get("backend")
    test_file = Path(state["test_location"]).as_posix()
    source_module = Path(state["spec"]["location"]).with_suffix("").as_posix().replace("/", ".")

    if backend:
        cmd = backend.test_command(Path(test_file), source_module)
    else:
        # Fallback: hardcoded pytest
        cmd = f"pytest {test_file} -v --cov={source_module} --cov-report=term-missing"

    logs = _run_in_environment(state, cmd)
    return {"test_results": logs}


def linter_node(state: ModuleState) -> dict:
    """LangGraph node: Executes linters in the configured environment."""
    backend = state.get("backend")
    target_file = Path(state["spec"]["location"]).as_posix()

    if backend:
        cmd = backend.lint_command(Path(target_file))
        success_prefixes = backend.lint_success_prefixes()
    else:
        # Fallback: hardcoded linters
        cmd = f"flake8 {target_file}; mypy {target_file}; bandit -r {target_file} --skip B101,B104"
        success_prefixes = ["Success: no issues found"]

    logs = _run_in_environment(state, cmd)

    # If the output is empty or matches success prefix, return empty to skip log_checker.
    if not logs.strip() or any(logs.strip().startswith(p) for p in success_prefixes):
        return {"lint_results": ""}
    return {"lint_results": logs}
