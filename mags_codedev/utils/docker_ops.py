import logging
import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import load_config
from mags_codedev.utils.logger import logger


# ---- Container runtime detection ----

def _get_runtime(preferred: str = "podman") -> Optional[str]:
    """Detect available container runtime.
    
    Returns the best available runtime matching *preferred*.
    Default priority: podman > docker (container runtimes only).
    """
    if preferred and shutil.which(preferred):
        return preferred
    if shutil.which("podman"):
        return "podman"
    if shutil.which("docker"):
        return "docker"
    return None


def _get_container_runtime(preferred: str = "podman") -> Optional[str]:
    """Detect any available container runtime (docker/podman/apptainer/singularity).
    
    Returns the runtime name or None. Default priority: podman > docker > apptainer > singularity.
    """
    if preferred:
        if shutil.which(preferred):
            return preferred
        # Normalize singularity -> apptainer for preference matching
        if preferred == "singularity" and shutil.which("apptainer"):
            return "apptainer"
    
    for runtime in ["podman", "docker", "apptainer", "singularity"]:
        if shutil.which(runtime):
            return runtime
    return None


def _generate_dockerfile_content(config: dict, backend) -> str:
    """Generates Dockerfile content based on project dependencies and backend."""
    dockerfile_parts = ["FROM python:3.11-slim", "WORKDIR /project"]
    pip_flags = "--no-cache-dir"

    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        dockerfile_parts.append("COPY requirements.txt .")
        dockerfile_parts.append(f"RUN pip install {pip_flags} -r requirements.txt")
    else:
        install_pkgs = config.get("settings", {}).get("install_packages", [])
        if install_pkgs:
            dockerfile_parts.append(
                f"RUN pip install {pip_flags} {' '.join(install_pkgs)}"
            )

    if backend:
        dockerfile_parts.append(backend.dockerfile_content())

    return "\n".join(dockerfile_parts)


def _generate_apptainer_def_content(config: dict, backend) -> str:
    """Generates Apptainer definition file content based on backend."""
    bootstrap = config.get("settings", {}).get("apptainer_bootstrap", "docker")
    from_image = config.get("settings", {}).get("apptainer_from_image", "python:3.11-slim")

    return f"""Bootstrap: {bootstrap}
From: {from_image}

%post
    pip install --no-cache-dir pytest flake8 mypy

%files
    requirements.txt /tmp/requirements.txt

%post
    pip install --no-cache-dir -r /tmp/requirements.txt

%environment
    export SINGULARITY_ROOTFS=/project
""".strip()


def _run_with_docker(state: dict, command: str, config: dict, func_logger: logging.Logger, backend=None) -> str:
    """Runs a command inside a Docker/Podman container, building the image if necessary."""
    runtime = _get_runtime()
    if runtime is None:
        return "ERROR: No container runtime (docker/podman) found."

    runtime_bin = runtime  # 'docker' or 'podman'

    try:
        worktree_path = state["worktree_path"]
        image_tag = config.get("settings", {}).get("docker_test_image", f"mags-codedev-{runtime}:latest")

        # Check if image exists; if not, build it
        check_cmd = [runtime_bin, "image", "inspect", image_tag]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        needs_build = result.returncode != 0

        if needs_build:
            func_logger.info(f"Building {runtime} image...")
            dockerfile_content = _generate_dockerfile_content(config, backend)
            dockerfile_path = Path(worktree_path) / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)

            build_cmd = [
                runtime_bin, "build", "-t", image_tag,
                "-f", str(dockerfile_path),
                worktree_path
            ]
            build_result = subprocess.run(
                build_cmd, capture_output=True, text=True, cwd=worktree_path
            )
            if build_result.returncode != 0:
                func_logger.error(f"{runtime} build failed:\n{build_result.stderr}")
                return build_result.stderr
            func_logger.info(f"{runtime} image built successfully.")

        # Run the command
        run_cmd = [
            runtime_bin, "run", "--rm",
            "-v", f"{worktree_path}:/project",
            "-w", "/project",
            "--env", f"PYTHONPATH=/project",
            image_tag,
            "/bin/bash", "-c", command,
        ]
        result = subprocess.run(
            run_cmd, capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        func_logger.debug(f"{runtime} output:\n{output}")
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out (120s limit)."
    except Exception as e:
        return f"ERROR: {runtime} execution failed: {e}"


def _run_with_apptainer(state: dict, command: str, config: dict, func_logger: logging.Logger, backend=None) -> str:
    """Runs a command inside an Apptainer container, building the image if necessary."""
    try:
        worktree_path = state["worktree_path"]
        image_name = config.get("settings", {}).get("apptainer_test_image", "mags-dev-env.sif")
        image_path = Path(worktree_path) / image_name

        # Check if image exists; if not, build it
        needs_build = not image_path.exists()

        if needs_build:
            func_logger.info("Building Apptainer image...")
            def_content = _generate_apptainer_def_content(config, backend)
            def_path = Path(worktree_path) / "project.def"
            def_path.write_text(def_content)

            # Copy requirements.txt if it exists
            req_path = Path("requirements.txt")
            if req_path.exists():
                shutil.copy(req_path, worktree_path)

            build_cmd = [
                "apptainer", "build", str(image_path),
                str(def_path)
            ]
            build_result = subprocess.run(
                build_cmd, capture_output=True, text=True, cwd=worktree_path
            )
            if build_result.returncode != 0:
                func_logger.error(f"Apptainer build failed:\n{build_result.stderr}")
                return build_result.stderr

        # Run the command
        run_cmd = [
            "apptainer", "exec",
            "--bind", f"{worktree_path}:/project",
            "--env", "PYTHONPATH=/project",
            str(image_path),
            "/bin/bash", "-c", command,
        ]
        result = subprocess.run(
            run_cmd, capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        func_logger.debug(f"Apptainer output:\n{output}")
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out (120s limit)."
    except Exception as e:
        return f"ERROR: Apptainer execution failed: {e}"


def _run_locally(state: dict, command: str, config: dict, func_logger: logging.Logger, backend=None) -> str:
    """Runs a command in the local environment."""
    try:
        worktree_path = state["worktree_path"]
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=worktree_path,
            timeout=120,
            env={**os.environ, "PYTHONPATH": worktree_path},
        )
        output = result.stdout + result.stderr
        func_logger.debug(f"Local output:\n{output}")
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out (120s limit)."
    except Exception as e:
        return f"ERROR: Local execution failed: {e}"


def _run_in_environment(state: ModuleState, command: str) -> str:
    """Helper to run a command in the configured environment (docker, apptainer, or local).
    
    Config key: settings.test_runner (values: auto, podman, docker, apptainer, singularity, local).
    Default: 'auto' — detects best available runtime (podman > docker > apptainer > singularity > local).
    """
    config_path = state["config_path"]
    config = load_config(config_path)

    backend = state.get("backend")
    if state.get("log_filepath"):
        import os
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger

    # FIX C1: Use 'test_runner' config key, default to 'auto' (not 'local')
    env_setting = config.get("settings", {}).get("test_runner", "auto")

    if env_setting == "auto":
        # Auto-detect best available runtime
        runtime = _get_container_runtime("podman")
        if runtime in ("podman", "docker"):
            return _run_with_docker(state, command, config, func_logger, backend)
        if runtime in ("apptainer", "singularity"):
            return _run_with_apptainer(state, command, config, func_logger, backend)
        func_logger.info("No container runtime available. Running locally.")
        return _run_locally(state, command, config, func_logger, backend)

    elif env_setting in ("docker", "podman"):
        runtime = _get_runtime(env_setting)
        if runtime:
            return _run_with_docker(state, command, config, func_logger, backend)
        # Fallback: try any docker-compatible runtime
        runtime = _get_runtime()
        if runtime:
            func_logger.warning(
                f"Requested '{env_setting}' but only '{runtime}' available. Using '{runtime}'."
            )
            return _run_with_docker(state, command, config, func_logger, backend)
        func_logger.warning(
            f"Container runtime '{env_setting}' requested but not available. Falling back to local."
        )
        return _run_locally(state, command, config, func_logger, backend)

    elif env_setting in ("apptainer", "singularity"):  # FIX M2: accept singularity alias
        if shutil.which("apptainer") or shutil.which("singularity"):
            return _run_with_apptainer(state, command, config, func_logger, backend)
        func_logger.warning(
            "Apptainer/Singularity requested but not available. Falling back to local."
        )
        return _run_locally(state, command, config, func_logger, backend)

    elif env_setting == "local":
        return _run_locally(state, command, config, func_logger, backend)

    else:
        func_logger.warning(
            f"Unknown test_runner setting '{env_setting}'. Falling back to local execution."
        )
        return _run_locally(state, command, config, func_logger, backend)


def run_command_in_project_env(command: str, config_path: Path, project_root: str, func_logger: logging.Logger, backend=None) -> str:
    """Helper to run a command in the configured environment against the whole project."""
    state = ModuleState(
        worktree_path=project_root,
        config_path=config_path,
        log_filepath=None,
    )
    return _run_in_environment(state, command)


def _write_worktree_files(state: ModuleState, func_logger: logging.Logger) -> None:
    """Write code and tests to the worktree, creating __init__.py files as needed.

    Must be called before running test/lint commands so the container or local
    runner sees the latest generated files.
    """
    backend = state.get("backend")
    worktree_path = state["worktree_path"]

    code = state.get("code", "")
    tests = state.get("tests", "")
    spec_location = state.get("module_location", "")
    test_location = state.get("test_location", "")

    # Write source code
    if code and spec_location:
        code_abs_path = os.path.join(worktree_path, spec_location)
        os.makedirs(os.path.dirname(code_abs_path), exist_ok=True)
        with open(code_abs_path, "w") as f:
            f.write(code)

    # Write tests
    if tests and test_location:
        test_abs_path = os.path.join(worktree_path, test_location)
        os.makedirs(os.path.dirname(test_abs_path), exist_ok=True)
        with open(test_abs_path, "w") as f:
            f.write(tests)

    # Create __init__.py files up the directory tree
    if spec_location:
        source_dir = Path(os.path.join(worktree_path, spec_location)).parent
        worktree_root = Path(worktree_path)

        current_dir = source_dir
        while worktree_root in current_dir.parents or current_dir == worktree_root:
            if backend:
                for pattern in backend.init_file_patterns():
                    init_file = current_dir / pattern
                    if not init_file.exists():
                        func_logger.debug(f"Creating missing {pattern} at {init_file}")
                        init_file.touch()
            else:
                init_py = current_dir / "__init__.py"
                if not init_py.exists():
                    func_logger.debug(f"Creating missing __init__.py at {init_py}")
                    init_py.touch()
            current_dir = current_dir.parent


def test_node(state: ModuleState) -> dict:
    """LangGraph node: Executes tests in the configured environment."""
    backend = state.get("backend")
    test_file = state.get("test_location", "tests/test_placeholder.py")

    if backend:
        command = backend.get_test_command(test_file)
    else:
        command = f"python -m pytest {test_file} -v --tb=short"

    # Write code/tests to worktree before running
    if state.get("log_filepath"):
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
    _write_worktree_files(state, func_logger)

    logs = _run_in_environment(state, command)
    return {"test_results": logs}


def linter_node(state: ModuleState) -> dict:
    """LangGraph node: Executes linters in the configured environment."""
    backend = state.get("backend")
    source_file = state.get("module_location", "src/placeholder.py")

    if backend:
        command = backend.get_lint_command(source_file)
    else:
        command = f"python -m flake8 {source_file} --max-line-length=120; python -m mypy {source_file} --ignore-missing-imports"

    # Write code/tests to worktree before running
    if state.get("log_filepath"):
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
    _write_worktree_files(state, func_logger)

    logs = _run_in_environment(state, command)
    return {"lint_results": logs}
