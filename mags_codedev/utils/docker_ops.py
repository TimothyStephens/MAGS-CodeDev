"""Container ops: Docker/Podman/Apptainer test and lint execution."""

import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import load_config
from mags_codedev.utils.logger import logger


# ---- Container runtime detection ----

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
    # B15: Use backend's default_base_image
    base_image = backend.default_base_image if backend else "python:3.11-slim"
    dockerfile_parts = [f"FROM {base_image}", "WORKDIR /project"]
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
    # B11: Use _get_container_runtime() instead of dead _get_runtime()
    runtime = _get_container_runtime()
    if runtime is None or runtime in ("apptainer", "singularity"):
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

        # B14: Use backend container_env_vars if available
        env_vars = backend.container_env_vars() if backend else {"PYTHONPATH": "/project"}
        env_args = []
        for k, v in env_vars.items():
            env_args.extend(["--env", f"{k}={v}"])

        # Run the command
        run_cmd = [
            runtime_bin, "run", "--rm",
            "-v", f"{worktree_path}:/project",
            "-w", "/project",
            *env_args,
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
    """Runs a command inside an Apptainer/Singularity container."""
    # B12: Detect actual binary (apptainer or singularity)
    runtime_bin = "apptainer" if shutil.which("apptainer") else "singularity"
    runtime_name = runtime_bin  # For logging

    try:
        worktree_path = state["worktree_path"]
        image_name = config.get("settings", {}).get("apptainer_test_image", "mags-dev-env.sif")
        image_path = Path(worktree_path) / image_name

        # Check if image exists; if not, build it
        needs_build = not image_path.exists()

        if needs_build:
            func_logger.info(f"Building {runtime_name} image...")
            def_content = _generate_apptainer_def_content(config, backend)
            def_path = Path(worktree_path) / "project.def"
            def_path.write_text(def_content)

            # Copy requirements.txt if it exists in worktree
            req_path = Path(worktree_path) / "requirements.txt"
            if req_path.exists():
                shutil.copy(req_path, Path(worktree_path) / "reqs_copy.txt")

            build_cmd = [
                runtime_bin, "build", str(image_path),
                str(def_path)
            ]
            build_result = subprocess.run(
                build_cmd, capture_output=True, text=True, cwd=worktree_path
            )
            if build_result.returncode != 0:
                func_logger.error(f"{runtime_name} build failed:\n{build_result.stderr}")
                return build_result.stderr

        # B14: Use backend container_env_vars if available
        env_vars = backend.container_env_vars() if backend else {"PYTHONPATH": "/project"}
        env_args = []
        for k, v in env_vars.items():
            env_args.extend(["--env", f"{k}={v}"])

        # Run the command
        run_cmd = [
            runtime_bin, "exec",
            "--bind", f"{worktree_path}:/project",
            *env_args,
            str(image_path),
            "/bin/bash", "-c", command,
        ]
        result = subprocess.run(
            run_cmd, capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        func_logger.debug(f"{runtime_name} output:\n{output}")
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out (120s limit)."
    except Exception as e:
        return f"ERROR: {runtime_name} execution failed: {e}"


def _run_locally(state: dict, command: str, config: dict, func_logger: logging.Logger, backend=None) -> str:
    """Runs a command in the local environment."""
    try:
        worktree_path = state["worktree_path"]
        # B14: Use backend env_vars for local runner too
        extra_env = backend.env_vars(worktree_path) if backend else {"PYTHONPATH": worktree_path}
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            capture_output=True,
            text=True,
            cwd=worktree_path,
            timeout=120,
            env={**os.environ, **extra_env},
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
        # B13: os already imported at module level
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags_codedev.func.{log_hash}")
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
        runtime = _get_container_runtime(env_setting)
        if runtime and runtime in ("podman", "docker"):
            return _run_with_docker(state, command, config, func_logger, backend)
        # Fallback: try any docker-compatible runtime
        runtime = _get_container_runtime()
        if runtime and runtime in ("podman", "docker"):
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


def run_command_in_project_env(
    command: str,
    config_path: Path,
    project_root: str,
    func_logger: logging.Logger,
    backend=None,
) -> str:
    """Helper to run a command in the configured environment against the whole project."""
    state = ModuleState(
        worktree_path=project_root,
        config_path=config_path,
        log_filepath=None,
    )
    return _run_in_environment(state, command)



def _get_func_logger(state: ModuleState) -> logging.Logger:
    """Extract the function logger from state, falling back to the root logger."""
    if state.get("log_filepath"):
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        return logging.getLogger(f"mags_codedev.func.{log_hash}")
    return logger


def _write_worktree_files(state: ModuleState, func_logger: logging.Logger) -> None:
    """Write code and tests to the worktree, creating __init__.py files as needed.

    Must be called before running test/lint commands so the container or local
    runner sees the latest generated files.

    E4: Error handling for all file operations.
    E3: Also walks test directory for __init__.py creation.
    """
    backend = state.get("backend")
    worktree_path = state["worktree_path"]

    code = state.get("code", "")
    tests = state.get("tests", "")
    spec_location = state.get("module_location", "")
    test_location = state.get("test_location", "")

    # Write source code
    if code and spec_location:
        try:
            code_abs_path = os.path.join(worktree_path, spec_location)
            os.makedirs(os.path.dirname(code_abs_path), exist_ok=True)
            with open(code_abs_path, "w") as f:
                f.write(code)
        except OSError as e:
            func_logger.error(f"Failed to write source code to {code_abs_path}: {e}")

    # Write tests
    if tests and test_location:
        try:
            test_abs_path = os.path.join(worktree_path, test_location)
            os.makedirs(os.path.dirname(test_abs_path), exist_ok=True)
            with open(test_abs_path, "w") as f:
                f.write(tests)
        except OSError as e:
            func_logger.error(f"Failed to write tests to {test_abs_path}: {e}")

    def _ensure_init_files(target_dir: Path, worktree_root: Path) -> None:
        """Walk up from target_dir to worktree_root, creating init files."""
        current_dir = target_dir
        while worktree_root in current_dir.parents or current_dir == worktree_root:
            if backend:
                for pattern in backend.init_file_patterns():
                    init_file = current_dir / pattern
                    if not init_file.exists():
                        func_logger.debug(f"Creating missing {pattern} at {init_file}")
                        try:
                            init_file.touch()
                        except OSError as e:
                            func_logger.error(f"Failed to create {init_file}: {e}")
            else:
                init_py = current_dir / "__init__.py"
                if not init_py.exists():
                    func_logger.debug(f"Creating missing __init__.py at {init_py}")
                    try:
                        init_py.touch()
                    except OSError as e:
                        func_logger.error(f"Failed to create {init_py}: {e}")
            current_dir = current_dir.parent

    # Create __init__.py files for source directory
    if spec_location:
        try:
            source_dir = Path(os.path.join(worktree_path, spec_location)).parent
            _ensure_init_files(source_dir, Path(worktree_path))
        except Exception as e:
            func_logger.error(f"Failed to create __init__.py for source dir: {e}")

    # E3: Create __init__.py files for test directory
    if test_location:
        try:
            test_dir = Path(os.path.join(worktree_path, test_location)).parent
            _ensure_init_files(test_dir, Path(worktree_path))
        except Exception as e:
            func_logger.error(f"Failed to create __init__.py for test dir: {e}")


def test_node(state: ModuleState) -> dict:
    """LangGraph node: Executes tests in the configured environment."""
    backend = state.get("backend")
    test_file = state.get("test_location", "tests/test_placeholder.py")

    if backend:
        command = backend.test_command(Path(test_file), state.get("module_location", ""))
    else:
        command = f"python3 -m pytest {test_file} -v --tb=short"

    # Write code/tests to worktree before running
    func_logger = _get_func_logger(state)
    _write_worktree_files(state, func_logger)

    logs = _run_in_environment(state, command)
    return {"test_results": logs}


def linter_node(state: ModuleState) -> dict:
    """LangGraph node: Executes linters in the configured environment."""
    backend = state.get("backend")
    source_file = state.get("module_location", "src/placeholder.py")

    if backend:
        command = backend.lint_command(Path(source_file))
    else:
        command = (
            f"python3 -m flake8 {source_file} --max-line-length=120 --extend-ignore=E302,E303,E305; "
            f"python3 -m mypy {source_file} --ignore-missing-imports"
        )

    # Write code/tests to worktree before running
    func_logger = _get_func_logger(state)
    _write_worktree_files(state, func_logger)

    logs = _run_in_environment(state, command)
    return {"lint_results": logs}
