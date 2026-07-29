"""Python 3 language backend.

Extracts all Python-specific commands, prompts, and environment
configuration so the generic orchestration layer never touches them.
"""

from __future__ import annotations

from pathlib import Path



class PythonBackend:
    """Python 3 implementation of :class:`LanguageBackend`."""

    # ── Identification ─────────────────────────────────────────────

    name: str = "python"
    display_name: str = "Python"

    # ── Container Environment ──────────────────────────────────────

    default_base_image: str = "python:3.11-slim"

    deps_filename: str = "requirements.txt"

    def dockerfile_content(self) -> str:
        """Extra Dockerfile lines: install the Python test/lint toolchain."""
        return "RUN pip install --no-cache-dir pytest pytest-cov flake8 mypy bandit"

    # ── Local Runner ───────────────────────────────────────────────

    local_install_command: str = (
        "pip install pytest pytest-cov flake8 mypy bandit"
    )

    local_project_install_command: str = "pip install -r requirements.txt"

    # ── Tool Commands ──────────────────────────────────────────────

    def test_command(self, test_file: Path, source_module: str) -> str:
        """Build the pytest command for a single test file."""
        return f"python3 -m pytest {test_file} -v"

    def test_command_project(self) -> str:
        """Build the pytest command to run all tests in the project."""
        return "python3 -m pytest -v"

    def lint_command(self, target_file: Path) -> str:
        """Build semicolon-separated lint, type-check, and security scan commands."""
        return (
            f"flake8 {target_file} --max-line-length=120 --extend-ignore=E302,E303,E305; "
            f"mypy {target_file} --ignore-missing-imports"
        )

    def lint_success_prefixes(self) -> list[str]:
        return ["Success: no issues found"]

    def test_failure_keywords(self) -> list[str]:
        """Keywords whose presence indicates test failure."""
        return ["FAILED", "ERROR"]

    # ── Environment Setup ──────────────────────────────────────────

    def init_file_patterns(self) -> list[str]:
        return ["__init__.py"]

    def env_vars(self, worktree_path: str) -> dict[str, str]:
        """Environment variables for local execution (uses host path).
        Container callers must translate paths (e.g., /app) themselves."""
        return {"PYTHONPATH": f"{worktree_path}"}

    def container_env_vars(self) -> dict[str, str]:
        """Environment variables inside the container (uses /app mount)."""
        return {"PYTHONPATH": "/app"}

    # ── Agent Prompt Templates ─────────────────────────────────────

    def coder_system_prompt(self) -> str:
        return (
            "You are an expert Software Engineer.\n"
            "Write a full, clean, production-ready Python module. "
            "Follow all instructions exactly.\n"
            "Return ONLY valid Python code for the entire file. "
            "Do not include markdown formatting like ```python."
        )

    def tester_system_prompt(self, module_path: str, import_hint: str) -> str:
        return (
            "You are a strict QA Automation Engineer.\n"
            "Write robust `pytest` unit tests for the provided Python module.\n"
            "The code is part of a larger project, so ensure imports are correct.\n"
            f"{import_hint}\n"
            "Your tests should cover all functions and classes in the module.\n"
            "Include edge cases, type boundary checks, and failure scenarios.\n"
            f"The module to be tested is located at the path '{module_path}'.\n"
            "Return ONLY valid Python code for the test file. "
            "Do not include markdown formatting."
        )

    def get_import_hint(self, source_location: Path) -> str:
        module_path = source_location.with_suffix("").as_posix().replace("/", ".")
        return (
            f"The module to test is '{source_location}'. "
            f"You can import from it using `from {module_path} import ...`."
        )

    def test_human_template_initial(self) -> str:
        return (
            "Module Specification:\n{spec}\n\n"
            "Generated Code to Test:\n{code}"
        )

    def test_human_template_fix(self) -> str:
        return (
            "The previous attempt to write tests failed. Please fix them.\n\n"
            "Module Specification:\n{spec}\n\n"
            "Generated Code to Test:\n{code}\n\n"
            "PREVIOUS (BROKEN) TESTS:\n{previous_tests}\n\n"
            "DIAGNOSIS OF FAILURE:\n{test_error_summary}\n\n"
            "Your task is to provide a new, corrected version of the pytest unit tests."
        )
