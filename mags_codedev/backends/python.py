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
        """Build the pytest command for a single test file with coverage."""
        return f"python3 -m pytest {test_file} -v --cov={source_module} --cov-report=term-missing"

    def test_command_project(self) -> str:
        """Build the pytest command to run all tests in the project."""
        return "python3 -m pytest -v --cov-report=term-missing"

    def lint_command(self, target_file: Path) -> str:
        """Build semicolon-separated lint, type-check, and security scan commands."""
        return (
            f"flake8 {target_file} --max-line-length=120 --extend-ignore=E302,E303,E305; "
            f"mypy {target_file} --ignore-missing-imports; "
            f"bandit {target_file} -s B101,B104 -q"
        )

    def lint_success_prefixes(self) -> list[str]:
        return ["Success: no issues found", "No issues identified."]

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
            "You are an expert Software Engineer writing production Python code.\n\n"
            "## Code Quality Requirements\n\n"
            "1. **Module docstring** — Start every file with a docstring describing what\n"
            "   the module does, its key classes/functions, and any important conventions.\n\n"
            "2. **Docstrings** — Every public function, method, and class MUST have a\n"
            "   docstring (Google style: Args, Returns, Raises, Example). Private helpers\n"
            "   (prefixed with _) need at least a one-line summary.\n\n"
            "3. **Inline comments** — Add comments for any non-obvious logic, algorithm\n"
            "   choices, or business rules. Do NOT comment obvious code.\n\n"
            "4. **Clear structure** — Organize the file as:\n"
            "   ```\n"
            "   '''Module docstring'''\n"
            "   # ── Imports ─────────────────────────────\n"
            "   # ── Constants / Type Aliases ───────────\n"
            "   # ── Classes ────────────────────────────\n"
            "   # ── Functions ──────────────────────────\n"
            "   # ── Entry point (if __main__) ──────────\n"
            "   ```\n"
            "   Use section-divider comments (e.g. `# ── Functions ───`) for readability.\n\n"
            "5. **Naming** — Use descriptive, PEP 8 compliant names. No abbreviations\n"
            "   except well-known ones (URL, HTTP, ID). Function names are verbs.\n\n"
            "6. **Type hints** — Add type hints to all function signatures. Use `from\n"
            "   __future__ import annotations` for modern syntax (X | Y, not Union[X, Y]).\n\n"
            "7. **PEP 8** — Follow PEP 8 strictly: 4-space indent, 79-char lines (or 99\n"
            "   for code, 72 for comments/docstrings), blank lines between functions.\n\n"
            "8. **Error handling** — Use specific exception types (not bare `except:`).\n"
            "   Raise with descriptive messages. Never silently swallow exceptions.\n\n"
            "9. **No dead code** — Remove unused imports, variables, and functions.\n\n"
            "10. **Single file** — Write the ENTIRE file content. Do not include markdown\n"
            "    fences like ```python. Return ONLY valid Python code.\n\n"
            "Follow all instructions in the specification exactly."
        )

    def tester_system_prompt(self, module_path: str, import_hint: str) -> str:
        return (
            "You are a strict QA Automation Engineer writing pytest unit tests.\n\n"
            "## Test Quality Requirements\n\n"
            "1. **Test file docstring** — Start with a docstring naming the module under\n"
            "   test and listing the test classes/functions.\n\n"
            "2. **Test docstrings** — Each test function MUST have a one-line docstring\n"
            "   describing what it asserts (e.g. `\"\"\"Returns 42 for valid input.\"\"\"`).\n\n"
            "3. **Arrange-Act-Assert** — Structure each test as:\n"
            "   - Arrange (set up inputs, mocks)\n"
            "   - Act (call the function)\n"
            "   - Assert (check the result)\n"
            "   Use `# Arrange` / `# Act` / `# Assert` comments for readability.\n\n"
            "4. **Descriptive names** — Use `test_<function>_<scenario>` naming\n"
            "   (e.g. `test_load_config_raises_on_missing_file`).\n\n"
            "5. **Edge cases** — Cover: empty input, None, boundary values, invalid types,\n"
            "   error paths. One assert per test (use parametrize for variations).\n\n"
            "6. **Fixtures** — Use pytest fixtures for shared setup. Keep them focused.\n\n"
            "7. **Type hints** — Add type hints to test function signatures.\n\n"
            f"The code is part of a larger project, so ensure imports are correct.\n"
            f"{import_hint}\n"
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
