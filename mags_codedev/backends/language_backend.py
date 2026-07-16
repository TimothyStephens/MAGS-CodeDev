"""Abstract protocol for language-specific tooling and prompts.

Each backend implements this interface so the generic orchestration
layer (graph, CLI, docker_ops, agents) never touches language-specific
commands, prompts, or environment variables directly.
"""


from abc import abstractmethod
from pathlib import Path
from typing import Protocol


class LanguageBackend(Protocol):
    """Language-specific tool invocation, environment setup, and prompts."""

    # ── Identification ─────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Lowercase identifier used in config, e.g. 'python', 'typescript'."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name, e.g. 'Python'."""
        ...

    # ── Container Environment ──────────────────────────────────────

    @property
    @abstractmethod
    def default_base_image(self) -> str:
        """Docker base image, e.g. 'python:3.11-slim'."""
        ...

    @property
    @abstractmethod
    def deps_filename(self) -> str:
        """Canonical dependency file name, e.g. 'requirements.txt'."""
        ...

    # ── Local Runner ───────────────────────────────────────────────

    @property
    @abstractmethod
    def local_install_command(self) -> str:
        """Command to install core deps on the host, e.g. 'pip install pytest ...'."""
        ...

    @property
    @abstractmethod
    def local_project_install_command(self) -> str:
        """Command to install the project itself locally."""
        ...

    # ── Tool Commands ──────────────────────────────────────────────

    @abstractmethod
    def test_command(self, test_file: Path, source_module: str) -> str:
        """Command to run tests with coverage for a single file."""
        ...

    @abstractmethod
    def test_command_project(self) -> str:
        """Command to run all tests in the project (for CLI test command)."""
        ...

    @abstractmethod
    def lint_command(self, target_file: Path) -> str:
        """Semicolon-separated lint/type-check/security commands."""
        ...

    @abstractmethod
    def lint_success_prefixes(self) -> list[str]:
        """Output prefixes that indicate lint success (skip log_checker)."""
        ...

    @abstractmethod
    def test_failure_keywords(self) -> list[str]:
        """Keywords indicating test failure (checked in graph.py)."""
        ...

    # ── Environment Setup ──────────────────────────────────────────

    @abstractmethod
    def init_file_patterns(self) -> list[str]:
        """Files to create for package init, e.g. ['__init__.py']."""
        ...

    @abstractmethod
    def env_vars(self, worktree_path: str) -> dict[str, str]:
        """Environment variables for local execution (host paths)."""
        ...

    @abstractmethod
    def container_env_vars(self) -> dict[str, str]:
        """Environment variables inside the container (e.g., PYTHONPATH=/app)."""
        ...

    # ── Agent Prompt Templates ─────────────────────────────────────

    @abstractmethod
    def coder_system_prompt(self) -> str:
        """System prompt for the coder agent."""
        ...

    @abstractmethod
    def tester_system_prompt(self, module_path: str, import_hint: str) -> str:
        """System prompt for the tester agent."""
        ...

    @abstractmethod
    def get_import_hint(self, source_location: Path) -> str:
        """How to import a module, e.g. 'from pkg.module import ...'."""
        ...

    @abstractmethod
    def test_human_template_initial(self) -> str:
        """Human template for initial test generation."""
        ...

    @abstractmethod
    def test_human_template_fix(self) -> str:
        """Human template for test fix cycle."""
        ...
