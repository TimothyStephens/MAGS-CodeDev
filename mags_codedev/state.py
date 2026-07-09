from typing import TYPE_CHECKING, TypedDict, List, Dict, Any, Optional
from pathlib import Path

if TYPE_CHECKING:
    from mags_codedev.backends.language_backend import LanguageBackend


class ModuleState(TypedDict):
    """
    The state dictionary passed between nodes in the LangGraph workflow.
    Each parallel execution of a module gets its own isolated state.
    """
    module_location: str          # The path to the module, e.g. "src/pricing.py"
    spec: Dict[str, Any]          # The description and dependencies from manifest.json

    # Generated Artifacts
    code: str                     # The current iteration of the module's code
    tests: str                    # The current iteration of the module's test code

    # Execution Feedback
    test_results: str             # stdout/stderr from the isolated test run
    lint_results: str             # Output from linters/type-checkers

    # Agent Feedback
    review_comments: List[str]    # Aggregated feedback from the multi-LLM review
    error_summary: str            # The log_checker's human-readable summary of failures
    error_location: Optional[str]  # Set by log_checker: 'SOURCE_CODE' or 'TEST_CODE'

    # Metadata
    backend: "LanguageBackend"    # Language-specific tooling (injected at graph init)
    config_path: Path             # Path to the config.yaml file
    worktree_path: str            # Path to the isolated git worktree
    test_location: str            # Relative path for the test file (e.g. tests/src/test_foo.py)
    iteration_count: int          # Tracks how many times we've looped back to the Coder
    max_iterations: int           # Hard stop to prevent infinite agent loops
    status: str                   # 'in_progress', 'success', 'failed'
    log_filepath: str             # Path to the module-specific log file
