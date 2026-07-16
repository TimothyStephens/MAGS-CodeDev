"""ModuleState TypedDict: shared state schema for LangGraph nodes."""

from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict, List, Dict, Any, Optional
from pathlib import Path

if TYPE_CHECKING:
    from mags_codedev.backends.language_backend import LanguageBackend


class ModuleState(TypedDict, total=False):
    """State dictionary passed between LangGraph nodes. Each parallel module gets its own isolated state."""

    # Input
    module_location: str          # Path to the module, e.g. "src/pricing.py"
    spec: Dict[str, Any]          # Description and dependencies from manifest.json
    config_path: Path             # Path to config.yaml
    worktree_path: str            # Path to isolated git worktree
    test_location: str            # Relative path for test file, e.g. "tests/src/test_foo.py"
    backend: "LanguageBackend"    # Language-specific tooling (injected at graph init)
    log_filepath: str             # Path to module-specific log file
    base_dir: str                 # Base artifact directory (e.g. ".mags-codedev")

    # Generated Artifacts
    code: str                     # Current iteration of module code
    tests: str                    # Current iteration of test code

    # Execution Feedback
    test_results: str             # stdout/stderr from isolated test run
    lint_results: str             # Output from linters/type-checkers

    # Agent Feedback (scoped — cleared after log_checker processes)
    test_error_summary: str       # Error summary from test phase only
    review_comments: List[str]    # Aggregated feedback from multi-LLM review
    error_location: Optional[str] # Set by log_checker: 'SOURCE_CODE' or 'TEST_CODE'

    # Convergence Detection
    previous_code_hash: Optional[str]   # SHA-256 of code from prior iteration
    previous_test_hash: Optional[str]   # SHA-256 of tests from prior iteration

    # Metadata
    iteration_count: int          # Times looped back to coder/tester
    max_test_fix_iterations: int  # Budget for test/lint fix cycles
    max_review_rounds: int        # Budget for review revision cycles
    review_round_count: int       # Tracks review-specific iterations
    status: str                   # 'in_progress', 'success', 'failed'
    # Dependency Context
    dependency_code: Dict[str, str]   # Maps dep location -> source code
    project_instructions: str         # Project-level instructions from AGENT.md
    offline: bool                     # Skip LLM API calls (use stubs)
