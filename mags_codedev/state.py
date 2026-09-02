"""ModuleState TypedDict: shared state schema for LangGraph nodes."""

from typing import TypedDict, List, Dict, Any, Optional, Required
from pathlib import Path

from mags_codedev.backends.language_backend import LanguageBackend


class ModuleState(TypedDict, total=False):
    """State dictionary passed between LangGraph nodes. Each parallel module gets its own isolated state."""

    # Input — always provided at graph init
    module_location: Required[str]          # Path to the module, e.g. "src/pricing.py"
    spec: Required[Dict[str, Any]]          # Description and dependencies from manifest.json
    config_path: Required[Path]             # Path to config.yaml
    worktree_path: Required[str]            # Path to isolated git worktree
    test_location: Required[str]            # Relative path for test file, e.g. "tests/src/test_foo.py"
    backend: Required["LanguageBackend"]    # Language-specific tooling (injected at graph init)
    log_filepath: Required[str]             # Path to module-specific log file
    base_dir: Required[str]                 # Base artifact directory (e.g. ".mags-codedev")

    # Generated Artifacts — always initialized (may be empty string)
    code: Required[str]                     # Current iteration of module code
    tests: Required[str]                    # Current iteration of test code

    # Execution Feedback — always initialized
    test_results: Required[str]             # stdout/stderr from isolated test run
    lint_results: Required[str]             # Output from linters/type-checkers

    # Agent Feedback (scoped — cleared after log_checker processes)
    test_error_summary: Required[str]       # Error summary from test phase only
    review_comments: Required[List[str]]    # Aggregated feedback from multi-LLM review
    error_location: Optional[str]           # Set by log_checker: 'SOURCE_CODE' or 'TEST_CODE'

    # Convergence Detection
    previous_code_hash: Required[Optional[str]]   # SHA-256 of code from prior iteration
    previous_test_hash: Required[Optional[str]]   # SHA-256 of tests from prior iteration

    # Metadata — always initialized
    iteration_count: Required[int]          # Times looped back to coder/tester
    max_test_fix_iterations: Required[int]  # Budget for test/lint fix cycles
    max_review_rounds: Required[int]        # Budget for review revision cycles
    review_round_count: Required[int]       # Tracks review-specific iterations
    status: Required[str]                   # 'in_progress', 'success', 'failed'

    # Session Tracking (lifecycle logging)
    _previous_iterations: Required[int]     # Total iterations from previous build sessions (from DB)
    _session_number: Required[int]          # Which build session this is (1-based, increments on restart)
    _exit_reason: Optional[str]             # Set by session_end_node: 'SUCCESS', 'MAX_TEST_ITERATIONS', etc.
    # Dependency Context
    dependency_code: Required[Dict[str, str]]   # Maps dep location -> source code
    project_instructions: Required[str]         # Project-level instructions from AGENT.md
    _next_reason: str                 # Routing reason for INFO logging
    test_returncode: int              # exit code of last test run; 0 = pass
    log_level: Required[str]          # Configured log level: 'info'/'debug'/'trace' (controls per-module log detail)
