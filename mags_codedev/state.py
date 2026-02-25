from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

class FunctionState(TypedDict):
    """
    The state dictionary passed between nodes in the LangGraph workflow.
    Each parallel execution of a function gets its own isolated state.
    """
    function_name: str
    spec: Dict[str, Any]          # The purpose, input, output from manifest.json
    
    # Generated Artifacts
    code: str                     # The current iteration of the function code
    tests: str                    # The current iteration of the pytest code
    
    # Execution Feedback
    test_results: str             # stdout/stderr from the Docker isolated test run
    lint_results: str             # Output from tools like Flake8/MyPy
    
    # Agent Feedback
    review_comments: List[str]    # Aggregated feedback from the multi-LLM review
    error_summary: str            # The log_checker's human-readable summary of failures
    
    # Metadata
    config_path: Path             # Path to the config.yaml file
    worktree_path: str            # Path to the isolated git worktree
    test_location: str            # Relative path for the test file (e.g. tests/src/test_foo.py)
    iteration_count: int          # Tracks how many times we've looped back to the Coder
    max_iterations: int           # Hard stop to prevent infinite agent loops
    status: str                   # 'in_progress', 'success', 'failed'
    log_filepath: str             # Path to the function-specific log file