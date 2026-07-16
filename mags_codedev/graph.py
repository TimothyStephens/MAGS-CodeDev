"""LangGraph workflow: build function graph with test/lint/review routes."""

import hashlib
from typing import Any

from langgraph.constants import END
from langgraph.graph import StateGraph

from mags_codedev.state import ModuleState

from mags_codedev.agents.coder import coder_node
from mags_codedev.agents.tester import tester_node
from mags_codedev.utils.docker_ops import test_node, linter_node
from mags_codedev.agents.log_checker import log_checker_node
from mags_codedev.agents.reviewer import multi_llm_review_node

# ---- Module-level edge functions (extracted for testability) ----

def evaluate_test_results(state: ModuleState) -> str:
    """Check if tests passed, failed, or max iterations reached."""
    max_iters = state.get("max_test_fix_iterations", 5)
    if max_iters > 0 and state["iteration_count"] >= max_iters:
        return "max_iterations_reached"

    backend = state.get("backend")
    failure_keywords = (
        backend.test_failure_keywords() if backend
        else ["FAILED", "ERROR"]
    )
    test_results = state.get("test_results", "")
    test_upper = test_results.upper()

    # Check for failure keywords first
    if any(kw in test_upper for kw in failure_keywords):
        return "tests_failed"

    # Check for "no tests collected" (pytest exit code 5)
    # M14: Removed "deselected" — false positive with pytest -k filter
    no_tests_indicators = [
        "collected 0 items",
        "no tests ran",
        "no tests collected",
    ]
    if any(ind in test_results.lower() for ind in no_tests_indicators):
        return "tests_failed"

    # Empty results = something went wrong
    if not test_results.strip():
        return "tests_failed"

    return "tests_passed"


def evaluate_logs(state: ModuleState) -> str:
    """Check if log checker found issues, and route accordingly.

    Uses test_error_summary (scoped) which combines test and lint analysis.
    Trusts error_location from log_checker for routing decisions.
    Returns 'clean' when no actionable errors found.
    """
    max_iters = state.get("max_test_fix_iterations", 5)
    if max_iters > 0 and state["iteration_count"] >= max_iters:
        return "max_iterations_reached"

    # Use test_error_summary (scoped tracking — includes test + lint analysis)
    error_summary = state.get("test_error_summary", "")

    # If log_checker explicitly said no issues, we're clean
    if error_summary and "no clear issues" in error_summary.lower():
        return "clean"

    # If there's a test error summary, route based on error_location
    if error_summary:
        error_location = state.get("error_location")
        # If location is explicitly set, trust it
        if error_location == "TEST_CODE":
            return "fix_tests"
        if error_location == "SOURCE_CODE":
            return "fix_source"
        # Linter-only issue (no explicit location) → cosmetic, let review handle it
        if "linter" in error_summary.lower() and "test" not in error_summary.lower():
            return "clean"
        # Other actionable issues without explicit location → default to source
        return "fix_source"

    return "clean"


def evaluate_reviews(state: ModuleState) -> str:
    """Check if review found issues, and route accordingly."""
    max_rounds = state.get("max_review_rounds", 3)
    if max_rounds > 0 and state.get("review_round_count", 0) >= max_rounds:
        return "max_review_rounds_reached"

    if state.get("review_comments"):
        return "revise"
    return "approved"


def check_convergence(state: ModuleState) -> dict:
    """Node: Check if code/tests have converged (identical to previous iteration).

    If code hash matches previous_code_hash for 2+ consecutive iterations, mark as failed.
    Same check for test hash. Otherwise, update hashes for next comparison.
    """
    code = state.get("code", "")
    tests = state.get("tests", "")
    current_code_hash = hashlib.sha256(code.encode()).hexdigest() if code else ""
    current_test_hash = hashlib.sha256(tests.encode()).hexdigest() if tests else ""

    previous_code_hash = state.get("previous_code_hash")
    previous_test_hash = state.get("previous_test_hash")

    # Check both code and test convergence
    if previous_code_hash and current_code_hash and current_code_hash == previous_code_hash:
        return {
            "status": "failed",
            "test_error_summary": "Coder convergence failed: code unchanged from previous iteration.",
        }
    if previous_test_hash and current_test_hash and current_test_hash == previous_test_hash:
        return {
            "status": "failed",
            "test_error_summary": "Tester convergence failed: tests unchanged from previous iteration.",
        }

    # Update hash tracking for next cycle
    return {
        "previous_code_hash": current_code_hash,
        "previous_test_hash": current_test_hash,
    }


# L2: Moved to module level to avoid recreating on every build_function_graph() call
def check_convergence_route(state: ModuleState) -> str:
    """Route after convergence check: end if failed, else proceed to review."""
    if state.get("status") == "failed":
        return "__end__"
    return "multi_llm_review"


# ---- Graph construction ----

def build_function_graph() -> Any:
    """Build the LangGraph state machine for module generation."""
    workflow = StateGraph(ModuleState)

    # Define Nodes
    workflow.add_node("coder", coder_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("run_tests", test_node)
    workflow.add_node("run_linters", linter_node)
    workflow.add_node("log_checker", log_checker_node)
    workflow.add_node("multi_llm_review", multi_llm_review_node)
    workflow.add_node("check_convergence", check_convergence)

    # Entry point and linear edges
    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "tester")
    workflow.add_edge("tester", "run_tests")

    # A. Test result evaluation
    workflow.add_conditional_edges(
        "run_tests",
        evaluate_test_results,
        {
            "tests_passed": "run_linters",
            "tests_failed": "log_checker",
            "max_iterations_reached": END,
        }
    )

    # Linters feed into log_checker
    workflow.add_edge("run_linters", "log_checker")

    # B. Log evaluation
    workflow.add_conditional_edges(
        "log_checker",
        evaluate_logs,
        {
            "clean": "check_convergence",
            "fix_source": "coder",
            "fix_tests": "tester",
            "max_iterations_reached": END,
        }
    )

    # C. Convergence check → review or fail
    workflow.add_conditional_edges(
        "check_convergence",
        check_convergence_route,
        {
            "__end__": END,
            "multi_llm_review": "multi_llm_review",
        }
    )

    # D. Review evaluation
    workflow.add_conditional_edges(
        "multi_llm_review",
        evaluate_reviews,
        {
            "approved": END,
            "revise": "coder",
            "max_review_rounds_reached": END,
        }
    )

    return workflow.compile()
