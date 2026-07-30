"""LangGraph workflow: build function graph with test/lint/review routes."""

import hashlib
import logging
from typing import Any
from langgraph.constants import END
from langgraph.graph import StateGraph
from mags_codedev.utils.logger import get_dual_loggers
from mags_codedev.utils.db import add_iterations_to_module, get_total_iterations

from mags_codedev.state import ModuleState

from mags_codedev.agents.coder import coder_node
from mags_codedev.agents.tester import tester_node
from mags_codedev.utils.docker_ops import test_node, linter_node
from mags_codedev.agents.log_checker import log_checker_node
from mags_codedev.agents.reviewer import multi_llm_review_node

# ---- Module-level edge functions (extracted for testability) ----
def evaluate_test_results(state: ModuleState) -> str:
    """Route after run_tests: tests passed → linters, failed → log_checker, max → end.

    Pure function — does NOT mutate state. The routing reason is logged to
    the module logger (propagates to workflow.log) for traceability.
    """
    _log = logging.getLogger("mags_codedev")
    max_iters = state.get("max_test_fix_iterations", 5)
    if max_iters > 0 and state.get("iteration_count", 0) >= max_iters:
        _log.info("[Route] Max test fix iterations (%d) reached.", max_iters)
        return "max_iterations_reached"

    backend = state.get("backend")
    failure_keywords = (
        backend.test_failure_keywords() if backend
        else ["FAILED", "ERROR"]
    )
    test_results = state.get("test_results", "")
    test_upper = test_results.upper()

    if any(kw in test_upper for kw in failure_keywords):
        _log.info("[Route] Tests failed, sending to log checker.")
        return "tests_failed"

    no_tests_indicators = ["collected 0 items", "no tests ran", "no tests collected"]
    if any(ind in test_results.lower() for ind in no_tests_indicators):
        _log.info("[Route] No tests collected, sending to log checker.")
        return "tests_failed"

    if not test_results.strip():
        _log.info("[Route] Empty test results, sending to log checker.")
        return "tests_failed"

    _log.info("[Route] Tests passed, proceeding to linting.")
    return "tests_passed"

def evaluate_logs(state: ModuleState) -> str:
    """Route after log_checker: clean → convergence check, fix → coder/tester.

    Pure function — does NOT mutate state.
    """
    _log = logging.getLogger("mags_codedev")
    max_iters = state.get("max_test_fix_iterations", 5)
    if max_iters > 0 and state.get("iteration_count", 0) >= max_iters:
        _log.info("[Route] Max test fix iterations (%d) reached.", max_iters)
        return "max_iterations_reached"

    error_summary = state.get("test_error_summary", "")

    if error_summary:
        summary_lower = error_summary.lower()
        no_issue_phrases = [
            "no clear issues", "no immediate fixes", "no fixes required",
            "passed successfully", "functioning correctly", "no issues found",
            "no issues detected", "no changes required", "all tests passed",
        ]
        if any(phrase in summary_lower for phrase in no_issue_phrases):
            _log.info("[Route] No issues found, proceeding to convergence check.")
            return "clean"

        if (
            ("linter" in summary_lower or "linting" in summary_lower
             or "style" in summary_lower or "pep 8" in summary_lower)
            and "assertion" not in summary_lower
            and "test failed" not in summary_lower
            and "test error" not in summary_lower
        ):
            _log.info("[Route] Linting only issue (cosmetic), proceeding to review.")
            return "clean"

        error_location = state.get("error_location")
        if error_location == "TEST_CODE":
            _log.info("[Route] Test error found: %s", error_summary[:120])
            return "fix_tests"
        if error_location == "SOURCE_CODE":
            _log.info("[Route] Source error found: %s", error_summary[:120])
            return "fix_source"

    _log.info("[Route] No issues found, proceeding to convergence check.")
    return "clean"

def evaluate_reviews(state: ModuleState) -> str:
    """Route after multi_llm_review: approved → end, revise → coder.

    Pure function — does NOT mutate state.
    """
    _log = logging.getLogger("mags_codedev")
    max_rounds = state.get("max_review_rounds", 3)
    if max_rounds > 0 and state.get("review_round_count", 0) >= max_rounds:
        _log.info("[Route] Max review rounds (%d) reached.", max_rounds)
        return "max_review_rounds_reached"

    review_comments = state.get("review_comments")
    if review_comments:
        _log.info(
            "[Route] Reviewer comments found (%d), sending to Coder for fix.",
            len(review_comments),
        )
        return "revise"
    _log.info("[Route] All reviewers approved, proceeding to session end.")
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


def check_convergence_route(state: ModuleState) -> str:
    """Route after convergence check: failed → end, changed → review.

    Pure function — does NOT mutate state.
    """
    _log = logging.getLogger("mags_codedev")
    if state.get("status") == "failed":
        _log.info("[Route] Convergence check failed, ending session.")
        return "__end__"
    _log.info("[Route] Code/tests changed, sending to multi-LLM review.")
    return "multi_llm_review"




# ---- Session lifecycle nodes ----

def _session_banner(char: str = "=") -> str:
    return char * 80


def session_start_node(state: ModuleState) -> dict:
    """Log build session start. Always the first node in the graph."""
    func_logger, resp_logger = get_dual_loggers(
        state.get("log_filepath"),
        base_dir=state.get("base_dir", ".mags-codedev"),
    )
    session = state.get("_session_number", 1)
    previous = state.get("_previous_iterations", 0)
    max_test = state.get("max_test_fix_iterations", 5)
    max_review = state.get("max_review_rounds", 3)
    module = state.get("module_location", "unknown")

    lines = [
        _session_banner(),
        f"Build session started: {module}",
        f"  Session: {session}"
        + (f" ({previous} iterations from previous session)" if previous else ""),
        f"  Max test iterations: {max_test} | Max review rounds: {max_review}",
        _session_banner(),
    ]
    banner = "\n".join(lines)
    func_logger.info(banner)
    resp_logger.debug(banner)

    return {}


def session_end_node(state: ModuleState) -> dict:
    """Log build session end. Always the last node before END."""
    func_logger, resp_logger = get_dual_loggers(
        state.get("log_filepath"),
        base_dir=state.get("base_dir", ".mags-codedev"),
    )
    session = state.get("_session_number", 1)
    previous = state.get("_previous_iterations", 0)
    iteration_count = state.get("iteration_count", 0)
    review_rounds = state.get("review_round_count", 0)
    module = state.get("module_location", "unknown")
    status = state.get("status", "unknown")
    exit_reason = state.get("_exit_reason", "UNKNOWN")

    cumulative = previous + iteration_count

    lines = [
        _session_banner(),
        f"Build session ended: {exit_reason}",
        f"  Module: {module}",
        f"  Session: {session}",
        f"  Iterations this session: {iteration_count}",
        f"  Cumulative iterations: {cumulative}",
        f"  Review rounds: {review_rounds}",
        f"  Final status: {status}",
        _session_banner(),
    ]
    banner = "\n".join(lines)
    func_logger.info(banner)
    resp_logger.debug(banner)

    return {}


# ---- Graph construction ----

def build_function_graph() -> Any:
    """Build the LangGraph state machine for module generation."""
    workflow = StateGraph(ModuleState)

    # Define Nodes
    workflow.add_node("session_start", session_start_node)
    workflow.add_node("session_end", session_end_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("run_tests", test_node)
    workflow.add_node("run_linters", linter_node)
    workflow.add_node("log_checker", log_checker_node)
    workflow.add_node("multi_llm_review", multi_llm_review_node)
    workflow.add_node("check_convergence", check_convergence)

    # Entry: session_start → coder
    workflow.set_entry_point("session_start")
    workflow.add_edge("session_start", "coder")

    # Exit: all terminal routes go through session_end first
    workflow.add_edge("coder", "tester")
    workflow.add_edge("tester", "run_tests")

    # A. Test result evaluation
    workflow.add_conditional_edges(
        "run_tests",
        evaluate_test_results,
        {
            "tests_passed": "run_linters",
            "tests_failed": "log_checker",
            "max_iterations_reached": "session_end",
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
            "max_iterations_reached": "session_end",
        }
    )

    # C. Convergence check → review or fail
    workflow.add_conditional_edges(
        "check_convergence",
        check_convergence_route,
        {
            "__end__": "session_end",
            "multi_llm_review": "multi_llm_review",
        }
    )

    # D. Review evaluation
    workflow.add_conditional_edges(
        "multi_llm_review",
        evaluate_reviews,
        {
            "approved": "session_end",
            "revise": "coder",
            "max_review_rounds_reached": "session_end",
        }
    )

    # Session end → END
    workflow.add_edge("session_end", END)

    return workflow.compile()
