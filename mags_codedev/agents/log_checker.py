"""Log checker agent: analyzes test/lint logs for bug diagnosis."""

import json

from mags_codedev.state import ModuleState
from mags_codedev.utils.llm_call import build_context_blocks, call_llm
from mags_codedev.utils.logger import get_dual_loggers, logger


def log_checker_node(state: ModuleState) -> dict:
    """Analyze raw test/lint logs and output a concise bug-fix strategy.

    The LLM call + conversation logging are delegated to :func:`call_llm`. This
    node owns the JSON contract (``{location, summary}``) and the routing
    decision. As a *diagnostic* node (not a deliverable generator), it degrades
    to keyword-based analysis on a hard LLM failure rather than failing the
    whole task — the coder/tester will still retry based on that routing.
    """
    backend = state.get("backend")
    failure_keywords = backend.test_failure_keywords() if backend else ["FAILED", "ERROR"]

    # If tests passed and lint output is empty or indicates success, skip analysis.
    test_upper = state.get("test_results", "").upper()
    lint_results = state.get("lint_results", "")
    lint_clean = not lint_results or (
        backend and any(lint_results.startswith(p) for p in backend.lint_success_prefixes())
    )
    if not any(kw in test_upper for kw in failure_keywords) and lint_clean:
        return {
            "test_error_summary": "",
            "test_results": "",  # consume the passing logs
            "lint_results": "",
        }

    system_prompt = (
        "You are a Senior Diagnostic Engineer. Your output MUST be a valid JSON object.\n"
        "Read the code, dependency context, test traceback, and linter warnings.\n"
        "Consider project conventions and dependency interfaces when diagnosing issues.\n\n"
        "Output a JSON object with two keys:\n"
        '1. "location": A string: "SOURCE_CODE", "TEST_CODE", or "NONE" if no issues found.\n'
        '2. "summary": A string explaining what failed, or "No issues detected" if all is clean.\n\n'
        "Example of issues found:\n"
        '{{"location": "SOURCE_CODE", "summary": "Fix the bug."}}\n'
        "Example of no issues:\n"
        '{{"location": "NONE", "summary": "No issues detected."}}'
    )

    proj_block, dep_block = build_context_blocks(state)
    human_template = (
        proj_block + dep_block
        + "\nCode:\n{code}\n\n"
        + "Test Traceback:\n{test_results}\n\n"
        + "Linter Warnings:\n{lint_results}\n"
    )
    params = {
        "code": state["code"],
        "test_results": state.get("test_results", "No test errors."),
        "lint_results": state.get("lint_results", "No linting errors."),
    }

    try:
        response_content = call_llm(
            role="log_checker",
            system_prompt=system_prompt,
            human_template=human_template,
            params=params,
            state=state,
            narrative="Diagnose test/lint failures and locate the fault.",
        )
    except Exception:
        logger.warning("log_checker LLM call failed; using keyword fallback")
        return _basic_analysis(state, failure_keywords)

    try:
        data = json.loads(response_content)
        error_location = data.get("location", "SOURCE_CODE").upper()
        if error_location == "NONE":
            error_location = None
        elif error_location not in ("SOURCE_CODE", "TEST_CODE"):
            error_location = "SOURCE_CODE"
        error_summary = data.get("summary", "No summary provided.")
    except (json.JSONDecodeError, AttributeError):
        # Malformed LLM output: blame the source conservatively.
        error_location = "SOURCE_CODE"
        error_summary = response_content[:500] + ("..." if len(response_content) > 500 else "")
    func_logger, _ = get_dual_loggers(
        state.get("log_filepath"),
        base_dir=state.get("base_dir", ".mags-codedev"),
        log_level=state.get("log_level", "info"),
    )
    func_logger.info(
        "─── Diagnosis ───\nLocation: %s\nSummary: %s",
        error_location or "NONE", error_summary,
    )

    return {
        "test_error_summary": error_summary,
        "error_location": error_location,
        "test_results": "",
        "lint_results": "",
    }


def _basic_analysis(state: ModuleState, failure_keywords: list[str]) -> dict:
    """Keyword-based fallback diagnosis when the LLM is unavailable."""
    test_results = state.get("test_results", "")
    lint_results = state.get("lint_results", "")
    test_upper = test_results.upper()
    no_tests_indicators = ["collected 0 items", "no tests ran", "no tests collected"]

    if any(kw in test_upper for kw in failure_keywords):
        error_summary = f"Test failures detected: {test_results[:200]}"
        error_location = "SOURCE_CODE"
    elif any(ind in test_results.lower() for ind in no_tests_indicators):
        error_summary = "No tests were collected. Check test file location and naming."
        error_location = "SOURCE_CODE"
    elif lint_results and "Success: no issues found" not in lint_results:
        error_summary = f"Linter/type-checker issues detected: {lint_results[:200]}"
        error_location = "SOURCE_CODE"
    else:
        error_summary = "LLM unavailable; basic analysis inconclusive."
        error_location = None

    return {
        "test_error_summary": error_summary,
        "error_location": error_location,
        "test_results": "",
        "lint_results": "",
    }
