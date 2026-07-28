"""Log checker agent: analyzes test/lint logs for bug diagnosis."""

import json
import re
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.retry import invoke_with_retry
from mags_codedev.utils.llm_helpers import resolve_logger, resolve_response_logger, extract_content, strip_markdown_code


def log_checker_node(state: ModuleState) -> dict:
    """Analyzes raw test/lint logs and outputs a concise bug-fix strategy."""
    backend = state.get("backend")

    # Get failure keywords from backend (fallback to defaults)
    failure_keywords = (
        backend.test_failure_keywords() if backend
        else ["FAILED", "ERROR"]
    )

    # If the docker_test_node already determined tests passed, skip analysis
    test_upper = state.get("test_results", "").upper()
    if (
        not any(kw in test_upper for kw in failure_keywords)
        and not state.get("lint_results")
    ):
        return {
            "test_error_summary": "",
            "test_results": "",  # Consume the passing logs
            "lint_results": "",  # Consume the (empty) lint results
        }

    func_logger = resolve_logger(state.get("log_filepath"))
    resp_logger = resolve_response_logger(state.get("log_filepath"))
    session = state.get("_session_number", 1)
    config_path = state["config_path"]

    # Offline mode: use basic log analysis without API call
    if state.get("offline"):
        error_summary, error_location = _basic_log_analysis(
            state.get("test_results", ""),
            state.get("lint_results", "")
        )
        func_logger.info(f"Offline mode: basic log analysis for '{state['module_location']}'.")
        return {
            "test_error_summary": error_summary,
            "error_location": error_location,
            "test_results": "",
            "lint_results": "",
        }

    # FIX M1: Get LLM only when actually needed (not in offline mode)
    llm = get_llm(role="log_checker", config_path=config_path)

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
    # Build project instructions block
    project_instructions = state.get("project_instructions", "")
    project_instructions_block = ""
    if project_instructions:
        project_instructions_block = (
            "PROJECT INSTRUCTIONS\n"
            "These are project-wide guidelines and conventions.\n\n"
            + project_instructions + "\n\n"
        )

    # Build dependency context block
    dep_code = state.get("dependency_code", {})
    dependency_context = ""
    if dep_code:
        dep_parts = []
        for loc, source in dep_code.items():
            dep_parts.append(f"--- File: {loc} ---\n```python\n{source.replace('{', '{{').replace('}', '}}')}\n```")
        dependency_context = (
            "DEPENDENCY CONTEXT\n"
            "These are the source files this module depends on.\n\n"
            + "\n\n".join(dep_parts) + "\n\n"
        )

    human_template = (
        project_instructions_block
        + dependency_context
        + "\nCode:\n{code}\n\n"
        + "Test Traceback:\n{test_results}\n\n"
        + "Linter Warnings:\n{lint_results}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])

    invoke_params = {
        "code": state["code"],
        "test_results": state.get("test_results", "No test errors."),
        "lint_results": state.get("lint_results", "No linting errors."),
    }
    iteration = state.get("iteration_count", 0) + 1
    func_logger.info(
        "[Session %d, Iteration %d] Log Checker: sending prompt for '%s'.",
        session,
        iteration,
        state['module_location'],
    )
    func_logger.debug(
        f"Log Checker Prompt:\n{prompt.format(**invoke_params)}"
    )

    try:
        chain = prompt | llm
        response = invoke_with_retry(chain, invoke_params)
        func_logger.debug(f"Log Checker Response:\n{response.content}")

        # INFO: log analysis decision
        response_text = extract_content(response.content)
        first_line = response_text.strip().split("\n")[0]
        func_logger.info(
            "[Session %d, Iteration %d] Log Checker: received analysis (%s).",
            session,
            iteration,
            first_line[:100] if len(first_line) > 100 else first_line,
        )

        # TRACE: log token usage if available
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            func_logger.trace(
                "[Session %d, Iteration %d] Log Checker tokens: input=%s, output=%s",
                session,
                iteration,
                meta.get("input_tokens", "?"),
                meta.get("output_tokens", "?"),
            )

        response_content = strip_markdown_code(response_text)

        resp_logger.info(
            f"[Session {session}, Iteration {iteration}] Log Checker Response:\n{response_content}"
        )
    except Exception as e:
        func_logger.warning(
            f"Log checker API call failed: {e}. Using basic log analysis."
        )
        error_summary, error_location = _basic_log_analysis(
            state.get("test_results", ""),
            state.get("lint_results", "")
        )
        return {
            "test_error_summary": error_summary,
            "error_location": error_location,
            "test_results": "",
            "lint_results": "",
        }

    try:
        data = json.loads(response_content)
        error_location = data.get("location", "SOURCE_CODE").upper()
        if error_location == "NONE":
            error_location = None
        elif error_location not in ("SOURCE_CODE", "TEST_CODE"):
            error_location = "SOURCE_CODE"
        error_summary = data.get("summary", "No summary provided.")
    except (json.JSONDecodeError, AttributeError):
        error_location = "SOURCE_CODE"
        error_summary = response_content[:500] + ("..." if len(response_content) > 500 else "")
    return {
        "test_error_summary": error_summary,
        "error_location": error_location,
        "test_results": "",
        "lint_results": "",
    }


def _basic_log_analysis(test_results: str, lint_results: str) -> tuple:
    """Basic log analysis when LLM is unavailable.

    Returns (error_summary, error_location).
    FIX m2: Improved error_location heuristic — analyzes traceback patterns
    to distinguish test code errors from source code errors.
    """
    parts = []
    # Only check for actual test failures (not pip install errors)
    test_failed = bool(
        re.search(r"\d+ failed", test_results) or re.search(r"\bFAILED\b", test_results)
    )
    if test_failed:
        parts.append("Tests failed. Check the test output for specific failures.")
    if lint_results and re.search(r"(flake8|mypy|pylint|E\d{3}|W\d{3})", lint_results):
        parts.append("Linter warnings found. Address code quality issues.")

    summary = " ".join(parts) if parts else "No clear issues detected in logs."

    # Determine error location from test traceback patterns
    error_location = _infer_error_location(test_results)

    return summary, error_location


def _infer_error_location(test_results: str) -> str:
    """Infer whether the error is in SOURCE_CODE or TEST_CODE from test output.

    Heuristic: if the traceback ONLY points to test files (no source frames),
    it's likely TEST_CODE. If source code frames appear, default to SOURCE_CODE.
    """
    # Only mark as TEST_CODE if traceback is entirely within test files
    # and the error is a test-specific framework error
    test_specific_errors = [
        r"pytest\.skip",
        r"test\s+function\s+.*not\s+found",
        r"no\s+tests\s+ran",
        # Match ModuleNotFoundError only for missing test modules
        r"ModuleNotFoundError.*No module named ['\"]test",
    ]
    for pattern in test_specific_errors:
        if re.search(pattern, test_results, re.IGNORECASE):
            return "TEST_CODE"

    return "SOURCE_CODE"
