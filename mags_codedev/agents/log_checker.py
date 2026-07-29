"""Log checker agent: analyzes test/lint logs for bug diagnosis."""

import json
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.retry import invoke_with_retry
from mags_codedev.utils.llm_helpers import extract_content, strip_markdown_code
from mags_codedev.utils.logger import get_dual_loggers


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

    func_logger, resp_logger = get_dual_loggers(state.get("log_filepath"))
    # Log the reason for this node being invoked
    reason = state.get("_next_reason", "")
    if reason:
        func_logger.info(f"[Reason] {reason}")
    session = state.get("_session_number", 1)
    config_path = state["config_path"]

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

        resp_logger.debug(
            f"[Session {session}, Iteration {iteration}] Log Checker Response:\n{response_content}"
        )
    except Exception as e:
        func_logger.warning(
            f"Log checker API call failed: {e}. Falling back to basic analysis."
        )
        test_results = state.get("test_results", "")
        lint_results = state.get("lint_results", "")
        test_upper = test_results.upper()

        # Check for test failure keywords
        failure_keywords = ["FAILED", "ERROR"]
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
            error_summary = f"LLM unavailable; basic analysis inconclusive: {e}"
            error_location = None

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

