import json
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.retry import invoke_with_retry
from mags_codedev.utils.llm_helpers import get_function_logger, extract_content, strip_markdown_code


def log_checker_node(state: ModuleState) -> dict:
    """Analyzes raw test/lint logs and outputs a concise bug-fix strategy."""
    backend = state.get("backend")

    # Get failure keywords from backend (fallback to defaults)
    failure_keywords = (
        backend.test_success_keywords() if backend
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

    func_logger = get_function_logger(state.get("log_filepath"))
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

    system_prompt = """You are a Senior Diagnostic Engineer. Your output MUST be a valid JSON object.
        Read the code, test traceback, and linter warnings.

        Output a JSON object with two keys:
        1. "location": A string, either "SOURCE_CODE" or "TEST_CODE".
        2. "summary": A string explaining why the code failed and providing a brief, actionable strategy for the Coder or Tester to fix it (under 5 sentences).

        Example:
        {{
          "location": "SOURCE_CODE",
          "summary": "The function fails because it does not handle division by zero. Add a check at the beginning of the function to validate the divisor."
        }}"""
    # Build project instructions block
    project_instructions = state.get("project_instructions", "")
    project_instructions_block = ""
    if project_instructions:
        project_instructions_block = (
            "PROJECT INSTRUCTIONS\n"
            "These are project-wide guidelines and conventions.\n\n"
            + project_instructions + "\n\n"
        )

    human_template = (
        project_instructions_block
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
    func_logger.info(f"Log Checker: Sending prompt for '{state['module_location']}'.")
    func_logger.debug(
        f"Log Checker Prompt:\n{prompt.format(**invoke_params)}"
    )

    try:
        chain = prompt | llm
        response = invoke_with_retry(chain, invoke_params)
        func_logger.debug(f"Log Checker Response:\n{response.content}")
        response_content = strip_markdown_code(extract_content(response.content))
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
        if error_location not in ("SOURCE_CODE", "TEST_CODE"):
            error_location = "SOURCE_CODE"
        error_summary = data.get("summary", "No summary provided.")
    except (json.JSONDecodeError, AttributeError):
        error_location = "SOURCE_CODE"
        error_summary = response_content

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
    import re
    parts = []
    # Only check for actual test failures (not pip install errors)
    test_failed = bool(
        re.search(r"\d+ failed", test_results) or re.search(r"FAILED\]", test_results)
    )
    if test_failed:
        parts.append("Tests failed. Check the test output for specific failures.")
    if lint_results and re.search(r"(error|warning|E\d+|W\d+)", lint_results):
        parts.append("Linter warnings found. Address code quality issues.")
    
    summary = " ".join(parts) if parts else "No clear issues detected in logs."
    
    # Determine error location from test traceback patterns
    error_location = _infer_error_location(test_results)
    
    return summary, error_location


def _infer_error_location(test_results: str) -> str:
    """Infer whether the error is in SOURCE_CODE or TEST_CODE from test output.
    
    Heuristic: if the traceback points to a file in tests/ or the error
    mentions 'assert' in a test function context, it's likely TEST_CODE.
    Otherwise default to SOURCE_CODE.
    """
    import re
    
    # Check if the traceback points to a test file
    if re.search(r"File\s+['\"].*[/\\]tests?[/\\]", test_results, re.IGNORECASE):
        # Further check: was the assertion itself wrong?
        if re.search(r"(AssertionError|assert\s)", test_results):
            return "TEST_CODE"
    
    # Check for common test framework errors that indicate test issues
    test_specific_errors = [
        r"pytest\.skip",
        r"test\s+function\s+.*not\s+found",
        r"no\s+tests\s+ran",
        # P3 fix: Only match if the MISSING module name itself is a test module
        r"ModuleNotFoundError.*No module named ['\"](?!.*src).*['\"]",
    ]
    for pattern in test_specific_errors:
        if re.search(pattern, test_results, re.IGNORECASE):
            return "TEST_CODE"
    
    return "SOURCE_CODE"
