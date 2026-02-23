from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import FunctionState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.logger import logger
from mags_codedev.utils.db import log_token_usage

def log_checker_node(state: FunctionState) -> dict:
    """Analyzes raw test/lint logs and outputs a concise bug-fix strategy."""
    # If the docker_test_node already determined tests passed, skip analysis
    if "FAILED" not in state.get("test_results", "").upper() and "ERROR" not in state.get("test_results", "").upper() and not state.get("lint_results"):
        return {
            "error_summary": "",
            "test_results": "", # Consume the passing logs
            "lint_results": ""  # Consume the (empty) lint results
        }
        
    config_path = state["config_path"]
    llm = get_llm(role="log_checker", config_path=config_path)
    
    system_prompt = """You are a Senior Diagnostic Engineer.
    Read the following code, test traceback, and linter warnings.

    First, on a single line, classify the error's primary location. Is it in the 'SOURCE_CODE' or the 'TEST_CODE'? Respond with only one of those two strings.

    Then, on subsequent lines, explain EXACTLY why the code failed and provide a brief, actionable strategy for the Coder or Tester to fix it.
    Keep your summary under 5 sentences."""

    human_template = """
    Code:
    {code}
    
    Test Traceback:
    {test_results}
    
    Linter Warnings:
    {lint_results}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    logger.info(f"Log Checker: Sending prompt for '{state['function_name']}'.")
    logger.debug(f"Log Checker Prompt:\n{prompt.format(code=state['code'], test_results=state.get('test_results', 'No test errors.'), lint_results=state.get('lint_results', 'No linting errors.'))}")

    chain = prompt | llm
    response = chain.invoke({
        "code": state['code'],
        "test_results": state.get('test_results', 'No test errors.'),
        "lint_results": state.get('lint_results', 'No linting errors.')
    })
    
    # Parse the response to extract the location and summary
    response_content = response.content
    lines = response_content.strip().split('\n')
    error_location = "SOURCE_CODE"  # Default to source code if classification fails
    error_summary = response_content  # Fallback to the full response

    if len(lines) > 1:
        first_line = lines[0].strip().upper()
        if "TEST_CODE" in first_line:
            error_location = "TEST_CODE"
            error_summary = "\n".join(lines[1:]).strip()
        elif "SOURCE_CODE" in first_line:
            error_summary = "\n".join(lines[1:]).strip()

    # Log token usage
    try:
        usage = response.response_metadata.get("token_usage", {})
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        log_token_usage(
            role="log_checker",
            model=llm.model_name,
            in_tokens=in_tokens,
            out_tokens=out_tokens
        )
        logger.debug(f"Log Checker Token Usage: In={in_tokens}, Out={out_tokens}")
    except Exception as e:
        logger.warning(f"Could not log token usage for log_checker: {e}")

    logger.debug(f"Log Checker Response:\n{response.content}")

    return {
        "error_summary": error_summary,
        "error_location": error_location,
        "test_results": "", # Consume the test results after analysis
        "lint_results": ""  # Consume the lint results after analysis
    }