import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import FunctionState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.logger import logger

def log_checker_node(state: FunctionState) -> dict:
    """Analyzes raw test/lint logs and outputs a concise bug-fix strategy."""
    # If the docker_test_node already determined tests passed, skip analysis
    if "FAILED" not in state.get("test_results", "").upper() and "ERROR" not in state.get("test_results", "").upper() and not state.get("lint_results"):
        return {
            "error_summary": "",
            "test_results": "", # Consume the passing logs
            "lint_results": ""  # Consume the (empty) lint results
        }
        
    if state.get("log_filepath"):
        import os
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
        
    config_path = state["config_path"]
    llm = get_llm(role="log_checker", config_path=config_path)
    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
    
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
    
    func_logger.info(f"Log Checker: Sending prompt for '{state['function_name']}'.")
    func_logger.debug(f"Log Checker Prompt:\n{prompt.format(code=state['code'], test_results=state.get('test_results', 'No test errors.'), lint_results=state.get('lint_results', 'No linting errors.'))}")

    chain = prompt | llm
    response = chain.invoke({
        "code": state['code'],
        "test_results": state.get('test_results', 'No test errors.'),
        "lint_results": state.get('lint_results', 'No linting errors.')
    })
    
    # Parse the JSON response with robust fallback
    response_content = response.content.strip()
    if "```json" in response_content:
        response_content = response_content.split("```json")[1].split("```")[0].strip()
    
    try:
        data = json.loads(response_content)
        error_location = data.get("location", "SOURCE_CODE").upper()
        if error_location not in ["SOURCE_CODE", "TEST_CODE"]:
            error_location = "SOURCE_CODE" # Default if invalid value
        error_summary = data.get("summary", "No summary provided.")
    except (json.JSONDecodeError, AttributeError):
        # Fallback for models that fail to produce valid JSON.
        error_location = "SOURCE_CODE"
        error_summary = response_content
        
    func_logger.debug(f"Log Checker Response:\n{response.content}")

    return {
        "error_summary": error_summary,
        "error_location": error_location,
        "test_results": "", # Consume the test results after analysis
        "lint_results": ""  # Consume the lint results after analysis
    }