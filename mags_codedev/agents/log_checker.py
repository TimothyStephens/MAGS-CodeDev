import json
import logging
from langchain_core.prompts import ChatPromptTemplate
import httpx
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.logger import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from google.genai.errors import ServerError

def _is_retryable_error(exception):
    """Check if the exception is a transient API error."""
    # Explicitly handle Google GenAI ServerError
    if isinstance(exception, ServerError):
        try:
            error_details = (getattr(exception, 'response_json', None) or {}).get('error', {})
            message = error_details.get('message', '').lower()
            status = error_details.get('status', '').lower()
            code = error_details.get('code')
            
            # Retry on 503 UNAVAILABLE, high demand, or resource exhausted
            if (code == 503 and status == 'unavailable') or \
               "high demand" in message or \
               "resource_exhausted" in message or \
               "rate limit" in message or \
               code == 429: # Explicitly check for 429
                return True
        except Exception:
            pass
            
    # Explicitly handle httpx.RemoteProtocolError
    if isinstance(exception, httpx.RemoteProtocolError):
        return True

    # Fallback for other exceptions that might contain these keywords in their string representation
    msg = str(exception).lower()
    return ("503" in msg or "unavailable" in msg or "rate limit" in msg or "429" in msg or "resource_exhausted" in msg
            or "server disconnected" in msg)

@retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True
)
def _invoke_with_retry(chain, inputs):
    return chain.invoke(inputs)

def log_checker_node(state: ModuleState) -> dict:
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
    
    func_logger.info(f"Log Checker: Sending prompt for '{state['module_location']}'.")
    func_logger.debug(f"Log Checker Prompt:\n{prompt.format(code=state['code'], test_results=state.get('test_results', 'No test errors.'), lint_results=state.get('lint_results', 'No linting errors.'))}")

    chain = prompt | llm
    response = _invoke_with_retry(chain, {
        "code": state['code'],
        "test_results": state.get('test_results', 'No test errors.'),
        "lint_results": state.get('lint_results', 'No linting errors.')
    })
    
    # Parse the JSON response with robust fallback
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block if isinstance(block, str) else 
            block.get("text", "") if isinstance(block, dict) else 
            getattr(block, "text", str(block))
            for block in content
        )
    response_content = str(content).strip()
    if "```json" in response_content:
        response_content = response_content.split("```json")[1].split("```")[0].strip()
    elif "```" in response_content:
        response_content = response_content.split("```")[1].split("```")[0].strip()
    
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