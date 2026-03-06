import logging
import httpx
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
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

def tester_node(state: ModuleState) -> dict:
    """Writes comprehensive unit tests using pytest."""
    config_path = state["config_path"]
    llm = get_llm(role="tester", config_path=config_path)
    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
    
    if state.get("log_filepath"):
        import os
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
    
    # Determine the correct import path from the project root to guide the LLM
    source_location = state['module_location']
    # e.g., "src/utils/helpers.py" -> "src.utils.helpers"
    module_path = Path(source_location).with_suffix('').as_posix().replace('/', '.')
    import_instruction = f"The module to test is '{source_location}'. You can import from it using `from {module_path} import ...`."

    # Check if we are in a fix cycle for tests
    is_fix = bool(state.get("error_summary")) and state.get("error_location") == "TEST_CODE"
    
    system_prompt = f"""You are a strict QA Automation Engineer.
    Write robust `pytest` unit tests for the provided Python module.
    The code is part of a larger project, so ensure imports are correct.
    {import_instruction}
    Your tests should cover all functions and classes in the module.
    Include edge cases, type boundary checks, and failure scenarios.
    The module to be tested is located at the path '{source_location}'.
    Return ONLY valid Python code for the test file. Do not include markdown formatting."""

    if is_fix:
        human_template = """The previous attempt to write tests failed. Please fix them.
        
Module Specification:
{spec}

Generated Code to Test:
{code}

PREVIOUS (BROKEN) TESTS:
{previous_tests}

DIAGNOSIS OF FAILURE:
{error_summary}

Your task is to provide a new, corrected version of the pytest unit tests."""
        invoke_params = {
            "spec": str(state['spec']),
            "code": state['code'],
            "previous_tests": state['tests'],
            "error_summary": state['error_summary']
        }
    else:
        human_template = """Module Specification:
{spec}

Generated Code to Test:
{code}"""
        invoke_params = {
            "spec": str(state['spec']),
            "code": state['code']
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    func_logger.info(f"Tester: Sending prompt for '{state['module_location']}'.")
    func_logger.debug(f"Tester Prompt:\n{prompt.format(**invoke_params)}")

    chain = prompt | llm
    response = _invoke_with_retry(chain, invoke_params)

    func_logger.debug(f"Tester Response:\n{response.content}")

    # Handle potential list content from LLM (e.g. text blocks + tool calls)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block if isinstance(block, str) else 
            block.get("text", "") if isinstance(block, dict) else 
            getattr(block, "text", str(block))
            for block in content
        )

    # The LLM may wrap the code in markdown blocks or add extraneous text. Extract the code.
    response_content = str(content).strip()
    if "```python" in response_content:
        response_content = response_content.split("```python")[1].split("```")[0].strip()
    elif "```" in response_content:
        # Fallback for ``` with no language specified
        response_content = response_content.split("```")[1].split("```")[0].strip()

    return {
        "tests": response_content + "\n",
    }