import logging
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import FunctionState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.logger import logger

def tester_node(state: FunctionState) -> dict:
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
    
    # Check if we are in a fix cycle for tests
    is_fix = bool(state.get("error_summary")) and state.get("error_location") == "TEST_CODE"
    
    system_prompt = """You are a strict QA Automation Engineer.
    Write robust `pytest` unit tests for the provided Python code.
    The code is part of a larger project, so ensure imports are correct.
    Include edge cases, type boundary checks, and failure scenarios.
    The code to be tested is located at the path specified in the 'location' field of the spec.
    Return ONLY valid Python code for the test file. Do not include markdown formatting."""
    
    if is_fix:
        human_template = """The previous attempt to write tests failed. Please fix them.
        
Function Specification:
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
        human_template = """Function Specification:
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
    
    func_logger.info(f"Tester: Sending prompt for '{state['function_name']}'.")
    func_logger.debug(f"Tester Prompt:\n{prompt.format(**invoke_params)}")

    chain = prompt | llm
    response = chain.invoke(invoke_params)

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

    return {
        "tests": str(content).strip(),
        "error_summary": "" # Clear the error summary after addressing it
    }