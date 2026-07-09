from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.retry import invoke_with_retry
from mags_codedev.utils.llm_helpers import get_function_logger, strip_markdown_code


def tester_node(state: ModuleState) -> dict:
    """Writes comprehensive unit tests."""
    config_path = state["config_path"]
    llm = get_llm(role="tester", config_path=config_path)
    func_logger = get_function_logger(state.get("log_filepath"))
    backend = state.get("backend")

    source_location = state["module_location"]

    # Import hint from backend
    if backend:
        import_instruction = backend.get_import_hint(Path(source_location))
    else:
        module_path = Path(source_location).with_suffix("").as_posix().replace("/", ".")
        import_instruction = (
            f"The module to test is '{source_location}'. "
            f"You can import from it using `from {module_path} import ...`."
        )

    # Check if we are in a fix cycle for tests
    is_fix = bool(state.get("error_summary")) and state.get("error_location") == "TEST_CODE"

    # System prompt: prefer backend
    if backend:
        system_prompt = backend.tester_system_prompt(source_location, import_instruction)
    else:
        system_prompt = (
            "You are a strict QA Automation Engineer.\n"
            "Write robust unit tests for the provided module.\n"
            "The code is part of a larger project, so ensure imports are correct.\n"
            f"{import_instruction}\n"
            "Your tests should cover all functions and classes in the module.\n"
            "Include edge cases, type boundary checks, and failure scenarios.\n"
            f"The module to be tested is located at the path '{source_location}'.\n"
            "Return ONLY valid code for the test file. Do not include markdown formatting."
        )

    # Human template: prefer backend
    if is_fix:
        if backend:
            human_template = backend.test_human_template_fix()
        else:
            human_template = (
                "The previous attempt to write tests failed. Please fix them.\n\n"
                "Module Specification:\n{spec}\n\n"
                "Generated Code to Test:\n{code}\n\n"
                "PREVIOUS (BROKEN) TESTS:\n{previous_tests}\n\n"
                "DIAGNOSIS OF FAILURE:\n{error_summary}\n\n"
                "Your task is to provide a new, corrected version of the unit tests."
            )
        invoke_params = {
            "spec": str(state["spec"]),
            "code": state["code"],
            "previous_tests": state["tests"],
            "error_summary": state["error_summary"],
        }
    else:
        if backend:
            human_template = backend.test_human_template_initial()
        else:
            human_template = (
                "Module Specification:\n{spec}\n\n"
                "Generated Code to Test:\n{code}"
            )
        invoke_params = {
            "spec": str(state["spec"]),
            "code": state["code"],
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])

    func_logger.info(f"Tester: Sending prompt for '{state['module_location']}'.")
    func_logger.debug(f"Tester Prompt:\n{prompt.format(**invoke_params)}")

    chain = prompt | llm
    response = invoke_with_retry(chain, invoke_params)

    func_logger.debug(f"Tester Response:\n{response.content}")

    response_content = strip_markdown_code(response.content)

    return {
        "tests": response_content + "\n",
    }
