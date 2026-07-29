"""Tester agent: generates or fixes pytest unit tests via LLM."""

from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.retry import invoke_with_retry
from mags_codedev.utils.llm_helpers import strip_markdown_code
from mags_codedev.utils.logger import get_dual_loggers


def tester_node(state: ModuleState) -> dict:
    """Writes comprehensive unit tests."""
    config_path = state["config_path"]
    func_logger, resp_logger = get_dual_loggers(state.get("log_filepath"))
    reason = state.get("_next_reason", "")
    if reason:
        func_logger.info(f"[Reason] {reason}")
    session = state.get("_session_number", 1)
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
            "These are the source files this module depends on.\n"
            "Use them to understand available functions, classes, and types.\n\n"
            + "\n\n".join(dep_parts) + "\n\n"
        )
    # Check if we are in a fix cycle for tests
    # Use test_error_summary instead of error_summary (scoped tracking)
    is_fix = bool(state.get("test_error_summary")) and state.get("error_location") == "TEST_CODE"

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
            human_template = project_instructions_block + dependency_context + backend.test_human_template_fix()
        else:
            human_template = (
                project_instructions_block +
                dependency_context +
                "The previous attempt to write tests failed. Please fix them.\n\n"
                "Module Specification:\n{spec}\n\n"
                "Generated Code to Test:\n{code}\n\n"
                "PREVIOUS (BROKEN) TESTS:\n{previous_tests}\n\n"
                "DIAGNOSIS OF FAILURE:\n{test_error_summary}\n\n"
                "{review_feedback}\n"
                "Your task is to provide a new, corrected version of the unit tests."
            )
        # Build review feedback if review comments mention tests
        review_feedback = ""
        review_comments = state.get("review_comments", [])
        test_reviews = [c for c in review_comments if "test" in c.lower()]
        if test_reviews:
            review_feedback = "REVIEW COMMENTS ABOUT TESTS:\n" + "\n".join(f"- {c}" for c in test_reviews) + "\n\n"
        invoke_params = {
            "spec": str(state["spec"]),
            "code": state["code"],
            "previous_tests": state["tests"],
            "test_error_summary": state["test_error_summary"],
            "review_feedback": review_feedback,
        }
    else:
        if backend:
            human_template = project_instructions_block + dependency_context + backend.test_human_template_initial()
        else:
            human_template = (
                project_instructions_block +
                dependency_context +
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

    iteration = state.get("iteration_count", 0) + 1
    func_logger.info(
        "[Session %d, Iteration %d] Tester: sending prompt for '%s'.",
        session,
        iteration,
        state['module_location'],
    )
    func_logger.debug(f"Tester Prompt:\n{prompt.format(**invoke_params)}")
    llm = get_llm(role="tester", config_path=config_path)
    try:
        chain = prompt | llm
        response = invoke_with_retry(chain, invoke_params)
        func_logger.debug(f"Tester Response:\n{response.content}")

        # INFO: log response summary
        first_line = response.content.strip().split("\n")[0]
        func_logger.info(
            "[Session %d, Iteration %d] Tester: received response (%s).",
            session,
            iteration,
            first_line[:100] if len(first_line) > 100 else first_line,
        )

        # TRACE: log token usage if available
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            func_logger.trace(
                "[Session %d, Iteration %d] Tester tokens: input=%s, output=%s",
                session,
                iteration,
                meta.get("input_tokens", "?"),
                meta.get("output_tokens", "?"),
            )

        response_content = strip_markdown_code(response.content)
    except Exception as e:
        func_logger.warning(
            f"LLM API call failed for tests '{state['module_location']}': "
            f"{e}. Generating stub tests."
        )
        module_name = Path(state["module_location"]).stem
        response_content = (
            f"import pytest\n\n\n"
            f"class Test{module_name.replace('_', ' ').title().replace(' ', '')}:\n"
            f'    """Test {module_name}."""\n'
            f"\n"
            f"    def test_{module_name}_main(self):\n"
            f'        """Test {module_name}_main."""\n'
            f"        # TODO: Implement test for {module_name}_main\n"
            f"        pass\n"
        )
        state.setdefault("last_error", str(e))

    resp_logger.debug(f"[Session {session}, Iteration {iteration}] Tester Response:\n{response_content}")

    return {
        "tests": response_content + "\n",
        "iteration_count": state.get("iteration_count", 0),
    }
