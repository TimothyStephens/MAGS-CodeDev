from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.retry import invoke_with_retry
from mags_codedev.utils.llm_helpers import get_function_logger, strip_markdown_code


def coder_node(state: ModuleState) -> dict:
    """Generates or updates the code based on specifications and feedback."""
    config_path = state["config_path"]
    func_logger = get_function_logger(state.get("log_filepath"))
    backend = state.get("backend")

    # Determine context based on whether this is a first run or a fix
    is_fix = state.get("iteration_count", 0) > 0
    feedback = ""
    prompt_narrative = "Write the initial implementation of this module."

    if backend:
        prompt_narrative = backend.coder_system_prompt()

    if is_fix:
        # FIX Bug #1: Include review_comments AND test_error_summary in feedback
        feedback_parts = []

        # Include test error summary if present (from log_checker)
        test_error = state.get("test_error_summary", "")
        if test_error and "no clear issues" not in test_error.lower():
            feedback_parts.append(f"TEST/LINT ERRORS:\n{test_error}")

        # Include review comments if present (from multi_llm_review) — FIX Bug #1
        review_comments = state.get("review_comments", [])
        if review_comments:
            feedback_parts.append(f"REVIEW COMMENTS:\n" + "\n".join(f"- {c}" for c in review_comments))

        # Always include existing code so the coder knows what to change
        if state.get("code"):
            feedback_parts.append(f"EXISTING CODE:\n{state['code']}")

        # Include current tests for context
        if state.get("tests"):
            feedback_parts.append(f"CURRENT TESTS:\n{state['tests']}")

        feedback = "\n\n".join(feedback_parts) if feedback_parts else ""

        # Determine narrative based on what feedback we have
        if review_comments and not test_error:
            prompt_narrative = "Revise the following code based on the review feedback."
        elif test_error:
            prompt_narrative = "Fix the following code based on the test/lint errors."
        else:
            prompt_narrative = "Update the following code based on the specification."
    elif state.get("code"):
        prompt_narrative = "Update the following module based on the specification."
        feedback = f"EXISTING CODE:\n{state.get('code', '')}"

    # System prompt: prefer backend, fallback to generic
    if backend:
        system_prompt = backend.coder_system_prompt()
    else:
        system_prompt = (
            "You are an expert Software Engineer.\n"
            "Write a full, clean, production-ready module. "
            "Follow all instructions exactly.\n"
            "Return ONLY valid code for the entire file. "
            "Do not include markdown formatting."
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
            dep_parts.append(f"--- File: {loc} ---\n```python\n{source}\n```")
        dependency_context = (
            "DEPENDENCY CONTEXT\n"
            "These are the source files this module depends on.\n"
            "Use them to understand available functions, classes, and types.\n\n"
            + "\n\n".join(dep_parts) + "\n\n"
        )
    human_template = (
        "\nModule Location: {module_location}\n"
        "Specification: {spec}\n\n"
        "{project_instructions_block}"
        "{dependency_context}"
        "{prompt_narrative}\n\n"
        "{feedback}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])

    func_logger.info(
        f"Coder: Sending prompt for '{state['module_location']}' "
        f"(iteration {state.get('iteration_count', 0) + 1})."
    )
    func_logger.debug(
        f"Coder Prompt:\n{prompt.format(module_location=state['module_location'], spec=str(state['spec']), project_instructions_block=project_instructions_block, dependency_context=dependency_context, prompt_narrative=prompt_narrative, feedback=feedback)}"
    )

    # Offline mode: generate stub without API call
    if state.get("offline"):
        module_name = Path(state["module_location"]).stem
        response_content = (
            f'"""\n{state["spec"].get("description", "")}\n"""\n\n'
            f"# Stub implementation (offline mode)\n"
            f"# Module: {state['module_location']}\n\n\n"
            f"def {module_name}_main():\n"
            f'    """Main function for {module_name}."""\n'
            f"    pass\n\n\n"
            f'if __name__ == "__main__":\n'
            f"    {module_name}_main()\n"
        )
        func_logger.info(f"Offline mode: generated stub for '{state['module_location']}'.")
    else:
        # FIX M1: Get LLM only when actually needed (not in offline mode)
        llm = get_llm(role="coder", config_path=config_path)
        try:
            chain = prompt | llm
            response = invoke_with_retry(chain, {
                "module_location": state["module_location"],
                "spec": str(state["spec"]),
                "project_instructions_block": project_instructions_block,
                "dependency_context": dependency_context,
                "prompt_narrative": prompt_narrative,
                "feedback": feedback
            })
            func_logger.debug(f"Coder Response:\n{response.content}")
            response_content = strip_markdown_code(response.content)
        except Exception as e:
            func_logger.warning(
                f"LLM API call failed for '{state['module_location']}': "
                f"{e}. Generating stub implementation."
            )
            module_name = Path(state["module_location"]).stem
            response_content = (
                f'"""\n{state["spec"].get("description", "")}\n"""\n\n'
                f"# Stub implementation - generated when LLM API unavailable\n"
                f"# Module: {state['module_location']}\n\n\n"
                f"def {module_name}_main():\n"
                f'    """Main function for {module_name}."""\n'
                f"    pass\n\n\n"
                f'if __name__ == "__main__":\n'
                f"    {module_name}_main()\n"
            )
            state.setdefault("last_error", str(e))

    return {
        "code": response_content + "\n",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "review_comments": []
    }
