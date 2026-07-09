from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.retry import invoke_with_retry
from mags_codedev.utils.llm_helpers import get_function_logger, strip_markdown_code


def coder_node(state: ModuleState) -> dict:
    """Generates or updates the code based on specifications and feedback."""
    config_path = state["config_path"]
    llm = get_llm(role="coder", config_path=config_path)
    func_logger = get_function_logger(state.get("log_filepath"))
    backend = state.get("backend")

    # Determine context based on whether this is a first run or a fix
    is_fix = state.get("iteration_count", 0) > 0
    feedback = ""
    prompt_narrative = "Write the initial implementation of this module."

    if backend:
        prompt_narrative = backend.coder_system_prompt()

    if is_fix:
        prompt_narrative = "Fix the following code based on the provided logs and reviews."
        feedback_parts = []
        if state.get("code"):
            feedback_parts.append(f"PREVIOUS CODE:\n{state.get('code')}")
        if state.get("tests"):
            feedback_parts.append(f"TESTS THAT FAILED:\n{state.get('tests')}")
        if state.get("error_summary"):
            feedback_parts.append(f"DIAGNOSIS:\n{state.get('error_summary')}")
        if state.get("review_comments"):
            feedback_parts.append(
                f"PEER REVIEW COMMENTS:\n{chr(10).join(state.get('review_comments'))}"
            )
        feedback = "\n\n".join(feedback_parts)
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

    human_template = (
        "\nModule Location: {module_location}\n"
        "Specification: {spec}\n\n"
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
        f"Coder Prompt:\n{prompt.format(module_location=state['module_location'], spec=str(state['spec']), prompt_narrative=prompt_narrative, feedback=feedback)}"
    )

    chain = prompt | llm
    response = invoke_with_retry(chain, {
        "module_location": state["module_location"],
        "spec": str(state["spec"]),
        "prompt_narrative": prompt_narrative,
        "feedback": feedback
    })

    func_logger.debug(f"Coder Response:\n{response.content}")

    response_content = strip_markdown_code(response.content)

    return {
        "code": response_content + "\n",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "review_comments": []
    }
