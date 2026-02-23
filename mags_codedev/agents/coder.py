from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import FunctionState
# (Assuming a helper function that loads the configured LLM for a specific role)
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.logger import logger
from mags_codedev.utils.db import log_token_usage

def coder_node(state: FunctionState) -> dict:
    """Generates or updates the code based on specifications and feedback."""
    config_path = state["config_path"]
    llm = get_llm(role="coder", config_path=config_path)
    
    # Determine context based on whether this is a first run or a fix
    is_fix = state.get("iteration_count", 0) > 0
    feedback = ""
    prompt_narrative = "Write the initial implementation of this function."
    
    # Build the feedback block for fixes or provide existing code for updates.
    if is_fix:
        prompt_narrative = "Fix the following code based on the provided logs and reviews."
        feedback_parts = []
        if state.get("code"):
            feedback_parts.append(f"PREVIOUS CODE:\n{state.get('code')}")
        if state.get("tests"): # Also show the tests that were run
            feedback_parts.append(f"TESTS THAT FAILED:\n{state.get('tests')}")
        if state.get("error_summary"):
            feedback_parts.append(f"DIAGNOSIS:\n{state.get('error_summary')}")
        if state.get("review_comments"):
            feedback_parts.append(f"PEER REVIEW COMMENTS:\n{chr(10).join(state.get('review_comments'))}")
        
        feedback = "\n\n".join(feedback_parts)
    elif state.get("code"): # This is an update of existing code, not a fix in this run
        prompt_narrative = "Update the following function based on the specification."
        feedback = f"EXISTING CODE:\n{state.get('code', '')}"
        
    system_prompt = """You are an expert Software Engineer.
    Write clean, production-ready Python code. Follow all instructions exactly.
    Return ONLY valid Python code. Do not include markdown formatting like ```python."""

    human_template = """
    Function Name: {function_name}
    Specification: {spec}
    
    {prompt_narrative}
    
    {feedback}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    logger.info(f"Coder: Sending prompt for '{state['function_name']}' (iteration {state.get('iteration_count', 0) + 1}).")
    logger.debug(f"Coder Prompt:\n{prompt.format(function_name=state['function_name'], spec=str(state['spec']), prompt_narrative=prompt_narrative, feedback=feedback)}")

    chain = prompt | llm
    response = chain.invoke({
        "function_name": state['function_name'],
        "spec": str(state['spec']),
        "prompt_narrative": prompt_narrative,
        "feedback": feedback
    })
    
    # Log token usage
    try:
        usage = response.response_metadata.get("token_usage", {})
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        log_token_usage(
            role="coder",
            model=llm.model_name,
            in_tokens=in_tokens,
            out_tokens=out_tokens
        )
        logger.debug(f"Coder Token Usage: In={in_tokens}, Out={out_tokens}")
    except Exception as e:
        logger.warning(f"Could not log token usage for coder: {e}")

    logger.debug(f"Coder Response:\n{response.content}")

    return {
        "code": response.content.strip(),
        # Increment the iteration counter
        "iteration_count": state.get("iteration_count", 0) + 1,
        # Clear previous feedback
        "error_summary": "",
        "review_comments": []
    }