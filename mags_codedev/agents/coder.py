import logging
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
# (Assuming a helper function that loads the configured LLM for a specific role)
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.logger import logger

def coder_node(state: ModuleState) -> dict:
    """Generates or updates the code based on specifications and feedback."""
    config_path = state["config_path"]
    llm = get_llm(role="coder", config_path=config_path)
    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
    
    # Use function-specific logger if available
    if state.get("log_filepath"):
        # Re-acquire the logger based on the hash logic used in cli.py
        # Since we don't have the hash here easily without re-hashing, we can rely on the fact that
        # we set up the logger with a specific name in cli.py.
        # However, passing the logger name in state would be cleaner.
        # For now, let's just use the global logger which is what was requested to be changed.
        # Actually, to write to the specific file, we need that specific logger.
        # Let's assume we update the agents to use a logger passed in state or derived.
        # Since we didn't add logger object to state (can't pickle easily), we rely on the file handler setup in the process.
        # But wait, `logger` imported above is the global one.
        # We need to get the logger for this function.
        # Let's assume we use the hash from the log filename to get the logger name.
        import os
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
    
    # Determine context based on whether this is a first run or a fix
    is_fix = state.get("iteration_count", 0) > 0
    feedback = ""
    prompt_narrative = "Write the initial implementation of this Python module."
    
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
        prompt_narrative = "Update the following module based on the specification."
        feedback = f"EXISTING CODE:\n{state.get('code', '')}"
        
    system_prompt = """You are an expert Software Engineer.
    Write a full, clean, production-ready Python module. Follow all instructions exactly.
    Return ONLY valid Python code for the entire file. Do not include markdown formatting like ```python."""

    human_template = """
    Module Location: {module_location}
    Specification: {spec}
    
    {prompt_narrative}
    
    {feedback}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    func_logger.info(f"Coder: Sending prompt for '{state['module_location']}' (iteration {state.get('iteration_count', 0) + 1}).")
    func_logger.debug(f"Coder Prompt:\n{prompt.format(module_location=state['module_location'], spec=str(state['spec']), prompt_narrative=prompt_narrative, feedback=feedback)}")

    chain = prompt | llm
    response = chain.invoke({
        "module_location": state['module_location'],
        "spec": str(state['spec']),
        "prompt_narrative": prompt_narrative,
        "feedback": feedback
    })

    func_logger.debug(f"Coder Response:\n{response.content}")

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
        "code": response_content + "\n",
        # Increment the iteration counter
        "iteration_count": state.get("iteration_count", 0) + 1,
        "review_comments": []
    }