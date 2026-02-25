import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import FunctionState
from mags_codedev.utils.config_parser import get_reviewer_llms
from mags_codedev.utils.logger import logger

async def _get_review(llm, state: FunctionState) -> str:
    """Helper function to execute a single review asynchronously."""
    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
    
    if state.get("log_filepath"):
        import os
        log_hash = os.path.basename(state["log_filepath"]).replace(".log", "")
        func_logger = logging.getLogger(f"mags.func.{log_hash}")
    else:
        func_logger = logger
    system_prompt = """You are a strict Code Reviewer.
    Analyze this code for security flaws, performance bottlenecks, and best practices.
    If the code is perfect, reply EXACTLY with 'LGTM'.
    If there are issues, list them clearly."""
    
    human_template = "Spec: {spec}\nCode:\n{code}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    func_logger.info(f"Reviewer ({model_name}): Sending prompt for '{state['function_name']}'.")
    # The debug log will go to the file, not the console, per logger.py setup
    func_logger.debug(f"Reviewer Prompt for {model_name}:\n{prompt.format(spec=str(state['spec']), code=state['code'])}")
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "spec": str(state['spec']),
        "code": state['code']
    })

    func_logger.debug(f"Reviewer ({model_name}) Response:\n{response.content}")
    return response.content

async def multi_llm_review_node(state: FunctionState) -> dict:
    """Runs multiple LLMs concurrently to review the final code."""
    config_path = state["config_path"]
    llms = get_reviewer_llms(config_path=config_path) # Returns a list of configured models
    
    # Run all reviewers concurrently
    tasks = [_get_review(llm, state) for llm in llms]
    reviews = await asyncio.gather(*tasks)
    
    # Filter out approvals ('LGTM') to leave only actionable critiques
    actionable_comments = [
        rev for rev in reviews if "LGTM" not in rev.upper()
    ]
    
    # If there are no actionable comments, the code is approved.
    status = "success" if not actionable_comments else "in_progress"
    
    return {
        "review_comments": actionable_comments,
        "status": status
    }