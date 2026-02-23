import asyncio
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import FunctionState
from mags_codedev.utils.config_parser import get_reviewer_llms
from mags_codedev.utils.logger import logger
from mags_codedev.utils.db import log_token_usage

async def _get_review(llm, state: FunctionState) -> str:
    """Helper function to execute a single review asynchronously."""
    system_prompt = """You are a strict Code Reviewer.
    Analyze this code for security flaws, performance bottlenecks, and best practices.
    If the code is perfect, reply EXACTLY with 'LGTM'.
    If there are issues, list them clearly."""
    
    human_template = "Spec: {spec}\nCode:\n{code}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    logger.info(f"Reviewer ({llm.model_name}): Sending prompt for '{state['function_name']}'.")
    # The debug log will go to the file, not the console, per logger.py setup
    logger.debug(f"Reviewer Prompt for {llm.model_name}:\n{prompt.format(spec=str(state['spec']), code=state['code'])}")
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "spec": str(state['spec']),
        "code": state['code']
    })
    
    # Log token usage from response metadata
    try:
        usage = response.response_metadata.get("token_usage", {})
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        log_token_usage(
            role=f"reviewer_{llm.model_name}",
            model=llm.model_name,
            in_tokens=in_tokens,
            out_tokens=out_tokens
        )
        logger.debug(f"Reviewer ({llm.model_name}) Token Usage: In={in_tokens}, Out={out_tokens}")
    except Exception as e:
        logger.warning(f"Could not log token usage for reviewer {llm.model_name}: {e}")

    logger.debug(f"Reviewer ({llm.model_name}) Response:\n{response.content}")
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