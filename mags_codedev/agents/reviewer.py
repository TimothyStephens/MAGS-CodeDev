import asyncio
import logging
import httpx
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_reviewer_llms
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
async def _get_review(llm, state: ModuleState) -> str:
    """Helper function to execute a single review asynchronously with retry logic."""
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
    
    func_logger.info(f"Reviewer ({model_name}): Sending prompt for '{state['module_location']}'.")
    # The debug log will go to the file, not the console, per logger.py setup
    func_logger.debug(f"Reviewer Prompt for {model_name}:\n{prompt.format(spec=str(state['spec']), code=state['code'])}")
    
    chain = prompt | llm
    response = await chain.ainvoke({
        "spec": str(state['spec']),
        "code": state['code']
    })

    func_logger.debug(f"Reviewer ({model_name}) Response:\n{response.content}")
    content = response.content
    if isinstance(content, list):
        content = "".join(
            block if isinstance(block, str) else 
            block.get("text", "") if isinstance(block, dict) else 
            getattr(block, "text", str(block))
            for block in content
        )
    return str(content)

async def multi_llm_review_node(state: ModuleState) -> dict:
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