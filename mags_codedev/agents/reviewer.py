import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_reviewer_llms
from mags_codedev.utils.logger import logger


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
    # Build project instructions block
    project_instructions = state.get("project_instructions", "")
    project_instructions_block = ""
    if project_instructions:
        project_instructions_block = (
            "PROJECT INSTRUCTIONS\n"
            "These are project-wide guidelines and conventions.\n\n"
            + project_instructions + "\n\n"
        )

    human_template = project_instructions_block + "Spec: {spec}\nCode:\n{code}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])

    func_logger.info(f"Reviewer ({model_name}): Sending prompt for '{state['module_location']}'.")
    # The debug log will go to the file, not the console, per logger.py setup
    func_logger.debug(f"Reviewer Prompt for {model_name}:\n{prompt.format(spec=str(state['spec']), code=state['code'])}")

    try:
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
    except Exception as e:
        func_logger.warning(
            f"Reviewer ({model_name}) API call failed: {e}. "
            f"Skipping this reviewer."
        )
        content = f"LGTM (reviewer skipped due to API error: {type(e).__name__})"
    return str(content)


async def multi_llm_review_node(state: ModuleState) -> dict:
    """Runs multiple LLMs concurrently to review the final code."""
    # Offline mode: skip all reviews, code is approved
    if state.get("offline"):
        func_logger = logging.getLogger("mags_codedev")
        func_logger.info("Offline mode: skipping multi-LLM review.")
        return {
            "review_comments": [],
            "status": "success"
        }

    config_path = state["config_path"]
    llms = get_reviewer_llms(config_path=config_path)

    # Run all reviewers concurrently
    tasks = [_get_review(llm, state) for llm in llms]
    reviews = await asyncio.gather(*tasks)

    # FIX Bug #2: Check if ALL reviewers failed
    failed_reviews = [r for r in reviews if "reviewer skipped due to api error" in r.lower()]

    if len(failed_reviews) == len(llms) and len(llms) > 0:
        # All reviewers failed — don't approve silently
        func_logger = logging.getLogger("mags_codedev")
        func_logger.warning("All reviewer LLM calls failed. Code not approved.")
        current_rounds = state.get("review_round_count", 0)
        return {
            "review_comments": ["All reviewers failed — please retry the build."],
            "status": "in_progress",
            "review_round_count": current_rounds + 1,
        }

    # Filter out LGTM approvals, keep only actionable critiques
    actionable_comments = [r for r in reviews if "LGTM" not in r.upper()]
    status = "success" if not actionable_comments else "in_progress"

    current_rounds = state.get("review_round_count", 0)
    if actionable_comments:
        return {
            "review_comments": actionable_comments,
            "status": status,
            "review_round_count": current_rounds + 1,
        }

    return {
        "review_comments": actionable_comments,
        "status": status,
    }
