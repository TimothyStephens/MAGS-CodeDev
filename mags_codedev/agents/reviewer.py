"""Reviewer agent: multi-LLM code review with LGTM/vote aggregation."""

import asyncio
import re
from langchain_core.prompts import ChatPromptTemplate
from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_reviewer_llms
from mags_codedev.utils.logger import get_dual_loggers, logger
from mags_codedev.utils.retry import ainvoke_with_retry
from mags_codedev.utils.llm_call import build_context_blocks
# Sentinel returned by ``_get_review`` when a reviewer's API call failed.
# Deliberately contains no "LGTM" so a skipped reviewer is never counted as an
# approval vote (see the quorum logic in ``multi_llm_review_node``).
_REVIEW_SKIPPED = "__REVIEW_SKIPPED__"


async def _get_review(llm, state: ModuleState) -> str:
    """Helper function to execute a single review asynchronously with retry logic."""
    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))

    func_logger, resp_logger = get_dual_loggers(state.get("log_filepath"))
    session = state.get("_session_number", 1)
    system_prompt = (
        "You are a strict Code Reviewer.\n"
        "Review this code for: correctness, edge case handling, naming conventions,\n"
        "code complexity, security flaws, performance bottlenecks, and best practices.\n\n"
        "ALSO check documentation quality:\n"
        "- Does every public function/class have a docstring (Google style)?\n"
        "- Are non-obvious logic and business rules commented?\n"
        "- Is the file organized clearly (imports, constants, classes, functions)?\n"
        "- Are there unused imports or dead code?\n"
        "- Do all functions have type hints?\n\n"
        "If the code is perfect, reply EXACTLY with 'LGTM'.\n"
        "If there are issues, list them clearly with specific line references.\n"
        "Do NOT flag style preferences — only flag things that affect\n"
        "correctness, maintainability, or readability."
    )
    # Shared context blocks (project instructions + dependency source).
    project_instructions_block, dependency_context = build_context_blocks(state)
    human_template = project_instructions_block + dependency_context + "Spec: {spec}\nCode:\n{code}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])

    review_round = state.get("review_round_count", 0) + 1
    func_logger.info(
        "[Session %d, Review Round %d] Reviewer (%s): sending prompt for '%s'.",
        session,
        review_round,
        model_name,
        state['module_location'],
    )
    # The debug log will go to the file, not the console, per logger.py setup
    func_logger.debug(
        "Reviewer Prompt for %s:\n%s",
        model_name,
        prompt.format(spec=str(state["spec"]), code=state["code"]),
    )
    try:
        chain = prompt | llm
        response = await ainvoke_with_retry(chain, {
            "spec": str(state['spec']),
            "code": state['code']
        })
        func_logger.debug(f"Reviewer ({model_name}) Response:\n{response.content}")

        # INFO: log response summary
        raw_content = response.content
        if isinstance(raw_content, list):
            raw_text = "".join(
                block if isinstance(block, str) else
                block.get("text", "") if isinstance(block, dict) else
                getattr(block, "text", str(block))
                for block in raw_content
            )
        else:
            raw_text = str(raw_content)

        first_line = raw_text.strip().split("\n")[0]
        func_logger.info(
            "[Session %d, Review Round %d] Reviewer (%s): received review (%s).",
            session,
            review_round,
            model_name,
            first_line[:100] if len(first_line) > 100 else first_line,
        )

        # TRACE: log token usage if available
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            func_logger.trace(
                "[Session %d, Review Round %d] Reviewer (%s) tokens: input=%s, output=%s",
                session,
                review_round,
                model_name,
                meta.get("input_tokens", "?"),
                meta.get("output_tokens", "?"),
            )

        content = response.content
        if isinstance(content, list):
            content = "".join(
                block if isinstance(block, str) else
                block.get("text", "") if isinstance(block, dict) else
                getattr(block, "text", str(block))
                for block in content
            )

        resp_logger.info(
            "[Session %d, Review Round %d] Reviewer (%s) full review:\n%s",
            session, review_round, model_name, content,
        )
    except Exception as e:
        func_logger.warning(
            f"Reviewer ({model_name}) API call failed: {e}. "
            f"Skipping this reviewer."
        )
        content = f"{_REVIEW_SKIPPED} ({type(e).__name__})"
    return str(content)


async def multi_llm_review_node(state: ModuleState) -> dict:
    """Runs multiple LLMs concurrently to review the final code."""
    config_path = state["config_path"]
    llms = get_reviewer_llms(config_path=config_path)

    # Run all reviewers concurrently
    tasks = [_get_review(llm, state) for llm in llms]
    reviews = await asyncio.gather(*tasks)

    # A skipped reviewer (API failure) is NEUTRAL — neither an approval nor
    # actionable feedback. Approval requires a strict majority of ALL
    # configured reviewers to reply LGTM, so a partial outage can never grant
    # a 1-of-N approval. Bounded by max_review_rounds in the graph.
    skipped = [r for r in reviews if r.startswith(_REVIEW_SKIPPED)]
    successful = [r for r in reviews if not r.startswith(_REVIEW_SKIPPED)]

    func_logger, _ = get_dual_loggers(state.get("log_filepath"))
    reason = state.get("_next_reason", "")
    if reason:
        func_logger.info(f"[Reason] {reason}")

    current_rounds = state.get("review_round_count", 0)

    # All reviewers failed: surface a clear retry signal.
    if llms and not successful:
        func_logger.warning(
            "All reviewer LLM calls failed. Code not approved. "
            "(%d/%d skipped)", len(skipped), len(llms),
        )
        return {
            "review_comments": ["All reviewers failed — please retry the build."],
            "status": "in_progress",
            "review_round_count": current_rounds + 1,
        }

    # Actionable comments come only from reviewers that actually responded.
    actionable_comments = [
        r for r in successful if not re.search(r"\bLGTM\b", r, re.IGNORECASE)
    ]
    if actionable_comments:
        func_logger.info(
            "─── Review Decision ───\n%d/%d reviewers gave actionable feedback "
            "(%d skipped). Sending back to coder for revision.",
            len(actionable_comments), len(llms), len(skipped),
        )
        return {
            "review_comments": actionable_comments,
            "status": "in_progress",
            "review_round_count": current_rounds + 1,
        }

    # No actionable comments: count LGTM votes. A strict majority of ALL
    # configured reviewers must approve; otherwise too many were skipped
    # (quorum unmet) and we ask for another round.
    approvals = [r for r in successful if re.search(r"\bLGTM\b", r, re.IGNORECASE)]
    if len(approvals) * 2 > len(llms):
        func_logger.info(
            "─── Review Decision ───\n%d/%d reviewers approved (LGTM). "
            "Strict majority met — code accepted.",
            len(approvals), len(llms),
        )
        return {
            "review_comments": [],
            "status": "success",
        }

    func_logger.warning(
        "Reviewer quorum unmet: %d/%d approved (%d skipped).",
        len(approvals), len(llms), len(skipped),
    )
    return {
        "review_comments": [
            f"Insufficient reviewer quorum: {len(approvals)}/{len(llms)} "
            f"approved ({len(skipped)} skipped). Please retry."
        ],
        "status": "in_progress",
        "review_round_count": current_rounds + 1,
    }
