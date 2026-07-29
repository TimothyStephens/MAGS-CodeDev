"""Shared LLM-call infrastructure for the agent nodes.

Centralizes the boilerplate that was duplicated verbatim across the coder,
tester, and log-checker agents:

* shared prompt context blocks (project instructions + dependency source),
* LLM invocation with exponential-backoff retry (:func:`invoke_with_retry`),
* response content flattening + markdown-fence stripping,
* structured conversation logging to the per-module log.

Conversation logging (the "failed-task chat log"):
    * INFO (default) writes a concise exchange summary to the module log —
      the task narrative, payload line-counts (code/tests/deps elided to
      counts, NOT dumped), and the inter-agent diagnostics
      (``test_error_summary`` / ``review_comments``) which ARE the chat.
    * DEBUG (``-v``) additionally writes the full prompt + full response
      (including embedded file contents) to the module log.

Adding a new agent: write a role-specific system/human prompt, assemble
``params``, and call :func:`call_llm`. No per-agent retry/logging/content
logic is required. For async / multi-LLM patterns (e.g. the reviewer), use
:func:`build_context_blocks` directly with :func:`ainvoke_with_retry`.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from mags_codedev.state import ModuleState
from mags_codedev.utils.config_parser import get_llm
from mags_codedev.utils.llm_helpers import extract_content, strip_markdown_code
from mags_codedev.utils.logger import get_dual_loggers
from mags_codedev.utils.retry import invoke_with_retry


def build_context_blocks(state: ModuleState) -> tuple[str, str]:
    """Build the shared PROJECT INSTRUCTIONS and DEPENDENCY CONTEXT blocks.

    Returns ``(project_block, dependency_block)`` ready to be concatenated
    into a human prompt template. Dependency source is brace-escaped
    (``{{``/``}}``) so it can sit inside a :class:`ChatPromptTemplate` that
    uses ``{...}`` placeholders for the role-specific fields.
    """
    project_instructions = state.get("project_instructions", "")
    proj_block = ""
    if project_instructions:
        # Brace-escape so the text is safe inside a ChatPromptTemplate (some
        # callers concatenate the block into the template string itself).
        safe = project_instructions.replace("{", "{{").replace("}", "}}")
        proj_block = (
            "PROJECT INSTRUCTIONS\n"
            "These are project-wide guidelines and conventions.\n\n"
            + safe + "\n\n"
        )

    dep_code = state.get("dependency_code", {})
    dep_block = ""
    if dep_code:
        parts = [
            f"--- File: {loc} ---\n```python\n"
            f"{source.replace('{', '{{').replace('}', '}}')}\n```"
            for loc, source in dep_code.items()
        ]
        dep_block = (
            "DEPENDENCY CONTEXT\n"
            "These are the source files this module depends on.\n"
            "Use them to understand available functions, classes, and types.\n\n"
            + "\n\n".join(parts) + "\n\n"
        )
    return proj_block, dep_block


def _line_count(text) -> int:
    if not text:
        return 0
    return len(str(text).splitlines())


def _exchange_summary(state: ModuleState, params: dict, narrative: str) -> str:
    """Concise, file-content-free summary of a prompt for the INFO chat log."""
    bits = []
    if narrative:
        bits.append(f"task={narrative}")

    code = params.get("code") or state.get("code")
    if code:
        bits.append(f"code={_line_count(code)}L")
    prev_tests = params.get("previous_tests") or state.get("tests")
    if prev_tests:
        bits.append(f"tests={_line_count(prev_tests)}L")
    deps = state.get("dependency_code", {})
    if deps:
        bits.append(f"deps={len(deps)} ({', '.join(deps.keys())})")

    payloads = " | ".join(bits) if bits else "no payloads"

    # Inter-agent diagnostics ARE the chat — surface them, not the file dumps.
    diag = []
    err = state.get("test_error_summary", "")
    if err:
        diag.append(f"errors: {err[:300]}")
    reviews = state.get("review_comments", [])
    if reviews:
        preview = reviews[0][:200] if reviews else ""
        diag.append(f"reviews: {len(reviews)} (e.g. {preview!r})")
    diagnostics = " | ".join(diag) if diag else "diagnostics: none"

    return f"{payloads} — {diagnostics}"


def call_llm(
    *,
    role: str,
    system_prompt: str,
    human_template: str,
    params: dict,
    state: ModuleState,
    narrative: str = "",
) -> str:
    """Run a single synchronous agent LLM call.

    Parameters
    ----------
    role :
        Config key under ``models.build_workflow`` (e.g. ``"coder"``).
    system_prompt, human_template :
        The two prompt halves; ``human_template`` must contain ``{...}``
        placeholders matching keys in *params*.
    params :
        Values substituted into ``human_template``.
    state :
        The current :class:`ModuleState` (used for logging + config path).
    narrative :
        Short description of the task for the INFO chat log.

    Returns the cleaned response text (markdown fences stripped).

    Raises on a hard (non-retryable) LLM failure so the task fails loudly;
    transient errors are retried by :func:`invoke_with_retry`.
    """
    log_level = state.get("log_level", "info")
    base_dir = state.get("base_dir", ".mags-codedev")
    func_logger, resp_logger = get_dual_loggers(
        state.get("log_filepath"), base_dir=base_dir, log_level=log_level,
    )

    reason = state.get("_next_reason", "")
    if reason:
        func_logger.info(f"[Reason] {reason}")

    session = state.get("_session_number", 1)
    iteration = state.get("iteration_count", 0) + 1
    module = state["module_location"]
    config_path = state["config_path"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template),
    ])

    func_logger.info(
        "[Session %d, Iteration %d] %s: sending prompt for '%s'.",
        session, iteration, role.capitalize(), module,
    )
    # INFO: concise exchange summary (module log only, no file contents).
    resp_logger.info(
        "[Session %d, Iteration %d] %s prompt for '%s' — %s",
        session, iteration, role.capitalize(), module,
        _exchange_summary(state, params, narrative),
    )
    # DEBUG (-v): full prompt incl. embedded file contents. Guarded so the
    # (large) formatted string is only built when debug logging is enabled.
    if func_logger.isEnabledFor(logging.DEBUG):
        func_logger.debug("%s Prompt:\n%s", role.capitalize(), prompt.format(**params))

    llm = get_llm(role=role, config_path=config_path)
    chain = prompt | llm
    try:
        response = invoke_with_retry(chain, params)
    except Exception as e:
        # Retry-exhausted (transient) or hard (auth/quota) failure: fail the
        # task with a real, logged error rather than a silent stub.
        func_logger.error(
            "[Session %d, Iteration %d] %s LLM call failed for '%s': %s",
            session, iteration, role.capitalize(), module, e,
        )
        raise

    content = extract_content(response.content)
    first_line = content.strip().split("\n")[0]
    func_logger.info(
        "[Session %d, Iteration %d] %s: received response (%s).",
        session, iteration, role.capitalize(),
        first_line[:100] if len(first_line) > 100 else first_line,
    )
    # DEBUG (-v): full response to both the propagating and module-only logs.
    if func_logger.isEnabledFor(logging.DEBUG):
        func_logger.debug("%s Response:\n%s", role.capitalize(), content)
        resp_logger.debug(
            "[Session %d, Iteration %d] %s Response:\n%s",
            session, iteration, role.capitalize(), content,
        )
    return strip_markdown_code(content)


__all__ = ["build_context_blocks", "call_llm"]
