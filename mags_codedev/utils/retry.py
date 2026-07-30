"""Shared retry helpers for LLM agent calls."""

import logging
import re
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
try:
    from google.genai.errors import ServerError
except ImportError:
    ServerError = None  # type: ignore[assignment, misc]


# Retry configuration
DEFAULT_MAX_RETRIES: int = 5
EXPONENTIAL_MULTIPLIER: float = 2
EXPONENTIAL_MIN_WAIT: float = 4
EXPONENTIAL_MAX_WAIT: float = 60

def _is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is a transient API error worth retrying."""
    # Explicitly handle Google GenAI ServerError
    if ServerError is not None and isinstance(exception, ServerError):
        try:
            error_details = (getattr(exception, "response_json", None) or {}).get("error", {})
            message = error_details.get("message", "").lower()
            status = error_details.get("status", "").lower()
            code = error_details.get("code")

            if (
                (code == 503 and status == "unavailable")
                or "high demand" in message
                or "resource_exhausted" in message
                or "rate limit" in message
                or code == 429
            ):
                return True
        except Exception:
            pass

    # Explicitly handle httpx.RemoteProtocolError
    if isinstance(exception, httpx.RemoteProtocolError):
        return True

    # Fallback: check error message string (use word boundaries to avoid false positives)
    msg = str(exception).lower()
    return bool(
        re.search(r"\b503\b", msg)
        or "unavailable" in msg
        or "rate limit" in msg
        or re.search(r"\b429\b", msg)
        or "resource_exhausted" in msg
        or "server disconnected" in msg
    )


def _log_token_usage(result, _log) -> None:
    """TRACE-log token usage from an LLM result (multi-provider)."""
    if hasattr(result, "usage_metadata"):
        meta = result.usage_metadata
        _log(
            "Token usage: input=%s, output=%s",
            meta.get("input_tokens", "?"),
            meta.get("output_tokens", "?"),
        )
    elif hasattr(result, "response_metadata"):
        tokens = result.response_metadata.get("token_usage", {})
        _log(
            "Token usage: prompt=%s, completion=%s",
            tokens.get("prompt_tokens", "?"),
            tokens.get("completion_tokens", "?"),
        )


def _make_retry(max_attempts, min_wait, max_wait, on_retry):
    """Build a tenacity retry decorator bound to call-time configuration.

    Building the decorator per call (rather than at module load) lets tests
    and callers override the backoff without monkeypatching module constants.
    """
    return retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=EXPONENTIAL_MULTIPLIER,
            min=min_wait,
            max=max_wait,
        ),
        before_sleep=on_retry,
        reraise=True,
    )


def invoke_with_retry(
    chain, inputs: dict, *,
    max_attempts: int = DEFAULT_MAX_RETRIES,
    min_wait: float = EXPONENTIAL_MIN_WAIT,
    max_wait: float = EXPONENTIAL_MAX_WAIT,
):
    """Invoke an LLM chain with exponential-backoff retry on transient errors.

    Transient errors (rate limits, 503s, server disconnects) are retried up to
    ``max_attempts`` times with exponential backoff; any other exception
    propagates to the caller (agents should fail the task on hard errors).

    TRACE logging (when enabled): attempt/elapsed/delay on each retry, error
    type on failure, and token usage on success.
    """
    _log = logging.getLogger("mags_codedev").trace  # type: ignore[attr-defined]
    start = time.monotonic()
    attempt = [0]

    def _on_retry(retry_state):
        attempt[0] += 1
        elapsed = time.monotonic() - start
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        delay = retry_state.next_action.sleep if retry_state.next_action else 0
        _log(
            "Retry attempt %d failed after %.1fs: %s — backing off %.1fs",
            attempt[0],
            elapsed,
            type(exception).__name__ if exception else "unknown",
            delay,
        )

    @_make_retry(max_attempts, min_wait, max_wait, _on_retry)
    def _invoke():
        attempt[0] += 1
        return chain.invoke(inputs)

    result = _invoke()
    elapsed = time.monotonic() - start
    _log("LLM call succeeded in %.1fs (%d attempt(s)).", elapsed, attempt[0])
    _log_token_usage(result, _log)
    return result


async def ainvoke_with_retry(
    chain, inputs: dict, *,
    max_attempts: int = DEFAULT_MAX_RETRIES,
    min_wait: float = EXPONENTIAL_MIN_WAIT,
    max_wait: float = EXPONENTIAL_MAX_WAIT,
):
    """Async twin of :func:`invoke_with_retry` using ``chain.ainvoke``.

    Used by the reviewer node, which runs several reviewers concurrently and
    must retry transient errors independently per reviewer.
    """
    _log = logging.getLogger("mags_codedev").trace  # type: ignore[attr-defined]
    start = time.monotonic()
    attempt = [0]

    def _on_retry(retry_state):
        attempt[0] += 1
        elapsed = time.monotonic() - start
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        delay = retry_state.next_action.sleep if retry_state.next_action else 0
        _log(
            "Retry attempt %d failed after %.1fs: %s — backing off %.1fs",
            attempt[0],
            elapsed,
            type(exception).__name__ if exception else "unknown",
            delay,
        )

    @_make_retry(max_attempts, min_wait, max_wait, _on_retry)
    async def _ainvoke():
        attempt[0] += 1
        return await chain.ainvoke(inputs)

    result = await _ainvoke()
    elapsed = time.monotonic() - start
    _log("Async LLM call succeeded in %.1fs (%d attempt(s)).", elapsed, attempt[0])
    _log_token_usage(result, _log)
    return result


__all__ = [
    "invoke_with_retry",
    "ainvoke_with_retry",
]

