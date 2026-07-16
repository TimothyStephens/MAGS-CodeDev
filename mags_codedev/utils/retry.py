"""Shared retry helpers for LLM agent calls."""

import re
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
    return (
        re.search(r"\b503\b", msg)
        or "unavailable" in msg
        or "rate limit" in msg
        or re.search(r"\b429\b", msg)
        or "resource_exhausted" in msg
        or "server disconnected" in msg
    )


def invoke_with_retry(chain, inputs: dict):
    """Invoke an LLM chain with exponential-backoff retry on transient errors."""

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=EXPONENTIAL_MULTIPLIER, min=EXPONENTIAL_MIN_WAIT, max=EXPONENTIAL_MAX_WAIT),
        reraise=True,
    )
    def _invoke():
        return chain.invoke(inputs)

    return _invoke()
