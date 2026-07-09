"""Shared retry helpers for LLM agent calls."""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

try:
    from google.genai.errors import ServerError
except ImportError:
    ServerError = None  # type: ignore[assignment, misc]


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

    # Fallback: check error message string
    msg = str(exception).lower()
    return (
        "503" in msg
        or "unavailable" in msg
        or "rate limit" in msg
        or "429" in msg
        or "resource_exhausted" in msg
        or "server disconnected" in msg
    )


def invoke_with_retry(chain, inputs: dict):
    """Invoke an LLM chain with exponential-backoff retry on transient errors."""

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _invoke():
        return chain.invoke(inputs)

    return _invoke()
