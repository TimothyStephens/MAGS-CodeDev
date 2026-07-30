"""CLI helper utilities: error formatting, editor."""

import os
import subprocess
from typing import Optional



def format_llm_error(error: Exception, role: str = "", provider: str = "") -> str:
    """Format LLM error for display with actionable guidance.

    Maps common SDK/library errors to concise, user-actionable messages.
    When *role* and *provider* are known, includes the relevant config key
    and environment variable name so the user knows exactly what to fix.
    """
    msg = str(error)
    msg_lower = msg.lower()

    # --- credential / auth failures ---
    if any(
        kw in msg_lower
        for kw in (
            "missing credentials", "missing api key", "api key",
            "authentication", "unauthorized", "invalid api key",
        )
    ):
        hint = _auth_hint(role, provider)
        return f"Missing or invalid API key{hint}"

    # --- placeholder / obviously-fake keys ---
    if any(marker in msg for marker in ("sk-...", "sk-ant-...", "/looks")):
        hint = _auth_hint(role, provider)
        return f"Placeholder API key detected in config{hint}"

    # --- rate limiting ---
    if "rate limit" in msg_lower:
        return "Rate limit hit. Wait before retrying."

    # --- context / token limits ---
    if any(kw in msg_lower for kw in ("context length", "max tokens", "token limit")):
        return "Context length exceeded. Reduce prompt size or use a model with a larger context window."

    # --- network / connectivity ---
    if any(kw in msg_lower for kw in ("connection", "timeout", "refused", "unreachable", "network")):
        return f"Could not reach the model endpoint. Check your network and base_url."

    # --- pydantic / validation errors (often missing required fields) ---
    if "validation error" in msg_lower:
        # Extract the field name if possible
        field = ""
        for word in msg.split():
            if word.isidentifier() and "api_key" in word.replace("-", "_"):
                field = word
                break
        hint = _auth_hint(role, provider)
        if field:
            return f"Configuration validation failed for '{field}'{hint}"
        return f"Configuration validation failed{hint}"

    # --- fallback: show the raw message ---
    return f"LLM error: {msg}"


_PROVIDER_KEY_MAP = {
    "openai": ("openai", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
    "google": ("gemini", "GOOGLE_API_KEY"),
    "mistral": ("mistral", "MISTRAL_API_KEY"),
    "cohere": ("cohere", "COHERE_API_KEY"),
    "ollama": ("ollama", "OLLAMA_API_KEY"),
    "local": ("openai", "OPENAI_API_KEY"),
    "custom_openai": ("openai", "OPENAI_API_KEY"),
}


def _auth_hint(role: str, provider: str) -> str:
    """Build a config/env-var hint string for credential errors."""
    cfg_key, env_var = _PROVIDER_KEY_MAP.get(
        provider.lower(), (None, None)
    )
    parts = []
    if role:
        parts.append(f" for {role}")
    hints = []
    if cfg_key:
        hints.append(f"config.api_keys.{cfg_key}")
    if env_var:
        hints.append(f"${env_var}")
    if hints:
        parts.append(f" (set {' or '.join(hints)})")
    return "".join(parts) if parts else ""


def _open_in_editor(path: str) -> Optional[str]:
    """Open file in the user's preferred editor. Returns editor path or None."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        # Probe for common editors
        for candidate in ["vim", "nano", "code"]:
            try:
                subprocess.run(
                    ["which", candidate],
                    capture_output=True,
                    check=True,
                )
                editor = candidate
                break
            except subprocess.CalledProcessError:
                continue

    if not editor:
        return None

    if editor == "code":
        # VS Code needs --wait to block
        subprocess.run(["code", "--wait", path])
    else:
        subprocess.run(editor.split() + [path])

    return editor
