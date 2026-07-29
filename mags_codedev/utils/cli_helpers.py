"""CLI helper utilities: error formatting, editor."""

import os
import subprocess
from typing import Optional



def format_llm_error(error: Exception) -> str:
    """Format LLM error for display."""
    msg = str(error)
    if "rate limit" in msg.lower():
        return "Rate limit hit. Waiting before retry..."
    if "authentication" in msg.lower() or "api key" in msg.lower():
        return "Authentication error. Check API key configuration."
    if "context length" in msg.lower() or "max tokens" in msg.lower():
        return "Context length exceeded. Consider reducing prompt size."
    return f"LLM error: {msg}"


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
