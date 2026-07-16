"""CLI helper utilities: content extraction, error formatting, editor."""

import os
import subprocess
import re
from typing import Optional


def extract_content(content: str) -> str:
    """Extract content from LLM response, stripping markdown code fences if present."""
    # If the content is wrapped in triple backticks, extract the inner content
    match = re.search(r"```(?:\w*)\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


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

    if editor == "code" and "--wait" not in editor:
        # VS Code needs --wait to block
        subprocess.run(["code", "--wait", path])
    else:
        subprocess.run(editor.split() + [path])

    return editor
