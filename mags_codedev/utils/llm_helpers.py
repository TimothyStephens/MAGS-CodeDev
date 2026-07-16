"""Shared helpers for processing LLM responses."""

import logging
import os
import re
from mags_codedev.utils.logger import logger


def resolve_logger(log_filepath: str | None) -> logging.Logger:
    """Return the function-specific logger for a given log file path.

    If *log_filepath* is ``None`` or missing, falls back to the global logger.
    """
    if log_filepath:
        log_hash = os.path.basename(log_filepath).replace(".log", "")
        return logging.getLogger(f"mags_codedev.func.{log_hash}")
    return logger


def extract_content(message_content) -> str:
    """Extract a plain-text string from LangChain message content.

    Handles ``str``, ``list`` (text blocks + tool calls), and arbitrary objects.
    """
    if message_content is None:
        return ""
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return "".join(
            block
            if isinstance(block, str)
            else block.get("text", "") if isinstance(block, dict)
            else getattr(block, "text", str(block))
            for block in message_content
        )
    return str(message_content)


def strip_markdown_code(content: str) -> str:
    """Remove surrounding `` ```python `` (or bare `` ``` ``) fences from *content*."""
    match = re.search(r"```(?:\w*)\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()
