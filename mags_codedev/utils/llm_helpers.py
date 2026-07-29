"""Shared helpers for processing LLM responses."""


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
    start = content.find("```")
    if start != -1:
        rest = content[start + 3:]
        # Strip language tag if present
        tag_end = rest.find("\n")
        if tag_end != -1:
            rest = rest[tag_end + 1:]
        else:
            rest = rest.strip()
        end = rest.rfind("```")
        if end != -1:
            return rest[:end].strip()
    return content.strip()
