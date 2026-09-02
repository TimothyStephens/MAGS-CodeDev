"""Chat agent: interactive REPL with file read/write tools for debugging."""

import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from mags_codedev.utils.config_parser import get_llm, load_config


@tool
def read_file(filepath: str) -> str:
    """Reads the contents of a file from the project directory."""
    project_dir = os.path.abspath(os.getcwd())
    target_path = os.path.abspath(os.path.join(project_dir, filepath))

    if target_path != project_dir and not target_path.startswith(project_dir + os.sep):
        return f"Error: Path traversal detected. Cannot read from '{filepath}'."

    try:
        with open(target_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at '{filepath}'."
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(filepath: str, content: str) -> str:
    """Writes content to a file in the project directory. Path traversal is not allowed."""
    project_dir = os.path.abspath(os.getcwd())
    target_path = os.path.abspath(os.path.join(project_dir, filepath))

    if target_path != project_dir and not target_path.startswith(project_dir + os.sep):
        return f"Error: Path traversal detected. Cannot write to '{filepath}'."

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


def start_chat_repl(
    config_path: Path,
    system_message_override: Optional[str] = None,
    command_name: str = "chat",
) -> Any:
    """Initializes and returns the Chat Agent graph for the CLI."""
    _ = load_config(config_path)

    llm = get_llm(role="chat", config_path=config_path)

    tools = [read_file, write_file]

    default_system_message = (
        "You are the MAGs-CodeDev interactive assistant. "
        "You help the user debug and refine their project. "
        "You can read and write files directly."
    )

    system_message = system_message_override or default_system_message

    # MemorySaver replaces ConversationBufferMemory for persisting state
    memory = MemorySaver()

    # create_react_agent builds a StateGraph pre-configured for tool calling
    try:
        return create_react_agent(
            llm, tools, state_modifier=system_message, checkpointer=memory,
        )
    except TypeError as e:
        # Fallback chain for different versions of langgraph
        if "state_modifier" in str(e):
            try:
                return create_react_agent(
                    llm, tools,
                    messages_modifier=system_message, checkpointer=memory,
                )
            except TypeError as e2:
                if "messages_modifier" in str(e2):
                    return create_react_agent(
                        llm, tools,
                        prompt=SystemMessage(content=system_message),
                        checkpointer=memory,
                    )
                raise e2
        raise e
