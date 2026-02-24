from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os
from pathlib import Path
from langchain_core.tools import tool
from mags_codedev.utils.config_parser import get_llm, load_config

@tool
def read_file(filepath: str) -> str:
    """Reads the contents of a file from the project directory."""
    project_dir = os.path.abspath(os.getcwd())
    target_path = os.path.abspath(os.path.join(project_dir, filepath))

    if not target_path.startswith(project_dir):
        return f"Error: Path traversal detected. Cannot read from '{filepath}'."

    try:
        with open(target_path, 'r') as f:
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

    if not target_path.startswith(project_dir):
        return f"Error: Path traversal detected. Cannot write to '{filepath}'."

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"

def start_chat_repl(config_path: Path, system_message_override: str = None):
    """Initializes and returns the Chat Agent graph for the CLI."""
    config = load_config(config_path)
    llm = get_llm(role="chat", config_path=config_path)

    tools = [read_file, write_file]
    
    default_system_message = "You are the MAGs-CodeDev interactive assistant. You help the user debug and refine their project. You can read and write files directly."
    
    system_message = system_message_override or default_system_message
    
    # MemorySaver replaces ConversationBufferMemory for persisting state between turns
    memory = MemorySaver()
    
    # create_react_agent builds a StateGraph pre-configured for tool calling
    return create_react_agent(llm, tools, state_modifier=system_message, checkpointer=memory)