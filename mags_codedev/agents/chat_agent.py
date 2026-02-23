from langchain.agents import create_agent
from langchain.agents.agent_executor import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
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

def start_chat_repl(config_path: Path):
    """Initializes and returns the Chat Agent executor for the CLI."""
    config = load_config(config_path)
    verbose_chat = config.get("settings", {}).get("verbose_chat", False)
    llm = get_llm(role="chat", config_path=config_path)

    tools = [read_file, write_file]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the MAGs-CodeDev interactive assistant. You help the user debug and refine their project. You can read and write files directly."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose_chat, memory=memory)