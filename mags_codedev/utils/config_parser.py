import os
import yaml
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None
try:
    from langchain_cohere import ChatCohere
except ImportError:
    ChatCohere = None

from mags_codedev.utils.db import TokenLoggingCallbackHandler

def ensure_config_structure(config: dict) -> dict:
    """Ensures the config dictionary has the modern structure, migrating if necessary."""
    models = config.get("models", {})
    
    # Check if migration is needed (i.e., flat structure is present)
    needs_migration = False
    flat_keys = ["coder", "tester", "log_checker", "reviewers"]
    for key in flat_keys:
        if key in models:
            needs_migration = True
            break
            
    if not needs_migration and ("build_workflow" in models or "interactive_commands" in models):
        return config

    # Create new structure
    new_models = {
        "interactive_commands": {
            "chat": models.get("chat", {"provider": "openai", "model": "gpt-4o"})
        },
        "build_workflow": {
            "coder": models.get("coder", {"provider": "openai", "model": "gpt-4o"}),
            "tester": models.get("tester", {"provider": "anthropic", "model": "claude-3-5-sonnet-20240620"}),
            "log_checker": models.get("log_checker", {"provider": "google", "model": "gemini-2.5-pro"}),
            "reviewers": models.get("reviewers", [])
        }
    }
    
    # Preserve any other keys in models
    for k, v in models.items():
        if k not in flat_keys and k != "chat":
            new_models[k] = v
            
    config["models"] = new_models
    return config

def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """Loads configuration from yaml and overrides with VS Code settings if present."""
    config = {}
    
    # Load base config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
            
    # Ensure structure is up to date
    config = ensure_config_structure(config)

    # Override with VS Code settings
    vscode_path = Path(".vscode/settings.json")
    if vscode_path.exists():
        try:
            with open(vscode_path, "r") as f:
                vscode_settings = json.load(f)
                # Parse "mags.api_keys.openai" -> config['api_keys']['openai']
                # Parse "mags-codedev.api_keys.openai" -> config['api_keys']['openai']
                for k, v in vscode_settings.items():
                    if k.startswith("mags-codedev.") or k.startswith("mags."):
                        parts = k.split(".")[1:]
                        d = config
                        for part in parts[:-1]:
                            d = d.setdefault(part, {})
                        d[parts[-1]] = v
        except Exception as e:
            logger.warning(f"Could not parse VS Code settings override from .vscode/settings.json: {e}")
            
    return config

def _create_llm_instance(model_config: dict, api_keys: dict, role: str = None):
    provider = model_config.get("provider", "openai").lower()
    model_name = model_config.get("model", "gpt-4o")
    
    llm = None
    if provider == "openai":
        llm = ChatOpenAI(api_key=api_keys.get("openai"), model=model_name)
    elif provider == "anthropic":
        llm = ChatAnthropic(api_key=api_keys.get("anthropic"), model=model_name)
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(google_api_key=api_keys.get("gemini"), model=model_name)
    elif provider == "mistral":
        if ChatMistralAI is None:
            raise ImportError("Mistral provider requires 'langchain-mistralai'. Please install it with `pip install langchain-mistralai`.")
        llm = ChatMistralAI(api_key=api_keys.get("mistral"), model=model_name)
    elif provider == "cohere":
        if ChatCohere is None:
            raise ImportError("Cohere provider requires 'langchain-cohere'. Please install it with `pip install langchain-cohere`.")
        llm = ChatCohere(api_key=api_keys.get("cohere"), model=model_name)
    elif provider == "custom_openai":
        # For local models like Ollama, vLLM, LM Studio
        llm = ChatOpenAI(
            api_key=model_config.get("api_key", "dummy"),
            base_url=model_config.get("base_url"),
            model=model_name
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
        
    if role and llm:
        # Attach the token logging callback automatically
        llm.callbacks = [TokenLoggingCallbackHandler(role=role, model_name=model_name)]
        
    return llm

def get_llm(role: str, config_path: Path = Path("config.yaml")):
    """Returns the instantiated LangChain model for a specific agent role."""
    config = load_config(config_path)
    models_config = config.get("models", {})
    
    # For backward compatibility, check new structure first, then old.
    build_config = models_config.get("build_workflow", {})
    interactive_config = models_config.get("interactive_commands", {})
    
    model_config = build_config.get(role) or interactive_config.get(role) or models_config.get(role)
    
    if not model_config:
        # Fallback to a default if the role is not defined anywhere
        model_config = {"provider": "openai", "model": "gpt-4o"}
        
    api_keys = config.get("api_keys", {})
    return _create_llm_instance(model_config, api_keys, role=role)

def get_reviewer_llms(config_path: Path = Path("config.yaml")) -> list:
    """Returns a list of instantiated LangChain models for parallel review."""
    config = load_config(config_path)
    models_config = config.get("models", {})
    build_config = models_config.get("build_workflow", {})
    
    # For backward compatibility, check new structure first, then old.
    reviewers_config = build_config.get("reviewers", []) or models_config.get("reviewers", [])
    api_keys = config.get("api_keys", {})
    # We assign a generic role name for reviewers, or we could index them
    return [_create_llm_instance(r, api_keys, role=f"reviewer_{r.get('model', 'unknown')}") for r in reviewers_config]