import os
import yaml
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """Loads configuration from yaml and overrides with VS Code settings if present."""
    config = {}
    
    # Load base config
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

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
        except Exception:
            pass # Handle malformed JSON silently or log it
            
    return config

def _create_llm_instance(model_config: dict, api_keys: dict):
    provider = model_config.get("provider", "openai").lower()
    model_name = model_config.get("model", "gpt-4o")
    
    if provider == "openai":
        return ChatOpenAI(api_key=api_keys.get("openai"), model=model_name)
    elif provider == "anthropic":
        return ChatAnthropic(api_key=api_keys.get("anthropic"), model=model_name)
    elif provider == "google":
        return ChatGoogleGenerativeAI(google_api_key=api_keys.get("gemini"), model=model_name)
    elif provider == "custom_openai":
        # For local models like Ollama, vLLM, LM Studio
        return ChatOpenAI(
            api_key=model_config.get("api_key", "dummy"),
            base_url=model_config.get("base_url"),
            model=model_name
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_llm(role: str, config_path: Path = Path("config.yaml")):
    """Returns the instantiated LangChain model for a specific agent role."""
    config = load_config(config_path)
    model_config = config.get("models", {}).get(role, {"provider": "openai", "model": "gpt-4o"})
    api_keys = config.get("api_keys", {})
    return _create_llm_instance(model_config, api_keys)

def get_reviewer_llms(config_path: Path = Path("config.yaml")) -> list:
    """Returns a list of instantiated LangChain models for parallel review."""
    config = load_config(config_path)
    reviewers_config = config.get("models", {}).get("reviewers", [])
    api_keys = config.get("api_keys", {})
    return [_create_llm_instance(r, api_keys) for r in reviewers_config]