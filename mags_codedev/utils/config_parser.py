"""Configuration loader: YAML config, VS Code overrides, environment variables."""

import os
import yaml
import json
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from langchain_cohere import ChatCohere
except ImportError:
    ChatCohere = None

from mags_codedev.utils.db import TokenLoggingCallbackHandler
from mags_codedev.utils.logger import logger


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


def _resolve_env_api_keys(config: dict) -> None:
    """Resolve API keys from environment variables. Env vars take highest priority."""
    api_keys = config.setdefault("api_keys", {})
    env_map = {
        "OPENAI_API_KEY": "openai",
        "ANTHROPIC_API_KEY": "anthropic",
        "GOOGLE_API_KEY": "gemini",
        "MISTRAL_API_KEY": "mistral",
        "COHERE_API_KEY": "cohere",
        "OLLAMA_API_KEY": "ollama",
    }
    for env_var, key_name in env_map.items():
        val = os.environ.get(env_var)
        if val:
            api_keys[key_name] = val

def _resolve_env_models(config: dict) -> None:
    """Resolve models from environment variables.

    Global override: MAGS_MODEL, MAGS_PROVIDER, MAGS_BASE_URL
    Role-specific override: MAGS_MODEL_<ROLE> (e.g., MAGS_MODEL_CODER)
    """
    models_config = config.get("models", {})
    build_config = models_config.get("build_workflow", {})
    interactive_config = models_config.get("interactive_commands", {})

    # Global model override
    global_model = os.environ.get("MAGS_MODEL")
    global_provider = os.environ.get("MAGS_PROVIDER")
    global_base_url = os.environ.get("MAGS_BASE_URL")

    # Apply global override to all roles
    if global_model or global_provider:
        global_overrides = {}
        if global_model:
            global_overrides["model"] = global_model
        if global_provider:
            global_overrides["provider"] = global_provider
        if global_base_url:
            global_overrides["base_url"] = global_base_url

        # Override build_workflow roles
        for role_key in ("coder", "tester", "log_checker"):
            role_config = build_config.get(role_key, {})
            role_config.update(global_overrides)
            build_config[role_key] = role_config

        # Override reviewers
        reviewers = build_config.get("reviewers", [])
        if reviewers:
            for r in reviewers:
                r.update(global_overrides)

        # Override interactive chat
        chat_config = interactive_config.get("chat", {})
        chat_config.update(global_overrides)
        interactive_config["chat"] = chat_config

    # Role-specific overrides (higher priority than global)
    role_prefix = "MAGS_MODEL_"
    for env_key, env_val in os.environ.items():
        if env_key.startswith(role_prefix):
            role = env_key[len(role_prefix):].lower()
            role_config = build_config.get(role, {})
            if not role_config:
                role_config = interactive_config.get(role, {})
            if env_val:
                role_config["model"] = env_val
            build_config[role] = role_config


def load_config(config_path: Path = Path("config.yaml")) -> dict:
    """Loads configuration from yaml and overrides with VS Code settings if present."""
    config = {}

    # Load base config
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")

    # Ensure structure is up to date
    config = ensure_config_structure(config)

    # Override with VS Code settings
    vscode_path = Path(".vscode/settings.json")
    if vscode_path.exists():
        try:
            with open(vscode_path, "r") as f:
                raw = f.read()
            import re
            # Strip // comments (JSONC format)
            raw = re.sub(r'//.*?$', '', raw, flags=re.MULTILINE)
            # Strip trailing commas (JSONC format)
            raw = re.sub(r',(\s*[}\]])', r'\1', raw)
            vscode_settings = json.loads(raw)
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

    # Override with environment variables (highest priority)
    _resolve_env_api_keys(config)
    _resolve_env_models(config)

    return config


def get_base_dir(config_path: Path = Path("config.yaml")) -> str:
    """Returns the base artifact directory from config (default '.mags-codedev')."""
    config = load_config(config_path)
    return config.get("settings", {}).get("base_dir", ".mags-codedev")


def get_log_level(config_path: Path = Path("config.yaml")) -> str:
    """Returns the log level from config (default 'info')."""
    config = load_config(config_path)
    return config.get("settings", {}).get("log_level", "info")


# ------------------------------------------------------------------ #
#  Provider Registry
# ------------------------------------------------------------------ #
# Each entry is a factory ``(model_config, api_keys, model_name) -> llm``.
# To add a new provider: write a factory and register it here — no if/elif
# chain to edit. Factories reference the module-level client class (with an
# optional-import guard) so test patches like ``config_parser.ChatOllama =``
# continue to intercept the call.

def _make_openai(model_config: dict, api_keys: dict, model_name: str):
    """OpenAI ChatCompletion."""
    return ChatOpenAI(
        api_key=api_keys.get("openai") or os.environ.get("OPENAI_API_KEY"),
        model=model_name,
        base_url=model_config.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
    )


def _make_anthropic(model_config: dict, api_keys: dict, model_name: str):
    return ChatAnthropic(
        api_key=api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY"),
        model=model_name,
        base_url=model_config.get("base_url") or os.environ.get("ANTHROPIC_BASE_URL"),
    )


def _make_google(model_config: dict, api_keys: dict, model_name: str):
    return ChatGoogleGenerativeAI(
        google_api_key=api_keys.get("gemini") or os.environ.get("GOOGLE_API_KEY"),
        model=model_name,
    )


def _make_mistral(model_config: dict, api_keys: dict, model_name: str):
    if ChatMistralAI is None:
        raise ImportError(
            "Mistral provider requires 'langchain-mistralai'. "
            "Install it with `pip install langchain-mistralai`."
        )
    return ChatMistralAI(
        api_key=api_keys.get("mistral") or os.environ.get("MISTRAL_API_KEY"),
        model=model_name,
    )


def _make_cohere(model_config: dict, api_keys: dict, model_name: str):
    if ChatCohere is None:
        raise ImportError(
            "Cohere provider requires 'langchain-cohere'. "
            "Install it with `pip install langchain-cohere`."
        )
    return ChatCohere(
        api_key=api_keys.get("cohere") or os.environ.get("COHERE_API_KEY"),
        model=model_name,
    )


def _make_ollama(model_config: dict, api_keys: dict, model_name: str):
    if ChatOllama is None:
        raise ImportError(
            "Ollama provider requires 'langchain-ollama'. "
            "Install it with `pip install langchain-ollama`."
        )
    return ChatOllama(
        model=model_name,
        base_url=(
            model_config.get("base_url")
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ),
        num_ctx=model_config.get("num_ctx", 8192),
    )


def _make_local(model_config: dict, api_keys: dict, model_name: str):
    """Any OpenAI-compatible server (vLLM, LM Studio, Ollama /v1)."""
    base_url = (
        model_config.get("base_url")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OLLAMA_BASE_URL")
    )
    if not base_url:
        logger.warning(
            "Provider '%s' has no base_url configured. "
            "Set 'base_url' in config or OPENAI_BASE_URL/OLLAMA_BASE_URL env var.",
            model_config.get("provider", "local"),
        )
        base_url = "http://localhost:11434/v1"
        logger.info("Falling back to default: %s", base_url)
    api_key = (
        model_config.get("api_key")
        or api_keys.get("ollama")
        or api_keys.get("openai")
        or os.environ.get("OLLAMA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "dummy"
    )
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )


# Registry: provider name -> factory. Add a new provider by appending here.
_PROVIDERS = {
    "openai": _make_openai,
    "anthropic": _make_anthropic,
    "google": _make_google,
    "mistral": _make_mistral,
    "cohere": _make_cohere,
    "ollama": _make_ollama,
    "local": _make_local,
    "custom_openai": _make_local,  # legacy alias
}


def _create_llm_instance(
    model_config: dict, api_keys: dict,
    role: Optional[str] = None,
    base_dir: str = ".mags-codedev",
):
    """Instantiate a LangChain chat model from a provider config entry.

    The provider is looked up in :data:`_PROVIDERS`; adding a new provider means
    writing a factory and registering it — no ``if/elif`` chain to edit. A
    :class:`TokenLoggingCallbackHandler` is attached automatically when *role*
    is given so token usage is persisted to the DB.
    """
    provider = model_config.get("provider", "openai").lower()
    model_name = model_config.get("model", "gpt-4o")

    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDERS))}."
        )

    llm = factory(model_config, api_keys, model_name)

    if role and llm:
        # Attach the token logging callback (append, don't overwrite).
        if llm.callbacks is None:
            llm.callbacks = []
        llm.callbacks.append(
            TokenLoggingCallbackHandler(role=role, model_name=model_name, base_dir=base_dir)
        )

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
    base_dir = config.get("settings", {}).get("base_dir", ".mags-codedev")
    return _create_llm_instance(model_config, api_keys, role=role, base_dir=base_dir)


def get_reviewer_llms(config_path: Path = Path("config.yaml")) -> list:
    """Returns a list of instantiated LangChain models for parallel review."""
    config = load_config(config_path)
    models_config = config.get("models", {})
    build_config = models_config.get("build_workflow", {})

    # For backward compatibility, check new structure first, then old.
    reviewers_config = build_config.get("reviewers", []) or models_config.get("reviewers", [])
    api_keys = config.get("api_keys", {})
    base_dir = config.get("settings", {}).get("base_dir", ".mags-codedev")
    # We assign a generic role name for reviewers, or we could index them
    return [
        _create_llm_instance(
            r, api_keys,
            role=f"reviewer_{r.get('model', 'unknown')}",
            base_dir=base_dir,
        )
        for r in reviewers_config
    ]
