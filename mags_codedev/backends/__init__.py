"""Language backends for MAGs-CodeDev.

Each language has its own backend that provides:
- Container environment setup (Dockerfile/Apptainer generation, env vars)
- Tool commands (test, lint, security scan)
- Agent prompt templates (coder, tester)
- Package initialization conventions (e.g. __init__.py)
"""

from pathlib import Path
from typing import TYPE_CHECKING

from mags_codedev.backends.language_backend import LanguageBackend
from mags_codedev.backends.python import PythonBackend

if TYPE_CHECKING:
    pass

# Registry: language identifier → backend instance
BACKEND_REGISTRY: dict[str, LanguageBackend] = {
    "python": PythonBackend(),
}

# Human-readable name map
LANGUAGE_DISPLAY: dict[str, str] = {
    "python": "Python",
}


def get_backend(config_path: Path) -> LanguageBackend:
    """Load the language backend from config or default to Python.

    Reads ``settings.language`` from the config YAML. If the key is
    absent or empty the default ("python") is used.
    """
    from mags_codedev.utils.config_parser import load_config

    config = load_config(config_path)
    lang = config.get("settings", {}).get("language", "python").lower()

    backend = BACKEND_REGISTRY.get(lang)
    if backend is None:
        available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported language: '{lang}'. Available: {available}"
        )
    return backend
