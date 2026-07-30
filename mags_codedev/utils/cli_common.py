"""Shared CLI constants and helpers used by command modules."""

import shutil
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

console = Console()

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

_DEFAULT_BASE_DIR = ".mags-codedev"
_USER_DEFAULT_CONFIG = Path.home() / ".omp" / "agent" / "mags-codedev.yaml"


def _find_package_config_path() -> Optional[Path]:
    """Find the config template shipped with the package."""
    # We need to locate the package directory dynamically
    try:
        import mags_codedev
        here = Path(mags_codedev.__file__).resolve().parent
    except Exception:
        here = Path(__file__).resolve().parent.parent
    for candidate in [here / "config.template.yaml", here.parent / "config.template.yaml"]:
        if candidate.exists():
            return candidate
    return None


_PACKAGE_CONFIG_PATH = _find_package_config_path()
_CONFIG_HELP_TEXT = (
    f"Path to the configuration YAML file. Default: '{_DEFAULT_BASE_DIR}/config.yaml'. "
    "On first init, copied from '~/.omp/agent/mags-codedev.yaml' (user default) or "
    f"the package template ('{_PACKAGE_CONFIG_PATH}')."
) if _PACKAGE_CONFIG_PATH else (
    f"Path to the configuration YAML file. Default: '{_DEFAULT_BASE_DIR}/config.yaml'."
)


# -------------------------------------------------------------------
# Config bootstrap
# -------------------------------------------------------------------

def _bootstrap_config(base_dir: str) -> tuple[Path, bool]:
    """Copy user default or template to <base_dir>/config.yaml.

    Returns (config_path, was_created).
    """
    import logging
    logger = logging.getLogger("mags")

    config_path = Path(base_dir) / "config.yaml"
    if config_path.exists():
        logger.debug(f"Config already exists: {config_path}")
        return config_path, False
    config_path.parent.mkdir(parents=True, exist_ok=True)

    package_template = _find_package_config_path()

    for source, label in [
        (_USER_DEFAULT_CONFIG, "user default (~/.omp/agent/mags-codedev.yaml)"),
        (package_template, "package template"),
    ]:
        if source and source.exists():
            try:
                shutil.copy2(source, config_path)
                console.print(
                    f"[green]Created {config_path} from {label}[/green]"
                )
                logger.info(f"Created config from {label}: {source} -> {config_path}")
                return config_path, True
            except OSError as e:
                logger.warning(f"Failed to copy {source}: {e}")
                continue

    logger.warning("No config source found; writing minimal stub")
    console.print("[yellow]No config template found; writing minimal stub.[/yellow]")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("api_keys: {}\nmodels: {}\nsettings: {}\n")
    return config_path, True


# -------------------------------------------------------------------
# Config path resolution
# -------------------------------------------------------------------

def find_default_config_path() -> Path:
    """Find the config file.

    Lookup order:
    1. <base_dir>/config.yaml  (e.g., .mags-codedev/config.yaml)
    2. config.yaml  (legacy cwd fallback)
    """
    import logging
    logger = logging.getLogger("mags")

    base_config = Path(_DEFAULT_BASE_DIR) / "config.yaml"
    if base_config.exists():
        logger.debug(f"Found config at {base_config}")
        return base_config

    cwd_config = Path("config.yaml")
    if cwd_config.exists():
        logger.warning(
            "Using legacy config location '%s'. "
            "Consider moving it to '%s'.", cwd_config, base_config
        )
        console.print(
            f"[yellow]Using legacy config location '{cwd_config}'. "
            f"Consider moving it to '{base_config}'.[/yellow]"
        )
        return cwd_config

    return base_config


def resolve_base_dir(config_path: Path) -> str:
    """Resolve the base artifact directory from config."""
    from mags_codedev.utils.config_parser import get_base_dir
    return get_base_dir(config_path)


# -------------------------------------------------------------------
# Config validation
# -------------------------------------------------------------------

def validate_config_connections(config_path: Path) -> bool:
    """Verifies that the API keys and Models defined in config.yaml are valid.

    Separates **construction errors** (bad config, missing keys) from
    **invoke errors** (network, rate limits) so the user gets a precise
    diagnosis instead of a raw SDK traceback.
    """
    from langchain_core.messages import HumanMessage
    from mags_codedev.utils.config_parser import get_llm, get_reviewer_llms
    from mags_codedev.utils.cli_helpers import format_llm_error

    import logging
    logger = logging.getLogger("mags")

    console.print(Panel("[bold cyan]Validating LLM Connections...[/bold cyan]"))
    logger.debug("Starting validation of LLM connections.")

    failures: list[tuple[str, str]] = []  # (label, error_message)
    roles = ["coder", "tester", "log_checker"]

    for role in roles:
        label = role
        provider = _role_provider(role, config_path)
        try:
            llm = get_llm(role, config_path)
            model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
            console.print(f"Checking [bold]{label}[/bold] ({model_name})...", end=" ")
            llm.invoke([HumanMessage(content="Test")])
            console.print("[green]OK[/green]")
            logger.debug(f"Connection verified for {label} ({model_name}).")
        except Exception as e:
            console.print("[red]FAILED[/red]")
            err_msg = format_llm_error(e, role=role, provider=provider)
            console.print(f"  [red]Error: {err_msg}[/red]")
            logger.warning(f"Connection failed for {label}: {err_msg}")
            failures.append((label, err_msg))

    try:
        reviewers = get_reviewer_llms(config_path)
        for i, llm in enumerate(reviewers):
            label = f"Reviewer {i + 1}"
            provider = _infer_provider(llm)
            try:
                model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
                console.print(f"Checking [bold]{label}[/bold] ({model_name})...", end=" ")
                llm.invoke([HumanMessage(content="Test")])
                console.print("[green]OK[/green]")
                logger.debug(f"Connection verified for {label} ({model_name}).")
            except Exception as e:
                console.print("[red]FAILED[/red]")
                err_msg = format_llm_error(e, role="reviewer", provider=provider)
                console.print(f"  [red]Error: {err_msg}[/red]")
                logger.warning(f"Connection failed for {label}: {err_msg}")
                failures.append((label, err_msg))
    except Exception as e:
        err_msg = format_llm_error(e)
        console.print(f"[red]Error loading reviewers: {err_msg}[/red]")
        logger.warning(f"Error loading reviewers: {err_msg}")
        failures.append(("reviewers", err_msg))

    if failures:
        _print_validation_summary(failures)

    return not failures


_KNOWN_LLM_CLASSES = {
    "ChatOpenAI": "openai",
    "ChatAnthropic": "anthropic",
    "ChatGoogleGenerativeAI": "google",
    "ChatOllama": "ollama",
    "ChatMistralAI": "mistral",
    "ChatCohere": "cohere",
}


def _infer_provider(llm) -> str:
    """Infer the provider name from an instantiated LLM object."""
    cls_name = type(llm).__name__
    return _KNOWN_LLM_CLASSES.get(cls_name, cls_name.lower().replace("chat", ""))


def _role_provider(role: str, config_path: Path) -> str:
    """Read the provider for a role from the config (before LLM construction)."""
    try:
        from mags_codedev.utils.config_parser import load_config
        config = load_config(config_path)
        models_config = config.get("models", {})
        build_config = models_config.get("build_workflow", {})
        interactive_config = models_config.get("interactive_commands", {})
        model_config = (
            build_config.get(role)
            or interactive_config.get(role)
            or models_config.get(role)
        )
        if model_config:
            return model_config.get("provider", "")
    except Exception:
        pass
    return ""


def _print_validation_summary(failures: list[tuple[str, str]]) -> None:
    """Print a compact summary of all validation failures."""
    lines = ["\n[bold red]Validation failed[/bold red] for the following roles:"]
    for label, err in failures:
        lines.append(f"  \u2022 [bold]{label}[/bold]: {err}")
    lines.append("")
    lines.append("Fix the issues above, or use [cyan]--skip-validation[/cyan] to bypass checks.")
    for line in lines:
        console.print(line)
