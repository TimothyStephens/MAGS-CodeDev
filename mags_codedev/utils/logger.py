"""Logging setup for MAGs-CodeDev.

Provides a root logger with console (INFO) and rotating file handlers,
plus per-function child loggers keyed on a hash derived from the log filepath.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

logging.addLevelName(5, "TRACE")


logger = logging.getLogger("mags_codedev")


def setup_logger(base_dir: str = ".mags-codedev", log_level: str = "info") -> logging.Logger:
    """Configure root + console + rotating file handlers. Returns root logger."""
    os.makedirs(base_dir, exist_ok=True)

    level_map = {"info": logging.INFO, "debug": logging.DEBUG, "trace": 5}
    level = level_map.get(log_level.lower(), logging.INFO)

    # Root logger
    root = logging.getLogger("mags_codedev")
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler - always INFO
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler - level from config
    log_file = os.path.join(base_dir, "workflow.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=3)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return root


def get_function_logger(log_filepath: str | None) -> logging.Logger:
    """Return per-module logger. Falls back to root if log_filepath is None."""
    if log_filepath:
        log_hash = os.path.basename(log_filepath).replace(".log", "")
        return logging.getLogger(f"mags.func.{log_hash}")
    return logger


def is_debug_enabled(logger: logging.Logger) -> bool:
    """Check if DEBUG level is enabled — guards expensive formatting."""
    return logger.isEnabledFor(logging.DEBUG)
