"""Logging setup for MAGs-CodeDev.

Provides a root logger with console and rotating file handlers,
plus per-module and per-session child loggers that propagate upward
so all logs flow into workflow.log while also being written to
individual files in the logs/ subdirectory.

Logger hierarchy:

    mags_codedev                          ← root logger
    ├── console (StreamHandler)           ← level follows verbosity flag
    └── workflow.log (RotatingFileHandler)← 10 MB × 3 backups, level follows verbosity

    mags_codedev.func.<hash>              ← per-module logger (build)
    └── logs/<hash>.log (RotatingFileHandler) ← 5 MB × 5 backups, propagate=True

    mags_codedev.session.<id>             ← per-session logger (chat, debug, etc.)
    └── logs/<id>.log (RotatingFileHandler) ← 5 MB × 5 backups, propagate=True
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

logging.addLevelName(5, "TRACE")

# Module-level root logger (lazy setup via setup_logger).
logger = logging.getLogger("mags_codedev")

# --------------- constants ---------------

_LOG_DIR = "logs"
_WORKFLOW_LOG = "workflow.log"
# No rotation — logs grow unbounded (backupCount=0 disables rotation entirely)
_MAX_BYTES_MODULE = 0
_BACKUP_COUNT_MODULE = 0
_MAX_BYTES_WORKFLOW = 0
_BACKUP_COUNT_WORKFLOW = 0

_LEVEL_MAP = {"info": logging.INFO, "debug": logging.DEBUG, "trace": 5}


# --------------- helpers ---------------

def _level_for(level: str) -> int:
    return _LEVEL_MAP.get(level.lower(), logging.INFO)


def _fmt() -> logging.Formatter:
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _make_handler(
    filepath: str,
    level: int,
    *,
    max_bytes: int = _MAX_BYTES_WORKFLOW,
    backup_count: int = _BACKUP_COUNT_WORKFLOW,
) -> RotatingFileHandler:
    """Return a RotatingFileHandler with append mode and rotation."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    handler = RotatingFileHandler(
        filepath,
        mode="a",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setLevel(level)
    handler.setFormatter(_fmt())
    return handler


# --------------- public API ---------------


def setup_logger(
    base_dir: str = ".mags-codedev",
    log_level: str = "info",
    console_level: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger with console + rotating file handlers.

    Parameters
    ----------
    base_dir :
        Directory for log files (default ``.mags-codedev``).
    log_level :
        Level for the file handler (``info`` / ``debug`` / ``trace``).
    console_level :
        Level for the console handler.  Defaults to *log_level* when ``None``.

    Returns the root ``mags_codedev`` logger.  All child loggers
    (``mags_codedev.func.*``, ``mags_codedev.session.*``) propagate
    upward so their messages also reach ``workflow.log`` and the console.
    """
    os.makedirs(base_dir, exist_ok=True)

    level = _level_for(log_level)
    clvl = _level_for(console_level) if console_level else level

    # Root logger
    root = logging.getLogger("mags_codedev")
    root.setLevel(level)

    # Clear existing handlers (avoid duplicates on repeated calls)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(clvl)
    console.setFormatter(_fmt())
    root.addHandler(console)

    # workflow.log — rotating, append
    wf_path = os.path.join(base_dir, _WORKFLOW_LOG)
    root.addHandler(
        _make_handler(
            wf_path,
            level,
            max_bytes=_MAX_BYTES_WORKFLOW,
            backup_count=_BACKUP_COUNT_WORKFLOW,
        )
    )

    return root


def get_function_logger(
    log_filepath: str | None,
    *,
    base_dir: str = ".mags-codedev",
    log_level: str = "info",
) -> logging.Logger:
    """Return a per-module logger.

    When *log_filepath* is provided a child logger
    ``mags_codedev.func.<hash>`` is created with a RotatingFileHandler
    writing to ``<base_dir>/logs/<hash>.log``.  The logger propagates
    upward so all messages also reach ``workflow.log``.

    When *log_filepath* is ``None`` the root logger is returned.

    Handlers are always replaced (never reused) so a changed *base_dir*
    or re-run produces a fresh handler.
    """
    if log_filepath is None:
        return logger

    log_hash = os.path.basename(log_filepath).replace(".log", "")
    child_name = f"mags_codedev.func.{log_hash}"
    child = logging.getLogger(child_name)
    level = _level_for(log_level)

    # Always replace handlers — prevents stale path on re-runs
    for h in list(child.handlers):
        h.close()
        child.removeHandler(h)
    child.setLevel(level)
    child.propagate = True  # flow into workflow.log

    # Per-module log file in logs/ subdirectory
    log_path = os.path.join(base_dir, _LOG_DIR, f"{log_hash}.log")
    child.addHandler(_make_handler(log_path, level))

    return child


def get_session_logger(
    session_id: str,
    *,
    base_dir: str = ".mags-codedev",
    log_level: str = "info",
) -> logging.Logger:
    """Return a per-session logger (chat, debug, init, etc.).

    Creates ``mags_codedev.session.<session_id>`` with a
    RotatingFileHandler writing to ``<base_dir>/logs/<session_id>.log``.
    Propagates upward so messages also reach ``workflow.log``.
    """
    child_name = f"mags_codedev.session.{session_id}"
    child = logging.getLogger(child_name)
    level = _level_for(log_level)

    # Always replace handlers
    for h in list(child.handlers):
        h.close()
        child.removeHandler(h)
    child.setLevel(level)
    child.propagate = True  # flow into workflow.log

    log_path = os.path.join(base_dir, _LOG_DIR, f"{session_id}.log")
    child.addHandler(_make_handler(log_path, level))

    return child


def is_debug_enabled(logger: logging.Logger) -> bool:
    """Check if DEBUG level is enabled — guards expensive formatting."""
    return logger.isEnabledFor(logging.DEBUG)
