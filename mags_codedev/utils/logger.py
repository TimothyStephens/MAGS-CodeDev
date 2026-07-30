"""Logging setup for MAGs-CodeDev.

Provides a root logger with console and rotating file handlers,
plus per-module child loggers that propagate upward
so all logs flow into workflow.log while also being written to
individual files in the logs/ subdirectory.

Logger hierarchy:

    mags_codedev                          ← root logger
    ├── console (StreamHandler)           ← level follows verbosity flag
    └── workflow.log (RotatingFileHandler)← 10 MB × 3 backups, level follows verbosity

    mags_codedev.func.<hash>              ← per-module logger (build)
    └── logs/<hash>.log (RotatingFileHandler) ← 5 MB × 5 backups, propagate=True
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Register TRACE level (5) and monkey-patch Logger.trace at runtime.
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def _trace_impl(self, message, *args, **kwargs) -> None:
    """Log a TRACE-level message (below DEBUG)."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)  # type: ignore[attr-defined]


logging.Logger.trace = _trace_impl  # type: ignore[attr-defined]
# Module-level root logger (lazy setup via setup_logger).
logger = logging.getLogger("mags_codedev")
# --------------- constants ---------------

_LOG_DIR = "logs"
_WORKFLOW_LOG = "workflow.log"
# No rotation — logs grow unbounded. Set _MAX_BYTES_MODULE > 0 for production use.
# backupCount=0 disables rotation entirely; set _BACKUP_COUNT_MODULE > 0 to retain
# rotated files (e.g., _BACKUP_COUNT_MODULE=5 keeps 5 rotated copies per module).
_MAX_BYTES_MODULE = 0
_BACKUP_COUNT_MODULE = 0
_MAX_BYTES_WORKFLOW = 0
_BACKUP_COUNT_WORKFLOW = 0

_LEVEL_MAP = {"info": logging.INFO, "debug": logging.DEBUG, "trace": 5}


# --------------- helpers ---------------

def _hash_from_filepath(filepath: str) -> str:
    """Extract the log hash from a file path string."""
    return Path(filepath).stem


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_FORMATTER = logging.Formatter(_LOG_FORMAT)


_VALID_LEVELS = {"info", "debug", "trace", "warning", "error", "critical"}


def _level_for(level: str) -> int:
    lower = level.lower()
    if lower not in _VALID_LEVELS:
        raise ValueError(f"Unknown log level: {level!r}. Use one of: {_VALID_LEVELS}")
    return _LEVEL_MAP.get(lower, logging.INFO)



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
    handler.setFormatter(_LOG_FORMATTER)
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
    (``mags_codedev.func.*``) propagate
    upward so their messages also reach ``workflow.log`` and the console.
    """
    os.makedirs(base_dir, exist_ok=True)

    level = _level_for(log_level)
    clvl = _level_for(console_level) if console_level else logging.INFO

    # Root logger
    root = logging.getLogger("mags_codedev")
    root.setLevel(level)

    # Clear existing handlers (avoid duplicates on repeated calls)
    root.handlers.clear()

    # Console handler — use console_level when provided
    console = logging.StreamHandler()
    console.setLevel(clvl)
    console.setFormatter(_LOG_FORMATTER)
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

    log_hash = _hash_from_filepath(log_filepath)
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


def get_response_logger(
    func_hash: str,
    *,
    base_dir: str = ".mags-codedev",
    log_level: str = "info",
) -> logging.Logger:
    """Return a non-propagating logger that writes ONLY to the module log file.

    Used for logging LLM response content to ``logs/<hash>.log`` without
    leaking into ``workflow.log`` or the console.

    Returns the root logger when *func_hash* is ``None`` or empty.
    """
    if not func_hash:
        return logger

    resp_name = f"mags_codedev.resp.{func_hash}"
    resp = logging.getLogger(resp_name)
    level = _level_for(log_level)

    # Always replace handlers — prevents stale path on re-runs
    for h in list(resp.handlers):
        h.close()
        resp.removeHandler(h)
    resp.setLevel(level)
    resp.propagate = False  # do NOT flow into workflow.log or console

    # Write to the same per-module log file
    log_path = os.path.join(base_dir, _LOG_DIR, f"{func_hash}.log")
    resp.addHandler(_make_handler(log_path, level))

    return resp




def get_dual_loggers(
    log_filepath: str | None,
    *,
    base_dir: str = ".mags-codedev",
    log_level: str = "info",
) -> tuple[logging.Logger, logging.Logger]:
    """Return (func_logger, resp_logger) pair.

    Handles the common pattern of creating both a propagating module logger
    and a non-propagating response logger with identical hash extraction.
    """
    func_logger = get_function_logger(log_filepath, base_dir=base_dir, log_level=log_level)
    hash = _hash_from_filepath(log_filepath) if log_filepath else ""
    resp_logger = get_response_logger(hash, base_dir=base_dir, log_level=log_level)
    return func_logger, resp_logger


from contextlib import contextmanager


@contextmanager
def suppress_console_logging(level: int = logging.WARNING):
    """Temporarily raise the console handler level so it doesn't garble the TUI.

    During a Rich ``Live`` render, INFO/DEBUG log lines printed to the console
    StreamHandler collide with the live-updated tree, producing garbled output.
    This context manager raises the console handler to ``level`` (default
    WARNING) for its duration, then restores it. File handlers are unaffected.
    """
    root = logging.getLogger("mags_codedev")
    original_levels: dict[int, int] = {}
    for i, handler in enumerate(root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            original_levels[i] = handler.level
            handler.setLevel(level)
    try:
        yield
    finally:
        for i, handler in enumerate(root.handlers):
            if i in original_levels:
                handler.setLevel(original_levels[i])


__all__ = [
    "setup_logger",
    "get_function_logger",
    "get_response_logger",
    "get_dual_loggers",
    "suppress_console_logging",
]


