"""JSONL status reporter for machine-readable build output.

When ``mags-codedev build --json`` is used, this module emits one JSON object
per line (JSONL) to stdout. Each line is a self-contained event that a
consumer (the OMP extension, a CI pipeline, or ``jq``) can parse incrementally.

Event types::

    build_start    — emitted once before the first wave
    module_start   — a module begins processing
    module_step    — a graph node is executing (coder, tester, run_tests, ...)
    module_tokens  — token usage for a completed module
    module_end     — a module finished (success or failure)
    wave_end       — a DAG wave completed (which modules succeeded/failed)
    build_end      — final summary

All events share a ``ts`` (ISO-8601 UTC timestamp) and ``event`` field.

Usage::

    reporter = JsonStatusReporter(enabled=True)
    reporter.build_start(manifest, total, built)
    reporter.module_start("src/foo.py", "a1b2...", session=1)
    # ...
    reporter.build_end(total, succeeded, failed, blocked, tokens_in, tokens_out)

When ``enabled=False``, every method is a no-op — callers don't need to
guard their calls with ``if json_output:`` checks.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Optional


def _now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class JsonStatusReporter:
    """Emit JSONL status events to stdout when enabled.

    Call ``emit()`` directly for custom events, or use the typed helpers
    (``build_start``, ``module_step``, etc.) for the standard event schema.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def emit(self, event: dict[str, Any]) -> None:
        """Write one JSONL line to stdout (flushed immediately)."""
        if not self.enabled:
            return
        payload = {"ts": _now(), **event}
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    # ── Build lifecycle ────────────────────────────────────────────

    def build_start(
        self, manifest: str, total_modules: int, already_built: int
    ) -> None:
        self.emit({
            "event": "build_start",
            "manifest": manifest,
            "total_modules": total_modules,
            "already_built": already_built,
        })

    def build_end(
        self,
        total_modules: int,
        succeeded: int,
        failed: int,
        blocked: int,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        self.emit({
            "event": "build_end",
            "total_modules": total_modules,
            "succeeded": succeeded,
            "failed": failed,
            "blocked": blocked,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        })

    # ── Module lifecycle ───────────────────────────────────────────

    def module_start(
        self, location: str, hash: str, session: int = 1
    ) -> None:
        self.emit({
            "event": "module_start",
            "location": location,
            "hash": hash,
            "session": session,
        })

    def module_step(
        self, location: str, step: str, iteration: int = 0
    ) -> None:
        self.emit({
            "event": "module_step",
            "location": location,
            "step": step,
            "iteration": iteration,
        })

    def module_tokens(
        self, location: str, tokens_in: int, tokens_out: int
    ) -> None:
        self.emit({
            "event": "module_tokens",
            "location": location,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        })

    def module_end(
        self,
        location: str,
        status: str,
        iterations: int,
        log_file: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        self.emit({
            "event": "module_end",
            "location": location,
            "status": status,
            "iterations": iterations,
            "log_file": log_file,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        })

    # ── Wave lifecycle ─────────────────────────────────────────────

    def wave_end(
        self,
        succeeded: list[str],
        failed: list[str],
    ) -> None:
        self.emit({
            "event": "wave_end",
            "succeeded": succeeded,
            "failed": failed,
        })


__all__ = ["JsonStatusReporter"]
