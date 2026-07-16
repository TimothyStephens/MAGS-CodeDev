"""SQLite-backed persistence for MAGs-CodeDev.

Tracks completed functions, token usage, module iteration counts,
module artifacts (code/tests), and per-iteration logs.

All functions accept `base_dir: str = ".mags-codedev"` as their first
parameter; the database lives at `os.path.join(base_dir, "cache.db")`.
"""

import sqlite3
import hashlib
import json
import os
from typing import Dict, List
from mags_codedev.utils.logger import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _db_path(base_dir: str) -> str:
    """Return the absolute path to the SQLite database."""
    return os.path.join(base_dir, "cache.db")


# ------------------------------------------------------------------ #
#  Schema
# ------------------------------------------------------------------ #

def init_db(base_dir: str = ".mags-codedev") -> None:
    """Create all tables in the database, creating the directory if needed."""
    os.makedirs(base_dir, exist_ok=True)
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS completed_functions (
                func_hash TEXT PRIMARY KEY,
                function_name TEXT,
                status TEXT DEFAULT 'success',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                model TEXT,
                in_tokens INTEGER,
                out_tokens INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS module_iterations (
                location TEXT PRIMARY KEY,
                total_iterations INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS module_artifacts (
                location TEXT PRIMARY KEY,
                code TEXT,
                tests TEXT,
                spec_hash TEXT,
                code_hash TEXT,
                previous_code_hash TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iteration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT,
                spec_hash TEXT,
                iteration INTEGER,
                node TEXT,
                action TEXT,
                error_summary TEXT,
                tokens_in INTEGER,
                tokens_out INTEGER,
                duration_ms INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


# ------------------------------------------------------------------ #
#  Completed Functions
# ------------------------------------------------------------------ #

def hash_spec(spec: dict) -> str:
    """Return a SHA-256 hex digest of the spec's location field only.

    Using location as the key means manifest description/dependency
    edits reuse existing worktrees, logs, and artifacts instead of
    spawning fresh ones.
    """
    return hashlib.sha256(spec.get("location", "").encode()).hexdigest()


def is_function_built(spec: dict, base_dir: str = ".mags-codedev") -> bool:
    """Check whether the function described by *spec* was already built."""
    spec_hash = hash_spec(spec)
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM completed_functions WHERE func_hash = ?", (spec_hash,)
        )
        return cursor.fetchone() is not None


def mark_function_built(
    function_name: str,
    spec: dict,
    status: str = "success",
    base_dir: str = ".mags-codedev",
) -> None:
    """Mark a function as built (or rebuilt) with the given status."""
    spec_hash = hash_spec(spec)
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO completed_functions (func_hash, function_name, status) VALUES (?, ?, ?)",
            (spec_hash, function_name, status),
        )
        conn.commit()


def get_completed_status(
    function_name: str,
    base_dir: str = ".mags-codedev",
) -> str | None:
    """Return the status of a previously completed function by name, or None."""
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM completed_functions WHERE function_name = ?",
            (function_name,),
        )
        row = cursor.fetchone()
        return row[0] if row else None


# ------------------------------------------------------------------ #
#  Module Iterations
# ------------------------------------------------------------------ #

def add_iterations_to_module(
    location: str,
    count: int,
    base_dir: str = ".mags-codedev",
) -> None:
    """Add *count* iterations to the module at *location*."""
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO module_iterations (location, total_iterations) VALUES (?, 0)",
            (location,),
        )
        cursor.execute(
            "UPDATE module_iterations SET total_iterations = total_iterations + ? WHERE location = ?",
            (count, location),
        )
        conn.commit()


def get_total_iterations(
    location: str,
    base_dir: str = ".mags-codedev",
) -> int:
    """Return the total number of iterations recorded for *location*."""
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT total_iterations FROM module_iterations WHERE location = ?",
            (location,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0


# ------------------------------------------------------------------ #
#  Token Usage
# ------------------------------------------------------------------ #

def log_token_usage(
    role: str,
    model: str,
    in_tokens: int,
    out_tokens: int,
    base_dir: str = ".mags-codedev",
) -> None:
    """Record a single LLM token-usage observation."""
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO token_usage (role, model, in_tokens, out_tokens) VALUES (?, ?, ?, ?)",
            (role, model, in_tokens, out_tokens),
        )
        conn.commit()


def get_token_summary(base_dir: str = ".mags-codedev") -> tuple:
    """Return aggregated token-usage statistics.

    Returns:
        (per_role_summary, per_model_summary, total)
        - per_role_summary: list of (role, model, sum_in, sum_out)
        - per_model_summary: list of (model, sum_in, sum_out)
        - total: (total_in, total_out)
    """
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()

        # Per-role breakdown
        cursor.execute("""
            SELECT role, model, SUM(in_tokens), SUM(out_tokens)
            FROM token_usage
            GROUP BY role, model
            ORDER BY role
        """)
        per_role_summary = cursor.fetchall()

        # Per-model breakdown
        cursor.execute("""
            SELECT model, SUM(in_tokens), SUM(out_tokens)
            FROM token_usage
            GROUP BY model
            ORDER BY model
        """)
        per_model_summary = cursor.fetchall()

        # Grand total
        cursor.execute(
            "SELECT COALESCE(SUM(in_tokens), 0), COALESCE(SUM(out_tokens), 0) FROM token_usage"
        )
        total = cursor.fetchone()

    return per_role_summary, per_model_summary, total or (0, 0)


# ------------------------------------------------------------------ #
#  Module Artifacts (code + tests persistence)
# ------------------------------------------------------------------ #

def save_artifact(
    location: str,
    code: str,
    tests: str,
    spec_hash: str,
    base_dir: str = ".mags-codedev",
) -> None:
    """Persist (or upsert) code and test artifacts for *location*.

    Computes a SHA-256 of the code and stores the previous code hash
    for convergence tracking.
    """
    code_hash = hashlib.sha256(code.encode()).hexdigest()

    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()

        # Capture previous_code_hash before overwriting
        cursor.execute(
            "SELECT code_hash FROM module_artifacts WHERE location = ?",
            (location,),
        )
        row = cursor.fetchone()
        previous_code_hash = row[0] if row else None

        cursor.execute("""
            INSERT INTO module_artifacts
                (location, code, tests, spec_hash, code_hash, previous_code_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(location) DO UPDATE SET
                code = excluded.code,
                tests = excluded.tests,
                spec_hash = excluded.spec_hash,
                code_hash = excluded.code_hash,
                previous_code_hash = excluded.previous_code_hash,
                updated_at = CURRENT_TIMESTAMP
        """, (location, code, tests, spec_hash, code_hash, previous_code_hash))

        conn.commit()


def load_artifact(
    location: str,
    base_dir: str = ".mags-codedev",
) -> dict | None:
    """Return persisted artifacts for *location*, or None if not found.

    Returns a dict with keys: code, tests, spec_hash, code_hash.
    """
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT code, tests, spec_hash, code_hash FROM module_artifacts WHERE location = ?",
            (location,),
        )
        row = cursor.fetchone()

    if row is None:
        return None

    return {
        "code": row[0],
        "tests": row[1],
        "spec_hash": row[2],
        "code_hash": row[3],
    }


def load_dependency_codes(
    dependency_locations: List[str],
    worktree_path: str,
    base_dir: str = ".mags-codedev",
) -> Dict[str, str]:
    """Load source code for each dependency location.

    Tries the worktree file first, falls back to the artifact DB.
    Returns a dict mapping location -> source code (only found deps).
    """
    result: Dict[str, str] = {}

    for dep_location in dependency_locations:
        # Try worktree file first
        dep_file_path = os.path.join(worktree_path, dep_location)
        try:
            abs_worktree = os.path.abspath(worktree_path)
            abs_dep = os.path.abspath(dep_file_path)
            if abs_dep.startswith(abs_worktree) and os.path.exists(dep_file_path):
                with open(dep_file_path, 'r') as f:
                    result[dep_location] = f.read()
                continue
        except OSError:
            pass
        # Fallback: artifact DB
        try:
            artifact_data = load_artifact(dep_location, base_dir=base_dir)
            if artifact_data and artifact_data.get("code"):
                result[dep_location] = artifact_data["code"]
        except Exception:
            pass  # DB may not exist or be initialized

    return result
# ------------------------------------------------------------------ #
#  Iteration Log
# ------------------------------------------------------------------ #

def log_iteration(
    location: str,
    spec_hash: str,
    iteration: int,
    node: str,
    action: str,
    error_summary: str,
    tokens_in: int,
    tokens_out: int,
    duration_ms: int,
    base_dir: str = ".mags-codedev",
) -> None:
    """Record a single iteration step in the iteration_log table."""
    with sqlite3.connect(_db_path(base_dir)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO iteration_log
                (location, spec_hash, iteration, node, action,
                 error_summary, tokens_in, tokens_out, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location,
                spec_hash,
                iteration,
                node,
                action,
                error_summary,
                tokens_in,
                tokens_out,
                duration_ms,
            ),
        )
        conn.commit()


# ------------------------------------------------------------------ #
#  TokenLoggingCallbackHandler
# ------------------------------------------------------------------ #

class TokenLoggingCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs token usage to the SQLite DB."""

    def __init__(
        self,
        role: str,
        model_name: str,
        base_dir: str = ".mags-codedev",
    ) -> None:
        self.role = role
        self.model_name = model_name
        self.base_dir = base_dir

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when LLM ends running."""
        in_tokens, out_tokens = 0, 0

        # 1. Check llm_output (Legacy & some providers)
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            usage_metadata = response.llm_output.get("usage_metadata", {})

            in_tokens = token_usage.get("input_tokens", 0)
            out_tokens = token_usage.get("output_tokens", 0)

            # Mistral fallback
            if in_tokens == 0 and out_tokens == 0:
                in_tokens = token_usage.get("prompt_tokens", 0)
                out_tokens = token_usage.get("completion_tokens", 0)

            # Google fallback
            if usage_metadata:
                in_tokens = usage_metadata.get("prompt_token_count", in_tokens)
                out_tokens = usage_metadata.get("candidates_token_count", out_tokens)

        # 2. Check generations (Newer LangChain / Google GenAI)
        if in_tokens == 0 and out_tokens == 0 and response.generations:
            for generation_list in response.generations:
                for gen in generation_list:
                    if hasattr(gen, "message"):
                        usage = getattr(gen.message, "usage_metadata", {})
                        if usage:
                            in_tokens += usage.get("input_tokens", 0)
                            out_tokens += usage.get("output_tokens", 0)
                            # Fallback for Google specific keys
                            if in_tokens == 0 and out_tokens == 0:
                                in_tokens += usage.get("prompt_token_count", 0)
                                out_tokens += usage.get("candidates_token_count", 0)

        if in_tokens > 0 or out_tokens > 0:
            log_token_usage(
                role=self.role,
                model=self.model_name,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
                base_dir=self.base_dir,
            )

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        """Run when LLM errors. Logs the failure but does not record token usage."""
        logger.warning(
            "LLM call failed for role '%s' (model: %s). Error: %s",
            self.role,
            self.model_name,
            error,
        )
class TokenCounter(BaseCallbackHandler):
    """Lightweight callback that accumulates token usage in a shared dict.

    Usage::

        counter = TokenCounter()
        graph_config = {"callbacks": [counter]}
        # ... run graph ...
        total = counter.total  # {"in": N, "out": M}
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, int] = {"in": 0, "out": 0}

    @property
    def total(self) -> Dict[str, int]:
        return dict(self._tokens)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        in_tokens, out_tokens = 0, 0

        # 1. Check llm_output
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            usage_metadata = response.llm_output.get("usage_metadata", {})
            in_tokens = token_usage.get("input_tokens", 0)
            out_tokens = token_usage.get("output_tokens", 0)
            if in_tokens == 0 and out_tokens == 0:
                in_tokens = token_usage.get("prompt_tokens", 0)
                out_tokens = token_usage.get("completion_tokens", 0)
            if usage_metadata:
                in_tokens = usage_metadata.get("prompt_token_count", in_tokens)
                out_tokens = usage_metadata.get("candidates_token_count", out_tokens)

        # 2. Check generations
        if in_tokens == 0 and out_tokens == 0 and response.generations:
            for generation_list in response.generations:
                for gen in generation_list:
                    if hasattr(gen, "message"):
                        usage = getattr(gen.message, "usage_metadata", {})
                        if usage:
                            in_tokens += usage.get("input_tokens", 0)
                            out_tokens += usage.get("output_tokens", 0)
                            if in_tokens == 0 and out_tokens == 0:
                                in_tokens += usage.get("prompt_token_count", 0)
                                out_tokens += usage.get("candidates_token_count", 0)

        if in_tokens > 0 or out_tokens > 0:
            self._tokens["in"] += in_tokens
            self._tokens["out"] += out_tokens
