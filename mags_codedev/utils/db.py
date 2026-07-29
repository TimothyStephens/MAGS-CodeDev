"""SQLite-backed persistence for MAGs-CodeDev.

Tracks completed functions, token usage, module iteration counts,
module artifacts (code/tests), and per-iteration logs.

All functions accept `base_dir: str = ".mags-codedev"` as their first
parameter; the database lives at `os.path.join(base_dir, "cache.db")`.
"""

import hashlib
import os
import sqlite3
from typing import Dict, List, Tuple, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from mags_codedev.utils.logger import logger


def _db_path(base_dir: str) -> str:
    """Return the absolute path to the SQLite database."""
    return os.path.join(base_dir, "cache.db")


def _ensure_dir(base_dir: str) -> None:
    """Ensure the base_dir exists. Called by all DB functions except init_db.

    C2: Non-init DB functions now ensure directory exists before connecting.
    """
    os.makedirs(base_dir, exist_ok=True)


# ------------------------------------------------------------------ #
#  Schema
# ------------------------------------------------------------------ #

def init_db(base_dir: str = ".mags-codedev") -> None:
    """Create all tables in the database, creating the directory if needed."""
    os.makedirs(base_dir, exist_ok=True)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode=WAL")
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
                test_hash TEXT,
                previous_test_hash TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate: add test_hash, previous_code_hash, previous_test_hash columns
        for col in ("test_hash", "previous_code_hash", "previous_test_hash"):
            cursor.execute(
                f"SELECT count(*) FROM pragma_table_info('module_artifacts') WHERE name = '{col}'",
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    f"ALTER TABLE module_artifacts ADD COLUMN {col} TEXT",
                )

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

    H3: Note: Spec changes (description/dependencies) that don't affect
    location will NOT trigger a rebuild. Use --force to rebuild.
    M1: Explicit UTF-8 encoding for portability.
    """
    return hashlib.sha256(spec.get("location", "").encode("utf-8")).hexdigest()


def is_function_built(spec: dict, base_dir: str = ".mags-codedev") -> bool:
    """Check whether the function described by *spec* was already built."""
    _ensure_dir(base_dir)
    spec_hash = hash_spec(spec)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
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
    """Mark a function as built (or rebuilt) with the given status.

    C1: Uses ON CONFLICT DO UPDATE instead of INSERT OR IGNORE,
    so status updates are applied even for existing func_hash.
    """
    _ensure_dir(base_dir)
    spec_hash = hash_spec(spec)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO completed_functions (func_hash, function_name, status) VALUES (?, ?, ?) "
            "ON CONFLICT(func_hash) DO UPDATE SET status=excluded.status, function_name=excluded.function_name",
            (spec_hash, function_name, status),
        )
        conn.commit()


# ------------------------------------------------------------------ #
#  Module Iterations
# ------------------------------------------------------------------ #

def add_iterations_to_module(
    location: str,
    count: int,
    base_dir: str = ".mags-codedev",
) -> None:
    """Add *count* iterations to the module at *location*."""
    _ensure_dir(base_dir)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
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
    _ensure_dir(base_dir)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
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
    _ensure_dir(base_dir)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
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
    _ensure_dir(base_dir)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
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

    M1: Explicit UTF-8 encoding.
    """
    _ensure_dir(base_dir)
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    test_hash = hashlib.sha256(tests.encode("utf-8")).hexdigest()

    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
        cursor = conn.cursor()

        # Capture previous hashes before overwriting
        cursor.execute(
            "SELECT code_hash, test_hash FROM module_artifacts WHERE location = ?",
            (location,),
        )
        row = cursor.fetchone()
        previous_code_hash = row[0] if row else None
        previous_test_hash = row[1] if row else None
        cursor.execute("""
            INSERT INTO module_artifacts
                (location, code, tests, spec_hash, code_hash, previous_code_hash, test_hash, previous_test_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(location) DO UPDATE SET
                code = excluded.code,
                tests = excluded.tests,
                spec_hash = excluded.spec_hash,
                code_hash = excluded.code_hash,
                previous_code_hash = excluded.previous_code_hash,
                test_hash = excluded.test_hash,
                previous_test_hash = excluded.previous_test_hash,
                updated_at = CURRENT_TIMESTAMP
        """, (location, code, tests, spec_hash, code_hash, previous_code_hash, test_hash, previous_test_hash))

        conn.commit()


def load_artifact(
    location: str,
    base_dir: str = ".mags-codedev",
) -> Optional[dict]:
    """Return persisted artifacts for *location*, or None if not found.

    Returns a dict with keys: code, tests, spec_hash, code_hash, test_hash.
    """
    _ensure_dir(base_dir)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT code, tests, spec_hash, code_hash, test_hash FROM module_artifacts WHERE location = ?",
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
        "test_hash": row[4],
    }


def load_dependency_codes(
    dependency_locations: List[str],
    worktree_path: str,
    base_dir: str = ".mags-codedev",
) -> Dict[str, str]:
    """Load source code for each dependency location.

    Tries the worktree file first, falls back to the artifact DB.
    Returns a dict mapping location -> source code (only found deps).

    H2: Path traversal fix — ensures abs_worktree ends with os.sep
    before startswith check.
    """
    result: Dict[str, str] = {}

    # H2: Ensure trailing separator for safe prefix check
    abs_worktree = os.path.abspath(worktree_path)
    if not abs_worktree.endswith(os.sep):
        abs_worktree_safe = abs_worktree + os.sep
    else:
        abs_worktree_safe = abs_worktree

    for dep_location in dependency_locations:
        # Try worktree file first
        dep_file_path = os.path.join(worktree_path, dep_location)
        try:
            abs_dep = os.path.abspath(dep_file_path)
            # H2: Use safe prefix with trailing separator
            if abs_dep.startswith(abs_worktree_safe) and os.path.exists(dep_file_path):
                with open(dep_file_path, "r", encoding="utf-8") as f:
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
#  Token Extraction Helper (shared by TokenLoggingCallbackHandler and TokenCounter)
# ------------------------------------------------------------------ #

def _extract_tokens(response: LLMResult) -> Tuple[int, int]:
    """Extract (input_tokens, output_tokens) from an LLMResult.

    M2: Deduplicated from TokenLoggingCallbackHandler and TokenCounter.
    Handles multiple provider formats (OpenAI, Anthropic, Google, Mistral).
    """
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

    return in_tokens, out_tokens


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
        in_tokens, out_tokens = _extract_tokens(response)

        if in_tokens > 0 or out_tokens > 0:
            logger.trace(
                "Token usage — role=%s, model=%s, input=%d, output=%d",
                self.role,
                self.model_name,
                in_tokens,
                out_tokens,
            )
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
        in_tokens, out_tokens = _extract_tokens(response)

        if in_tokens > 0 or out_tokens > 0:
            self._tokens["in"] += in_tokens
            self._tokens["out"] += out_tokens


# ------------------------------------------------------------------ #
#  Iteration Log (per-iteration audit trail)
# ------------------------------------------------------------------ #

def log_iteration(
    location: str,
    spec_hash: str,
    iteration: int,
    node: str,
    action: str,
    error_summary: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    duration_ms: int = 0,
    base_dir: str = ".mags-codedev",
) -> None:
    """Record a single iteration event for audit / debugging."""
    _ensure_dir(base_dir)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO iteration_log "
            "(location, spec_hash, iteration, node, action, error_summary, "
            "tokens_in, tokens_out, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (location, spec_hash, iteration, node, action, error_summary,
             tokens_in, tokens_out, duration_ms),
        )
        conn.commit()


# ------------------------------------------------------------------ #
#  Completed Functions (status queries)
# ------------------------------------------------------------------ #

def get_completed_status(
    base_dir: str = ".mags-codedev",
) -> List[Tuple[str, str, str, str]]:
    """Return all completed function records.

    Returns: list of (func_hash, function_name, status, timestamp).
    """
    _ensure_dir(base_dir)
    with sqlite3.connect(_db_path(base_dir), timeout=10) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT func_hash, function_name, status, timestamp "
            "FROM completed_functions ORDER BY timestamp"
        )
        return cursor.fetchall()


__all__ = [
    "init_db",
    "hash_spec",
    "is_function_built",
    "mark_function_built",
    "add_iterations_to_module",
    "get_total_iterations",
    "log_token_usage",
    "get_token_summary",
    "save_artifact",
    "load_artifact",
    "load_dependency_codes",
    "TokenLoggingCallbackHandler",
    "TokenCounter",
    "log_iteration",
    "get_completed_status",
]

