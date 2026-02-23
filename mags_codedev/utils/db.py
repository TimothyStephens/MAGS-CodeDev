import sqlite3
import hashlib
import json

DB_PATH = "mags_cache.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS completed_functions (
                func_hash TEXT PRIMARY KEY,
                function_name TEXT,
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
        conn.commit()

def _hash_spec(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

def is_function_built(spec: dict) -> bool:
    spec_hash = _hash_spec(spec)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM completed_functions WHERE func_hash = ?", (spec_hash,))
        return cursor.fetchone() is not None

def mark_function_built(function_name: str, spec: dict):
    spec_hash = _hash_spec(spec)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO completed_functions (func_hash, function_name) VALUES (?, ?)", 
                       (spec_hash, function_name))
        conn.commit()

def log_token_usage(role: str, model: str, in_tokens: int, out_tokens: int):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO token_usage (role, model, in_tokens, out_tokens) VALUES (?, ?, ?, ?)",
                       (role, model, in_tokens, out_tokens))
        conn.commit()

def get_token_summary():
    """Queries the database for aggregated token usage statistics."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Get per-role summary
        cursor.execute("""
            SELECT role, model, SUM(in_tokens), SUM(out_tokens)
            FROM token_usage
            GROUP BY role, model
            ORDER BY role
        """)
        summary = cursor.fetchall()
        # Get total
        cursor.execute("SELECT SUM(in_tokens), SUM(out_tokens) FROM token_usage")
        total = cursor.fetchone()
        return summary, total or (0, 0)