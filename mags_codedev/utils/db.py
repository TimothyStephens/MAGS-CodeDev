import sqlite3
import hashlib
import json
import pprint
import os
from mags_codedev.utils.logger import logger
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

DB_DIR = ".MAGS-CodeDev"
DB_PATH = os.path.join(DB_DIR, "cache.db")

def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
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

def hash_spec(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

def is_function_built(spec: dict) -> bool:
    spec_hash = hash_spec(spec)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM completed_functions WHERE func_hash = ?", (spec_hash,))
        return cursor.fetchone() is not None

def mark_function_built(function_name: str, spec: dict):
    spec_hash = hash_spec(spec)
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
        cursor.execute("SELECT COALESCE(SUM(in_tokens), 0), COALESCE(SUM(out_tokens), 0) FROM token_usage")
        total = cursor.fetchone()
        return summary, total or (0, 0)

class TokenLoggingCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs token usage to the SQLite DB."""
    def __init__(self, role: str, model_name: str):
        self.role = role
        self.model_name = model_name

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Run when LLM ends running."""
        # print(pprint.pformat(response))
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
                    if hasattr(gen, 'message'):
                        usage = getattr(gen.message, 'usage_metadata', {})
                        if usage:
                            in_tokens += usage.get("input_tokens", 0)
                            out_tokens += usage.get("output_tokens", 0)
                            # Fallback for Google specific keys
                            if in_tokens == 0 and out_tokens == 0:
                                in_tokens += usage.get("prompt_token_count", 0)
                                out_tokens += usage.get("candidates_token_count", 0)

        if in_tokens > 0 or out_tokens > 0:
            log_token_usage(role=self.role, model=self.model_name, in_tokens=in_tokens, out_tokens=out_tokens)