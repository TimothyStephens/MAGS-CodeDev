import pytest
import hashlib
import os
from mags_codedev.utils.db import (
    init_db, hash_spec, is_function_built, mark_function_built,
    add_iterations_to_module, get_total_iterations, log_token_usage, get_token_summary,
    save_artifact, load_artifact, log_iteration, get_completed_status
)


class TestDatabase:
    def test_init_db_creates_tables(self, temp_dir):
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)
        import sqlite3
        db_path = os.path.join(base_dir, "cache.db")
        assert os.path.exists(db_path)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
        assert "completed_functions" in tables
        assert "token_usage" in tables
        assert "module_iterations" in tables
        assert "module_artifacts" in tables
        assert "iteration_log" in tables

    def test_hash_spec(self):
        spec = {"description": "test", "dependencies": []}
        h1 = hash_spec(spec)
        h2 = hash_spec(spec)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_save_and_load_artifact(self, temp_dir):
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)

        code_content = "def foo(): return 42"
        tests_content = "def test_foo(): assert foo() == 42"
        save_artifact(
            location="src/foo.py",
            code=code_content,
            tests=tests_content,
            spec_hash="abc123",
            base_dir=base_dir,
        )

        expected_code_hash = hashlib.sha256(code_content.encode("utf-8")).hexdigest()
        expected_test_hash = hashlib.sha256(tests_content.encode("utf-8")).hexdigest()

        artifact = load_artifact("src/foo.py", base_dir=base_dir)
        assert artifact is not None
        assert artifact["code"] == "def foo(): return 42"
        assert artifact["tests"] == "def test_foo(): assert foo() == 42"
        assert artifact["code_hash"] == expected_code_hash
        assert artifact["test_hash"] == expected_test_hash

    def test_load_artifact_not_found(self, temp_dir):
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)
        artifact = load_artifact("nonexistent.py", base_dir=base_dir)
        assert artifact is None

    def test_add_and_get_iterations(self, temp_dir):
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)

        add_iterations_to_module("src/foo.py", 3, base_dir=base_dir)
        assert get_total_iterations("src/foo.py", base_dir=base_dir) == 3

        add_iterations_to_module("src/foo.py", 2, base_dir=base_dir)
        assert get_total_iterations("src/foo.py", base_dir=base_dir) == 5

    def test_mark_and_check_function_built(self, temp_dir):
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)

        spec = {"description": "test module"}
        assert not is_function_built(spec, base_dir=base_dir)

        mark_function_built("test_module", spec, base_dir=base_dir)
        assert is_function_built(spec, base_dir=base_dir)

    def test_log_token_usage(self, temp_dir):
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)

        log_token_usage("coder", "gpt-4", 100, 50, base_dir=base_dir)
        per_role, per_model, total = get_token_summary(base_dir=base_dir)
        assert total == (100, 50)

    def test_log_iteration(self, temp_dir):
        base_dir = os.path.join(temp_dir, ".mags-codedev")
        init_db(base_dir=base_dir)

        log_iteration(
            location="src/foo.py",
            spec_hash="abc123",
            iteration=1,
            node="coder",
            action="generate",
            error_summary="",
            tokens_in=100,
            tokens_out=50,
            duration_ms=2000,
            base_dir=base_dir,
        )

        import sqlite3
        db_path = os.path.join(base_dir, "cache.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM iteration_log")
            count = cursor.fetchone()[0]
        assert count == 1
