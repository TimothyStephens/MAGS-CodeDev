import pytest
from pathlib import Path
from mags_codedev.state import ModuleState


class TestStateInvariants:
    def test_module_state_has_all_required_fields(self):
        """ModuleState TypedDict should have all expected fields."""
        state: ModuleState = {
            "module_location": "src/foo.py",
            "spec": {"description": "test", "dependencies": []},
            "config_path": Path("/tmp/config.yaml"),
            "worktree_path": "/tmp/worktree",
            "test_location": "tests/test_foo.py",
            "backend": None,
            "log_filepath": "/tmp/foo.log",
            "base_dir": "/tmp/.mags-codedev",
            "code": "",
            "tests": "",
            "test_results": "",
            "lint_results": "",
            "test_error_summary": "",
            "lint_error_summary": "",
            "review_comments": [],
            "error_location": None,
            "previous_code_hash": None,
            "previous_test_hash": None,
            "iteration_count": 0,
            "max_test_fix_iterations": 5,
            "max_review_rounds": 3,
            "review_round_count": 0,
            "status": "in_progress",
            "offline": True,
        }
        # If this type-checks, all fields are present
        assert "review_comments" in state
        assert "test_error_summary" in state
        assert "max_review_rounds" in state

    def test_convergence_hashes_are_optional(self):
        """previous_code_hash and previous_test_hash should be Optional[str]."""
        state: ModuleState = {
            "code": "def foo(): pass",
            "previous_code_hash": None,
            "previous_test_hash": None,
        }
        assert state["previous_code_hash"] is None

    def test_module_state_total_false(self):
        """ModuleState with total=False should allow partial state."""
        state: ModuleState = {
            "module_location": "src/foo.py",
            "code": "print('hello')",
        }
        assert state["module_location"] == "src/foo.py"
        assert state["code"] == "print('hello')"
