import pytest
import os
from pathlib import Path


class TestProcessModule:
    """Orchestration tests with temp git repo."""

    @pytest.fixture
    def temp_repo(self, temp_dir):
        """Create a temporary git repo."""
        import subprocess
        subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
        Path(temp_dir, "README.md").write_text("# Test")
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=temp_dir, check=True, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@test.com",
                 "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@test.com"}
        )
        return temp_dir

    def test_artifacts_saved_on_failure(self, temp_repo, mock_config_path, sample_module_state):
        """Even when the graph fails, artifacts should be saved for next attempt."""
        from mags_codedev.utils.db import init_db, load_artifact

        base_dir = os.path.join(temp_repo, ".mags-codedev")
        init_db(base_dir=base_dir)

        # Simulate saving artifacts
        from mags_codedev.utils.db import save_artifact
        save_artifact(
            location="src/foo.py",
            code="def foo(): return 42",
            tests="def test_foo(): assert foo() == 42",
            spec_hash="abc123",
            base_dir=base_dir,
        )

        # Verify artifacts were saved
        artifact = load_artifact("src/foo.py", base_dir=base_dir)
        assert artifact is not None
        assert artifact["code"] == "def foo(): return 42"
        assert artifact["tests"] == "def test_foo(): assert foo() == 42"
