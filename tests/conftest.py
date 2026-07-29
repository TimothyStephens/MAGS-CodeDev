import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def temp_dir():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_config_path(temp_dir):
    """Create a minimal config.yaml in temp dir."""
    config = {
        "api_keys": {"openai": "test-key"},
        "models": {
            "build_workflow": {
                "coder": {"provider": "openai", "model": "gpt-4"},
                "tester": {"provider": "openai", "model": "gpt-4"},
                "log_checker": {"provider": "openai", "model": "gpt-4"},
                "reviewers": [{"provider": "openai", "model": "gpt-4"}],
            },
            "interactive_commands": {
                "chat": {"provider": "openai", "model": "gpt-4"},
            }
        },
        "settings": {
            "max_parallel_modules": 2,
            "max_test_fix_iterations": 5,
            "max_review_rounds": 3,
            "base_dir": os.path.join(temp_dir, ".mags-codedev"),
            "log_level": "debug"
        }
    }
    import yaml
    path = Path(temp_dir) / "config.yaml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def mock_llm():
    """Create a mock LLM instance."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "mock response"
    llm.invoke.return_value = response
    llm.ainvoke.return_value = response
    return llm


@pytest.fixture
def sample_module_state(temp_dir, mock_config_path):
    """Create a minimal ModuleState for testing."""
    from mags_codedev.state import ModuleState
    return ModuleState(
        module_location="src/foo.py",
        spec={"description": "test module", "dependencies": []},
        config_path=mock_config_path,
        worktree_path=temp_dir,
        test_location="tests/test_foo.py",
        backend=None,
        log_filepath=None,
        base_dir=os.path.join(temp_dir, ".mags-codedev"),
        code="",
        tests="",
        test_results="",
        lint_results="",
        test_error_summary="",
        review_comments=[],
        error_location=None,
        previous_code_hash=None,
        previous_test_hash=None,
        iteration_count=0,
        max_test_fix_iterations=5,
        max_review_rounds=3,
        review_round_count=0,
        status="in_progress",
    )
