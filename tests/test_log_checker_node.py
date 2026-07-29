import pytest
from unittest.mock import patch, MagicMock
from mags_codedev.agents.log_checker import log_checker_node


class TestLogCheckerNode:
    def test_returns_scoped_test_error_summary(self, sample_module_state):
        """Log checker should return test_error_summary, not error_summary."""
        state = sample_module_state
        state["test_results"] = "FAILED test_foo - AssertionError"
        state["lint_results"] = ""

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"summary": "AssertionError in test_foo", "location": "SOURCE_CODE"}'
        mock_llm.invoke.return_value = mock_response

        with patch("mags_codedev.agents.log_checker.get_llm", return_value=mock_llm):
            result = log_checker_node(state)

        assert "test_error_summary" in result, "Should return test_error_summary key"
        assert "error_summary" not in result, "Should not return old error_summary key"

    def test_tests_passed_skips_analysis(self, sample_module_state):
        """If tests passed and no lint issues, skip LLM call."""
        state = sample_module_state
        state["test_results"] = "===== 3 passed ====="
        state["lint_results"] = ""

        with patch("mags_codedev.agents.log_checker.get_llm") as mock_get_llm:
            result = log_checker_node(state)

        mock_get_llm.assert_not_called()
        assert result.get("test_error_summary", "") == ""
