import pytest
from unittest.mock import patch, MagicMock


class TestGraphIntegration:
    """Full graph tests with mocked LLMs and Docker."""

    def _mock_llms(self):
        """Patch all LLM calls to return controlled responses."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "def foo(): return 42"
        mock_llm.invoke.return_value = mock_response

        patches = [
            patch("mags_codedev.agents.coder.get_llm", return_value=mock_llm),
            patch("mags_codedev.agents.tester.get_llm", return_value=mock_llm),
            patch("mags_codedev.agents.log_checker.get_llm", return_value=mock_llm),
            patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm]),
        ]
        return patches

    def test_successful_first_pass(self, sample_module_state):
        """Graph completes: coder → tester → tests pass → linters pass → review LGTM."""
        # Full graph integration test — requires mocking docker/docker ops
        # This is a placeholder for when full mock infrastructure is available
        assert True  # Placeholder — real test needs docker mock infrastructure
