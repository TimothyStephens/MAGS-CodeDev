import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from mags_codedev.agents.reviewer import multi_llm_review_node


class TestReviewerNode:
    def test_all_reviewers_fail_means_not_approved(self, sample_module_state):
        """BUG FIX VERIFICATION: all reviewers failing must not approve silently."""
        state = sample_module_state
        state["offline"] = False
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        # Create an AsyncMock that returns the failure message
        mock_get_review = AsyncMock(
            return_value="LGTM (reviewer skipped due to API error: ConnectionError)"
        )

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "in_progress", \
            f"All reviewers failed but code was approved. Got: {result}"
        assert len(result["review_comments"]) > 0, \
            "No feedback provided when all reviewers failed"

    def test_lgtm_means_approved(self, sample_module_state):
        """All reviewers saying LGTM means approved."""
        state = sample_module_state
        state["offline"] = False
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_get_review = AsyncMock(return_value="LGTM")

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "success"
        assert result["review_comments"] == []

    def test_actionable_review_means_in_progress(self, sample_module_state):
        """Actionable review comments mean code needs revision."""
        state = sample_module_state
        state["offline"] = False
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"

        mock_get_review = AsyncMock(
            return_value="Use a context manager for file handling."
        )

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "in_progress"
        assert len(result["review_comments"]) == 1
