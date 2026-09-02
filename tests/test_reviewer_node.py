import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from mags_codedev.agents.reviewer import multi_llm_review_node, _REVIEW_SKIPPED

SKIP = f"{_REVIEW_SKIPPED} (ConnectionError)"


class TestReviewerNode:
    def test_all_reviewers_fail_means_not_approved(self, sample_module_state):
        """All reviewers failing must not approve silently."""
        state = sample_module_state
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_get_review = AsyncMock(return_value=SKIP)

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "in_progress", \
            f"All reviewers failed but code was approved. Got: {result}"
        assert len(result["review_comments"]) > 0, \
            "No feedback provided when all reviewers failed"

    def test_lgtm_means_approved(self, sample_module_state):
        """A single reviewer saying LGTM approves (strict majority of 1)."""
        state = sample_module_state
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
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_get_review = AsyncMock(return_value="Use a context manager for file handling.")

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "in_progress"
        assert len(result["review_comments"]) == 1

    def test_partial_skip_does_not_approve(self, sample_module_state):
        """P1-2: 1-of-2 reviewers skipped + 1 LGTM must NOT approve (quorum unmet)."""
        state = sample_module_state
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_get_review = AsyncMock(side_effect=["LGTM", SKIP])

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm, mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "in_progress", \
            f"Partial outage granted approval. Got: {result}"
        assert result["review_comments"], "Expected a quorum-unmet comment"

    def test_majority_lgtm_with_one_skip_approves(self, sample_module_state):
        """P1-2: 2-of-3 reviewers LGTM (1 skipped) meets strict majority → approve."""
        state = sample_module_state
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_get_review = AsyncMock(side_effect=["LGTM", "LGTM", SKIP])

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm, mock_llm, mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "success", \
            f"Majority LGTM should approve despite a skip. Got: {result}"
        assert result["review_comments"] == []

    def test_skipped_is_neutral_not_actionable(self, sample_module_state):
        """P1-2: a skipped reviewer is neutral — never counted as actionable feedback."""
        state = sample_module_state
        state["code"] = "def foo(): return 42"

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4"
        mock_get_review = AsyncMock(side_effect=[SKIP, "Fix the off-by-one error in line 3."])

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[mock_llm, mock_llm]):
            with patch("mags_codedev.agents.reviewer._get_review", mock_get_review):
                result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "in_progress"
        assert len(result["review_comments"]) == 1, \
            "Skip must not be counted as an actionable comment"

    def test_no_reviewers_configured_auto_accepts(self, sample_module_state):
        """M5: no reviewers configured -> skip review and auto-approve (no 0/0 quorum loop)."""
        state = sample_module_state
        state["code"] = "def foo(): return 42"

        with patch("mags_codedev.agents.reviewer.get_reviewer_llms", return_value=[]):
            result = asyncio.run(multi_llm_review_node(state))

        assert result["status"] == "success", \
            f"No reviewers must auto-accept, got: {result}"
        assert result["review_comments"] == []
