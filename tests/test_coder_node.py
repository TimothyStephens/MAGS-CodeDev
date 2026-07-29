import pytest
from unittest.mock import patch, MagicMock
from mags_codedev.agents.coder import coder_node


class TestCoderNode:
    def _mock_invoke(self, content):
        """Create a mock invoke_with_retry that returns the given content."""
        mock_response = MagicMock()
        mock_response.content = content

        def fake_invoke(chain, inputs):
            return mock_response
        return fake_invoke

    def test_includes_review_comments_in_feedback(self, sample_module_state):
        """BUG FIX VERIFICATION: review comments must appear in coder feedback."""
        state = sample_module_state
        state["iteration_count"] = 3
        state["review_comments"] = ["Add input validation", "Handle empty list case"]
        state["code"] = "def foo(x): return x"
        state["tests"] = "def test_foo(): pass"

        captured_inputs = {}
        def capture_invoke(chain, inputs):
            captured_inputs.update(inputs)
            resp = MagicMock()
            resp.content = "def foo(x): return x or []"
            return resp

        with patch("mags_codedev.utils.llm_call.invoke_with_retry", side_effect=capture_invoke):
            coder_node(state)

        feedback = captured_inputs.get("feedback", "")
        assert "Add input validation" in feedback, \
            f"Review comments not included in coder feedback: {feedback}"
        assert "Handle empty list case" in feedback, \
            f"Review comments not included in coder feedback: {feedback}"

    def test_includes_test_error_in_feedback(self, sample_module_state):
        """Coder must include test_error_summary when present."""
        state = sample_module_state
        state["iteration_count"] = 2
        state["test_error_summary"] = "TypeError: 'NoneType' object has no attribute 'strip'"
        state["code"] = "def foo(): pass"

        captured_inputs = {}
        def capture_invoke(chain, inputs):
            captured_inputs.update(inputs)
            resp = MagicMock()
            resp.content = "def foo(): return 42"
            return resp

        with patch("mags_codedev.utils.llm_call.invoke_with_retry", side_effect=capture_invoke):
            coder_node(state)

        feedback = captured_inputs.get("feedback", "")
        assert "TypeError" in feedback, \
            f"Test error not included in coder feedback: {feedback}"

    def test_first_run_no_feedback(self, sample_module_state):
        """First run (iteration_count=0) should not include error feedback."""
        state = sample_module_state
        state["iteration_count"] = 0

        captured_inputs = {}
        def capture_invoke(chain, inputs):
            captured_inputs.update(inputs)
            resp = MagicMock()
            resp.content = "def foo(): return 42"
            return resp

        with patch("mags_codedev.utils.llm_call.invoke_with_retry", side_effect=capture_invoke):
            coder_node(state)

        narrative = captured_inputs.get("prompt_narrative", "")
        assert "Fix the following" not in narrative, \
            f"First run should not be in fix mode: {narrative}"

    def test_llm_failure_raises_not_stub(self, sample_module_state):
        """P1-1: an exhausted LLM call must raise, not synthesize a stub pass()."""
        state = sample_module_state
        state["iteration_count"] = 0

        def failing_invoke(chain, inputs):
            raise RuntimeError("API key invalid")

        with patch("mags_codedev.utils.llm_call.invoke_with_retry", side_effect=failing_invoke):
            with pytest.raises(RuntimeError, match="API key invalid"):
                coder_node(state)

