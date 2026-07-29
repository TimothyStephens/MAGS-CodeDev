"""Tests for the tester agent node."""

import pytest
from unittest.mock import patch

# Import as a module attribute (not a bare `tester_node` name) so pytest does
# not collect the imported function itself (its name starts with "test").
from mags_codedev.agents import tester as tester_agent


class TestTesterNode:
    def test_llm_failure_raises_not_stub(self, sample_module_state):
        """P1-1: an exhausted LLM call must raise, not synthesize a passing stub test."""
        state = sample_module_state
        state["iteration_count"] = 0
        state["code"] = "def foo(): return 42"

        def failing_invoke(chain, inputs):
            raise RuntimeError("quota exhausted")

        with patch("mags_codedev.utils.llm_call.invoke_with_retry", side_effect=failing_invoke):
            with pytest.raises(RuntimeError, match="quota exhausted"):
                tester_agent.tester_node(state)
