"""Tests for the shared retry helpers (sync + async)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mags_codedev.utils.retry import ainvoke_with_retry, invoke_with_retry

# Fast retry config so tests don't actually sleep.
_FAST = {"min_wait": 0, "max_wait": 0}


class TestInvokeWithRetry:
    def test_retries_transient_then_succeeds(self):
        chain = MagicMock()
        ok = MagicMock(content="ok")
        chain.invoke.side_effect = [ValueError("rate limit exceeded"), ok]

        result = invoke_with_retry(chain, {}, **_FAST)

        assert chain.invoke.call_count == 2
        assert result is ok

    def test_non_retryable_raises_immediately(self):
        chain = MagicMock()
        chain.invoke.side_effect = ValueError("authentication error: bad key")

        with pytest.raises(ValueError, match="authentication"):
            invoke_with_retry(chain, {}, **_FAST)

        assert chain.invoke.call_count == 1


class TestAinvokeWithRetry:
    def test_retries_transient_then_succeeds(self):
        chain = MagicMock()
        ok = MagicMock(content="ok")
        chain.ainvoke = AsyncMock(
            side_effect=[ValueError("429 Too Many Requests"), ok]
        )

        result = asyncio.run(ainvoke_with_retry(chain, {}, **_FAST))

        assert chain.ainvoke.call_count == 2
        assert result is ok

    def test_non_retryable_raises_immediately(self):
        chain = MagicMock()
        chain.ainvoke = AsyncMock(side_effect=[ValueError("invalid_api_key")])

        with pytest.raises(ValueError, match="invalid_api_key"):
            asyncio.run(ainvoke_with_retry(chain, {}, **_FAST))

        assert chain.ainvoke.call_count == 1
