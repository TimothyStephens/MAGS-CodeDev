"""Tests for config_parser: local LLM providers and env var resolution."""
import pytest
import os
from unittest.mock import patch, MagicMock
from mags_codedev.utils.config_parser import (
    _create_llm_instance,
    _resolve_env_api_keys,
)


class TestOllamaProvider:
    """Test the ollama provider (ChatOllama)."""

    def test_ollama_raises_import_error_when_not_installed(self):
        """ollama provider should raise ImportError if langchain-ollama missing."""
        # ChatOllama is None when langchain-ollama is not installed
        with patch(
            "mags_codedev.utils.config_parser.ChatOllama", None
        ):
            with pytest.raises(ImportError, match="langchain-ollama"):
                _create_llm_instance(
                    {"provider": "ollama", "model": "llama3.1"},
                    api_keys={},
                )

    def test_ollama_uses_default_base_url(self):
        """ollama should default to http://localhost:11434."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOllama", mock_cls
        ):
            _create_llm_instance(
                {"provider": "ollama", "model": "llama3.1"},
                api_keys={},
            )
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:11434"

    def test_ollama_uses_config_base_url(self):
        """ollama should use base_url from config when provided."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOllama", mock_cls
        ):
            _create_llm_instance(
                {
                    "provider": "ollama",
                    "model": "llama3.1",
                    "base_url": "http://custom:1234",
                },
                api_keys={},
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://custom:1234"

    def test_ollama_uses_env_base_url(self):
        """ollama should use OLLAMA_BASE_URL env var when set."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOllama", mock_cls
        ), patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env:9999"}):
            _create_llm_instance(
                {"provider": "ollama", "model": "llama3.1"},
                api_keys={},
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://env:9999"

    def test_ollama_config_base_url_overrides_env(self):
        """ollama config base_url takes priority over env var."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOllama", mock_cls
        ), patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env:9999"}):
            _create_llm_instance(
                {
                    "provider": "ollama",
                    "model": "llama3.1",
                    "base_url": "http://config:1234",
                },
                api_keys={},
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://config:1234"

    def test_ollama_passes_model_name(self):
        """ollama should pass model name to ChatOllama."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOllama", mock_cls
        ):
            _create_llm_instance(
                {"provider": "ollama", "model": "mistral-nemo"},
                api_keys={},
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["model"] == "mistral-nemo"

    def test_ollama_passes_num_ctx(self):
        """ollama should pass num_ctx to ChatOllama."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOllama", mock_cls
        ):
            _create_llm_instance(
                {
                    "provider": "ollama",
                    "model": "llama3.1",
                    "num_ctx": 16384,
                },
                api_keys={},
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["num_ctx"] == 16384


class TestLocalProvider:
    """Test the local provider (alias for custom_openai)."""

    def test_local_provider_creates_chat_openai(self):
        """local provider should create ChatOpenAI instance."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOpenAI", mock_cls
        ):
            _create_llm_instance(
                {
                    "provider": "local",
                    "model": "Qwen2.5-72B",
                    "base_url": "http://localhost:8000/v1",
                },
                api_keys={},
            )
        mock_cls.assert_called_once()

    def test_local_provider_falls_back_to_ollama_base_url_env(self):
        """local should check OLLAMA_BASE_URL env var."""
        mock_cls = MagicMock()
        env = {"OLLAMA_BASE_URL": "http://env:11434/v1"}
        # Make sure OPENAI_BASE_URL is not set
        env_copy = dict(os.environ)
        env_copy.pop("OPENAI_BASE_URL", None)
        env_copy.pop("OLLAMA_BASE_URL", None)
        env_copy.update(env)
        with patch(
            "mags_codedev.utils.config_parser.ChatOpenAI", mock_cls
        ), patch.dict(os.environ, env_copy, clear=True):
            _create_llm_instance(
                {"provider": "local", "model": "test"},
                api_keys={},
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://env:11434/v1"

    def test_local_provider_uses_config_base_url(self):
        """local should use config base_url when provided."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOpenAI", mock_cls
        ):
            _create_llm_instance(
                {
                    "provider": "local",
                    "model": "test",
                    "base_url": "http://config:9999/v1",
                },
                api_keys={},
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["base_url"] == "http://config:9999/v1"

    def test_local_provider_warns_on_missing_base_url(self, caplog):
        """local should warn when no base_url is configured."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOpenAI", mock_cls
        ), patch.dict(os.environ, {}, clear=True):
            _create_llm_instance(
                {"provider": "local", "model": "test"},
                api_keys={},
            )
        assert "no base_url configured" in caplog.text


class TestCustomOpenaiProvider:
    """Test the custom_openai provider (legacy name)."""

    def test_custom_openai_creates_chat_openai(self):
        """custom_openai should create ChatOpenAI instance."""
        mock_cls = MagicMock()
        with patch(
            "mags_codedev.utils.config_parser.ChatOpenAI", mock_cls
        ):
            _create_llm_instance(
                {
                    "provider": "custom_openai",
                    "model": "llama3:70b",
                    "base_url": "http://localhost:11434/v1",
                },
                api_keys={},
            )
        mock_cls.assert_called_once()

    def test_custom_openai_aliases_same_as_local(self):
        """custom_openai and local should behave identically."""
        mock_cls = MagicMock()
        for provider in ("custom_openai", "local"):
            mock_cls.reset_mock()
            with patch(
                "mags_codedev.utils.config_parser.ChatOpenAI", mock_cls
            ):
                _create_llm_instance(
                    {
                        "provider": provider,
                        "model": "test",
                        "base_url": "http://localhost:11434/v1",
                    },
                    api_keys={},
                )
            mock_cls.assert_called_once()


class TestEnvVarResolution:
    """Test environment variable resolution for API keys."""

    def test_ollama_api_key_from_env(self):
        """OLLAMA_API_KEY env var should be resolved."""
        config = {}
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "ollama-key-123"}):
            _resolve_env_api_keys(config)
        assert config["api_keys"]["ollama"] == "ollama-key-123"

    def test_ollama_api_key_in_api_keys(self):
        """ollama key should be accessible via api_keys dict."""
        config = {}
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "my-ollama-key"}):
            _resolve_env_api_keys(config)
        api_keys = config["api_keys"]
        assert "ollama" in api_keys
        assert api_keys["ollama"] == "my-ollama-key"


class TestUnsupportedProvider:
    """Test error handling for unsupported providers."""

    def test_unsupported_provider_raises_error(self):
        """Unsupported provider should raise ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            _create_llm_instance(
                {"provider": "unknown_provider", "model": "test"},
                api_keys={},
            )

    def test_unsupported_provider_lists_supported(self):
        """Error message should list supported providers."""
        with pytest.raises(ValueError, match="ollama"):
            _create_llm_instance(
                {"provider": "bad", "model": "test"},
                api_keys={},
            )
