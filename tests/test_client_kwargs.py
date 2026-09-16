# Description: Tests for the proxy's per-arm credential selection (_client_kwargs).
# Description: Pins that each provider prefix receives its own key, DeepSeek included.

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from pdp_router import _proxy
from pdp_router._clients import DeepSeekClient, get_client
from pdp_router._proxy_config import ProxyConfig


def _config() -> ProxyConfig:
    return ProxyConfig(
        anthropic_api_key="ant-key",
        gemini_api_key="gem-key",
        openrouter_api_key="or-key",
        deepseek_api_key="ds-key",
    )


class TestClientKwargs:
    """_client_kwargs picks each arm's own credential by model-name prefix."""

    def test_deepseek_arm_takes_the_deepseek_key(self):
        kwargs = _proxy._client_kwargs("deepseek-chat", _config())
        assert kwargs["api_key"] == "ds-key"
        assert "ant-key" not in kwargs.values()

    def test_deepseek_client_authenticates_with_the_deepseek_key(self):
        client = get_client("deepseek-chat", **_proxy._client_kwargs("deepseek-chat", _config()))
        assert isinstance(client, DeepSeekClient)
        assert client._api_key == "ds-key"

    def test_deepseek_key_reads_the_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
        assert ProxyConfig().deepseek_api_key == "from-env"

    def test_other_arms_are_unchanged(self):
        config = _config()
        assert _proxy._client_kwargs("claude-sonnet-5", config)["api_key"] == "ant-key"
        assert _proxy._client_kwargs("gemini-2.5-flash", config)["api_key"] == "gem-key"
        assert _proxy._client_kwargs("meta/llama-4-scout", config)["api_key"] == "ant-key"
        assert _proxy._client_kwargs("openai/gpt-5.5", config)["api_key"] == "or-key"
        assert _proxy._client_kwargs("qwen/qwen3.7-plus", config)["api_key"] == "or-key"


class TestHealthProviders:
    """/health reports whether the DeepSeek arm has a credential, like the others."""

    def test_health_lists_deepseek_key_presence(self):
        section = _proxy._configured_providers(_config())
        assert section["deepseek"] is True
        assert _proxy._configured_providers(ProxyConfig(deepseek_api_key=""))["deepseek"] is False
