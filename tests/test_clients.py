# Description: Tests for LLM client wrappers and factory function.
# Description: Covers AnthropicClient, OllamaClient, get_client, and error handling.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdp_router._clients import (
    WEB_SEARCH_MAX_USES,
    WEB_SEARCH_TOOL_VERSION,
    AnthropicClient,
    CompletionResult,
    OllamaClient,
    get_client,
)
from pdp_router._models import CreditExhaustionError


class TestCompletionResult:
    def test_frozen_dataclass(self) -> None:
        result = CompletionResult(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            model="test",
            estimated_cost_usd=0.001,
        )
        assert result.text == "hello"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        with pytest.raises(AttributeError):
            result.text = "changed"  # type: ignore[misc]


class TestAnthropicClient:
    def test_init_with_api_key(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-test")

    def test_init_with_auth_token(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            AnthropicClient("claude-sonnet-4-20250514", auth_token="oat-t")
            mock_anthropic.Anthropic.assert_called_once_with(auth_token="oat-t")

    def test_init_prefers_api_key_over_auth_token(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            AnthropicClient(
                "claude-sonnet-4-20250514",
                api_key="sk-test",
                auth_token="oat-t",
            )
            mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-test")

    def test_init_no_credentials_raises(self) -> None:
        with pytest.raises(ValueError, match="No Anthropic credentials"):
            AnthropicClient("claude-sonnet-4-20250514")

    def test_complete_returns_completion_result(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(type="text", text="response text")]
            mock_message.usage.input_tokens = 100
            mock_message.usage.output_tokens = 50
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            result = client.complete("system prompt", "user message")

            assert isinstance(result, CompletionResult)
            assert result.text == "response text"
            assert result.input_tokens == 100
            assert result.output_tokens == 50
            assert result.model == "claude-sonnet-4-20250514"
            assert result.estimated_cost_usd > 0

    def test_complete_multi_returns_completion_result(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(type="text", text="multi response")]
            mock_message.usage.input_tokens = 200
            mock_message.usage.output_tokens = 100
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            messages = [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
            result = client.complete_multi("system", messages)

            assert result.text == "multi response"
            assert result.input_tokens == 200

    def test_credit_exhaustion_on_402(self) -> None:
        import anthropic as real_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.headers = {}
        error = real_anthropic.APIStatusError(
            "credit exhaustion",
            response=mock_response,
            body=None,
        )
        with patch("pdp_router._clients.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = error
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            with pytest.raises(CreditExhaustionError):
                client.complete("system", "message")

    def test_credit_exhaustion_on_billing_keyword(self) -> None:
        import anthropic as real_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}
        error = real_anthropic.APIStatusError(
            "insufficient credit balance",
            response=mock_response,
            body=None,
        )
        with patch("pdp_router._clients.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = error
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            with pytest.raises(CreditExhaustionError):
                client.complete("system", "message")

    def test_non_billing_error_reraises(self) -> None:
        import anthropic as real_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = real_anthropic.APIStatusError(
            "rate limited",
            response=mock_response,
            body=None,
        )
        with patch("pdp_router._clients.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = error
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            with pytest.raises(real_anthropic.APIStatusError):
                client.complete("system", "message")

    def test_langsmith_extra_passed_through(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            mock_message = MagicMock()
            mock_message.content = [MagicMock(type="text", text="ok")]
            mock_message.usage.input_tokens = 10
            mock_message.usage.output_tokens = 5
            mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_message

            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            client.complete("system", "message", langsmith_extra={"run_name": "test"})

            call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args
            assert call_kwargs.kwargs.get("langsmith_extra") == {"run_name": "test"}


class TestAnthropicWebSearch:
    """Web search server tool wiring (proxy_web_search_enabled cascade path)."""

    def _mock_text_message(self, text: str = "answer") -> MagicMock:
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text=text)]
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 5
        msg.usage.server_tool_use.web_search_requests = 0
        return msg

    def test_default_off_attaches_no_tools(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            create = mock_anthropic.Anthropic.return_value.messages.create
            create.return_value = self._mock_text_message()
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            client.complete("sys", "msg")
            assert "tools" not in create.call_args.kwargs

    def test_enable_web_search_attaches_pinned_tool(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            create = mock_anthropic.Anthropic.return_value.messages.create
            create.return_value = self._mock_text_message()
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            client.complete("sys", "msg", enable_web_search=True)
            tools = create.call_args.kwargs["tools"]
            assert tools == [
                {
                    "type": WEB_SEARCH_TOOL_VERSION,
                    "name": "web_search",
                    "max_uses": WEB_SEARCH_MAX_USES,
                }
            ]
            assert WEB_SEARCH_TOOL_VERSION == "web_search_20250305"

    def test_complete_multi_enable_web_search_attaches_tool(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            create = mock_anthropic.Anthropic.return_value.messages.create
            create.return_value = self._mock_text_message()
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            client.complete_multi(
                "sys", [{"role": "user", "content": "hi"}], enable_web_search=True
            )
            assert create.call_args.kwargs["tools"][0]["name"] == "web_search"

    def test_search_response_concatenates_text_blocks(self) -> None:
        """When search fires, content is [text, server_tool_use, result, text].

        content[0] is only the preamble; the answer lives in a later text block.
        The parser must join every text block, not read content[0].
        """
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            msg = MagicMock()
            msg.content = [
                MagicMock(type="text", text="I'll search. "),
                MagicMock(type="server_tool_use"),
                MagicMock(type="web_search_tool_result"),
                MagicMock(type="text", text="The answer is 42."),
            ]
            msg.usage.input_tokens = 6000
            msg.usage.output_tokens = 900
            msg.usage.server_tool_use.web_search_requests = 2
            mock_anthropic.Anthropic.return_value.messages.create.return_value = msg

            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            result = client.complete("sys", "msg", enable_web_search=True)

            assert result.text == "I'll search. The answer is 42."
            assert result.web_search_requests == 2

    def test_no_search_requests_defaults_to_zero(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            msg = MagicMock()
            msg.content = [MagicMock(type="text", text="plain answer")]
            msg.usage.input_tokens = 10
            msg.usage.output_tokens = 5
            # server_tool_use absent -> None on the real SDK; coerce to 0 here.
            msg.usage.server_tool_use = None
            mock_anthropic.Anthropic.return_value.messages.create.return_value = msg

            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            result = client.complete("sys", "msg")
            assert result.text == "plain answer"
            assert result.web_search_requests == 0


class TestOllamaClient:
    def test_complete_raises_not_available(self) -> None:
        client = OllamaClient("ollama/llama3")
        with pytest.raises(RuntimeError, match="not yet available"):
            client.complete("system", "message")

    def test_complete_multi_raises_not_available(self) -> None:
        client = OllamaClient("ollama/llama3")
        with pytest.raises(RuntimeError, match="not yet available"):
            client.complete_multi("system", [{"role": "user", "content": "hi"}])


class TestGetClient:
    def test_claude_prefix_returns_anthropic(self) -> None:
        with patch("pdp_router._clients.anthropic"):
            client = get_client("claude-sonnet-4-20250514", api_key="sk-test")
            assert isinstance(client, AnthropicClient)

    def test_ollama_prefix_returns_ollama(self) -> None:
        client = get_client("ollama/llama3")
        assert isinstance(client, OllamaClient)

    def test_local_prefix_returns_ollama(self) -> None:
        client = get_client("local/mistral")
        assert isinstance(client, OllamaClient)

    def test_gemini_prefix_returns_gemini_client(self) -> None:
        pytest.importorskip("google.genai")
        from unittest.mock import patch as _patch

        from pdp_router._clients import GeminiClient

        with _patch("google.genai.Client"):
            client = get_client("gemini-2.5-flash", api_key="test-key")
            assert isinstance(client, GeminiClient)

    def test_vertex_prefix_returns_gemini_client(self) -> None:
        pytest.importorskip("google.genai")
        from unittest.mock import patch as _patch

        from pdp_router._clients import GeminiClient

        with _patch("google.genai.Client"):
            client = get_client(
                "meta/llama-4-scout-17b-16e-instruct-maas",
                project="my-project",
                location="us-east5",
            )
            assert isinstance(client, GeminiClient)

    def test_unknown_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model provider"):
            get_client("gpt-4o")
