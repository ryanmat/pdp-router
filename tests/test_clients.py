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
    DeepSeekClient,
    OllamaClient,
    OpenAICompatibleClient,
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


class TestOpenAICompatibleClient:
    """Generalized OpenAI-compatible transport (OpenRouter arms + DeepSeek subclass)."""

    def test_no_api_key_raises_with_label(self) -> None:
        with pytest.raises(ValueError, match="No OpenRouter credentials"):
            OpenAICompatibleClient(
                "openai/gpt-5.5",
                api_key="",
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
            )

    def test_no_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="No base_url"):
            OpenAICompatibleClient("openai/gpt-5.5", api_key="k", base_url="")

    def test_chat_url_strips_trailing_slash(self) -> None:
        with patch("httpx.Client"):
            client = OpenAICompatibleClient(
                "openai/gpt-5.5", api_key="k", base_url="https://openrouter.ai/api/v1/"
            )
            assert client._chat_url == "https://openrouter.ai/api/v1/chat/completions"

    def test_complete_parses_content_and_usage(self) -> None:
        with patch("httpx.Client") as mock_cls:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": "hi there"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7},
            }
            mock_cls.return_value.post.return_value = resp
            client = OpenAICompatibleClient(
                "openai/gpt-5.5", api_key="k", base_url="https://openrouter.ai/api/v1"
            )
            result = client.complete("system", "user")
            assert result.text == "hi there"
            assert result.input_tokens == 12
            assert result.output_tokens == 7
            assert result.model == "openai/gpt-5.5"
            # POSTs to the absolute chat URL, not a relative path.
            assert (
                mock_cls.return_value.post.call_args.args[0]
                == "https://openrouter.ai/api/v1/chat/completions"
            )

    def test_enable_web_search_accepted_and_not_forwarded(self) -> None:
        with patch("httpx.Client") as mock_cls:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            mock_cls.return_value.post.return_value = resp
            client = OpenAICompatibleClient(
                "openai/gpt-5.5", api_key="k", base_url="https://openrouter.ai/api/v1"
            )
            client.complete("system", "user", enable_web_search=True)
            sent = mock_cls.return_value.post.call_args.kwargs["json"]
            assert set(sent) == {"model", "messages", "max_tokens"}

    def test_complete_402_raises_credit_exhaustion(self) -> None:
        with patch("httpx.Client") as mock_cls:
            resp = MagicMock()
            resp.status_code = 402
            resp.text = "payment required"
            mock_cls.return_value.post.return_value = resp
            client = OpenAICompatibleClient(
                "openai/gpt-5.5",
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
            )
            with pytest.raises(CreditExhaustionError, match="OpenRouter"):
                client.complete("system", "user")

    def test_complete_billing_keyword_raises(self) -> None:
        with patch("httpx.Client") as mock_cls:
            resp = MagicMock()
            resp.status_code = 400
            resp.text = "Insufficient balance to complete request"
            mock_cls.return_value.post.return_value = resp
            client = OpenAICompatibleClient(
                "qwen/qwen3.7-plus",
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
            )
            with pytest.raises(CreditExhaustionError):
                client.complete("system", "user")

    def test_deepseek_subclass_uses_deepseek_url(self) -> None:
        with patch("httpx.Client"):
            client = DeepSeekClient("deepseek-chat", api_key="k")
            assert isinstance(client, OpenAICompatibleClient)
            assert client._chat_url == "https://api.deepseek.com/v1/chat/completions"

    def test_deepseek_no_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="No DeepSeek credentials"):
            DeepSeekClient("deepseek-chat", api_key="")


class TestEffortKnob:
    """Per-call reasoning-effort dial (Thread 3). Anthropic -> output_config.effort;
    OpenRouter -> reasoning.effort; the DeepSeek subclass and effort=None omit it."""

    def _mock_anthropic_message(self) -> MagicMock:
        msg = MagicMock()
        msg.content = [MagicMock(type="text", text="ok")]
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 5
        msg.usage.server_tool_use.web_search_requests = 0
        return msg

    def test_anthropic_effort_sets_output_config(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            create = mock_anthropic.Anthropic.return_value.messages.create
            create.return_value = self._mock_anthropic_message()
            client = AnthropicClient("claude-sonnet-4-6", api_key="sk-test")
            client.complete("sys", "msg", effort="high")
            assert create.call_args.kwargs["output_config"] == {"effort": "high"}

    def test_anthropic_no_effort_omits_output_config(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            create = mock_anthropic.Anthropic.return_value.messages.create
            create.return_value = self._mock_anthropic_message()
            client = AnthropicClient("claude-sonnet-4-6", api_key="sk-test")
            client.complete("sys", "msg")
            assert "output_config" not in create.call_args.kwargs

    def test_anthropic_haiku_self_guards_drops_effort(self) -> None:
        # Client-layer defense: even if a caller passes effort to a Haiku model
        # (which rejects output_config.effort), the client must drop it.
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            create = mock_anthropic.Anthropic.return_value.messages.create
            create.return_value = self._mock_anthropic_message()
            client = AnthropicClient("claude-haiku-4-5-20251001", api_key="sk-test")
            client.complete("sys", "msg", effort="high")
            assert "output_config" not in create.call_args.kwargs

    def test_anthropic_complete_multi_effort(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            create = mock_anthropic.Anthropic.return_value.messages.create
            create.return_value = self._mock_anthropic_message()
            client = AnthropicClient("claude-opus-4-8", api_key="sk-test")
            client.complete_multi("sys", [{"role": "user", "content": "hi"}], effort="medium")
            assert create.call_args.kwargs["output_config"] == {"effort": "medium"}

    def _mock_httpx_ok(self, mock_cls: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_cls.return_value.post.return_value = resp

    def test_openrouter_effort_sets_reasoning(self) -> None:
        with patch("httpx.Client") as mock_cls:
            self._mock_httpx_ok(mock_cls)
            client = OpenAICompatibleClient(
                "openai/gpt-5.5", api_key="k", base_url="https://openrouter.ai/api/v1"
            )
            client.complete("system", "user", effort="high")
            sent = mock_cls.return_value.post.call_args.kwargs["json"]
            assert sent["reasoning"] == {"effort": "high"}

    def test_openrouter_no_effort_omits_reasoning(self) -> None:
        with patch("httpx.Client") as mock_cls:
            self._mock_httpx_ok(mock_cls)
            client = OpenAICompatibleClient(
                "openai/gpt-5.5", api_key="k", base_url="https://openrouter.ai/api/v1"
            )
            client.complete("system", "user")
            sent = mock_cls.return_value.post.call_args.kwargs["json"]
            assert "reasoning" not in sent

    def test_deepseek_subclass_never_sends_reasoning(self) -> None:
        with patch("httpx.Client") as mock_cls:
            self._mock_httpx_ok(mock_cls)
            client = DeepSeekClient("deepseek-chat", api_key="k")
            client.complete("system", "user", effort="high")
            sent = mock_cls.return_value.post.call_args.kwargs["json"]
            assert "reasoning" not in sent


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

    def test_openai_prefix_returns_openai_compatible(self) -> None:
        with patch("httpx.Client"):
            client = get_client(
                "openai/gpt-5.5", api_key="k", base_url="https://openrouter.ai/api/v1"
            )
            assert isinstance(client, OpenAICompatibleClient)
            assert client._chat_url == "https://openrouter.ai/api/v1/chat/completions"

    def test_qwen_prefix_returns_openai_compatible_with_default_base(self) -> None:
        with patch("httpx.Client"):
            client = get_client("qwen/qwen3.7-plus", api_key="k")
            assert isinstance(client, OpenAICompatibleClient)
            # No base_url passed -> defaults to OpenRouter.
            assert client._chat_url == "https://openrouter.ai/api/v1/chat/completions"

    def test_deepseek_prefix_returns_deepseek_subclass(self) -> None:
        with patch("httpx.Client"):
            client = get_client("deepseek-chat", api_key="k")
            assert isinstance(client, DeepSeekClient)
            assert isinstance(client, OpenAICompatibleClient)
            assert client._chat_url == "https://api.deepseek.com/v1/chat/completions"

    def test_unknown_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model provider"):
            get_client("gpt-4o")
