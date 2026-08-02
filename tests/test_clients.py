# Description: Tests for LLM client wrappers and factory function.
# Description: Covers AnthropicClient, OllamaClient, get_client, and error handling.

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
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
from pdp_router._models import CreditExhaustionError, UpstreamStreamError
from pdp_router._tools import StreamFinish, ToolCallDelta


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


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": "run a command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }
]


class TestAnthropicToolCalls:
    """complete_with_tools: OpenAI tool payloads in, CompletionResult tool calls out."""

    def _tool_use_block(self, block_id: str = "toolu_1", name: str = "run", **kwargs) -> MagicMock:
        """A tool_use content block. `name` cannot go through the MagicMock
        constructor -- it would set the mock's name instead of the attribute."""
        block = MagicMock(type="tool_use", id=block_id)
        block.name = name
        block.input = kwargs or {"cmd": "ls"}
        return block

    def _message(self, content: list, stop_reason: str = "tool_use") -> MagicMock:
        msg = MagicMock()
        msg.content = content
        msg.stop_reason = stop_reason
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 5
        msg.usage.server_tool_use.web_search_requests = 0
        return msg

    def _client_and_create(self, mock_anthropic, message: MagicMock):
        create = mock_anthropic.Anthropic.return_value.messages.create
        create.return_value = message
        return AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test"), create

    def test_tools_are_translated_into_the_request(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, create = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block()])
            )
            client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

            assert create.call_args.kwargs["tools"] == [
                {
                    "name": "run",
                    "description": "run a command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                    },
                }
            ]

    def test_tool_choice_is_translated(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, create = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block()])
            )
            client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS, tool_choice="required"
            )
            assert create.call_args.kwargs["tool_choice"] == {"type": "any"}

    def test_parallel_tool_calls_false_is_threaded(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, create = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block()])
            )
            client.complete_with_tools(
                "sys",
                [{"role": "user", "content": "hi"}],
                _TOOLS,
                parallel_tool_calls=False,
            )
            assert create.call_args.kwargs["tool_choice"]["disable_parallel_tool_use"] is True

    def test_tool_history_is_translated(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, create = self._client_and_create(
                mock_anthropic, self._message([MagicMock(type="text", text="a and b")], "end_turn")
            )
            client.complete_with_tools(
                "sys",
                [
                    {"role": "user", "content": "list"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "a\nb"},
                ],
                _TOOLS,
            )

            sent = create.call_args.kwargs["messages"]
            assert sent[1]["content"][0]["type"] == "tool_use"
            assert sent[2]["content"][0] == {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "a\nb",
            }

    def test_web_search_is_never_auto_attached(self) -> None:
        """The plain paths attach web_search when the flag is on. Doing that here
        would overwrite the caller's tools with the search tool."""
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, create = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block()])
            )
            client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)
            assert [t["name"] for t in create.call_args.kwargs["tools"]] == ["run"]

    def test_effort_is_still_applied(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, create = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block()])
            )
            client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS, effort="high"
            )
            assert "output_config" in create.call_args.kwargs

    def test_tool_use_response_becomes_tool_calls(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block(cmd="ls -la")])
            )
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )

            assert result.finish_reason == "tool_calls"
            assert len(result.tool_calls) == 1
            call = result.tool_calls[0]
            assert (call.id, call.name) == ("toolu_1", "run")
            assert json.loads(call.arguments) == {"cmd": "ls -la"}

    def test_tool_use_ids_are_byte_exact(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block("toolu_01AbC_xyz")])
            )
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )
            assert result.tool_calls[0].id == "toolu_01AbC_xyz"

    def test_interleaved_text_and_tool_use(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(
                mock_anthropic,
                self._message(
                    [
                        MagicMock(type="text", text="Let me look. "),
                        self._tool_use_block(),
                        MagicMock(type="text", text="One moment."),
                    ]
                ),
            )
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )
            assert result.text == "Let me look. One moment."
            assert len(result.tool_calls) == 1

    def test_parallel_tool_use_blocks_keep_order(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(
                mock_anthropic,
                self._message(
                    [self._tool_use_block("toolu_1"), self._tool_use_block("toolu_2", "read")]
                ),
            )
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )
            assert [c.id for c in result.tool_calls] == ["toolu_1", "toolu_2"]
            assert result.tool_calls[1].name == "read"

    def test_zero_argument_tool_use_serialises_to_empty_object(self) -> None:
        """A no-parameter tool_use block must serialise its arguments as "{}", a
        valid empty JSON object -- never "", which the second round could not
        parse. The non-stream sibling of the 13d6d5c streaming fix."""
        block = MagicMock(type="tool_use", id="toolu_0")
        block.name = "ping"
        block.input = {}
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(mock_anthropic, self._message([block]))
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )
            assert result.tool_calls[0].arguments == "{}"

    def test_end_turn_response_has_no_tool_calls(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(
                mock_anthropic, self._message([MagicMock(type="text", text="done")], "end_turn")
            )
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )
            assert result.tool_calls == ()
            assert result.finish_reason == "stop"
            assert result.text == "done"

    def test_max_tokens_stop_reason_maps_to_length(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block()], "max_tokens")
            )
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )
            assert result.finish_reason == "length"

    def test_usage_and_cost_are_populated(self) -> None:
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, _ = self._client_and_create(
                mock_anthropic, self._message([self._tool_use_block()])
            )
            result = client.complete_with_tools(
                "sys", [{"role": "user", "content": "hi"}], _TOOLS
            )
            assert (result.input_tokens, result.output_tokens) == (10, 5)
            assert result.estimated_cost_usd > 0
            assert result.model == "claude-sonnet-4-20250514"

    def test_plain_paths_send_no_tool_keys(self) -> None:
        """Regression: adding the with-tools path must not change the body of an
        ordinary completion."""
        with patch("pdp_router._clients.anthropic") as mock_anthropic:
            client, create = self._client_and_create(
                mock_anthropic, self._message([MagicMock(type="text", text="hi")], "end_turn")
            )
            client.complete_multi("sys", [{"role": "user", "content": "hi"}])
            assert "tools" not in create.call_args.kwargs
            assert "tool_choice" not in create.call_args.kwargs


class _FakeToolStream:
    """Stands in for the Anthropic async stream manager.

    Has to be a real class: the SDK object is used as an async context manager
    AND as an async iterator, and MagicMock implements neither protocol.
    raise_at=N raises after N events, exercising the mid-stream error path.
    """

    def __init__(self, events: list, raise_at: int | None = None) -> None:
        self._events = events
        self._raise_at = raise_at
        self._pos = 0

    async def __aenter__(self) -> _FakeToolStream:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    def __aiter__(self) -> _FakeToolStream:
        return self

    async def __anext__(self) -> object:
        if self._raise_at is not None and self._pos == self._raise_at:
            raise RuntimeError("backend exploded")
        if self._pos >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._pos]
        self._pos += 1
        return event


def _collect(agen) -> list:
    """Drain an async generator from a sync test (no pytest-asyncio here)."""

    async def _drive() -> list:
        return [item async for item in agen]

    return asyncio.run(_drive())


class TestAnthropicToolStreaming:
    """stream_with_tools: Anthropic stream events out as text, deltas and a finish."""

    def _text_delta(self, text: str, index: int = 0) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_delta",
            index=index,
            delta=SimpleNamespace(type="text_delta", text=text),
        )

    def _tool_start(self, index: int, block_id: str, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_start",
            index=index,
            content_block=SimpleNamespace(type="tool_use", id=block_id, name=name),
        )

    def _json_delta(self, index: int, partial: str) -> SimpleNamespace:
        return SimpleNamespace(
            type="content_block_delta",
            index=index,
            delta=SimpleNamespace(type="input_json_delta", partial_json=partial),
        )

    def _message_delta(self, stop_reason: str) -> SimpleNamespace:
        return SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason=stop_reason))

    def _client(
        self, mock_cls: MagicMock, events: list, raise_at: int | None = None
    ) -> AnthropicClient:
        mock_cls.return_value.messages.stream.return_value = _FakeToolStream(events, raise_at)
        return AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")

    def _stream_call(self, client: AnthropicClient, **kwargs) -> object:
        return client.stream_with_tools(
            "sys", [{"role": "user", "content": "hi"}], _TOOLS, **kwargs
        )

    def test_text_then_tool_stream_yields_text_deltas_and_finish(self) -> None:
        """The tool_use block sits at content index 1 behind a text block, which is
        the ordinary shape and the one that separates block index from ordinal."""
        events = [
            self._text_delta("Let me "),
            self._tool_start(1, "toolu_1", "run"),
            self._json_delta(1, '{"comm'),
            self._json_delta(1, 'and":"x"}'),
            self._message_delta("tool_use"),
        ]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        assert out == [
            "Let me ",
            ToolCallDelta(index=0, id="toolu_1", name="run", arguments=""),
            ToolCallDelta(index=0, arguments='{"comm'),
            ToolCallDelta(index=0, arguments='and":"x"}'),
            StreamFinish("tool_calls"),
        ]

    def test_two_tool_blocks_yield_ordinals_zero_and_one(self) -> None:
        """Content indices 1 and 2 become tool ordinals 0 and 1: a passthrough of
        event.index would emit 1 and 2 and break every OpenAI client."""
        events = [
            self._text_delta("ok"),
            self._tool_start(1, "toolu_1", "run"),
            self._json_delta(1, '{"a":1}'),
            self._tool_start(2, "toolu_2", "run"),
            self._json_delta(2, '{"b":2}'),
            self._message_delta("tool_use"),
        ]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        deltas = [item for item in out if isinstance(item, ToolCallDelta)]
        assert [d.index for d in deltas] == [0, 0, 1, 1]
        assert [d.id for d in deltas] == ["toolu_1", None, "toolu_2", None]

    def test_pure_text_stream_yields_stop_finish(self) -> None:
        events = [
            self._text_delta("hello "),
            self._text_delta("world"),
            self._message_delta("end_turn"),
        ]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        assert out == ["hello ", "world", StreamFinish("stop")]

    def test_empty_text_delta_is_skipped(self) -> None:
        events = [self._text_delta(""), self._text_delta("x"), self._message_delta("end_turn")]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        assert out == ["x", StreamFinish("stop")]

    def test_unmapped_input_json_delta_is_dropped(self) -> None:
        """A server_tool_use block (web search) streams input_json_delta too. With
        no tool_use start to map it, attributing it to a call would corrupt the
        arguments of an unrelated tool."""
        events = [
            self._tool_start(1, "toolu_1", "run"),
            self._json_delta(7, '{"not":"mine"}'),
            self._message_delta("tool_use"),
        ]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        # The dropped fragment leaves call 0 with nothing, so the empty-argument
        # normalisation below supplies "{}" rather than leaving it unparseable.
        assert out == [
            ToolCallDelta(index=0, id="toolu_1", name="run", arguments=""),
            ToolCallDelta(index=0, arguments="{}"),
            StreamFinish("tool_calls"),
        ]

    def test_zero_argument_tool_call_normalises_to_an_empty_object(self) -> None:
        """Anthropic sends a single input_json_delta carrying "" for a tool that
        takes no parameters, so the fragments concatenate to "" and no client can
        json.loads it. The non-stream sibling emits "{}" via json.dumps(input or
        {}); the two legs have to agree. Zero-parameter tools are ordinary in MCP
        servers, so this is on the live-acceptance path."""
        events = [
            self._tool_start(1, "toolu_1", "now"),
            self._json_delta(1, ""),
            self._message_delta("tool_use"),
        ]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        assert out == [
            ToolCallDelta(index=0, id="toolu_1", name="now", arguments=""),
            ToolCallDelta(index=0, arguments=""),
            ToolCallDelta(index=0, arguments="{}"),
            StreamFinish("tool_calls"),
        ]
        reassembled = "".join(d.arguments for d in out if isinstance(d, ToolCallDelta))
        assert json.loads(reassembled) == {}

    def test_tool_call_with_no_argument_fragments_normalises(self) -> None:
        events = [self._tool_start(0, "toolu_1", "now"), self._message_delta("tool_use")]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        assert out == [
            ToolCallDelta(index=0, id="toolu_1", name="now", arguments=""),
            ToolCallDelta(index=0, arguments="{}"),
            StreamFinish("tool_calls"),
        ]

    def test_only_the_argumentless_call_is_normalised(self) -> None:
        """A parallel turn mixing a real-argument call with a zero-argument one
        must repair only the empty one."""
        events = [
            self._tool_start(1, "toolu_1", "run"),
            self._json_delta(1, '{"cmd":"ls"}'),
            self._tool_start(2, "toolu_2", "now"),
            self._json_delta(2, ""),
            self._message_delta("tool_use"),
        ]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        joined: dict[int, str] = {}
        for item in out:
            if isinstance(item, ToolCallDelta):
                joined[item.index] = joined.get(item.index, "") + item.arguments
        assert joined == {0: '{"cmd":"ls"}', 1: "{}"}

    def test_synthesized_sdk_events_are_ignored(self) -> None:
        """The SDK's own helper emits a synthesized text/input_json event after
        each raw delta. Dispatching on attributes rather than the type string
        would yield every fragment twice."""
        events = [
            self._text_delta("hi"),
            SimpleNamespace(type="text", text="hi", snapshot="hi"),
            self._tool_start(1, "toolu_1", "run"),
            self._json_delta(1, '{"a":1}'),
            SimpleNamespace(type="input_json", partial_json='{"a":1}', snapshot={"a": 1}),
            self._message_delta("tool_use"),
        ]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events)
            out = _collect(self._stream_call(client))

        assert out == [
            "hi",
            ToolCallDelta(index=0, id="toolu_1", name="run", arguments=""),
            ToolCallDelta(index=0, arguments='{"a":1}'),
            StreamFinish("tool_calls"),
        ]

    def test_tools_and_choice_are_translated_into_the_stream_request(self) -> None:
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, [self._message_delta("end_turn")])
            _collect(self._stream_call(client, tool_choice="required"))

        kwargs = mock_cls.return_value.messages.stream.call_args.kwargs
        assert kwargs["tools"] == [
            {
                "name": "run",
                "description": "run a command",
                "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            }
        ]
        assert kwargs["tool_choice"] == {"type": "any"}

    def test_tool_history_is_translated_into_the_stream_request(self) -> None:
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, [self._message_delta("end_turn")])
            _collect(
                client.stream_with_tools(
                    "sys",
                    [
                        {"role": "user", "content": "list"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_1", "content": "a\nb"},
                    ],
                    _TOOLS,
                )
            )

        sent = mock_cls.return_value.messages.stream.call_args.kwargs["messages"]
        assert sent[1]["content"][0]["type"] == "tool_use"
        assert sent[2]["content"][0]["tool_use_id"] == "call_1"

    def test_web_search_is_never_auto_attached(self) -> None:
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, [self._message_delta("end_turn")])
            _collect(self._stream_call(client))

        kwargs = mock_cls.return_value.messages.stream.call_args.kwargs
        assert [t["name"] for t in kwargs["tools"]] == ["run"]

    def test_effort_is_applied_to_the_stream_request(self) -> None:
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, [self._message_delta("end_turn")])
            _collect(self._stream_call(client, effort="high"))

        assert "output_config" in mock_cls.return_value.messages.stream.call_args.kwargs

    def test_mid_stream_exception_propagates(self) -> None:
        events = [self._text_delta("partial"), self._message_delta("end_turn")]
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            client = self._client(mock_cls, events, raise_at=1)
            with pytest.raises(RuntimeError, match="backend exploded"):
                _collect(self._stream_call(client))

    def test_stream_credit_error_maps_to_credit_exhaustion(self) -> None:
        import anthropic as real_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.headers = {}
        error = real_anthropic.APIStatusError(
            "credit exhaustion", response=mock_response, body=None
        )
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.stream.side_effect = error
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            with pytest.raises(CreditExhaustionError):
                _collect(self._stream_call(client))

    def test_non_billing_stream_error_reraises(self) -> None:
        import anthropic as real_anthropic

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        error = real_anthropic.APIStatusError("rate limited", response=mock_response, body=None)
        with patch("pdp_router._clients.anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value.messages.stream.side_effect = error
            client = AnthropicClient("claude-sonnet-4-20250514", api_key="sk-test")
            with pytest.raises(real_anthropic.APIStatusError):
                _collect(self._stream_call(client))


class TestOllamaClient:
    def test_complete_raises_not_available(self) -> None:
        client = OllamaClient("ollama/llama3")
        with pytest.raises(RuntimeError, match="not yet available"):
            client.complete("system", "message")

    def test_complete_multi_raises_not_available(self) -> None:
        client = OllamaClient("ollama/llama3")
        with pytest.raises(RuntimeError, match="not yet available"):
            client.complete_multi("system", [{"role": "user", "content": "hi"}])

    def test_complete_with_tools_raises_not_implemented(self) -> None:
        """The driver floor guarantees this is unreachable. The stub is what turns
        a routing mistake into a clean error instead of an AttributeError."""
        client = OllamaClient("ollama/llama3")
        with pytest.raises(NotImplementedError):
            client.complete_with_tools("system", [{"role": "user", "content": "hi"}], [])

    def test_stream_with_tools_raises_not_implemented(self) -> None:
        client = OllamaClient("ollama/llama3")
        with pytest.raises(NotImplementedError):
            client.stream_with_tools("system", [{"role": "user", "content": "hi"}], [])


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


class _HTTPStatusError(Exception):
    """Stands in for httpx.HTTPStatusError, which needs real request/response objects."""


class _FakeSSEResponse:
    """Stands in for an httpx streaming response.

    A real class for the same reason as _FakeToolStream: the object is an async
    context manager whose body is drained by an async iterator, and MagicMock
    implements neither protocol.
    """

    def __init__(self, lines: list[str], status_code: int = 200, body: bytes = b"") -> None:
        self._lines = lines
        self._body = body
        self.status_code = status_code
        # Counted because "never read the body on a 2xx" is otherwise invisible
        # to every assertion here: reading it leaves aiter_lines working and the
        # yielded tokens identical, so a guard that buffers the whole upstream
        # response before emitting a token looks exactly like one that streams.
        self.aread_calls = 0

    async def __aenter__(self) -> _FakeSSEResponse:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def aread(self) -> bytes:
        self.aread_calls += 1
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _HTTPStatusError(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _ok_tool_response(content: object, tool_calls: list | None = None, finish: str = "tool_calls"):
    """An OpenAI-shaped non-stream response carrying tool calls."""
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }
    return resp


_OPENAI_CALL = [
    {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "run", "arguments": '{"cmd": "ls -la"}'},
    }
]


class TestOpenAICompatibleToolCalls:
    """complete_with_tools: native passthrough, no translation in either direction."""

    def _client(self, mock_cls: MagicMock, resp: MagicMock) -> OpenAICompatibleClient:
        mock_cls.return_value.post.return_value = resp
        return OpenAICompatibleClient(
            "openai/gpt-5.5",
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            label="OpenRouter",
        )

    def test_tools_tool_choice_parallel_flags_passthrough_verbatim(self) -> None:
        """Native endpoints speak OpenAI already, so the payload must arrive
        untouched -- identity, not merely equality, on the tools list."""
        with patch("httpx.Client") as mock_cls:
            client = self._client(mock_cls, _ok_tool_response(None, _OPENAI_CALL))
            client.complete_with_tools(
                "sys",
                [{"role": "user", "content": "hi"}],
                _TOOLS,
                tool_choice="required",
                parallel_tool_calls=False,
            )

        sent = mock_cls.return_value.post.call_args.kwargs["json"]
        assert sent["tools"] is _TOOLS
        assert sent["tool_choice"] == "required"
        assert sent["parallel_tool_calls"] is False
        assert sent["messages"][0] == {"role": "system", "content": "sys"}

    def test_omitted_tool_options_are_not_sent(self) -> None:
        with patch("httpx.Client") as mock_cls:
            client = self._client(mock_cls, _ok_tool_response(None, _OPENAI_CALL))
            client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        sent = mock_cls.return_value.post.call_args.kwargs["json"]
        assert "tool_choice" not in sent
        assert "parallel_tool_calls" not in sent

    def test_plain_complete_multi_body_has_no_tool_keys(self) -> None:
        """Regression: widening _call must not change an ordinary request body."""
        with patch("httpx.Client") as mock_cls:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            client = self._client(mock_cls, resp)
            result = client.complete_multi("sys", [{"role": "user", "content": "hi"}])

        sent = mock_cls.return_value.post.call_args.kwargs["json"]
        assert set(sent) == {"model", "messages", "max_tokens"}
        assert result.tool_calls == ()
        assert result.finish_reason == "stop"

    def test_response_tool_calls_parsed_verbatim(self) -> None:
        with patch("httpx.Client") as mock_cls:
            client = self._client(mock_cls, _ok_tool_response(None, _OPENAI_CALL))
            result = client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert (call.id, call.name) == ("call_abc", "run")
        # The raw argument string survives byte-exact: no parse-reserialise round
        # trip, so key order and any provider truncation reach the caller intact.
        assert call.arguments == '{"cmd": "ls -la"}'

    def test_null_content_with_tool_calls_yields_empty_text(self) -> None:
        """A tool-only turn returns content null, which the plain path's
        unguarded subscript would hand downstream as None."""
        with patch("httpx.Client") as mock_cls:
            client = self._client(mock_cls, _ok_tool_response(None, _OPENAI_CALL))
            result = client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        assert result.text == ""
        assert result.finish_reason == "tool_calls"

    def test_text_and_tool_calls_both_survive(self) -> None:
        with patch("httpx.Client") as mock_cls:
            client = self._client(mock_cls, _ok_tool_response("on it", _OPENAI_CALL))
            result = client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        assert result.text == "on it"
        assert len(result.tool_calls) == 1

    def test_finish_reason_passthrough_length(self) -> None:
        with patch("httpx.Client") as mock_cls:
            client = self._client(mock_cls, _ok_tool_response("cut off", None, finish="length"))
            result = client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        assert result.finish_reason == "length"
        assert result.tool_calls == ()

    def test_missing_finish_reason_defaults_to_stop(self) -> None:
        with patch("httpx.Client") as mock_cls:
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            client = self._client(mock_cls, resp)
            result = client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        assert result.finish_reason == "stop"

    def test_usage_and_cost_are_populated(self) -> None:
        with patch("httpx.Client") as mock_cls:
            client = self._client(mock_cls, _ok_tool_response(None, _OPENAI_CALL))
            result = client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        assert (result.input_tokens, result.output_tokens) == (11, 4)
        assert result.model == "openai/gpt-5.5"

    def test_billing_error_still_raises_on_the_tools_path(self) -> None:
        with patch("httpx.Client") as mock_cls:
            resp = MagicMock()
            resp.status_code = 402
            resp.text = "payment required"
            client = self._client(mock_cls, resp)
            with pytest.raises(CreditExhaustionError, match="OpenRouter"):
                client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

    def test_deepseek_inherits_the_tools_path(self) -> None:
        with patch("httpx.Client") as mock_cls:
            mock_cls.return_value.post.return_value = _ok_tool_response(None, _OPENAI_CALL)
            client = DeepSeekClient("deepseek-chat", api_key="k")
            result = client.complete_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)

        assert result.tool_calls[0].id == "call_abc"
        assert mock_cls.return_value.post.call_args.kwargs["json"]["tools"] is _TOOLS


class TestOpenAICompatibleToolStreaming:
    """stream_with_tools: SSE tool_calls fragments out as ToolCallDelta."""

    def _sse(self, payload: dict) -> str:
        return f"data: {json.dumps(payload)}"

    def _content(self, text: str) -> dict:
        return {"choices": [{"index": 0, "delta": {"content": text}}]}

    def _fragment(self, index: int, **frag: object) -> dict:
        return {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, **frag}]}}]}

    def _finish(self, reason: str) -> dict:
        return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}

    def _client(self, mock_cls: MagicMock, lines: list[str]) -> OpenAICompatibleClient:
        mock_cls.return_value.stream.return_value = _FakeSSEResponse(lines)
        return OpenAICompatibleClient(
            "openai/gpt-5.5",
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            label="OpenRouter",
        )

    def _stream_call(self, client: OpenAICompatibleClient, **kwargs) -> object:
        return client.stream_with_tools(
            "sys", [{"role": "user", "content": "hi"}], _TOOLS, **kwargs
        )

    def test_stream_tool_call_fragments_become_deltas(self) -> None:
        lines = [
            self._sse(
                self._fragment(
                    0, id="call_1", type="function", function={"name": "run", "arguments": ""}
                )
            ),
            "",
            self._sse(self._fragment(0, function={"arguments": '{"cmd"'})),
            self._sse(self._fragment(0, function={"arguments": ': "ls"}'})),
            self._sse(self._finish("tool_calls")),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(self._stream_call(client))

        assert out == [
            ToolCallDelta(index=0, id="call_1", name="run", arguments=""),
            ToolCallDelta(index=0, arguments='{"cmd"'),
            ToolCallDelta(index=0, arguments=': "ls"}'),
            StreamFinish("tool_calls"),
        ]

    def test_parallel_call_indices_pass_through_unrewritten(self) -> None:
        """Unlike the Anthropic path these indices are already tool ordinals."""
        lines = [
            self._sse(self._fragment(0, id="call_1", function={"name": "run", "arguments": ""})),
            self._sse(self._fragment(1, id="call_2", function={"name": "run", "arguments": ""})),
            self._sse(self._fragment(1, function={"arguments": "{}"})),
            self._sse(self._finish("tool_calls")),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(self._stream_call(client))

        deltas = [item for item in out if isinstance(item, ToolCallDelta)]
        assert [d.index for d in deltas] == [0, 1, 1]
        assert [d.id for d in deltas] == ["call_1", "call_2", None]

    def test_interleaved_content_deltas_yield_str(self) -> None:
        lines = [
            self._sse(self._content("Let me ")),
            self._sse(self._fragment(0, id="call_1", function={"name": "run", "arguments": ""})),
            self._sse(self._finish("tool_calls")),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(self._stream_call(client))

        assert out[0] == "Let me "
        assert isinstance(out[1], ToolCallDelta)

    def test_content_and_finish_on_same_chunk_yields_text_first(self) -> None:
        """Some arms attach finish_reason to the last content-bearing chunk
        instead of sending it alone; the text must not be lost behind it."""
        lines = [
            self._sse({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(self._stream_call(client))

        assert out == ["done", StreamFinish("stop")]

    def test_exotic_finish_reason_passes_through_unmapped(self) -> None:
        """Verbatim has to mean verbatim on this path, because it is the native
        one -- only the Anthropic leg translates a stop_reason.

        Every other streaming finish assertion here uses "tool_calls" or "stop",
        and both are fixed points of any plausible mapper, so none of them would
        notice a translation slipped in. "content_filter" is a real OpenAI
        finish_reason that no Anthropic stop_reason maps onto, so it moves only
        if something is rewriting it.
        """
        lines = [
            self._sse(self._content("blocked")),
            self._sse(self._finish("content_filter")),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(self._stream_call(client))

        assert out == ["blocked", StreamFinish("content_filter")]

    def test_usage_only_chunk_after_finish_is_tolerated(self) -> None:
        """OpenRouter emits a usage chunk with an empty choices list after the
        finish chunk, so the finish must not terminate the loop."""
        lines = [
            self._sse(self._content("hi")),
            self._sse(self._finish("stop")),
            self._sse({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(self._stream_call(client))

        assert out == ["hi", StreamFinish("stop")]

    def test_keepalive_and_unparseable_lines_are_skipped(self) -> None:
        lines = [
            ": OPENROUTER PROCESSING",
            "data: not json",
            self._sse(self._content("hi")),
            self._sse(self._finish("stop")),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(self._stream_call(client))

        assert out == ["hi", StreamFinish("stop")]

    def test_tools_ride_verbatim_on_the_stream_body(self) -> None:
        lines = [self._sse(self._finish("stop")), "data: [DONE]"]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            _collect(self._stream_call(client, tool_choice="auto"))

        sent = mock_cls.return_value.stream.call_args.kwargs["json"]
        assert sent["tools"] is _TOOLS
        assert sent["tool_choice"] == "auto"
        assert sent["stream"] is True

    def test_plain_stream_body_has_no_tool_keys(self) -> None:
        """Regression: the sibling generator must leave _stream's body alone."""
        lines = [self._sse(self._content("hi")), "data: [DONE]"]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            out = _collect(client.stream_complete_multi("sys", [{"role": "user", "content": "x"}]))

        sent = mock_cls.return_value.stream.call_args.kwargs["json"]
        assert set(sent) == {"model", "messages", "max_tokens", "stream"}
        assert out == ["hi"]

    def test_success_path_never_reads_the_body(self) -> None:
        """Same invariant as the plain leg: a 2xx body is never touched, because
        reading it buffers the whole generation ahead of the first token."""
        fake = _FakeSSEResponse(
            [self._sse(self._content("hi")), self._sse(self._finish("stop")), "data: [DONE]"]
        )
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.stream.return_value = fake
            client = OpenAICompatibleClient(
                "openai/gpt-5.5",
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
            )
            out = _collect(self._stream_call(client))

        assert out == ["hi", StreamFinish("stop")]
        assert fake.aread_calls == 0

    def test_stream_billing_error_maps_to_credit_exhaustion(self) -> None:
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.stream.return_value = _FakeSSEResponse([], status_code=402)
            client = OpenAICompatibleClient(
                "openai/gpt-5.5",
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
            )
            with pytest.raises(CreditExhaustionError, match="OpenRouter"):
                _collect(self._stream_call(client))

    def test_stream_billing_keyword_on_non_402_maps_to_credit_exhaustion(self) -> None:
        """OpenRouter signals an exhausted balance as 429 with a billing body as
        often as it does 402. The keyword branch is the only thing that catches
        it, and awaiting inside the any() generator expression makes it an async
        generator that any() cannot consume -- a TypeError, not a billing error."""
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.stream.return_value = _FakeSSEResponse(
                [], status_code=429, body=b"Insufficient credits remaining"
            )
            client = OpenAICompatibleClient(
                "openai/gpt-5.5",
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
            )
            with pytest.raises(CreditExhaustionError, match="OpenRouter"):
                _collect(self._stream_call(client))

    def test_stream_non_billing_error_falls_through_to_raise_for_status(self) -> None:
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.stream.return_value = _FakeSSEResponse(
                [], status_code=429, body=b"rate limited, slow down"
            )
            client = OpenAICompatibleClient(
                "openai/gpt-5.5",
                api_key="k",
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
            )
            with pytest.raises(_HTTPStatusError):
                _collect(self._stream_call(client))

    def test_deepseek_inherits_the_tool_stream(self) -> None:
        lines = [
            self._sse(self._fragment(0, id="call_1", function={"name": "run", "arguments": ""})),
            self._sse(self._finish("tool_calls")),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.stream.return_value = _FakeSSEResponse(lines)
            client = DeepSeekClient("deepseek-chat", api_key="k")
            out = _collect(
                client.stream_with_tools("sys", [{"role": "user", "content": "hi"}], _TOOLS)
            )

        assert out[0] == ToolCallDelta(index=0, id="call_1", name="run", arguments="")


class TestPlainStreamErrorGuard:
    """_stream's error guard, on the PLAIN streaming path.

    stream_complete/stream_complete_multi back live plain streaming on both
    proxy surfaces, and the guard there carried the same await-inside-any()
    defect the tool stream did. Only 402 worked, because it short-circuits
    ahead of the comprehension; every other error status raised TypeError
    before the billing check ran, which also left raise_for_status
    unreachable, so auth failures, rate limits and 5xx all surfaced as a type
    error rather than the real fault.
    """

    def _client(self, mock_cls: MagicMock, response: _FakeSSEResponse) -> OpenAICompatibleClient:
        mock_cls.return_value.stream.return_value = response
        return OpenAICompatibleClient(
            "openai/gpt-5.5",
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            label="OpenRouter",
        )

    def _sse_line(self, text: str) -> str:
        return f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': text}}]})}"

    def _plain_call(self, client: OpenAICompatibleClient) -> object:
        return client.stream_complete_multi("sys", [{"role": "user", "content": "x"}])

    def test_success_path_never_reads_the_body(self) -> None:
        """The status gate in front of the body read is load-bearing, not a
        micro-optimisation. Reading the body of a 2xx streaming response buffers
        the entire upstream generation before the first token escapes, which on
        the live surfaces means the client sees silence for the whole answer and
        then every token at once. Nothing else here can see that: aread() leaves
        the line iteration working and the yielded tokens identical, so the
        symptom is pure latency and the suite stays green.
        """
        fake = _FakeSSEResponse(
            [self._sse_line("hi"), self._sse_line(" there"), "data: [DONE]"], status_code=200
        )
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, fake)
            out = _collect(self._plain_call(client))

        assert out == ["hi", " there"]
        assert fake.aread_calls == 0

    def test_plain_stream_402_maps_to_credit_exhaustion(self) -> None:
        """The one leg that already worked, pinned before the guard is touched:
        no test drove a plain-path error status at all before this class."""
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, _FakeSSEResponse([], status_code=402))
            with pytest.raises(CreditExhaustionError, match="OpenRouter"):
                _collect(self._plain_call(client))

    def test_plain_stream_billing_keyword_on_non_402_maps_to_credit_exhaustion(self) -> None:
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(
                mock_cls,
                _FakeSSEResponse([], status_code=429, body=b"Insufficient credits remaining"),
            )
            with pytest.raises(CreditExhaustionError, match="OpenRouter"):
                _collect(self._plain_call(client))

    def test_plain_stream_non_billing_error_falls_through_to_raise_for_status(self) -> None:
        """A 5xx carries no billing keyword, so it has to reach
        raise_for_status and surface as the HTTP error it is."""
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(
                mock_cls, _FakeSSEResponse([], status_code=503, body=b"upstream unavailable")
            )
            with pytest.raises(_HTTPStatusError):
                _collect(self._plain_call(client))


_ERROR_FRAME = {"error": {"message": "upstream generation failed", "type": "server_error"}}


class TestMidStreamErrorFrame:
    """A provider failure that arrives INSIDE an already-200 stream.

    The status guard cannot catch this: the provider accepted the request,
    flushed 200 headers, streamed content, and only then failed, signalling it as
    a `data: {"error": ...}` frame. That frame carries no `choices`, so the
    choices check swallowed it and the generator returned normally -- a truncated
    answer delivered to the caller as a clean completion, on both the plain and
    the tool leg. Reproduced against a real socket with real httpx before this
    class existed; the Anthropic SDK already raises on the same event.
    """

    def _client(self, mock_cls: MagicMock, lines: list[str]) -> OpenAICompatibleClient:
        mock_cls.return_value.stream.return_value = _FakeSSEResponse(lines)
        return OpenAICompatibleClient(
            "openai/gpt-5.5",
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            label="OpenRouter",
        )

    @staticmethod
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}"

    def _content(self, text: str) -> dict:
        return {"choices": [{"index": 0, "delta": {"content": text}}]}

    def _plain(self, client: OpenAICompatibleClient) -> object:
        return client.stream_complete_multi("sys", [{"role": "user", "content": "x"}])

    def _tools(self, client: OpenAICompatibleClient) -> object:
        return client.stream_with_tools("sys", [{"role": "user", "content": "x"}], _TOOLS)

    # -- plain leg --

    def test_plain_error_frame_after_content_raises(self) -> None:
        """The shape that matters: the caller must not be handed a partial answer
        as a finished one."""
        lines = [self._sse(self._content("The answer ")), self._sse(_ERROR_FRAME)]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            with pytest.raises(UpstreamStreamError, match="upstream generation failed"):
                _collect(self._plain(client))

    def test_plain_error_frame_before_any_content_raises(self) -> None:
        """Otherwise the turn arrives as an empty but successful stream."""
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, [self._sse(_ERROR_FRAME)])
            with pytest.raises(UpstreamStreamError):
                _collect(self._plain(client))

    def test_plain_error_frame_wins_over_a_following_done(self) -> None:
        """A provider that closes politely still failed; [DONE] after an error
        frame must not launder the turn into a clean finish."""
        lines = [
            self._sse(self._content("partial")),
            self._sse(_ERROR_FRAME),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            with pytest.raises(UpstreamStreamError):
                _collect(self._plain(client))

    def test_plain_error_as_a_bare_string_raises_with_that_text(self) -> None:
        """Not every provider wraps the error in an object."""
        lines = [self._sse({"error": "model overloaded"})]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            with pytest.raises(UpstreamStreamError, match="model overloaded"):
                _collect(self._plain(client))

    def test_plain_null_error_key_on_a_normal_chunk_does_not_raise(self) -> None:
        """Over-trigger guard: providers send "error": null on healthy chunks, so
        the check has to be on truthiness, not on the key being present."""
        lines = [
            self._sse({"error": None, "choices": [{"index": 0, "delta": {"content": "ok"}}]}),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            assert _collect(self._plain(client)) == ["ok"]

    def test_plain_chunk_with_neither_choices_nor_error_is_still_skipped(self) -> None:
        """A usage-only trailer is not a failure and must stay ignored."""
        lines = [
            self._sse(self._content("hi")),
            self._sse({"usage": {"prompt_tokens": 1, "completion_tokens": 2}}),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            assert _collect(self._plain(client)) == ["hi"]

    # -- tool leg --

    def test_tool_error_frame_after_a_fragment_raises(self) -> None:
        lines = [
            self._sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "run", "arguments": '{"cm'},
                                    }
                                ]
                            },
                        }
                    ]
                }
            ),
            self._sse(_ERROR_FRAME),
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            with pytest.raises(UpstreamStreamError, match="upstream generation failed"):
                _collect(self._tools(client))

    def test_tool_error_frame_before_any_fragment_raises(self) -> None:
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, [self._sse(_ERROR_FRAME)])
            with pytest.raises(UpstreamStreamError):
                _collect(self._tools(client))

    def test_tool_null_error_key_does_not_raise(self) -> None:
        lines = [
            self._sse({"error": None, "choices": [{"index": 0, "delta": {"content": "ok"}}]}),
            self._sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, lines)
            assert _collect(self._tools(client)) == ["ok", StreamFinish("stop")]

    def test_error_frame_names_the_provider(self) -> None:
        """The proxy surfaces this string to the caller, so it has to say which
        arm failed rather than just that something did."""
        with patch("httpx.Client"), patch("httpx.AsyncClient") as mock_cls:
            client = self._client(mock_cls, [self._sse(_ERROR_FRAME)])
            with pytest.raises(UpstreamStreamError, match="OpenRouter"):
                _collect(self._plain(client))


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
