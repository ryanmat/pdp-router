# Description: Tests for the tool-passthrough value types and the /openai/v1 tool schemas.
# Description: Schema-level only -- the endpoint behavior these back lives in test_proxy.py.

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("fastapi")

from pydantic import ValidationError

from pdp_router._clients import CompletionResult
from pdp_router._proxy import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCallSpec,
    LenientToolChatCompletionRequest,
    LenientToolChatMessage,
    ToolCallSpec,
    ToolChatCompletionChoice,
    ToolChatCompletionRequest,
    ToolChatCompletionResponse,
    ToolChatMessage,
    ToolResponseMessage,
    Usage,
)
from pdp_router._tools import StreamFinish, ToolCall, ToolCallDelta


def _tool_call_spec(name: str = "run", arguments: str = '{"cmd": "ls"}') -> ToolCallSpec:
    return ToolCallSpec(id="call_1", function=FunctionCallSpec(name=name, arguments=arguments))


class TestToolValueTypes:
    """The frozen types in _tools.py that cross the client boundary.

    They are stdlib dataclasses, not pydantic: stream_with_tools yields one
    ToolCallDelta per argument fragment, so validating values the proxy built
    itself from already-validated SDK objects would be cost with no benefit.
    """

    def test_tool_call_keeps_arguments_as_a_raw_string(self) -> None:
        """Arguments are never parsed at this layer -- ids and payloads pass through
        unrewritten, which invariant 6 requires."""
        call = ToolCall(id="toolu_abc", name="run", arguments='{"cmd": "ls"}')
        assert call.id == "toolu_abc"
        assert call.name == "run"
        assert call.arguments == '{"cmd": "ls"}'
        assert isinstance(call.arguments, str)

    def test_tool_call_is_frozen(self) -> None:
        call = ToolCall(id="toolu_abc", name="run", arguments="{}")
        with pytest.raises(AttributeError):
            call.id = "toolu_xyz"  # type: ignore[misc]

    def test_tool_call_delta_defaults_id_and_name_to_none(self) -> None:
        """Only the first fragment of a call carries id/name; the rest are
        argument text against the same index."""
        first = ToolCallDelta(index=0, id="call_1", name="run", arguments="")
        later = ToolCallDelta(index=0, arguments='{"cmd')
        assert (first.id, first.name) == ("call_1", "run")
        assert (later.id, later.name) == (None, None)
        assert later.index == 0
        assert later.arguments == '{"cmd'

    def test_tool_call_delta_is_frozen(self) -> None:
        delta = ToolCallDelta(index=0, arguments="{}")
        with pytest.raises(AttributeError):
            delta.index = 1  # type: ignore[misc]

    def test_stream_finish_carries_the_finish_reason(self) -> None:
        """The terminal reason is yielded rather than inferred, because the
        with-tools stream cannot assume "stop" the way the plain path does."""
        assert StreamFinish(finish_reason="tool_calls").finish_reason == "tool_calls"

    def test_stream_finish_is_frozen(self) -> None:
        finish = StreamFinish(finish_reason="stop")
        with pytest.raises(AttributeError):
            finish.finish_reason = "length"  # type: ignore[misc]


class TestCompletionResultWidening:
    """CompletionResult gains tool fields without disturbing any caller.

    Every construction site in the package, its tests, and both downstream
    consumers is keyword-only, so defaulted fields appended last are invisible
    to all of them.
    """

    def test_existing_construction_signature_still_works(self) -> None:
        result = CompletionResult(
            text="hello",
            input_tokens=10,
            output_tokens=5,
            model="test",
            estimated_cost_usd=0.001,
        )
        assert result.text == "hello"
        assert result.tool_calls == ()
        assert result.finish_reason == "stop"

    def test_tool_fields_round_trip_when_populated(self) -> None:
        call = ToolCall(id="call_1", name="run", arguments='{"cmd": "ls"}')
        result = CompletionResult(
            text="",
            input_tokens=10,
            output_tokens=5,
            model="test",
            estimated_cost_usd=0.001,
            tool_calls=(call,),
            finish_reason="tool_calls",
        )
        assert result.tool_calls == (call,)
        assert result.finish_reason == "tool_calls"

    def test_new_fields_are_last_and_defaulted(self) -> None:
        """Guards the ordering that makes the widening safe: a new field placed
        before an existing defaulted one would break every positional caller and
        reorder the dataclass signature."""
        fields = [f.name for f in dataclasses.fields(CompletionResult)]
        assert fields[-2:] == ["tool_calls", "finish_reason"]
        assert fields[:6] == [
            "text",
            "input_tokens",
            "output_tokens",
            "model",
            "estimated_cost_usd",
            "web_search_requests",
        ]


class TestToolChatMessage:
    """The request-side message model for /openai/v1 only."""

    def test_plain_role_content_turn_parses(self) -> None:
        message = ToolChatMessage(role="user", content="hi")
        assert (message.role, message.content) == ("user", "hi")
        assert message.tool_calls is None

    def test_assistant_tool_calls_turn_with_content_omitted(self) -> None:
        """OpenAI clients omit the content key entirely on a tool_calls turn
        rather than sending null, so the field has to be defaulted, not just
        nullable."""
        message = ToolChatMessage.model_validate(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            }
        )
        assert message.content is None
        assert message.tool_calls is not None
        assert message.tool_calls[0].id == "call_1"

    def test_tool_result_turn_parses(self) -> None:
        message = ToolChatMessage(role="tool", content="file-a\nfile-b", tool_call_id="call_1")
        assert message.tool_call_id == "call_1"
        assert message.content == "file-a\nfile-b"

    def test_content_none_without_tool_calls_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolChatMessage(role="user", content=None)

    def test_content_none_with_tool_calls_is_accepted(self) -> None:
        message = ToolChatMessage(role="assistant", content=None, tool_calls=[_tool_call_spec()])
        assert message.content is None

    def test_rejection_is_message_level_and_names_content(self) -> None:
        """Pins the half of the flag-off 422 contract that survives.

        The validator is model-level, so pydantic reports loc ("messages", 0)
        rather than the field, and _openai_validation_handler folds that into
        "messages.0: ...". The status and the envelope shape are the contract;
        the message string is not, but it must still name the offending field
        or the folded body stops being debuggable.
        """
        with pytest.raises(ValidationError) as excinfo:
            ToolChatCompletionRequest(
                model="pdp-auto", messages=[ToolChatMessage.model_construct(role="user")]
            )
        error = excinfo.value.errors()[0]
        assert error["loc"] == ("messages", 0)
        assert "content" in error["msg"]


class TestToolCallSpec:
    """The faithful OpenAI tool-call shape. Arguments are transported, not parsed."""

    def test_arguments_stay_a_raw_string(self) -> None:
        spec = _tool_call_spec(arguments='{"cmd": "ls -la"}')
        assert spec.function.arguments == '{"cmd": "ls -la"}'
        assert isinstance(spec.function.arguments, str)

    def test_malformed_argument_json_is_not_rejected_here(self) -> None:
        """The schema layer never parses arguments, so a truncated payload (a
        max_tokens cut mid-JSON) survives to the translation layer, which owns
        that error."""
        spec = _tool_call_spec(arguments='{"cmd": "l')
        assert spec.function.arguments == '{"cmd": "l'

    def test_type_defaults_to_function(self) -> None:
        assert _tool_call_spec().type == "function"

    def test_function_call_spec_carries_name_and_arguments(self) -> None:
        function = FunctionCallSpec(name="run", arguments="{}")
        assert (function.name, function.arguments) == ("run", "{}")


class TestToolChatCompletionRequest:
    """The request model /openai/v1 will bind in prompt 6."""

    def test_tools_survive_round_trip_verbatim(self) -> None:
        """JSON-Schema internals are never modelled -- an arbitrary nested schema
        has to come back out unchanged."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string", "minLength": 1}},
                        "required": ["cmd"],
                    },
                },
            }
        ]
        request = ToolChatCompletionRequest(
            model="pdp-auto", messages=[ToolChatMessage(role="user", content="hi")], tools=tools
        )
        assert request.model_dump()["tools"] == tools

    def test_tool_choice_accepts_both_the_string_and_dict_forms(self) -> None:
        base = {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]}
        plain = ToolChatCompletionRequest.model_validate({**base, "tool_choice": "auto"})
        assert plain.tool_choice == "auto"

        named = {"type": "function", "function": {"name": "run"}}
        forced = ToolChatCompletionRequest.model_validate({**base, "tool_choice": named})
        assert forced.tool_choice == named

    def test_tool_fields_default_to_none(self) -> None:
        request = ToolChatCompletionRequest(
            model="pdp-auto", messages=[ToolChatMessage(role="user", content="hi")]
        )
        assert request.tools is None
        assert request.tool_choice is None
        assert request.parallel_tool_calls is None

    def test_base_request_defaults_are_inherited(self) -> None:
        request = ToolChatCompletionRequest(messages=[ToolChatMessage(role="user", content="hi")])
        assert (request.model, request.max_tokens, request.stream) == ("pdp-auto", 4096, False)

    def test_unknown_fields_are_still_ignored(self) -> None:
        """Crush sends temperature/top_p/stream_options. Widening the endpoint type
        must not start rejecting them, so the subclass stays leaf-only with no
        ConfigDict anywhere in the chain."""
        request = ToolChatCompletionRequest.model_validate(
            {
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
                "stream_options": {"include_usage": True},
            }
        )
        assert request.model_extra is None


# A null id: valid JSON, refused by ToolCallSpec, dropped as an extra by the
# legacy models. One of the shapes whose flag-off 200 the lenient bind restores.
_MALFORMED_CALL = {"id": None, "type": "function", "function": {"name": "run"}}


class TestLenientToolModels:
    """The models /openai/v1 actually binds, and the two-pass design they enable.

    Tool validation cannot sit at the endpoint annotation: FastAPI's dependency
    layer runs before the passthrough flag is read, so a malformed tool field
    would be refused even with the flag off, where it has always been dropped and
    the request served. The lenient models carry those fields; the strict models
    above are applied afterwards, behind the flag.
    """

    def test_lenient_message_carries_a_malformed_tool_call_verbatim(self) -> None:
        message = LenientToolChatMessage.model_validate(
            {"role": "assistant", "content": "thinking", "tool_calls": [_MALFORMED_CALL]}
        )
        assert message.tool_calls == [_MALFORMED_CALL]

    def test_strict_message_still_refuses_the_same_call(self) -> None:
        """The strictness moved rather than vanished."""
        with pytest.raises(ValidationError):
            ToolChatMessage.model_validate(
                {"role": "assistant", "content": "thinking", "tool_calls": [_MALFORMED_CALL]}
            )

    def test_lenient_request_carries_malformed_tool_params_verbatim(self) -> None:
        request = LenientToolChatCompletionRequest.model_validate(
            {
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": {"not": "a list"},
                "tool_choice": True,
                "parallel_tool_calls": "maybe",
            }
        )
        assert request.tools == {"not": "a list"}
        assert request.tool_choice is True
        assert request.parallel_tool_calls == "maybe"

    def test_lenient_message_still_requires_content_or_tool_calls(self) -> None:
        """Flag-off contract shape 1 rests on this validator, so it has to live on
        the lenient model -- the strict one is never reached with the flag off."""
        with pytest.raises(ValidationError):
            LenientToolChatMessage(role="user", content=None)

    def test_lenient_request_still_ignores_unknown_fields(self) -> None:
        request = LenientToolChatCompletionRequest.model_validate(
            {
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.7,
            }
        )
        assert request.model_extra is None

    def test_the_two_passes_agree_on_a_canonical_payload(self) -> None:
        """What the whole design rests on: routing a request through the lenient
        bind first must hand the tools path exactly what a direct strict parse
        would have. If these ever diverge, the flag-on surface has drifted."""
        payload = {
            "model": "pdp-auto",
            "messages": [
                {"role": "user", "content": "list files"},
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
                {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
            ],
            "tools": [{"type": "function", "function": {"name": "run"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        direct = ToolChatCompletionRequest.model_validate(payload)
        two_pass = ToolChatCompletionRequest.model_validate(
            LenientToolChatCompletionRequest.model_validate(payload).model_dump()
        )

        assert two_pass == direct

    def test_strict_request_is_a_subclass_so_downstream_reads_are_unchanged(self) -> None:
        """_handle_chat rebinds the request to the strict model mid-flight; every
        field the rest of the handler reads has to be there afterwards."""
        assert issubclass(ToolChatCompletionRequest, LenientToolChatCompletionRequest)
        assert issubclass(ToolChatMessage, LenientToolChatMessage)


class TestToolResponseSerialization:
    """The response side, where pydantic's annotation-driven serialization bites."""

    def _response(
        self, message: ToolResponseMessage, finish_reason: str
    ) -> ToolChatCompletionResponse:
        choice = ToolChatCompletionChoice(
            index=0, message=message, finish_reason=finish_reason
        )
        return ToolChatCompletionResponse(
            id="chatcmpl-abc123",
            object="chat.completion",
            created=1700000000,
            model="claude-sonnet-4-6",
            choices=[choice],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def test_tool_calls_survive_the_top_level_response_dump(self) -> None:
        """The reason there are three response classes and not one.

        Pydantic serializes a field through its ANNOTATED type, so a
        ToolResponseMessage left in a `message: ChatMessage` slot dumps as
        {"role": ..., "content": null} and loses tool_calls silently -- no
        error, no warning. Asserting on the message alone passes either way,
        and so does asserting on the choice alone. Only the top-level dump
        tells a correct response model from a broken one, which is exactly
        how this would otherwise have reached Crush as a turn with no tools.
        """
        message = ToolResponseMessage(
            role="assistant", content=None, tool_calls=[_tool_call_spec()]
        )
        dumped = self._response(message, "tool_calls").model_dump()

        assert dumped["choices"][0]["message"]["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
            }
        ]
        assert dumped["choices"][0]["message"]["content"] is None
        assert dumped["choices"][0]["finish_reason"] == "tool_calls"

    def test_text_and_tool_calls_coexist_on_one_turn(self) -> None:
        message = ToolResponseMessage(
            role="assistant", content="Let me look.", tool_calls=[_tool_call_spec()]
        )
        dumped = self._response(message, "tool_calls").model_dump()
        assert dumped["choices"][0]["message"]["content"] == "Let me look."
        assert len(dumped["choices"][0]["message"]["tool_calls"]) == 1

    def test_plain_text_turn_still_serializes(self) -> None:
        message = ToolResponseMessage(role="assistant", content="done")
        dumped = self._response(message, "stop").model_dump()
        assert dumped["choices"][0]["message"]["content"] == "done"
        assert dumped["choices"][0]["message"]["tool_calls"] is None
        assert dumped["usage"]["total_tokens"] == 15


class TestLegacyModelsUnchanged:
    """/v1 keeps the original models forever (invariant 1)."""

    def test_chat_message_still_rejects_none_content(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content=None)

    def test_chat_completion_request_still_drops_tool_fields(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "run"}}],
                "tool_choice": "auto",
            }
        )
        assert request.model_extra is None
        assert not hasattr(request, "tools")

    def test_legacy_message_list_does_not_accept_tool_shaped_turns(self) -> None:
        """A tool result posted to /v1 fails at the schema boundary, so the legacy
        path can never receive one."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate(
                {"model": "pdp-auto", "messages": [{"role": "tool", "tool_call_id": "call_1"}]}
            )
