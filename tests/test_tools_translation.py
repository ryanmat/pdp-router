# Description: Tests for the OpenAI-to-Anthropic tool translation functions.
# Description: Pure functions over plain dicts -- no proxy, no client, no network.

from __future__ import annotations

import pytest

from pdp_router._tools import (
    ToolTranslationError,
    anthropic_stop_reason_to_finish_reason,
    openai_messages_to_anthropic,
    openai_tools_to_anthropic,
)

_RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}

_READ_TOOL = {
    "type": "function",
    "function": {"name": "read", "description": "read a file", "parameters": {"type": "object"}},
}


def _tool_call(
    call_id: str = "call_1", name: str = "run", arguments: str = '{"cmd": "ls"}'
) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class TestToolDefinitionTranslation:
    """OpenAI's nested function envelope flattens into Anthropic's tool shape."""

    def test_name_description_and_parameters_map_to_input_schema(self) -> None:
        tools, _ = openai_tools_to_anthropic([_RUN_TOOL])
        assert tools == [
            {
                "name": "run",
                "description": "run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            }
        ]

    def test_multiple_tools_keep_their_order(self) -> None:
        tools, _ = openai_tools_to_anthropic([_RUN_TOOL, _READ_TOOL])
        assert [t["name"] for t in tools] == ["run", "read"]

    def test_missing_description_is_omitted_not_nulled(self) -> None:
        """Anthropic treats description as optional; sending an explicit null is
        a validation error rather than a no-op."""
        tools, _ = openai_tools_to_anthropic(
            [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]
        )
        assert tools == [{"name": "ping", "input_schema": {"type": "object"}}]

    def test_missing_parameters_becomes_an_empty_object_schema(self) -> None:
        """input_schema is required, so a no-argument tool still needs a schema."""
        tools, _ = openai_tools_to_anthropic(
            [{"type": "function", "function": {"name": "ping"}}]
        )
        assert tools[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_nested_schema_internals_pass_through_untouched(self) -> None:
        deep = {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "properties": {"depth": {"type": "integer", "minimum": 0}},
                }
            },
            "additionalProperties": False,
        }
        tools, _ = openai_tools_to_anthropic(
            [{"type": "function", "function": {"name": "deep", "parameters": deep}}]
        )
        assert tools[0]["input_schema"] == deep

    def test_no_tools_translates_to_nothing(self) -> None:
        assert openai_tools_to_anthropic(None) == (None, None)
        assert openai_tools_to_anthropic([]) == (None, None)

    def test_a_tool_without_a_name_is_rejected(self) -> None:
        with pytest.raises(ToolTranslationError):
            openai_tools_to_anthropic([{"type": "function", "function": {"description": "x"}}])


class TestToolChoiceTranslation:
    """auto/required/named/none map onto Anthropic's four behaviors."""

    @pytest.mark.parametrize(
        ("openai_choice", "expected"),
        [
            (None, None),
            ("auto", {"type": "auto"}),
            ("required", {"type": "any"}),
        ],
    )
    def test_string_forms(self, openai_choice, expected) -> None:
        _, choice = openai_tools_to_anthropic([_RUN_TOOL], tool_choice=openai_choice)
        assert choice == expected

    def test_named_choice_forces_that_tool(self) -> None:
        _, choice = openai_tools_to_anthropic(
            [_RUN_TOOL], tool_choice={"type": "function", "function": {"name": "run"}}
        )
        assert choice == {"type": "tool", "name": "run"}

    def test_none_keeps_the_tools_and_forbids_a_call(self) -> None:
        """A "none" choice keeps the tool definitions and maps to Anthropic's
        {"type": "none"} (SDK 0.89.0 ToolChoiceNoneParam). Stripping the tools
        would leave any tool_use/tool_result blocks in the transcript orphaned,
        which the provider rejects."""
        tools, choice = openai_tools_to_anthropic([_RUN_TOOL], tool_choice="none")
        assert tools is not None
        assert len(tools) == 1
        assert choice == {"type": "none"}

    def test_none_with_no_tools_sends_nothing(self) -> None:
        assert openai_tools_to_anthropic(None, tool_choice="none") == (None, None)
        assert openai_tools_to_anthropic([], tool_choice="none") == (None, None)

    def test_none_over_a_tool_history_never_orphans_the_blocks(self) -> None:
        """The regression this fix closes: a "none" turn that carries prior tool
        calls must not send tool_use/tool_result blocks with the tools param
        stripped. Translating both halves together, the tools survive whenever
        the transcript still carries tool blocks."""
        tools, choice = openai_tools_to_anthropic([_RUN_TOOL], tool_choice="none")
        turns = openai_messages_to_anthropic(
            [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "run", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "a.txt"},
                {"role": "user", "content": "now just summarize, no tools"},
            ]
        )
        has_tool_blocks = any(
            isinstance(t.get("content"), list)
            and any(b.get("type") in ("tool_use", "tool_result") for b in t["content"])
            for t in turns
        )
        assert has_tool_blocks
        assert tools is not None  # blocks are never sent without their definitions
        assert choice == {"type": "none"}

    def test_parallel_tool_calls_false_disables_parallel_use(self) -> None:
        _, choice = openai_tools_to_anthropic(
            [_RUN_TOOL], tool_choice="auto", parallel_tool_calls=False
        )
        assert choice == {"type": "auto", "disable_parallel_tool_use": True}

    def test_parallel_tool_calls_false_without_a_choice_still_applies(self) -> None:
        """The flag rides on the choice object, so suppressing parallel calls
        requires materialising the default choice."""
        _, choice = openai_tools_to_anthropic([_RUN_TOOL], parallel_tool_calls=False)
        assert choice == {"type": "auto", "disable_parallel_tool_use": True}

    def test_parallel_tool_calls_true_is_the_default_and_adds_nothing(self) -> None:
        _, choice = openai_tools_to_anthropic(
            [_RUN_TOOL], tool_choice="auto", parallel_tool_calls=True
        )
        assert choice == {"type": "auto"}

    def test_an_unknown_choice_string_is_rejected(self) -> None:
        with pytest.raises(ToolTranslationError):
            openai_tools_to_anthropic([_RUN_TOOL], tool_choice="sometimes")


class TestMessageTranslation:
    """Transcript translation. The grouping rules are the fiddly part."""

    def test_plain_turns_pass_through(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert openai_messages_to_anthropic(messages) == messages

    def test_assistant_tool_calls_turn_becomes_tool_use_blocks(self) -> None:
        result = openai_messages_to_anthropic(
            [{"role": "assistant", "content": None, "tool_calls": [_tool_call()]}]
        )
        assert result == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "run",
                        "input": {"cmd": "ls"},
                    }
                ],
            }
        ]

    def test_tool_call_ids_are_byte_exact(self) -> None:
        """A result is bound to its call by id. Rewriting one breaks the second
        round trip in a way the provider reports as an unmatched tool_use."""
        result = openai_messages_to_anthropic(
            [{"role": "assistant", "tool_calls": [_tool_call(call_id="call_AbC-123_xyz")]}]
        )
        assert result[0]["content"][0]["id"] == "call_AbC-123_xyz"

    def test_text_and_tool_calls_become_a_text_block_then_tool_use_blocks(self) -> None:
        result = openai_messages_to_anthropic(
            [{"role": "assistant", "content": "Let me look.", "tool_calls": [_tool_call()]}]
        )
        assert result[0]["content"][0] == {"type": "text", "text": "Let me look."}
        assert result[0]["content"][1]["type"] == "tool_use"

    def test_parallel_tool_calls_keep_their_order(self) -> None:
        calls = [_tool_call("call_1", "run"), _tool_call("call_2", "read", '{"path": "a"}')]
        result = openai_messages_to_anthropic([{"role": "assistant", "tool_calls": calls}])
        assert [b["id"] for b in result[0]["content"]] == ["call_1", "call_2"]
        assert result[0]["content"][1]["input"] == {"path": "a"}

    def test_consecutive_tool_results_collapse_into_one_user_turn(self) -> None:
        """Anthropic requires every tool_result for a turn in a single user
        message; emitting one turn per result is a validation error."""
        result = openai_messages_to_anthropic(
            [
                {"role": "tool", "tool_call_id": "call_1", "content": "a-out"},
                {"role": "tool", "tool_call_id": "call_2", "content": "b-out"},
            ]
        )
        assert result == [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "a-out"},
                    {"type": "tool_result", "tool_use_id": "call_2", "content": "b-out"},
                ],
            }
        ]

    def test_a_tool_result_followed_by_user_text_stays_two_turns(self) -> None:
        result = openai_messages_to_anthropic(
            [
                {"role": "tool", "tool_call_id": "call_1", "content": "a-out"},
                {"role": "user", "content": "thanks"},
            ]
        )
        assert len(result) == 2
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[1] == {"role": "user", "content": "thanks"}

    def test_a_full_tool_round_trip_translates(self) -> None:
        result = openai_messages_to_anthropic(
            [
                {"role": "user", "content": "list the files"},
                {"role": "assistant", "tool_calls": [_tool_call()]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a\nb"},
                {"role": "user", "content": "and the first one?"},
            ]
        )
        assert [m["role"] for m in result] == ["user", "assistant", "user", "user"]
        assert result[1]["content"][0]["type"] == "tool_use"
        assert result[2]["content"][0]["type"] == "tool_result"

    def test_a_separated_pair_of_tool_results_does_not_merge(self) -> None:
        """Only ADJACENT results collapse; merging across an intervening turn
        would reorder the transcript."""
        result = openai_messages_to_anthropic(
            [
                {"role": "tool", "tool_call_id": "call_1", "content": "a"},
                {"role": "assistant", "content": "thinking"},
                {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            ]
        )
        assert len(result) == 3
        assert result[0]["content"][0]["tool_use_id"] == "call_1"
        assert result[2]["content"][0]["tool_use_id"] == "call_2"

    def test_omitted_content_on_a_tool_calls_turn_is_not_an_empty_text_block(self) -> None:
        """An empty text block is a provider-side validation error, so the
        omitted-content case must produce tool_use blocks only."""
        result = openai_messages_to_anthropic(
            [{"role": "assistant", "tool_calls": [_tool_call()]}]
        )
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_use"

    def test_empty_string_content_is_also_dropped(self) -> None:
        result = openai_messages_to_anthropic(
            [{"role": "assistant", "content": "", "tool_calls": [_tool_call()]}]
        )
        assert [b["type"] for b in result[0]["content"]] == ["tool_use"]

    def test_arguments_are_parsed_into_input(self) -> None:
        result = openai_messages_to_anthropic(
            [{"role": "assistant", "tool_calls": [_tool_call(arguments='{"a": 1, "b": [2, 3]}')]}]
        )
        assert result[0]["content"][0]["input"] == {"a": 1, "b": [2, 3]}

    def test_empty_arguments_string_becomes_an_empty_input(self) -> None:
        """A no-argument call is commonly serialised as "" rather than "{}"."""
        result = openai_messages_to_anthropic(
            [{"role": "assistant", "tool_calls": [_tool_call(arguments="")]}]
        )
        assert result[0]["content"][0]["input"] == {}

    def test_malformed_arguments_json_is_rejected(self) -> None:
        """Truncated argument JSON (a max_tokens cut mid-object) must fail here
        and loudly, not reach the provider as an unparseable tool_use."""
        with pytest.raises(ToolTranslationError) as excinfo:
            openai_messages_to_anthropic(
                [{"role": "assistant", "tool_calls": [_tool_call(arguments='{"cmd": "l')]}]
            )
        assert "call_1" in str(excinfo.value)

    def test_non_object_arguments_json_is_rejected(self) -> None:
        with pytest.raises(ToolTranslationError):
            openai_messages_to_anthropic(
                [{"role": "assistant", "tool_calls": [_tool_call(arguments="[1, 2]")]}]
            )

    def test_a_tool_result_without_a_call_id_is_rejected(self) -> None:
        with pytest.raises(ToolTranslationError):
            openai_messages_to_anthropic([{"role": "tool", "content": "orphan"}])

    def test_an_unknown_role_is_rejected(self) -> None:
        """System turns are split off upstream, so anything unrecognised here is
        a caller error worth surfacing rather than forwarding."""
        with pytest.raises(ToolTranslationError):
            openai_messages_to_anthropic([{"role": "developer", "content": "x"}])

    def test_input_is_not_shared_between_translated_calls(self) -> None:
        """Guards against a shared-default bug: mutating one call's input must
        not reach another."""
        calls = [_tool_call("call_1", "ping", ""), _tool_call("call_2", "ping", "")]
        result = openai_messages_to_anthropic([{"role": "assistant", "tool_calls": calls}])
        result[0]["content"][0]["input"]["x"] = 1
        assert result[0]["content"][1]["input"] == {}


class TestStopReasonMapping:
    """Anthropic stop_reason onto OpenAI finish_reason."""

    @pytest.mark.parametrize(
        ("stop_reason", "expected"),
        [
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("something_new", "stop"),
            (None, "stop"),
        ],
    )
    def test_mapping(self, stop_reason, expected) -> None:
        assert anthropic_stop_reason_to_finish_reason(stop_reason) == expected

    def test_unknown_reasons_default_to_stop_rather_than_raising(self) -> None:
        """A provider adding a stop_reason must not take the surface down; the
        turn did end, and "stop" is the honest fallback."""
        assert anthropic_stop_reason_to_finish_reason("refusal") == "stop"
