# Description: Value types for OpenAI-compatible tool calling on the faithful surface.
# Description: Frozen and dependency-free so both the clients and the proxy can import them.

from __future__ import annotations

import json
from dataclasses import dataclass

# This module must stay a leaf: it imports nothing from the rest of the package.
# _clients imports ToolCall from here for CompletionResult, and the package
# __init__ imports _clients early, so an import back the other way would surface
# as an ImportError on every test and on the next service start.


@dataclass(frozen=True)
class ToolCall:
    """One completed tool call the model asked for."""

    # Provider-assigned identifier. Passed through byte-exact in both directions:
    # a tool result is matched to its call by this id, so rewriting it silently
    # breaks the second round trip.
    id: str
    name: str
    # Raw JSON text, never parsed here. Transporting it verbatim keeps key order
    # and any truncation the provider produced, both of which the client must see.
    arguments: str


@dataclass(frozen=True)
class ToolCallDelta:
    """One fragment of a tool call arriving on a stream."""

    # Which call this fragment belongs to. Parallel calls interleave, so the
    # index is the only thing tying fragments back together.
    index: int
    # Present on the first fragment of a call and None on every later one, which
    # is how a provider announces a new call mid-stream.
    id: str | None = None
    name: str | None = None
    # Partial JSON. Fragments concatenate to the complete argument string; no
    # single fragment is required to parse on its own.
    arguments: str = ""


@dataclass(frozen=True)
class StreamUsage:
    """Final token usage a provider reported for one streamed turn.

    Cache fields are Anthropic-only today; providers without prompt caching
    leave them zero. Consumed server-side for cost accounting -- never
    serialized onto the SSE wire.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class StreamFinish:
    """The terminal reason a with-tools stream ended."""

    # Yielded rather than inferred. The plain streaming path can hardcode "stop";
    # a tool turn ends in "tool_calls", and a truncated one in "length".
    finish_reason: str
    # None means the provider reported no usage for this turn -- tellable from
    # a reported all-zero StreamUsage. Defaulted so every existing literal
    # (StreamFinish("stop")) keeps its meaning and equality.
    usage: StreamUsage | None = None


class ToolTranslationError(ValueError):
    """An OpenAI-shaped tool payload that cannot be expressed to Anthropic."""


# -- OpenAI <-> Anthropic translation --
#
# Pure functions over plain dicts. They deliberately do not take the proxy's
# pydantic models: keeping the signatures at dict level is what lets this module
# stay a leaf, and lets the translation be tested without fastapi installed.
# System turns are split off upstream and never reach here.

_STOP_REASONS = {
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
}


def anthropic_stop_reason_to_finish_reason(stop_reason: str | None) -> str:
    """Map an Anthropic stop_reason onto an OpenAI finish_reason.

    Unknown reasons fall back to "stop" rather than raising. A provider adding a
    stop_reason should not take the surface down, and the turn did end.
    """
    return _STOP_REASONS.get(stop_reason or "", "stop")


def openai_tools_to_anthropic(
    tools: list[dict] | None,
    tool_choice: str | dict | None = None,
    parallel_tool_calls: bool | None = None,
) -> tuple[list[dict] | None, dict | None]:
    """Translate OpenAI tool definitions and choice into the Anthropic pair.

    Returns (tools, tool_choice), either of which may be None meaning "do not
    send this key at all". With no tools at all, both are None. tool_choice
    "none" keeps the tools and emits Anthropic's {"type": "none"}: withholding
    the tools instead would strip the definitions while a transcript's own
    tool_use / tool_result blocks stayed in the body, which Anthropic rejects --
    the exact shape the tool-history guard exists to prevent.
    """
    if not tools:
        return None, None

    translated = []
    for tool in tools:
        function = tool.get("function") or {}
        name = function.get("name")
        if not name:
            raise ToolTranslationError("every tool needs a function.name")
        entry: dict = {"name": name}
        description = function.get("description")
        if description is not None:
            entry["description"] = description
        # input_schema is required even for a no-argument tool.
        entry["input_schema"] = function.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        translated.append(entry)

    if tool_choice == "none":
        # Verified against anthropic 0.89.0 (ToolChoiceNoneParam): the model is
        # not allowed to use tools this turn. disable_parallel_tool_use does not
        # apply when no call may happen, so it is not attached here.
        return translated, {"type": "none"}

    choice = _translate_tool_choice(tool_choice)
    if parallel_tool_calls is False:
        choice = {**(choice or {"type": "auto"}), "disable_parallel_tool_use": True}
    return translated, choice


def _translate_tool_choice(tool_choice: str | dict | None) -> dict | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if not name:
            raise ToolTranslationError("a named tool_choice needs function.name")
        return {"type": "tool", "name": name}
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "required":
        return {"type": "any"}
    raise ToolTranslationError(f"unsupported tool_choice {tool_choice!r}")


def openai_messages_to_anthropic(messages: list[dict]) -> list[dict]:
    """Translate an OpenAI transcript into Anthropic turns.

    Two shape changes carry the work. An assistant turn with tool_calls becomes a
    block list (optional text, then one tool_use per call). Consecutive
    role:"tool" results collapse into ONE user turn of tool_result blocks, which
    Anthropic requires -- one turn per result is a validation error -- while a
    result followed by anything else stays its own turn.
    """
    turns: list[dict] = []
    pending: list[dict] = []

    def flush() -> None:
        if pending:
            turns.append({"role": "user", "content": list(pending)})
            pending.clear()

    for message in messages:
        role = message.get("role")

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                raise ToolTranslationError("a tool result needs a tool_call_id")
            pending.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": message.get("content") or "",
                }
            )
            continue

        flush()
        if role not in ("user", "assistant"):
            raise ToolTranslationError(f"unsupported message role {role!r}")

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            turns.append({"role": role, "content": message.get("content") or ""})
            continue

        blocks = []
        text = message.get("content")
        # Only when non-empty: an empty text block is a provider-side error, and
        # the omitted-content tool turn is the common case, not the exception.
        if text:
            blocks.append({"type": "text", "text": text})
        blocks.extend(_tool_use_block(call) for call in tool_calls)
        turns.append({"role": role, "content": blocks})

    flush()
    return turns


def _tool_use_block(call: dict) -> dict:
    call_id = call.get("id")
    function = call.get("function") or {}
    name = function.get("name")
    if not call_id or not name:
        raise ToolTranslationError("a tool call needs an id and a function.name")

    raw = function.get("arguments") or ""
    if not raw.strip():
        # A no-argument call is commonly serialised as "" rather than "{}".
        arguments: object = {}
    else:
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolTranslationError(
                f"tool call {call_id} has unparseable arguments: {exc}"
            ) from exc
    if not isinstance(arguments, dict):
        raise ToolTranslationError(f"tool call {call_id} arguments must be a JSON object")

    # id byte-exact: a result is bound to its call by this value.
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
