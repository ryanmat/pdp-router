# Description: Tests for OpenAI-compatible SSE streaming in the PDP Router Proxy.
# Description: Covers the default route_info surface and the /openai/v1 faithful surface.

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pdp_router import _proxy
from pdp_router._clients import CompletionResult
from pdp_router._proxy import (
    ChatCompletionRequest,
    ChatMessage,
    _iter_sse,
    _RouteProvenance,
    app,
)
from pdp_router._tools import StreamFinish, ToolCallDelta, ToolTranslationError


@pytest.fixture()
def client():
    """TestClient with ANTHROPIC_API_KEY set and streaming flag enabled."""
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "GEMINI_API_KEY": "gm-test"}),
        patch("pdp_router._proxy._streaming_enabled", return_value=True),
        TestClient(app) as c,
    ):
        yield c


@pytest.fixture()
def client_streaming_disabled():
    """TestClient with the proxy_streaming_enabled flag OFF."""
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "GEMINI_API_KEY": "gm-test"}),
        patch("pdp_router._proxy._streaming_enabled", return_value=False),
        TestClient(app) as c,
    ):
        yield c


def _mock_completion(text: str = "Hello!") -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=10,
        output_tokens=5,
        model="gemini-2.5-flash",
        estimated_cost_usd=0.0001,
    )


def _make_streaming_mock_client(tokens: list[str], raise_at: int | None = None) -> MagicMock:
    """Build a MagicMock LLM client whose stream_complete yields the given tokens.

    raise_at=N causes a RuntimeError after yielding N tokens, exercising the
    mid-stream error path in _iter_sse.
    """
    mock_client = MagicMock()

    async def _stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        for i, tok in enumerate(tokens):
            yield tok
            if raise_at is not None and (i + 1) == raise_at:
                raise RuntimeError("backend exploded")

    mock_client.stream_complete = _stream
    mock_client.stream_complete_multi = _stream
    return mock_client


def _parse_sse(body: str) -> list[str]:
    """Return the payload portion of each `data:` line in an SSE body."""
    out: list[str] = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            out.append(line[5:].strip())
    return out


class TestNonStreamingRegression:
    """Default callers MUST keep working after we add the streaming branch."""

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_nonstream_still_returns_json(self, mock_get_client, _mock_classify, client) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("plain reply")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "plain reply"

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_stream_default_false_returns_json(
        self, mock_get_client, _mock_classify, client
    ) -> None:
        """No stream key in the body == non-streaming."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("hi")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")


class TestStreamingResponseShape:
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_stream_returns_sse_content_type(self, mock_get_client, _mock_classify, client) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["a", "b"])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_stream_first_event_is_route_info(
        self, mock_get_client, _mock_classify, client
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["hi"])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        first = json.loads(events[0])
        assert first["type"] == "route_info"
        assert first["object"] == "pdp.route_info"
        assert first["model"]
        assert "confidence" in first
        assert "score" in first

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_stream_yields_content_deltas(self, mock_get_client, _mock_classify, client) -> None:
        tokens = ["hello", " ", "world"]
        mock_get_client.return_value = _make_streaming_mock_client(tokens)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        # First is route_info; subsequent deltas should be N=len(tokens) chunks.
        delta_events = [
            json.loads(e)
            for e in events[1:]
            if e != "[DONE]" and json.loads(e).get("object") == "chat.completion.chunk"
        ]
        assert len(delta_events) == 3
        recovered = "".join(d["choices"][0]["delta"]["content"] for d in delta_events)
        assert recovered == "hello world"

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_stream_done_terminator(self, mock_get_client, _mock_classify, client) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["one"])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_stream_multi_turn_uses_stream_complete_multi(
        self, mock_get_client, _mock_classify, client
    ) -> None:
        """Multi-message conversations route through stream_complete_multi."""
        mock_llm = MagicMock()
        captured: dict = {}

        async def _stream_multi(**kwargs: object) -> AsyncIterator[str]:
            captured.update(kwargs)
            for tok in ("multi", "-", "reply"):
                yield tok

        async def _stream_single(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
            yield "should-not-be-called"

        mock_llm.stream_complete_multi = _stream_multi
        mock_llm.stream_complete = _stream_single
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ack"},
                    {"role": "user", "content": "second"},
                ],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        delta_events = [
            json.loads(e)
            for e in events[1:]
            if e != "[DONE]" and json.loads(e).get("object") == "chat.completion.chunk"
        ]
        assert len(delta_events) == 3
        assert "messages" in captured


class TestStreamingErrorPath:
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_stream_mid_error_emits_error_event(
        self, mock_get_client, _mock_classify, client
    ) -> None:
        # Yields one token then raises -- emulates a backend dropping mid-stream.
        mock_get_client.return_value = _make_streaming_mock_client(["first"], raise_at=1)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        parsed = [json.loads(e) if e != "[DONE]" else "[DONE]" for e in events]
        # Expected event order:
        #   route_info -> 1 chunk -> error -> [DONE]
        assert parsed[0]["type"] == "route_info"
        assert parsed[1]["object"] == "chat.completion.chunk"
        error_events = [e for e in parsed if isinstance(e, dict) and e.get("type") == "error"]
        assert len(error_events) == 1
        assert error_events[0]["object"] == "pdp.error"
        assert "backend exploded" in error_events[0]["message"]
        assert parsed[-1] == "[DONE]"


class TestStreamFlagOff:
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_flag_off_falls_back_to_json(
        self, mock_get_client, _mock_classify, client_streaming_disabled
    ) -> None:
        """When pipeline.proxy_streaming_enabled is False, stream=true is ignored."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("non-stream fallback")
        mock_get_client.return_value = mock_llm

        resp = client_streaming_disabled.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "non-stream fallback"


class TestOpenAIFaithfulSurface:
    """The /openai/v1 surface omits the route_info event and frames the stream
    OpenAI-faithfully (leading assistant-role delta + terminal finish_reason:stop)
    so strict agent clients like Crush stop rejecting it as 'unexpected EOF'.
    See claude-code-concerns.md #44."""

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_faithful_first_event_is_role_chunk_not_route_info(
        self, mock_get_client, _mock_classify, client
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["hi"])

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)
        first = json.loads(events[0])
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"] == {"role": "assistant"}
        # No route_info event anywhere on the faithful surface.
        for e in events:
            if e == "[DONE]":
                continue
            payload = json.loads(e)
            assert payload.get("object") != "pdp.route_info"
            assert payload.get("type") != "route_info"

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_faithful_terminal_chunk_has_finish_reason_stop(
        self, mock_get_client, _mock_classify, _footer, client
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["hello", " world"])

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"
        # The chunk immediately before [DONE] carries finish_reason="stop".
        last_chunk = json.loads(events[-2])
        assert last_chunk["object"] == "chat.completion.chunk"
        assert last_chunk["choices"][0]["finish_reason"] == "stop"
        assert last_chunk["choices"][0]["delta"] == {}
        # Content is still recoverable from the in-flight (finish_reason=None) chunks.
        content = "".join(
            json.loads(e)["choices"][0]["delta"].get("content", "")
            for e in events
            if e != "[DONE]" and json.loads(e)["choices"][0]["finish_reason"] is None
        )
        assert content == "hello world"

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_faithful_mid_error_emits_error_frame_not_clean_stop(
        self, mock_get_client, _mock_classify, client
    ) -> None:
        """A mid-stream failure must surface as an OpenAI-style error frame, never as
        a finish_reason:stop a client would read as a clean (empty/truncated) turn."""
        mock_get_client.return_value = _make_streaming_mock_client(["first"], raise_at=1)

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        parsed = [json.loads(e) if e != "[DONE]" else "[DONE]" for e in events]
        # No non-standard pdp.error event on the faithful surface.
        assert all(not (isinstance(e, dict) and e.get("object") == "pdp.error") for e in parsed)
        # The failure surfaces as an OpenAI-style {"error": {...}} frame.
        error_frames = [e for e in parsed if isinstance(e, dict) and "error" in e]
        assert len(error_frames) == 1
        assert "backend exploded" in error_frames[0]["error"]["message"]
        # The errored turn must NOT be stamped finish_reason:"stop".
        stop_chunks = [
            e
            for e in parsed
            if isinstance(e, dict)
            and e.get("object") == "chat.completion.chunk"
            and e["choices"][0].get("finish_reason") == "stop"
        ]
        assert stop_chunks == []
        assert parsed[-1] == "[DONE]"

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_default_v1_surface_still_leads_with_route_info(
        self, mock_get_client, _mock_classify, client
    ) -> None:
        """Regression guard: /v1 keeps the route_info first event (Rust CLI contract)."""
        mock_get_client.return_value = _make_streaming_mock_client(["hi"])

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        events = _parse_sse(resp.text)
        assert json.loads(events[0])["object"] == "pdp.route_info"

    @patch("pdp_router._proxy._route_footer_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_faithful_footer_when_enabled(
        self, mock_get_client, _mock_classify, _footer, client
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["answer"])
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        full = "".join(
            json.loads(e)["choices"][0]["delta"].get("content", "")
            for e in _parse_sse(resp.text)
            if e != "[DONE]" and json.loads(e)["choices"][0].get("finish_reason") is None
        )
        assert "[routed:" in full
        assert "score 3" in full
        # The model answer is intact ahead of the footer.
        assert full.startswith("answer")

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_faithful_no_footer_when_disabled(
        self, mock_get_client, _mock_classify, _footer, client
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["answer"])
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert "[routed:" not in resp.text

    @patch("pdp_router._proxy._route_footer_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_default_v1_never_gets_footer(
        self, mock_get_client, _mock_classify, _footer, client
    ) -> None:
        """The footer is faithful-surface only -- /v1 (Rust CLI / bot) never gets it."""
        mock_get_client.return_value = _make_streaming_mock_client(["answer"])
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert "[routed:" not in resp.text

    def test_openai_models_endpoint_lists_pdp_auto(self, client) -> None:
        resp = client.get("/openai/v1/models")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.json()["data"]}
        assert "pdp-auto" in ids


class TestFaithfulPlainStreamGolden:
    """The exact wire sequence /openai/v1 emits for a plain (no-tools) turn.

    A characterization wall rather than a behavior test: it asserts the whole
    frame in one literal so that any future change to the faithful surface has
    to be deliberate. The tool-passthrough work adds a sibling generator for
    tool turns, and the plain path is required to come through it byte-identical;
    a golden captured afterwards could only compare the new code against itself,
    which is why this lands before any of it.

    `id` and `created` are the only per-request values and are asserted for shape
    and stability instead of content.

    _route_request is patched rather than _classify_request: the cascade's
    epsilon-greedy explore step can pick a different arm on any given run, so
    patching only the classifier leaves `model` free to vary and the golden
    intermittently red. Same reason and same 8-tuple shape as
    test_proxy.py::TestEffortRouting._ROUTE.
    """

    _ROUTE = (
        "claude-sonnet-4-6",  # model_name
        0.55,  # confidence
        3,  # score
        0,  # panel_score
        False,  # search_intent
        "",  # system
        [ChatMessage(role="user", content="hi")],  # non_system
        _RouteProvenance(mode="cascade", explored=False),  # provenance
    )

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._route_request", return_value=_ROUTE)
    @patch("pdp_router._proxy.get_client")
    def test_plain_stream_chunk_sequence_is_unchanged(
        self, mock_get_client, _route, _footer, client
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["hello", " world"])

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]

        # One id for the whole turn, OpenAI-prefixed; created is a unix stamp.
        ids = {c.pop("id") for c in chunks}
        assert len(ids) == 1
        assert ids.pop().startswith("chatcmpl-")
        assert all(isinstance(c.pop("created"), int) for c in chunks)

        assert chunks == [
            {
                "object": "chat.completion.chunk",
                "model": "claude-sonnet-4-6",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            },
            {
                "object": "chat.completion.chunk",
                "model": "claude-sonnet-4-6",
                "choices": [
                    {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                ],
            },
            {
                "object": "chat.completion.chunk",
                "model": "claude-sonnet-4-6",
                "choices": [
                    {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
                ],
            },
            {
                "object": "chat.completion.chunk",
                "model": "claude-sonnet-4-6",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._route_request", return_value=_ROUTE)
    @patch("pdp_router._proxy.get_client")
    def test_plain_stream_framing_is_unchanged(
        self, mock_get_client, _route, _footer, client
    ) -> None:
        """SSE framing, not payloads: `data: ` prefix, blank-line separator, [DONE] last.

        _parse_sse strips all of that away, so the chunk-sequence golden above
        would still pass if the separator or the terminator changed.
        """
        mock_get_client.return_value = _make_streaming_mock_client(["hello", " world"])

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert resp.text.endswith("data: [DONE]\n\n")
        frames = resp.text.split("\n\n")
        assert frames[-1] == ""
        frames = frames[:-1]
        assert len(frames) == 5
        assert all(f.startswith("data: ") for f in frames)
        assert all("\n" not in f for f in frames)


class TestIterSseSymbol:
    """Smoke check that the streaming helper is importable for downstream tooling."""

    def test_iter_sse_is_callable(self) -> None:
        assert callable(_iter_sse)


_TOOLS_PARAM = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }
]


def _make_tool_streaming_mock_client(events: list, raise_at: int | None = None) -> MagicMock:
    """Build a MagicMock whose stream_with_tools yields the given events.

    The sibling of _make_streaming_mock_client for the with-tools leg. Events are
    the client's yield union: str for text, ToolCallDelta for a call fragment,
    StreamFinish to end the turn. raise_at=N raises after the Nth event.
    """
    mock_client = MagicMock()

    async def _stream(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        for i, event in enumerate(events):
            yield event
            if raise_at is not None and (i + 1) == raise_at:
                raise RuntimeError("backend exploded")

    mock_client.stream_with_tools = _stream
    return mock_client


class TestToolStreaming:
    """Prompt 7: /openai/v1 streams a turn that calls tools.

    The tool leg gets its own generator rather than a mode on _iter_sse, for the
    reason the clients give: _iter_sse backs live plain streaming on both
    surfaces and its bytes are pinned by TestFaithfulPlainStreamGolden.
    """

    _ROUTE = (
        "claude-sonnet-4-6",
        0.55,
        3,
        0,
        False,
        "",
        [ChatMessage(role="user", content="list files")],
        _RouteProvenance(mode="cascade", explored=False),
    )

    def _body(self, **overrides) -> dict:
        body: dict = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": _TOOLS_PARAM,
            "stream": True,
        }
        body.update(overrides)
        return body

    def _post(self, client, events: list, *, raise_at: int | None = None, footer: bool = False):
        mock_llm = _make_tool_streaming_mock_client(events, raise_at=raise_at)
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_footer_enabled", return_value=footer),
            patch("pdp_router._proxy._route_request", return_value=self._ROUTE),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            return client.post("/openai/v1/chat/completions", json=self._body())

    @staticmethod
    def _chunks(resp) -> list[dict]:
        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"
        return [json.loads(e) for e in events[:-1]]

    @staticmethod
    def _deltas(chunks: list[dict]) -> list[dict]:
        return [c["choices"][0]["delta"] for c in chunks if "choices" in c]

    def test_tool_call_stream_emits_the_openai_fragment_sequence(self, client) -> None:
        """The wire shape an agent client reassembles: the first fragment of a
        call carries id/type/function.name, later fragments carry only argument
        text against the same index, and the turn ends on tool_calls."""
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=0, arguments='{"cmd"'),
            ToolCallDelta(index=0, arguments=': "ls"}'),
            StreamFinish("tool_calls"),
        ]
        resp = self._post(client, events)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        chunks = self._chunks(resp)
        assert self._deltas(chunks) == [
            {"role": "assistant"},
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run", "arguments": ""},
                    }
                ]
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd"'}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": ': "ls"}'}}]},
            {},
        ]
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])

    def test_arguments_concatenate_to_the_original_json(self, client) -> None:
        """Fragments are transported, never reassembled or reparsed here, so the
        client's join has to reproduce the argument string exactly."""
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=0, arguments='{"cmd": "ls '),
            ToolCallDelta(index=0, arguments='-la /tmp"}'),
            StreamFinish("tool_calls"),
        ]
        chunks = self._chunks(self._post(client, events))

        joined = "".join(
            frag["function"].get("arguments", "")
            for delta in self._deltas(chunks)
            for frag in delta.get("tool_calls", ())
        )
        assert joined == '{"cmd": "ls -la /tmp"}'

    def test_parallel_calls_keep_their_own_indices(self, client) -> None:
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=1, id="call_2", name="read"),
            ToolCallDelta(index=0, arguments='{"cmd": "ls"}'),
            ToolCallDelta(index=1, arguments='{"path": "a"}'),
            StreamFinish("tool_calls"),
        ]
        chunks = self._chunks(self._post(client, events))

        by_index: dict[int, list] = {}
        for delta in self._deltas(chunks):
            for frag in delta.get("tool_calls", ()):
                by_index.setdefault(frag["index"], []).append(frag)
        assert set(by_index) == {0, 1}
        assert by_index[0][0]["id"] == "call_1"
        assert by_index[1][0]["id"] == "call_2"
        assert by_index[0][1]["function"]["arguments"] == '{"cmd": "ls"}'
        assert by_index[1][1]["function"]["arguments"] == '{"path": "a"}'

    def test_text_and_tool_calls_interleave_in_arrival_order(self, client) -> None:
        """Content sits BETWEEN two fragments on purpose. With text first, a
        generator that buffered every fragment to the end of the turn would emit
        a byte-identical stream and this test could not tell the difference."""
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            "Let me look. ",
            ToolCallDelta(index=0, arguments="{}"),
            StreamFinish("tool_calls"),
        ]
        deltas = self._deltas(self._chunks(self._post(client, events)))

        assert deltas[1]["tool_calls"][0]["id"] == "call_1"
        assert deltas[2] == {"content": "Let me look. "}
        assert deltas[3]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_pure_text_through_the_tool_path_finishes_stop(self, client) -> None:
        """A tools request whose model answers in prose is a normal turn and has
        to close like one, or a client waits forever for a call that never comes."""
        events = ["hello", " world", StreamFinish("stop")]
        chunks = self._chunks(self._post(client, events))

        assert self._deltas(chunks) == [
            {"role": "assistant"},
            {"content": "hello"},
            {"content": " world"},
            {},
        ]
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_no_footer_on_a_tool_calls_turn(self, client) -> None:
        """The footer is display text. Appending it to a turn whose payload is a
        tool call corrupts the agent's loop, so it is suppressed there even with
        the flag on."""
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=0, arguments="{}"),
            StreamFinish("tool_calls"),
        ]
        chunks = self._chunks(self._post(client, events, footer=True))

        assert not any("routed:" in (d.get("content") or "") for d in self._deltas(chunks))

    def test_footer_still_rides_a_plain_stop_turn_through_the_tool_path(self, client) -> None:
        events = ["hello", StreamFinish("stop")]
        chunks = self._chunks(self._post(client, events, footer=True))

        contents = [d.get("content", "") for d in self._deltas(chunks)]
        # The exact payload, not a substring: this is the shared
        # _route_footer_content, so drift here also moves the live plain path.
        assert "\n\n`[routed: claude-sonnet-4-6 | score 3]`" in contents

    def test_no_footer_when_the_provider_labels_a_tool_turn_stop(self, client) -> None:
        """The footer guard cannot key on the provider's label.

        A provider can stream tool calls and still report "stop", and the default
        Anthropic arm does it routinely: _STOP_REASONS maps only four reasons, so
        a real stop_reason outside it (pause_turn, refusal) becomes "stop" on a
        genuine tool turn. Appending display text there hands the agent loop a
        turn the model never wrote.
        """
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=0, arguments="{}"),
            StreamFinish("stop"),
        ]
        chunks = self._chunks(self._post(client, events, footer=True))

        assert not any("routed:" in (d.get("content") or "") for d in self._deltas(chunks))
        # The reported reason still rides through verbatim; only the footer is
        # decided by what this generator actually emitted.
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_a_tool_turn_with_no_reported_reason_closes_on_tool_calls(self, client) -> None:
        """A provider that streams calls and then just stops has produced a tool
        turn. Inventing "stop" for it labels a call as finished prose, and with
        the footer on it would also collect display text."""
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=0, arguments="{}"),
        ]
        chunks = self._chunks(self._post(client, events, footer=True))

        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert not any("routed:" in (d.get("content") or "") for d in self._deltas(chunks))

    def test_a_text_turn_with_no_reported_reason_still_closes_on_stop(self, client) -> None:
        """The other half of the inferred default: no fragments means prose."""
        chunks = self._chunks(self._post(client, ["hello"]))

        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_mid_stream_error_emits_an_error_frame_and_no_finish_chunk(self, client) -> None:
        """Same convention as the plain faithful path: a failure ends the turn
        with an error frame and NO finish_reason chunk, so a strict client reads
        a failure rather than a truncated call as a completed one."""
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=0, arguments='{"cm'),
        ]
        resp = self._post(client, events, raise_at=2)

        events_out = _parse_sse(resp.text)
        assert events_out[-1] == "[DONE]"
        payloads = [json.loads(e) for e in events_out[:-1]]
        errors = [p for p in payloads if "error" in p]
        assert len(errors) == 1
        assert errors[0]["error"]["type"] == "upstream_error"
        assert "backend exploded" in errors[0]["error"]["message"]
        assert not any(
            p.get("choices", [{}])[0].get("finish_reason") for p in payloads if "choices" in p
        )

    def test_no_route_info_event_on_the_tool_stream(self, client) -> None:
        """The tool surface is /openai/v1 only, and Crush rejects route_info."""
        events = [ToolCallDelta(index=0, id="call_1", name="run"), StreamFinish("tool_calls")]
        chunks = self._chunks(self._post(client, events))

        assert all(c.get("object") == "chat.completion.chunk" for c in chunks)

    def test_tools_ride_verbatim_to_the_streaming_client(self, client) -> None:
        mock_llm = _make_tool_streaming_mock_client([StreamFinish("stop")])
        seen: dict = {}

        async def _capture(*args: object, **kwargs: object):
            seen.update(kwargs)
            seen["args"] = args
            for event in (StreamFinish("stop"),):
                yield event

        mock_llm.stream_with_tools = _capture
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._ROUTE),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            client.post(
                "/openai/v1/chat/completions",
                json=self._body(tool_choice="auto", parallel_tool_calls=False),
            )

        assert seen["tools"] == _TOOLS_PARAM
        assert seen["tool_choice"] == "auto"
        assert seen["parallel_tool_calls"] is False
        # The rest of what the branch is responsible for handing the client.
        # Asserting only the tool trio leaves the system prompt, the token
        # budget and the routed effort droppable with the suite green.
        assert seen["system"] == ""
        assert seen["max_tokens"] == 4096
        # Effort flag is off here, so the routed level is None; the flag-on
        # value is pinned in test_routed_effort_value_reaches_the_stream_client.
        assert seen["effort"] is None

    def test_a_tool_shaped_transcript_reaches_the_streaming_client(self, client) -> None:
        """Pins the exclude_none dump on this leg: the assistant turn goes out
        carrying tool_calls and NO content key, which is the shape OpenAI clients
        send, and the result turn keeps the id binding it to its call."""
        history = [
            ChatMessage(role="user", content="list files"),
            _proxy.ToolChatMessage(
                role="assistant",
                tool_calls=[
                    _proxy.ToolCallSpec(
                        id="call_abc",
                        function=_proxy.FunctionCallSpec(name="run", arguments='{"cmd": "ls"}'),
                    )
                ],
            ),
            _proxy.ToolChatMessage(role="tool", tool_call_id="call_abc", content="a.txt"),
        ]
        route = (*self._ROUTE[:6], history, self._ROUTE[7])
        mock_llm = _make_tool_streaming_mock_client([StreamFinish("stop")])
        seen: dict = {}

        async def _capture(*_args: object, **kwargs: object):
            seen.update(kwargs)
            for event in (StreamFinish("stop"),):
                yield event

        mock_llm.stream_with_tools = _capture
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=route),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            client.post("/openai/v1/chat/completions", json=self._body())

        assistant = seen["messages"][1]
        assert "content" not in assistant
        assert assistant["tool_calls"][0]["id"] == "call_abc"
        assert seen["messages"][2]["tool_call_id"] == "call_abc"

    def test_the_stream_response_carries_the_prediction_id(self, client) -> None:
        """It is what ties the SSE response back to the routing row just written."""
        events = [ToolCallDelta(index=0, id="call_1", name="run"), StreamFinish("tool_calls")]
        resp = self._post(client, events)

        assert resp.headers["X-PDP-Prediction-Id"]

    def test_routed_effort_value_reaches_the_stream_client(self, client) -> None:
        """The routed effort LEVEL, not just the key, must reach the client. The
        _ROUTE score is 3 and sonnet supports the dial, so with the effort flag
        on it arrives as "medium"; asserting only key presence let effort=None
        pass."""
        mock_llm = _make_tool_streaming_mock_client([StreamFinish("stop")])
        seen: dict = {}

        async def _capture(*_args: object, **kwargs: object):
            seen.update(kwargs)
            yield StreamFinish("stop")

        mock_llm.stream_with_tools = _capture
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._effort_routing_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._ROUTE),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            client.post("/openai/v1/chat/completions", json=self._body())

        assert seen["effort"] == "medium"

    def test_system_prompt_reaches_the_stream_client(self, client) -> None:
        """A non-empty system turn is threaded to stream_with_tools; the
        empty-system route in the sibling tests let a dropped system pass."""
        route = (*self._ROUTE[:5], "You are a careful tool runner.", *self._ROUTE[6:])
        mock_llm = _make_tool_streaming_mock_client([StreamFinish("stop")])
        seen: dict = {}

        async def _capture(*_args: object, **kwargs: object):
            seen.update(kwargs)
            yield StreamFinish("stop")

        mock_llm.stream_with_tools = _capture
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=route),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            client.post("/openai/v1/chat/completions", json=self._body())

        assert seen["system"] == "You are a careful tool runner."

    def test_completion_id_is_stable_across_every_chunk(self, client) -> None:
        """One id per turn: an agent client correlates a streamed turn by its id,
        so a fresh id on any chunk would fragment the turn."""
        events = [
            ToolCallDelta(index=0, id="call_1", name="run"),
            ToolCallDelta(index=0, arguments='{"cmd": "ls"}'),
            StreamFinish("tool_calls"),
        ]
        chunks = self._chunks(self._post(client, events))
        ids = {c["id"] for c in chunks}
        assert len(ids) == 1

    def test_no_footer_on_a_truncated_text_turn(self, client) -> None:
        """The footer rides only a clean stop turn. A turn that emitted no tool
        call but ended on "length" (truncated) must not gain the footer, or the
        footer half of the guard is untested -- the finish_reason conjunct."""
        events = ["partial answer", StreamFinish("length")]
        chunks = self._chunks(self._post(client, events, footer=True))
        contents = [
            c["choices"][0]["delta"].get("content")
            for c in chunks
            if c.get("choices") and "content" in c["choices"][0]["delta"]
        ]
        assert contents == ["partial answer"]
        assert not any("routed:" in (t or "") for t in contents)
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"

    def test_translation_error_frame_matches_the_non_stream_classification(self, client) -> None:
        """A ToolTranslationError cannot be a 400 once the stream is open (status
        already flushed), so it rides an error frame -- but with the same
        invalid_request_error type the non-stream leg returns, not the generic
        upstream_error used for backend failures."""
        mock_llm = MagicMock()

        async def _raise(*_args: object, **_kwargs: object):
            raise ToolTranslationError("tool call c1 has unparseable arguments")
            yield  # pragma: no cover -- makes this an async generator

        mock_llm.stream_with_tools = _raise
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_footer_enabled", return_value=False),
            patch("pdp_router._proxy._route_request", return_value=self._ROUTE),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._body())

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        errors = [c for c in chunks if "error" in c]
        assert len(errors) == 1
        assert errors[0]["error"]["type"] == "invalid_request_error"
        assert "unparseable arguments" in errors[0]["error"]["message"]
        # No finish chunk after an error frame.
        assert all(c["choices"][0]["finish_reason"] is None for c in chunks if "choices" in c)


# -- panel-into-Crush: streaming the auto-panel chair synthesis on /openai/v1 --


@pytest.fixture()
def inbox_dir(client, tmp_path):
    """Override _config.routing_inbox_dir for the test (mirrors test_proxy.py)."""
    from pdp_router import _proxy

    inbox = tmp_path / "inbox"
    assert _proxy._config is not None
    orig = _proxy._config.routing_inbox_dir
    object.__setattr__(_proxy._config, "routing_inbox_dir", inbox)
    try:
        yield inbox
    finally:
        object.__setattr__(_proxy._config, "routing_inbox_dir", orig)


def _read_inbox_rows(inbox_dir):
    files = list(inbox_dir.glob("proxy-*.jsonl"))
    assert len(files) == 1, f"expected 1 inbox file, got {len(files)}"
    return [json.loads(line) for line in files[0].read_text().splitlines() if line]


_PANEL_MEMBERS = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]


def _panel_member_completion(text: str = "member answer") -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=10,
        output_tokens=5,
        model="panel-member",
        estimated_cost_usd=0.0002,
    )


def _make_panel_client_factory(
    *,
    chair_tokens: tuple[str, ...] = ("synth ", "answer"),
    member_text: str = "member answer",
    fail_substr: tuple[str, ...] = (),
    member_sleep: float = 0.0,
    chair_raise_at: int | None = None,
):
    """get_client side_effect: each call returns a mock serving BOTH a panel member
    (sync .complete, run in a thread) and the streaming chair (.stream_complete)."""

    def _factory(model_id, *_args, **_kwargs):
        m = MagicMock()
        if any(s in model_id for s in fail_substr):
            m.complete.side_effect = RuntimeError("member down")
        else:

            def _complete(*_ca, **_ck):
                if member_sleep:
                    import time as _t

                    _t.sleep(member_sleep)
                return _panel_member_completion(member_text)

            m.complete.side_effect = _complete

        async def _chair_stream(*_sa, **_sk):
            for i, tok in enumerate(chair_tokens):
                yield tok
                if chair_raise_at is not None and (i + 1) == chair_raise_at:
                    raise RuntimeError("chair exploded")

        m.stream_complete = _chair_stream
        m.stream_complete_multi = _chair_stream
        return m

    return _factory


def _content(events: list[str]) -> str:
    """Concatenate the in-flight (finish_reason=None) content deltas."""
    out = []
    for e in events:
        if e == "[DONE]":
            continue
        payload = json.loads(e)
        if payload.get("object") != "chat.completion.chunk":
            continue
        choice = payload["choices"][0]
        if choice.get("finish_reason") is None:
            out.append(choice["delta"].get("content", ""))
    return "".join(out)


class TestFaithfulPanelStreaming:
    """When panel-eligible AND faithful (/openai/v1) AND streaming, the proxy runs
    the auto-panel and streams the chair synthesis. See ROADMAP Tier B."""

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_panels_and_streams_chair(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        mock_get_client.side_effect = _make_panel_client_factory()
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y tradeoffs"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)
        # First event is the assistant-role delta (no route_info on the faithful surface).
        first = json.loads(events[0])
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"] == {"role": "assistant"}
        for e in events:
            if e == "[DONE]":
                continue
            assert json.loads(e).get("object") != "pdp.route_info"
        # Chair synthesis is what gets streamed; the chunk model marks the panel.
        assert _content(events) == "synth answer"
        assert "pdp-panel-3+claude-sonnet-4-6" in resp.text
        # Terminal stop then [DONE].
        assert events[-1] == "[DONE]"
        last_chunk = json.loads(events[-2])
        assert last_chunk["choices"][0]["finish_reason"] == "stop"
        assert last_chunk["choices"][0]["delta"] == {}
        # Routing rows: 3 panel members + 1 chair, sharing one chat_request_id.
        rows = _read_inbox_rows(inbox_dir)
        assert len(rows) == 4
        ctx = [json.loads(r["context_json"]) for r in rows]
        roles = [c["role"] for c in ctx]
        assert roles.count("panel_member") == 3
        assert roles.count("chair") == 1
        assert len({c["chat_request_id"] for c in ctx}) == 1
        modes = {r["routing_mode"] for r in rows}
        assert modes == {"panel", "panel_chair"}

    @patch("pdp_router._proxy._route_footer_enabled", return_value=True)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_footer_names_panel_members(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        mock_get_client.side_effect = _make_panel_client_factory()
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        full = _content(_parse_sse(resp.text))
        assert full.startswith("synth answer")
        assert "[panel:" in full
        for member in _PANEL_MEMBERS:
            assert member in full
        assert "chair: claude-sonnet-4-6" in full

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_zero_survivors_emits_error_frame_no_stop(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        mock_get_client.side_effect = _make_panel_client_factory(
            fail_substr=("opus", "gemini", "deepseek")
        )
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        parsed = [json.loads(e) if e != "[DONE]" else "[DONE]" for e in events]
        error_frames = [e for e in parsed if isinstance(e, dict) and "error" in e]
        assert len(error_frames) == 1
        assert "all panel members failed" in error_frames[0]["error"]["message"]
        stop_chunks = [
            e
            for e in parsed
            if isinstance(e, dict)
            and e.get("object") == "chat.completion.chunk"
            and e["choices"][0].get("finish_reason") == "stop"
        ]
        assert stop_chunks == []
        assert parsed[-1] == "[DONE]"
        # No survivors -> no routing rows written.
        assert list(inbox_dir.glob("proxy-*.jsonl")) == []

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_chair_empty_falls_back_to_first_survivor(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        # Empty chair stream -> the first survivor's text is streamed instead.
        mock_get_client.side_effect = _make_panel_client_factory(
            chair_tokens=(), member_text="survivor text"
        )
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        assert _content(events) == "survivor text"
        # The chair-empty fallback relabels the closing frames to +chair_fallback so a
        # reader of the chunk model field does not see single-arm output as synthesis.
        assert "pdp-panel-3+chair_fallback" in resp.text
        last_chunk = json.loads(events[-2])
        assert last_chunk["choices"][0]["finish_reason"] == "stop"
        assert last_chunk["model"] == "pdp-panel-3+chair_fallback"
        # Rows still written (the panel ran).
        assert len(_read_inbox_rows(inbox_dir)) == 4

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_chair_stream_error_emits_error_frame(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        mock_get_client.side_effect = _make_panel_client_factory(
            chair_tokens=("partial",), chair_raise_at=1
        )
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        parsed = [json.loads(e) if e != "[DONE]" else "[DONE]" for e in events]
        # Partial content was streamed before the failure.
        assert "partial" in _content(events)
        error_frames = [e for e in parsed if isinstance(e, dict) and "error" in e]
        assert len(error_frames) == 1
        stop_chunks = [
            e
            for e in parsed
            if isinstance(e, dict)
            and e.get("object") == "chat.completion.chunk"
            and e["choices"][0].get("finish_reason") == "stop"
        ]
        assert stop_chunks == []

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=[])
    @patch("pdp_router._proxy.get_client")
    def test_empty_members_streams_cascade_fallback(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["cascade ", "reply"])
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        assert _content(events) == "cascade reply"
        assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
        rows = _read_inbox_rows(inbox_dir)
        assert len(rows) == 1
        assert rows[0]["routing_mode"] == "cascade_panel_empty_fallback"

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=[])
    @patch("pdp_router._proxy.get_client")
    def test_empty_members_fallback_records_the_executed_policy(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        tmp_path,
    ) -> None:
        """This fallback runs the bandit when one is configured, so the label must say so.

        It passed routing_mode=_config.routing_mode into confidence_cascade but
        then wrote a flat "cascade_panel_empty_fallback" literal, mislabelling a
        Thompson-sampled pick as a cascade one.

        Builds its own client rather than using the `client`/`inbox_dir` pair,
        because that fixture mutates the outer _config while a nested TestClient
        re-runs the lifespan and replaces it. The inbox is therefore set through
        the environment so the fresh config picks it up.
        """
        import sqlite3

        db = tmp_path / "trust.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE model_trust (model_id TEXT, weight REAL);"
            "CREATE TABLE bandit_state (model_id TEXT, mu REAL, sigma REAL,"
            " n_obs INTEGER, sum_reward REAL, sum_sq_reward REAL,"
            " effective_n REAL, effective_sum REAL);"
        )
        conn.execute(
            "INSERT INTO bandit_state VALUES "
            "('claude-opus-4',0.98,0.005,900,880.0,870.0,900.0,882.0)"
        )
        conn.commit()
        conn.close()

        inbox = tmp_path / "inbox"
        mock_get_client.return_value = _make_streaming_mock_client(["ok"])
        with (
            patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "sk-test",
                    "GEMINI_API_KEY": "gm-test",
                    "ROUTING_MODE": "bandit",
                    "PROXY_TRUST_DB": str(db),
                    "PROXY_ROUTING_INBOX_DIR": str(inbox),
                    "PROXY_EXPLORE_RATE": "0",
                },
            ),
            patch("pdp_router._proxy._streaming_enabled", return_value=True),
            TestClient(app) as c,
        ):
            c.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "compare X vs Y"}],
                    "stream": True,
                },
            )
        rows = _read_inbox_rows(inbox)
        assert rows[0]["routing_mode"] == "bandit_panel_empty_fallback"

    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_keepalive_during_slow_fanout(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        client,
    ) -> None:
        mock_get_client.side_effect = _make_panel_client_factory(member_sleep=0.1)
        with patch.dict(os.environ, {"PROXY_PANEL_KEEPALIVE_S": "0.02"}):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "compare X vs Y"}],
                    "stream": True,
                },
            )
        # SSE comment keep-alives fired during the fan-out (ignored by data: parsers).
        assert ": panel-keepalive" in resp.text

    # -- regression guards: the panel-stream branch must not leak into other paths --

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.get_client")
    def test_default_v1_stream_high_panel_score_is_single_model_route_info(
        self, mock_get_client, _classify, _autopanel, client
    ) -> None:
        """/v1 (Rust CLI) keeps single-model route_info streaming even at panel_score 9."""
        mock_get_client.return_value = _make_streaming_mock_client(["hi"])
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        assert json.loads(events[0])["object"] == "pdp.route_info"
        assert "pdp-panel" not in resp.text

    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 4))
    @patch("pdp_router._proxy.get_client")
    def test_faithful_stream_low_panel_score_is_single_model(
        self, mock_get_client, _classify, _autopanel, _panelstream, client
    ) -> None:
        """panel_score below threshold -> single-model faithful stream, no panel."""
        mock_get_client.return_value = _make_streaming_mock_client(["plain"])
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "trivial"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        assert json.loads(events[0])["choices"][0]["delta"] == {"role": "assistant"}
        assert "pdp-panel" not in resp.text
        assert mock_get_client.call_count == 1

    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=False)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.get_client")
    def test_kill_switch_off_is_single_model(
        self, mock_get_client, _classify, _autopanel, _panelstream, client
    ) -> None:
        """pipeline.proxy_panel_streaming_enabled OFF -> faithful single-model stream."""
        mock_get_client.return_value = _make_streaming_mock_client(["plain"])
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        assert json.loads(events[0])["choices"][0]["delta"] == {"role": "assistant"}
        assert "pdp-panel" not in resp.text
        assert mock_get_client.call_count == 1

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_partial_survivor_excludes_errored_member_from_rows(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        """One failing member -> only survivors get panel rows (bandit-poisoning guard).

        The safety-critical invariant: an errored arm must NEVER get a routing row.
        Mirrors the non-streaming test_errored_member_excluded_from_routing_rows and
        locks model_selected on the new streaming code path.
        """
        mock_get_client.side_effect = _make_panel_client_factory(fail_substr=("gemini",))
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "pdp-panel-2+claude-sonnet-4-6" in resp.text  # 2 survivors, not 3
        rows = _read_inbox_rows(inbox_dir)
        member_rows = [r for r in rows if r["routing_mode"] == "panel"]
        chair_rows = [r for r in rows if r["routing_mode"] == "panel_chair"]
        member_models = {r["model_selected"] for r in member_rows}
        assert member_models == {"claude-opus-4-7", "deepseek-chat"}
        assert "gemini-2.5-pro" not in member_models  # errored arm excluded
        assert len(member_rows) == 2
        assert len(chair_rows) == 1
        assert chair_rows[0]["model_selected"] == "claude-sonnet-4-6"

    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_chair_client_build_failure_emits_error_frame(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        inbox_dir,
        client,
    ) -> None:
        """A client-build raise AFTER the role delta (here: the chair) must surface as a
        faithful error frame, not a torn stream with no finish_reason:stop."""

        def _factory(model_id, *_a, **_k):
            if model_id == "claude-sonnet-4-6":  # the chair
                raise RuntimeError("chair client boom")
            m = MagicMock()
            m.complete.side_effect = lambda *_ca, **_ck: _panel_member_completion("ok")
            return m

        mock_get_client.side_effect = _factory
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        parsed = [json.loads(e) if e != "[DONE]" else "[DONE]" for e in events]
        error_frames = [e for e in parsed if isinstance(e, dict) and "error" in e]
        assert len(error_frames) == 1
        assert "Chair setup failed" in error_frames[0]["error"]["message"]
        stop_chunks = [
            e
            for e in parsed
            if isinstance(e, dict)
            and e.get("object") == "chat.completion.chunk"
            and e["choices"][0].get("finish_reason") == "stop"
        ]
        assert stop_chunks == []
        assert parsed[-1] == "[DONE]"

    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", side_effect=RuntimeError("compose boom"))
    @patch("pdp_router._proxy.get_client")
    def test_fanout_machinery_failure_emits_error_frame(
        self, mock_get_client, _compose, _classify, _autopanel, _panelstream, client
    ) -> None:
        """compose_panel raising -> faithful 'Panel failed' error frame, no clean stop."""
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        events = _parse_sse(resp.text)
        parsed = [json.loads(e) if e != "[DONE]" else "[DONE]" for e in events]
        error_frames = [e for e in parsed if isinstance(e, dict) and "error" in e]
        assert len(error_frames) == 1
        assert "Panel failed" in error_frames[0]["error"]["message"]
        stop_chunks = [
            e
            for e in parsed
            if isinstance(e, dict)
            and e.get("object") == "chat.completion.chunk"
            and e["choices"][0].get("finish_reason") == "stop"
        ]
        assert stop_chunks == []
        assert parsed[-1] == "[DONE]"

    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.get_client")
    def test_streaming_disabled_high_panel_score_is_single_model_json(
        self, mock_get_client, _classify, _autopanel, _panelstream, client_streaming_disabled
    ) -> None:
        """Master streaming kill-switch OFF + stream:true + panel_score 9 -> single-model
        JSON, NOT the ~12x panel. Guards the gate-logic cost-coupling regression."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("single model")
        mock_get_client.return_value = mock_llm
        resp = client_streaming_disabled.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert "pdp-panel" not in data["model"]
        assert data["choices"][0]["message"]["content"] == "single model"
        mock_get_client.assert_called_once()


# -- panel-transcript capture on the streaming faithful path (chat-quality eval) --


@pytest.fixture()
def transcript_dir(client, tmp_path):
    """Override _config.panel_transcript_dir for the test (mirrors inbox_dir)."""
    from pdp_router import _proxy

    out = tmp_path / "transcripts"
    assert _proxy._config is not None
    orig = _proxy._config.panel_transcript_dir
    object.__setattr__(_proxy._config, "panel_transcript_dir", out)
    try:
        yield out
    finally:
        object.__setattr__(_proxy._config, "panel_transcript_dir", orig)


def _read_transcript_rows(transcript_dir):
    files = list(transcript_dir.glob("panel-*.jsonl"))
    assert len(files) == 1, f"expected 1 transcript file, got {len(files)}"
    return [json.loads(line) for line in files[0].read_text().splitlines() if line]


class TestPanelTranscriptStreaming:
    """proxy_panel_transcript_enabled on the streaming faithful panel path: the chair
    tokens are teed into the transcript even though _faithful_stream_tail consumes the
    source and never materializes it."""

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=True)
    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_stream_tee_captures_full_synthesis(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        mock_get_client.side_effect = _make_panel_client_factory()
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X vs Y tradeoffs"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        # The client received the full synthesis...
        assert _content(_parse_sse(resp.text)) == "synth answer"
        # ...and the tee captured the SAME synthesis into the transcript.
        rows = _read_transcript_rows(transcript_dir)
        assert len(rows) == 1
        rec = rows[0]
        assert rec["surface"] == "stream"
        assert rec["panel_score"] == 9
        assert rec["prompt"] == "compare X vs Y tradeoffs"
        assert rec["messages"] == [{"role": "user", "content": "compare X vs Y tradeoffs"}]
        assert rec["synthesis_text"] == "synth answer"
        assert rec["synthesis_status"] == "complete"
        assert [m["model_id"] for m in rec["members"]] == list(_PANEL_MEMBERS)
        assert all(m["text"] == "member answer" for m in rec["members"])

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=True)
    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_stream_chair_error_records_partial_with_error_status(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        # Chair yields "synth " then raises: _faithful_stream_tail emits an error frame
        # and returns, so the async-for ends normally and the finally writes the
        # transcript with the partial synthesis marked status="error" -- so the grader
        # can exclude this truncated turn rather than score it as a real synthesis.
        mock_get_client.side_effect = _make_panel_client_factory(chair_raise_at=1)
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hard q"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        rows = _read_transcript_rows(transcript_dir)
        assert len(rows) == 1
        rec = rows[0]
        assert rec["synthesis_text"] == "synth "  # partial, before the chair raised
        assert rec["synthesis_status"] == "error"
        assert len(rec["members"]) == 3
        assert all(m["text"] == "member answer" for m in rec["members"])

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=True)
    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_stream_chair_empty_records_empty_synthesis_not_fallback(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        # Chair streams nothing -> _faithful_stream_tail emits the first-survivor text
        # to the client DIRECTLY (not through the tee). chair_buf stays empty, so the
        # transcript must record synthesis_text='' (status chair_empty) and must NOT
        # absorb the single-arm survivor text -- invariant #5 on the stream path.
        mock_get_client.side_effect = _make_panel_client_factory(
            chair_tokens=(), member_text="survivor text"
        )
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        # Client received the survivor fallback...
        assert _content(_parse_sse(resp.text)) == "survivor text"
        rec = _read_transcript_rows(transcript_dir)[0]
        # ...but the recorded synthesis is empty, NOT the survivor text.
        assert rec["synthesis_text"] == ""
        assert rec["synthesis_status"] == "chair_empty"
        assert "survivor text" not in rec["synthesis_text"]

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=True)
    @patch("pdp_router._proxy._route_footer_enabled", return_value=True)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_stream_footer_reaches_client_but_not_synthesis(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        # Invariant #1: the route footer is emitted by _faithful_stream_tail, NOT through
        # the tee, so it reaches the client but must stay out of synthesis_text.
        mock_get_client.side_effect = _make_panel_client_factory()
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        client_content = _content(_parse_sse(resp.text))
        assert "[panel:" in client_content  # footer reached the client
        rec = _read_transcript_rows(transcript_dir)[0]
        assert rec["synthesis_text"] == "synth answer"  # chair tokens only
        assert "[panel:" not in rec["synthesis_text"]
        assert "chair:" not in rec["synthesis_text"]

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=False)
    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy._panel_streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.35, 4, 9))
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_stream_flag_off_writes_no_transcript_and_keeps_sse(
        self,
        mock_get_client,
        _compose,
        _classify,
        _autopanel,
        _panelstream,
        _footer,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        # The shipped default state on the primary (Crush) surface: flag OFF -> no file,
        # and the SSE the client receives is unchanged.
        mock_get_client.side_effect = _make_panel_client_factory()
        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert _content(_parse_sse(resp.text)) == "synth answer"
        assert not list(transcript_dir.glob("panel-*.jsonl"))

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=True)
    @patch("pdp_router._proxy._route_footer_enabled", return_value=False)
    @patch("pdp_router._proxy.compose_panel", return_value=list(_PANEL_MEMBERS))
    @patch("pdp_router._proxy.get_client")
    def test_stream_client_disconnect_records_partial_with_disconnect_status(
        self,
        mock_get_client,
        _compose,
        _footer,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        # The design rationale for the finally: drive _iter_panel_sse directly and
        # aclose() mid chair-stream (the real GeneratorExit a TestClient cannot
        # simulate). The finally must still write the record, marked "disconnect", with
        # the partial synthesis -- without suppressing GeneratorExit or emitting frames
        # after close.
        from pdp_router import _proxy

        mock_get_client.side_effect = _make_panel_client_factory()

        async def _drive():
            gen = _proxy._iter_panel_sse(
                request=ChatCompletionRequest(
                    model="pdp-auto",
                    messages=[ChatMessage(role="user", content="q")],
                    stream=True,
                ),
                chat_request_id="disc-1",
                confidence=0.35,
                score=4,
                panel_score=9,
                system="",
                non_system=[ChatMessage(role="user", content="q")],
            )
            seen: list[str] = []
            got_content = False
            for _ in range(20):
                ev = await gen.__anext__()
                seen.append(ev)
                if "synth " in ev:  # first chair content chunk pulled
                    got_content = True
                    break
            assert got_content, seen
            await gen.aclose()  # simulate client disconnect -> GeneratorExit
            return seen

        seen = asyncio.run(_drive())
        # No [DONE] was produced up to and including the disconnect.
        assert all("[DONE]" not in ev for ev in seen)
        rec = _read_transcript_rows(transcript_dir)[0]
        assert rec["synthesis_status"] == "disconnect"
        assert rec["synthesis_text"] == "synth "  # only the first token was pulled
        assert len(rec["members"]) == 3
        assert all(m["text"] == "member answer" for m in rec["members"])


class TestToolStreamingDriverFloor:
    """Prompt 8: the driver floor sits upstream of the stream dispatch too."""

    _NON_DRIVER_ROUTE = (
        "claude-haiku-4-5-20251001",
        0.55,
        3,
        0,
        False,
        "",
        [ChatMessage(role="user", content="what is the capital of france")],
        _RouteProvenance(mode="cascade", explored=False),
    )

    _EVENTS = (
        ToolCallDelta(index=0, id="call_1", name="run"),
        StreamFinish("tool_calls"),
    )

    def _stream_body(self) -> dict:
        return {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "what is the capital of france"}],
            "tools": _TOOLS_PARAM,
            "stream": True,
        }

    def test_floored_driver_reaches_the_stream_chunks(self, client) -> None:
        """A non-driver cascade pick is replaced before the stream opens: every
        chunk's model field is the pinned driver ("what is the capital of
        france" -> claude-opus-4-8), proving the floor runs upstream of
        _build_tool_stream_response rather than only on the non-stream leg."""
        mock_llm = _make_tool_streaming_mock_client(list(self._EVENTS))
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_footer_enabled", return_value=False),
            patch("pdp_router._proxy._route_request", return_value=self._NON_DRIVER_ROUTE),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_gc,
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._stream_body())

        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-opus-4-8"
        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        assert chunks and all(c["model"] == "claude-opus-4-8" for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_panel_worthy_stream_with_tools_takes_the_tool_path(self, client) -> None:
        """panel_score over the threshold + autopanel + panel streaming all on:
        a tools request still gets the tool stream. Pins the tool branch's
        position ABOVE the panel guard -- if the branch ever moves below it,
        this dispatches the panel builder and fails loudly."""
        route = (*self._NON_DRIVER_ROUTE[:3], 9, *self._NON_DRIVER_ROUTE[4:])
        mock_llm = _make_tool_streaming_mock_client(list(self._EVENTS))
        panel_spy = MagicMock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._autopanel_enabled", return_value=True),
            patch("pdp_router._proxy._route_footer_enabled", return_value=False),
            patch("pdp_router._proxy._build_panel_stream_response", panel_spy),
            patch("pdp_router._proxy._route_request", return_value=route),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._stream_body())

        assert resp.status_code == 200
        panel_spy.assert_not_called()
        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        assert deltas[0] == {"role": "assistant"}
        assert any("tool_calls" in d for d in deltas)
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
