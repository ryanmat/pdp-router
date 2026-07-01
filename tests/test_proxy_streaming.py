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

from pdp_router._clients import CompletionResult
from pdp_router._proxy import ChatCompletionRequest, ChatMessage, _iter_sse, app


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


class TestIterSseSymbol:
    """Smoke check that the streaming helper is importable for downstream tooling."""

    def test_iter_sse_is_callable(self) -> None:
        assert callable(_iter_sse)


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
