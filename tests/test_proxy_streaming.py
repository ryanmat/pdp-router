# Description: Tests for OpenAI-compatible SSE streaming in the PDP Router Proxy.
# Description: Covers route_info first event, delta chunks, [DONE] terminator, mid-stream errors.

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pdp_router._clients import CompletionResult
from pdp_router._proxy import _iter_sse, app


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


class TestIterSseSymbol:
    """Smoke check that the streaming helper is importable for downstream tooling."""

    def test_iter_sse_is_callable(self) -> None:
        assert callable(_iter_sse)
