# Description: Regression wall for tool passthrough: flag-off, /v1, and panel invariants.
# Description: Pins the surfaces Concern 44 must never move, plus the NotImplementedError backstop.

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pdp_router._clients import CompletionResult
from pdp_router._proxy import ChatMessage, _RouteProvenance, app
from pdp_router._tools import ToolCall

# Helpers are file-local like every sibling test file: pytest 9's importlib
# mode gives test modules no sys.path entry, so cross-test imports do not
# resolve. The twelve malformed shapes stay single-sourced in test_proxy.py,
# where their flag-off /v1 wall test lives beside them.

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


def _mock_completion(text: str = "Hello!") -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=50,
        output_tokens=20,
        model="gemini-2.5-flash",
        estimated_cost_usd=0.0001,
    )


def _tool_completion(text: str = "", finish_reason: str = "tool_calls") -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=50,
        output_tokens=20,
        model="claude-sonnet-4-6",
        estimated_cost_usd=0.0001,
        tool_calls=(ToolCall(id="call_abc", name="run", arguments='{"cmd": "ls -la"}'),),
        finish_reason=finish_reason,
    )


def _make_streaming_mock_client(tokens: list[str]) -> MagicMock:
    mock_client = MagicMock()

    async def _stream(*_args: object, **_kwargs: object) -> AsyncIterator[str]:
        for token in tokens:
            yield token

    mock_client.stream_complete = _stream
    return mock_client


def _parse_sse(body: str) -> list[str]:
    return [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]


@contextlib.contextmanager
def _credential_overrides(**fields):
    """object.__setattr__ bypass for the frozen live config, restore-on-exit
    (the inbox_dir fixture idiom from test_proxy.py)."""
    from pdp_router import _proxy

    assert _proxy._config is not None
    originals = {name: getattr(_proxy._config, name) for name in fields}
    for name, value in fields.items():
        object.__setattr__(_proxy._config, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            object.__setattr__(_proxy._config, name, value)


@pytest.fixture()
def client():
    """TestClient with Anthropic + Gemini keys, matching the sibling files."""
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "GEMINI_API_KEY": "gm-test"}),
        TestClient(app) as c,
    ):
        yield c


def _route(model: str = "claude-sonnet-4-6", panel_score: int = 0) -> tuple:
    return (
        model,
        0.55,
        3,
        panel_score,
        "general",
        False,
        "",
        [ChatMessage(role="user", content="hi")],
        _RouteProvenance(mode="cascade", explored=False),
    )


def _inbox_rows() -> list[dict]:
    inbox = Path(os.environ["PROXY_ROUTING_INBOX_DIR"])
    return [
        json.loads(line)
        for path in sorted(inbox.glob("*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


_SECOND_ROUND_HISTORY = [
    {"role": "user", "content": "list the files"},
    {
        "role": "assistant",
        "content": "Running it.",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
]


class TestFlagOffFourShapeContract:
    """The flag-off /openai/v1 contract, all four shapes in one place.

    Identity with the legacy models is the promise (spec invariant 2); each
    test names the shape and pins the exact answer it has always had.
    """

    def test_null_content_tool_calls_turn_is_a_422(self, client) -> None:
        """The one shape the legacy annotation refused; it must keep refusing
        with the legacy message, not the flag-on one."""
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [
                        {"role": "user", "content": "run it"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                },
            )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == (
            "messages.1: content is required because tool passthrough is disabled on this proxy"
        )

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
    @patch("pdp_router._proxy.confidence_cascade", return_value=("claude-sonnet-4-6", False))
    @patch("pdp_router._proxy.get_client")
    def test_tool_role_message_with_content_proceeds_to_the_provider(
        self, mock_get_client, _mock_cascade, _mock_classify, _flag, client
    ) -> None:
        """role:"tool" + string content is served on the legacy path, and the
        provider sees the turn as plain role/content. The real _route_request
        runs so non_system is built from the posted transcript."""
        mock_llm = MagicMock()
        mock_llm.complete_multi.return_value = _mock_completion("ok")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/openai/v1/chat/completions",
            json={"model": "pdp-auto", "messages": _SECOND_ROUND_HISTORY},
        )
        assert resp.status_code == 200
        mock_llm.complete_multi.assert_called_once()
        messages = mock_llm.complete_multi.call_args.kwargs["messages"]
        assert {"role": "tool", "content": "a.txt"} in messages

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
    @patch("pdp_router._proxy.confidence_cascade", return_value=("claude-sonnet-4-6", False))
    @patch("pdp_router._proxy.get_client")
    def test_assistant_tool_calls_with_content_serves_and_drops_the_tool_fields(
        self, mock_get_client, _mock_cascade, _mock_classify, _flag, client
    ) -> None:
        """200, and neither a tools kwarg nor a tool_calls key reaches the
        client mock -- the legacy path dumps role/content only."""
        mock_llm = MagicMock()
        mock_llm.complete_multi.return_value = _mock_completion("ok")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": _SECOND_ROUND_HISTORY,
                "tools": _TOOLS_PARAM,
                "tool_choice": "auto",
            },
        )
        assert resp.status_code == 200
        mock_llm.complete_with_tools.assert_not_called()
        kwargs = mock_llm.complete_multi.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
        for message in kwargs["messages"]:
            assert set(message) == {"role", "content"}

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False)
    @patch("pdp_router._proxy._route_request", return_value=_route())
    @patch("pdp_router._proxy.get_client")
    def test_bare_tools_are_stripped_into_a_plain_completion(
        self, mock_get_client, _mock_route, _flag, client
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ok")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _TOOLS_PARAM,
                "tool_choice": "auto",
            },
        )
        assert resp.status_code == 200
        mock_llm.complete.assert_called_once()
        assert "tools" not in mock_llm.complete.call_args.kwargs
        mock_llm.complete_with_tools.assert_not_called()

    @pytest.mark.parametrize("footer", [False, True], ids=["footer_off", "footer_on"])
    def test_bare_tools_leave_the_plain_stream_bytes_unchanged(self, footer, client) -> None:
        """Chunk-by-chunk equality against the no-tools control: the streamed
        frames for a plain request must not shift when ignored tool fields
        ride along flag-off. Both footer states, since production runs the
        footer on. id/created are the only per-request values."""

        def _post(with_tools: bool) -> list[dict]:
            body: dict = {
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
            if with_tools:
                body["tools"] = _TOOLS_PARAM
                body["tool_choice"] = "auto"
            with (
                patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False),
                patch("pdp_router._proxy._streaming_enabled", return_value=True),
                patch("pdp_router._proxy._route_footer_enabled", return_value=footer),
                patch("pdp_router._proxy._route_request", return_value=_route()),
                patch(
                    "pdp_router._proxy.get_client",
                    return_value=_make_streaming_mock_client(["hello", " world"]),
                ),
            ):
                resp = client.post("/openai/v1/chat/completions", json=body)
            assert resp.status_code == 200
            events = _parse_sse(resp.text)
            assert events[-1] == "[DONE]"
            chunks = [json.loads(e) for e in events[:-1]]
            for chunk in chunks:
                assert chunk.pop("id").startswith("chatcmpl-")
                assert isinstance(chunk.pop("created"), int)
            return chunks

        assert _post(with_tools=True) == _post(with_tools=False)


class TestV1SurfaceUntouched:
    """Invariant 1: the /v1 wire shape never moves, whatever rides in.

    The twelve malformed shapes' /v1 walls live beside the shapes in
    test_proxy.py; these pin the canonical (well-formed) tool payloads.
    """

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
    @patch("pdp_router._proxy.confidence_cascade", return_value=("claude-sonnet-4-6", False))
    @patch("pdp_router._proxy.get_client")
    def test_v1_ignores_a_canonical_tool_request_even_with_the_flag_on(
        self, mock_get_client, _mock_cascade, _mock_classify, _flag, client
    ) -> None:
        """The flag gates /openai/v1 only; a canonical second round on /v1
        takes the legacy path with the tool fields dropped."""
        mock_llm = MagicMock()
        mock_llm.complete_multi.return_value = _mock_completion("ok")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": _SECOND_ROUND_HISTORY,
                "tools": _TOOLS_PARAM,
                "tool_choice": "auto",
            },
        )
        assert resp.status_code == 200
        mock_llm.complete_with_tools.assert_not_called()
        assert "tools" not in mock_llm.complete_multi.call_args.kwargs

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True)
    @patch("pdp_router._proxy._route_request", return_value=_route())
    @patch("pdp_router._proxy.get_client")
    def test_no_tool_field_is_serializable_in_a_v1_response(
        self, mock_get_client, _mock_route, _flag, client
    ) -> None:
        """Even a provider result carrying tool calls cannot leak them: the /v1
        response models have no tool_calls slot to serialize."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _tool_completion(text="ok", finish_reason="stop")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert "tool_calls" not in json.dumps(resp.json())

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True)
    @patch("pdp_router._proxy._streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._route_request", return_value=_route())
    @patch("pdp_router._proxy.get_client")
    def test_v1_stream_keeps_route_info_first_with_tools_riding(
        self, mock_get_client, _mock_route, _mock_streaming, _flag, client
    ) -> None:
        mock_get_client.return_value = _make_streaming_mock_client(["answer"])
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _TOOLS_PARAM,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        first = json.loads(_parse_sse(resp.text)[0])
        assert first["type"] == "route_info"
        assert first["object"] == "pdp.route_info"


class TestPanelInvariants:
    """Invariant 3: panel paths are text-only forever."""

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._run_panel_members")
    @patch("pdp_router._proxy._route_request", return_value=_route(panel_score=9))
    @patch("pdp_router._proxy.get_client")
    def test_tools_bearing_panel_worthy_request_never_reaches_the_members(
        self, mock_get_client, _mock_route, panel_spy, _mock_panel, _flag, client
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.complete_with_tools.return_value = _tool_completion()
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _TOOLS_PARAM,
            },
        )
        assert resp.status_code == 200
        panel_spy.assert_not_called()
        mock_llm.complete_with_tools.assert_called_once()

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False)
    @patch("pdp_router._proxy._web_search_enabled", return_value=False)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_flag_off_panel_members_see_no_tools_kwarg_when_tools_ride(
        self, mock_get_client, mock_compose, _mock_classify, _mock_panel, _mock_ws, _flag, client
    ) -> None:
        """Flag off, tools riding, panel-worthy: the panel fires (legacy strip)
        and every member call is a plain completion with no tools kwarg."""
        mock_compose.return_value = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]
        created: list[MagicMock] = []

        def make_client(*args, **kwargs):
            m = MagicMock()
            m.complete.return_value = _mock_completion("panelist answer")
            created.append(m)
            return m

        mock_get_client.side_effect = make_client

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X and Y"}],
                "tools": _TOOLS_PARAM,
                "tool_choice": "auto",
                "max_tokens": 300,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"].startswith("pdp-panel-")
        assert created
        for member in created:
            member.complete_with_tools.assert_not_called()
            for call in member.complete.call_args_list:
                assert "tools" not in call.kwargs
                assert "tool_choice" not in call.kwargs


class TestNoDriverBackstop:
    """Invariant 5's failure surface. Invariant 5 is the ROUTER's to keep, not
    the client's: when no tool driver is usable the floor refuses before any
    dispatch, so no non-driver arm -- not Gemini, and crucially not Haiku,
    whose client WOULD serve tools -- ever receives the request."""

    @staticmethod
    def _no_driver_usable():
        """Strip the credentials for every driver family (Anthropic + OpenRouter),
        leaving no usable tool driver. Gemini's key stays set, which is why a
        gemini cascade pick is a realistic degraded case rather than a total
        outage."""
        return _credential_overrides(anthropic_api_key="", openrouter_api_key="")

    def test_no_usable_driver_refuses_before_dispatch_with_no_row(self, client) -> None:
        """A router-level 503, raised in the floor before the client is built or
        a routing row is written -- distinct from the old behavior where the
        pick was left and the client raised (which wrote a row)."""
        with (
            self._no_driver_usable(),
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=_route("gemini-2.5-pro")),
            patch("pdp_router._proxy.get_client") as mock_get_client,
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["type"] == "server_error"
        assert body["error"]["message"] == "No tool-capable model is currently available"
        assert resp.headers["X-PDP-Prediction-Id"]
        mock_get_client.assert_not_called()
        assert _inbox_rows() == []

    def test_haiku_pick_with_no_usable_driver_is_refused_not_served(self, client) -> None:
        """The defect this fix closes: Haiku's AnthropicClient has no model gate,
        so leaving a degraded Haiku pick would SERVE the tool call. The floor
        must refuse instead -- complete_with_tools is never reached."""
        mock_llm = MagicMock()
        mock_llm.complete_with_tools.return_value = _tool_completion()
        with (
            self._no_driver_usable(),
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch(
                "pdp_router._proxy._route_request",
                return_value=_route("claude-haiku-4-5-20251001"),
            ),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 503
        assert resp.json()["error"]["message"] == "No tool-capable model is currently available"
        mock_llm.complete_with_tools.assert_not_called()
        assert _inbox_rows() == []

    def test_retired_driver_pick_is_walked_past_not_dispatched(self, client) -> None:
        """The kept-pick path checks usability, not bare membership: a cascade
        pick that IS a driver but is retired (available=False) must be walked
        past to a usable one, never dispatched. Retiring sonnet 4.6 leaves the
        walk to land on the pinned driver for this turn."""
        from pdp_router._router import DEFAULT_REGISTRY

        cap = DEFAULT_REGISTRY.get("claude-sonnet-4-6")
        assert cap is not None and cap.available is True
        object.__setattr__(cap, "available", False)
        mock_llm = MagicMock()
        mock_llm.complete_with_tools.return_value = _tool_completion()
        try:
            with (
                patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
                patch(
                    "pdp_router._proxy._route_request",
                    return_value=_route("claude-sonnet-4-6"),
                ),
                patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_get_client,
            ):
                resp = client.post(
                    "/openai/v1/chat/completions",
                    json={
                        "model": "pdp-auto",
                        "messages": [{"role": "user", "content": "hi"}],
                        "tools": _TOOLS_PARAM,
                    },
                )
        finally:
            object.__setattr__(cap, "available", True)
        assert resp.status_code == 200
        # The load-bearing half: the retired arm is never the dispatched one.
        assert mock_get_client.call_args[0][0] != "claude-sonnet-4-6"
        assert mock_get_client.call_args[0][0] == "claude-sonnet-5"

    def test_no_usable_driver_refuses_a_stream_before_it_opens(self, client) -> None:
        """A streaming request degrades to the same JSON 503, not a 200 stream
        with an error frame: the refusal is raised in the floor before the
        stream response is built, so nothing is flushed."""
        with (
            self._no_driver_usable(),
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._streaming_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=_route("gemini-2.5-pro")),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": _TOOLS_PARAM,
                    "stream": True,
                },
            )
        assert resp.status_code == 503
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["message"] == "No tool-capable model is currently available"
        assert _inbox_rows() == []

    @patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True)
    @patch("pdp_router._proxy._route_request", return_value=_route())
    @patch("pdp_router._proxy.get_client")
    def test_bare_runtime_error_still_writes_the_exposure_row(
        self, mock_get_client, _mock_route, _flag, client
    ) -> None:
        """The failure row rides on _execute_single_with_tools wrapping every
        raise in HTTPException; if a bare raise ever escapes that wrap, this
        wall catches the silently dropped exposure."""
        mock_llm = MagicMock()
        mock_llm.complete_with_tools.side_effect = RuntimeError("boom")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": _TOOLS_PARAM,
            },
        )
        assert resp.status_code == 503
        assert "boom" in resp.json()["error"]["message"]
        rows = _inbox_rows()
        assert len(rows) == 1
        assert rows[0]["model_selected"] == "claude-sonnet-4-6"
