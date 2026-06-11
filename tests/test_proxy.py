# Description: Tests for the PDP Router Proxy FastAPI endpoints.
# Description: Uses TestClient to validate routing, classification, and error handling.

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pdp_router._clients import CompletionResult
from pdp_router._models import OPUS, CreditExhaustionError
from pdp_router._proxy import (
    _SCORE_TO_CONFIDENCE,
    TrustCache,
    _classify_request,
    _parse_classifier,
    app,
)
from pdp_router._proxy_config import ProxyConfig


@pytest.fixture()
def client():
    """TestClient with ANTHROPIC_API_KEY set."""
    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test", "GEMINI_API_KEY": "gm-test"}),
        TestClient(app) as c,
    ):
        yield c


def _mock_completion(
    text: str = "Hello!", input_tokens: int = 50, output_tokens: int = 20
) -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model="gemini-2.5-flash",
        estimated_cost_usd=0.0001,
    )


class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models"] > 0


class TestModelsEndpoint:
    """OpenAI-compat /v1/models endpoint added 2026-05-19 so Open WebUI / Pal
    Chat / OpenCat etc. can auto-discover the model list at config time and
    populate their dropdowns. Without it those clients refuse to send chat
    completions even though /v1/chat/completions works."""

    def test_models_returns_openai_shape(self, client) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

    def test_models_includes_pdp_auto(self, client) -> None:
        # `pdp-auto` is the cascade-routing virtual model -- the recommended
        # default for any OpenAI-compat client that wants Sibyl's routing
        # brain rather than forcing a specific model.
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert "pdp-auto" in ids

    def test_models_owned_by_sibyl_for_pdp_auto(self, client) -> None:
        resp = client.get("/v1/models")
        pdp_auto = next(m for m in resp.json()["data"] if m["id"] == "pdp-auto")
        assert pdp_auto["owned_by"] == "sibyl"
        assert pdp_auto["object"] == "model"
        assert isinstance(pdp_auto["created"], int)

    def test_models_includes_real_models(self, client) -> None:
        # Make sure the concrete model IDs from DEFAULT_REGISTRY are surfaced
        # so callers can force-route around the cascade when needed (e.g.
        # post-cascade A/B experiments).
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        # Sonnet is in DEFAULT_REGISTRY at tier 2 and is always available.
        assert any("sonnet" in i for i in ids)


class TestScoreToConfidence:
    def test_all_scores_mapped(self) -> None:
        assert set(_SCORE_TO_CONFIDENCE.keys()) == {1, 2, 3, 4, 5}

    def test_scores_decrease(self) -> None:
        confs = [_SCORE_TO_CONFIDENCE[i] for i in range(1, 6)]
        assert confs == sorted(confs, reverse=True)


class TestProxyConfigDefaults:
    def test_trust_db_defaults_to_pdp_router_home(self, monkeypatch) -> None:
        monkeypatch.delenv("PROXY_TRUST_DB", raising=False)
        config = ProxyConfig()
        assert str(config.trust_db_path) == os.path.expanduser("~/.pdp-router/pdp_tracker.db")

    def test_inbox_defaults_to_pdp_router_home(self, monkeypatch) -> None:
        monkeypatch.delenv("PROXY_ROUTING_INBOX_DIR", raising=False)
        config = ProxyConfig()
        assert str(config.routing_inbox_dir) == os.path.expanduser("~/.pdp-router/inbox")

    def test_trust_db_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("PROXY_TRUST_DB", "/tmp/custom/trust.db")
        config = ProxyConfig()
        assert str(config.trust_db_path) == "/tmp/custom/trust.db"

    def test_inbox_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("PROXY_ROUTING_INBOX_DIR", "/tmp/custom/inbox")
        config = ProxyConfig()
        assert str(config.routing_inbox_dir) == "/tmp/custom/inbox"


class TestClassifyRequest:
    @patch("pdp_router._proxy.get_client")
    def test_returns_confidence_and_score(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion(text="3")
        mock_get_client.return_value = mock_client

        config = ProxyConfig()
        from pdp_router._proxy import ChatMessage

        messages = [ChatMessage(role="user", content="Explain recursion")]
        _confidence, score, _panel_score = _classify_request(messages, config)

        assert score == 3
        assert _confidence == 0.55

    @patch("pdp_router._proxy.get_client")
    def test_classifier_failure_falls_back_to_3(self, mock_get_client) -> None:
        mock_get_client.side_effect = Exception("API down")

        config = ProxyConfig()
        from pdp_router._proxy import ChatMessage

        messages = [ChatMessage(role="user", content="test")]
        _confidence, score, _panel_score = _classify_request(messages, config)

        assert score == 3
        assert _confidence == 0.55

    @patch("pdp_router._proxy.get_client")
    def test_non_numeric_response_falls_back(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion(text="moderate")
        mock_get_client.return_value = mock_client

        config = ProxyConfig()
        from pdp_router._proxy import ChatMessage

        messages = [ChatMessage(role="user", content="test")]
        _confidence, score, _panel_score = _classify_request(messages, config)

        assert score == 3


class TestChatCompletions:
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_returns_openai_format(self, mock_get_client, mock_classify, client) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion("Routed response")
        mock_get_client.return_value = mock_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Routed response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 50
        assert data["usage"]["completion_tokens"] == 20
        assert data["usage"]["total_tokens"] == 70

    # Pin web search off: when the flag is on, the cascade path appends the
    # capability hint to the system prompt, which would break the exact-match
    # assertion below. This test is about system extraction, not web search.
    @patch("pdp_router._proxy._web_search_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_system_prompt_extracted(
        self, mock_get_client, mock_classify, _mock_ws, client
    ) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion("ok")
        mock_get_client.return_value = mock_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )

        assert resp.status_code == 200
        call_kwargs = mock_client.complete.call_args
        assert (
            call_kwargs.kwargs.get("system") == "You are helpful"
            or call_kwargs[1].get("system") == "You are helpful"
        )

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_multi_turn_uses_complete_multi(self, mock_get_client, mock_classify, client) -> None:
        mock_client = MagicMock()
        mock_client.complete_multi.return_value = _mock_completion("multi reply")
        mock_get_client.return_value = mock_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ],
            },
        )

        assert resp.status_code == 200
        mock_client.complete_multi.assert_called_once()

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_credit_exhaustion_returns_402(self, mock_get_client, mock_classify, client) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = CreditExhaustionError("out of credits")
        mock_get_client.return_value = mock_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        assert resp.status_code == 402

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_api_error_returns_503(self, mock_get_client, mock_classify, client) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("model not found")
        mock_get_client.return_value = mock_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        assert resp.status_code == 503


class TestExplicitModel:
    """Sibyl panel composer pre-selects models then calls the proxy with
    that ID. The proxy must honor the pick and skip classification --
    cascade is no longer mandatory, just the default for pdp-auto."""

    @patch("pdp_router._proxy._classify_request")
    @patch("pdp_router._proxy.get_client")
    def test_explicit_model_honored(self, mock_get_client, mock_classify, client) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion("haiku says hi")
        mock_get_client.return_value = mock_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "claude-haiku-4-5-20251001"
        # Classifier MUST NOT be called when caller picked explicitly.
        mock_classify.assert_not_called()
        # get_client called with the explicit model.
        call_args = mock_get_client.call_args
        assert call_args.args[0] == "claude-haiku-4-5-20251001"

    @patch("pdp_router._proxy._classify_request")
    @patch("pdp_router._proxy.get_client")
    def test_explicit_model_unknown_rejected(self, mock_get_client, mock_classify, client) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bogus-model-id",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 400
        # No client created, no classifier invoked.
        mock_classify.assert_not_called()
        mock_get_client.assert_not_called()

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_pdp_auto_still_classifies(self, mock_get_client, mock_classify, client) -> None:
        """Regression guard -- explicit-model branch must not break the
        default cascade path that every existing consumer relies on."""
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion("auto routed")
        mock_get_client.return_value = mock_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "explain recursion"}],
            },
        )

        assert resp.status_code == 200
        # Classifier IS called for pdp-auto.
        mock_classify.assert_called_once()


class TestTrustCache:
    def test_empty_db_returns_empty(self, tmp_path) -> None:
        cache = TrustCache(str(tmp_path / "nonexistent.db"))
        assert cache.get_weights() == {}

    def test_reads_db(self, tmp_path) -> None:
        import sqlite3

        db_path = tmp_path / "trust.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE model_trust (model_id TEXT, weight REAL)")
        conn.execute("INSERT INTO model_trust VALUES ('claude-haiku-4-5-20251001', 0.72)")
        conn.execute("INSERT INTO model_trust VALUES ('gemini-2.5-flash', 0.65)")
        conn.commit()
        conn.close()

        cache = TrustCache(str(db_path))
        # Force past the 5s check interval
        cache._last_check = 0.0
        weights = cache.get_weights()

        assert weights["claude-haiku-4-5-20251001"] == 0.72
        assert weights["gemini-2.5-flash"] == 0.65

    def test_initializes_last_check_to_monotonic(self) -> None:
        import time

        before = time.monotonic()
        cache = TrustCache("/tmp/nonexistent.db")
        after = time.monotonic()
        assert before <= cache._last_check <= after

    def test_expands_canonical_to_live_alias(self, tmp_path) -> None:
        """Post-S62 the table holds canonical keys; cache fans them back out
        to live registry aliases so the router's lookups still work."""
        import sqlite3

        db_path = tmp_path / "trust.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE model_trust (model_id TEXT, weight REAL)")
        conn.execute("INSERT INTO model_trust VALUES ('claude-opus-4', 0.81)")
        conn.execute("INSERT INTO model_trust VALUES ('claude-haiku-4', 0.55)")
        conn.execute("INSERT INTO model_trust VALUES ('gemini-2.5-flash-lite', 0.62)")
        conn.commit()
        conn.close()

        cache = TrustCache(str(db_path))
        cache._last_check = 0.0
        weights = cache.get_weights()

        # Canonical Anthropic IDs expand to the current live alias (OPUS).
        assert weights[OPUS] == 0.81
        assert weights["claude-haiku-4-5-20251001"] == 0.55
        # Non-Anthropic IDs stay as-is.
        assert weights["gemini-2.5-flash-lite"] == 0.62
        # Canonical storage keys are NOT exposed to the router.
        assert "claude-opus-4" not in weights
        assert "claude-haiku-4" not in weights


class TestBanditCacheCanonical:
    def test_expands_canonical_to_live_alias(self, tmp_path) -> None:
        """Post-S62 BanditCache fans canonical bandit_state rows back out to
        the live registry aliases the router uses for API calls."""
        import sqlite3

        from pdp_router._bandit import BanditState
        from pdp_router._proxy import BanditCache

        db_path = tmp_path / "trust.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE bandit_state (
                model_id TEXT, mu REAL, sigma REAL, n_obs INTEGER,
                sum_reward REAL, sum_sq_reward REAL,
                effective_n REAL DEFAULT 0.0, effective_sum REAL DEFAULT 0.0
            );
            INSERT INTO bandit_state VALUES
                ('claude-opus-4', 0.62, 0.05, 100, 62.0, 40.0, 20.0, 12.4),
                ('gemini-2.5-flash-lite', 0.55, 0.04, 500, 275.0, 160.0, 20.0, 11.0);
            """
        )
        conn.commit()
        conn.close()

        cache = BanditCache(str(db_path))
        cache._last_check = 0.0
        states = cache.get_states()

        # Live OPUS alias gets the canonical claude-opus-4 posterior.
        assert isinstance(states[OPUS], BanditState)
        assert states[OPUS].mu == 0.62
        assert states[OPUS].n_obs == 100
        # Non-Anthropic ID passes through.
        assert states["gemini-2.5-flash-lite"].mu == 0.55
        # Canonical key not exposed.
        assert "claude-opus-4" not in states


# -- Sprint X.K: parse_classifier + auto-panel gate --


class TestParseClassifier:
    def test_two_int_format(self) -> None:
        assert _parse_classifier("4 8") == (4, 8)
        assert _parse_classifier("1 0") == (1, 0)
        assert _parse_classifier("5 10") == (5, 10)

    def test_single_int_back_compat(self) -> None:
        assert _parse_classifier("4") == (4, 0)
        assert _parse_classifier("1") == (1, 0)

    def test_garbage_falls_back(self) -> None:
        assert _parse_classifier("moderate") == (3, 0)
        assert _parse_classifier("") == (3, 0)
        assert _parse_classifier("no clue") == (3, 0)

    def test_clamps_both_axes(self) -> None:
        assert _parse_classifier("9 99") == (5, 10)
        assert _parse_classifier("0 -5") == (1, 0)
        assert _parse_classifier("7 -1") == (5, 0)

    def test_strips_markdown_fences(self) -> None:
        assert _parse_classifier("```\n4 8\n```") == (4, 8)


@pytest.fixture()
def inbox_dir(client, tmp_path):
    """Per-test override of _config.routing_inbox_dir.

    _config is a frozen dataclass set during lifespan; bypass the freeze
    via object.__setattr__ and restore the original on teardown.
    """
    import json as _json  # local import keeps top namespace clean

    from pdp_router import _proxy

    inbox = tmp_path / "inbox"
    assert _proxy._config is not None
    orig = _proxy._config.routing_inbox_dir
    object.__setattr__(_proxy._config, "routing_inbox_dir", inbox)
    try:
        yield inbox
    finally:
        object.__setattr__(_proxy._config, "routing_inbox_dir", orig)
    # _json captured to keep the import in the fixture's local namespace
    # so tests reading the inbox don't re-import.
    _ = _json


def _read_inbox_rows(inbox_dir):
    """Helper: read all JSONL rows from the per-day inbox file."""
    import json as _json

    files = list(inbox_dir.glob("proxy-*.jsonl"))
    assert len(files) == 1, f"expected 1 inbox file, got {len(files)}"
    return [_json.loads(line) for line in files[0].read_text().splitlines() if line]


class TestAutoPanelGate:
    """Sprint X.K -- clawflag-gated proxy auto-panel decompose+synth path."""

    @patch("pdp_router._proxy._autopanel_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.get_client")
    def test_flag_off_high_panel_score_still_cascades(
        self, mock_get_client, _mock_classify, _mock_flag, client
    ) -> None:
        """Regression guard: with clawflag off, panel_score=9 must NOT route panel."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("cascade response")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X and Y"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "pdp-panel" not in data["model"]
        mock_llm.complete.assert_called_once()

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 4))
    @patch("pdp_router._proxy.get_client")
    def test_flag_on_low_panel_score_still_cascades(
        self, mock_get_client, _mock_classify, _mock_flag, client
    ) -> None:
        """Flag on but panel_score below threshold 7 -> cascade path."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("cascade")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "trivial"}],
            },
        )

        assert resp.status_code == 200
        assert "pdp-panel" not in resp.json()["model"]

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_flag_on_high_panel_score_triggers_panel(
        self,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        """Flag on + panel_score=9 + pdp-auto + non-stream -> 3 members + chair."""
        mock_compose.return_value = [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]

        def make_client(*args, **kwargs):
            m = MagicMock()
            m.complete.return_value = _mock_completion("panelist answer")
            return m

        mock_get_client.side_effect = make_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare lock-free vs mutex queue"}],
                "max_tokens": 500,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"].startswith("pdp-panel-")
        assert "claude-sonnet-4-6" in data["model"]
        # 3 panel members + 1 chair = 4 get_client calls.
        assert mock_get_client.call_count == 4

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request")
    @patch("pdp_router._proxy.get_client")
    def test_skipped_when_explicit_model(
        self, mock_get_client, mock_classify, _mock_flag, client
    ) -> None:
        """Explicit model -> classifier never called, panel never triggered."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("haiku")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "compare X and Y"}],
            },
        )

        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-haiku-4-5-20251001"
        mock_classify.assert_not_called()

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.get_client")
    def test_skipped_when_streaming(
        self,
        mock_get_client,
        _mock_classify,
        _mock_streaming,
        _mock_panel,
        client,
    ) -> None:
        """Stream + high panel_score -> SSE stream (cascade-streaming), not panel."""

        async def fake_stream(**_kwargs):
            yield "Hello"

        mock_llm = MagicMock()
        mock_llm.stream_complete = lambda **kw: fake_stream(**kw)
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X and Y"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_panel_member_failure_excluded_from_chair(
        self,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        """One failing member -> chair sees N-1; response model reflects survivors."""
        mock_compose.return_value = [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]

        def make_client(model_id, **kwargs):
            m = MagicMock()
            if "deepseek" in model_id:
                m.complete.side_effect = RuntimeError("upstream timeout")
            else:
                m.complete.return_value = _mock_completion(f"answer from {model_id}")
            return m

        mock_get_client.side_effect = make_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "complex Q"}],
            },
        )

        assert resp.status_code == 200
        assert "pdp-panel-2+" in resp.json()["model"]

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_panel_all_failures_returns_503(
        self,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        """Every panel member fails -> 503 (chair never invoked)."""
        mock_compose.return_value = [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]
        m = MagicMock()
        m.complete.side_effect = RuntimeError("all upstream down")
        mock_get_client.return_value = m

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "complex Q"}],
            },
        )

        assert resp.status_code == 503

    @patch("pdp_router._proxy._autopanel_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_cascade_writes_jsonl_row(
        self,
        mock_get_client,
        _mock_classify,
        _mock_flag,
        client,
        inbox_dir,
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("cascade")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 200
        rows = _read_inbox_rows(inbox_dir)
        assert len(rows) == 1
        assert rows[0]["routing_mode"] == "cascade"
        assert rows[0]["context_bucket"] == "chat:cascade"
        assert rows[0]["domain"] == "chat"
        assert rows[0]["prediction_id"] == 0

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_panel_writes_jsonl_rows(
        self,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
        inbox_dir,
    ) -> None:
        mock_compose.return_value = [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]

        def make_client(*args, **kwargs):
            m = MagicMock()
            m.complete.return_value = _mock_completion("answer")
            return m

        mock_get_client.side_effect = make_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "complex"}],
            },
        )

        assert resp.status_code == 200
        rows = _read_inbox_rows(inbox_dir)
        assert len(rows) == 4
        modes = [r["routing_mode"] for r in rows]
        assert modes.count("panel") == 3
        assert modes.count("panel_chair") == 1
        # All rows share the same chat_request_id.
        ctx_ids = {__import__("json").loads(r["context_json"])["chat_request_id"] for r in rows}
        assert len(ctx_ids) == 1

    @patch("pdp_router._proxy._autopanel_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_prediction_id_header_present_for_cascade(
        self,
        mock_get_client,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        m = MagicMock()
        m.complete.return_value = _mock_completion("cascade")
        mock_get_client.return_value = m

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 200
        # Header is case-insensitive but starlette lowercases on read.
        uid = resp.headers.get("x-pdp-prediction-id")
        assert uid is not None
        assert len(uid) == 36  # uuid4 hex shape

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_prediction_id_header_present_for_panel(
        self,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        mock_compose.return_value = [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]

        def make_client(*args, **kwargs):
            m = MagicMock()
            m.complete.return_value = _mock_completion("a")
            return m

        mock_get_client.side_effect = make_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "complex"}],
            },
        )

        assert resp.status_code == 200
        assert resp.headers.get("x-pdp-prediction-id") is not None

    @patch("pdp_router._proxy._autopanel_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_prediction_id_matches_chat_request_id_in_jsonl(
        self,
        mock_get_client,
        _mock_classify,
        _mock_flag,
        client,
        inbox_dir,
    ) -> None:
        import json as _json

        m = MagicMock()
        m.complete.return_value = _mock_completion("cascade")
        mock_get_client.return_value = m

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        uid = resp.headers["x-pdp-prediction-id"]
        rows = _read_inbox_rows(inbox_dir)
        ctx = _json.loads(rows[0]["context_json"])
        assert ctx["chat_request_id"] == uid

    @patch("pdp_router._proxy._autopanel_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_prediction_id_header_survives_cascade_503(
        self,
        mock_get_client,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        """FastAPI's HTTPException handler builds a fresh JSONResponse; the
        proxy must explicitly attach the header on error paths via the
        HTTPException(headers=...) param so the inbox drain can correlate failures."""
        m = MagicMock()
        m.complete.side_effect = Exception("upstream down")
        mock_get_client.return_value = m

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 503
        assert resp.headers.get("x-pdp-prediction-id") is not None
        assert len(resp.headers["x-pdp-prediction-id"]) == 36

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_prediction_id_header_survives_panel_all_failures_503(
        self,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        mock_compose.return_value = [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]
        m = MagicMock()
        m.complete.side_effect = RuntimeError("all upstream down")
        mock_get_client.return_value = m

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "complex"}],
            },
        )

        assert resp.status_code == 503
        assert resp.headers.get("x-pdp-prediction-id") is not None
        assert len(resp.headers["x-pdp-prediction-id"]) == 36

    def test_prediction_id_header_survives_unknown_model_400(self, client) -> None:
        """The 400 unknown-model branch raises HTTPException too; header must survive."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "totally-bogus-model-id",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 400
        assert resp.headers.get("x-pdp-prediction-id") is not None
        assert len(resp.headers["x-pdp-prediction-id"]) == 36

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    @patch("pdp_router._proxy.synthesize_chair")
    def test_chair_empty_uses_chair_fallback_label(
        self,
        mock_synth,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
    ) -> None:
        """When chair returns empty content, the response.model field must say
        'chair_fallback' so downstream grading doesn't attribute single-arm
        output to the synthesis path."""
        from pdp_router._panel import ChairSynthResult

        mock_compose.return_value = [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]

        def make_client(*args, **kwargs):
            m = MagicMock()
            m.complete.return_value = _mock_completion("a real answer")
            return m

        mock_get_client.side_effect = make_client

        mock_synth.return_value = ChairSynthResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            latency_ms=10.0,
            chair_model="claude-sonnet-4-6",
            error="empty_response",
        )

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "complex"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "chair_fallback" in data["model"]
        # Content is the first survivor's text, not empty.
        assert data["choices"][0]["message"]["content"] == "a real answer"


class TestWebSearchGate:
    """Tier B -- clawflag-gated proxy web search on the cascade path."""

    @patch("pdp_router._proxy._web_search_enabled", return_value=False)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_flag_off_cascade_no_web_search(
        self, mock_get_client, _mock_classify, _mock_flag, client
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert resp.status_code == 200
        assert mock_llm.complete.call_args.kwargs.get("enable_web_search") is False

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_flag_on_cascade_passes_flag_and_nudges_system(
        self, mock_get_client, _mock_classify, _mock_flag, client
    ) -> None:
        """Flag on -> complete(enable_web_search=True) and the system prompt is
        augmented with the capability hint while preserving the caller's prompt."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [
                    {"role": "system", "content": "You are Richard."},
                    {"role": "user", "content": "search for the latest on X"},
                ],
            },
        )

        assert resp.status_code == 200
        kwargs = mock_llm.complete.call_args.kwargs
        assert kwargs.get("enable_web_search") is True
        assert "web_search tool" in kwargs.get("system", "")
        assert "You are Richard." in kwargs.get("system", "")

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch("pdp_router._proxy.get_client")
    def test_flag_on_streaming_passes_flag(
        self, mock_get_client, _mock_classify, _mock_streaming, _mock_flag, client
    ) -> None:
        captured: dict = {}

        async def fake_stream(**kwargs):
            captured.update(kwargs)
            yield "Hello"

        mock_llm = MagicMock()
        mock_llm.stream_complete = lambda **kw: fake_stream(**kw)
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search for X"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        # Drain the stream so the generator body (and capture) actually runs.
        _ = resp.text
        assert captured.get("enable_web_search") is True

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_panel_path_stays_search_free(
        self, mock_get_client, mock_compose, _mock_classify, _mock_panel, _mock_ws, client
    ) -> None:
        """MVP defers panel: even with both flags on, no panel member or chair
        call is made with enable_web_search=True."""
        mock_compose.return_value = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]
        created: list = []

        def make_client(*args, **kwargs):
            m = MagicMock()
            m.complete.return_value = _mock_completion("panelist")
            created.append(m)
            return m

        mock_get_client.side_effect = make_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare X and Y"}],
                "max_tokens": 300,
            },
        )

        assert resp.status_code == 200
        assert resp.json()["model"].startswith("pdp-panel-")
        assert created  # clients were built
        for m in created:
            for call in m.complete.call_args_list:
                assert call.kwargs.get("enable_web_search", False) is False
