# Description: Tests for the PDP Router Proxy FastAPI endpoints.
# Description: Uses TestClient to validate routing, classification, and error handling.

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pdp_router import _proxy
from pdp_router._clients import CompletionResult
from pdp_router._models import OPUS, CreditExhaustionError
from pdp_router._proxy import (
    _SCORE_TO_CONFIDENCE,
    ChatMessage,
    TrustCache,
    _classify_request,
    _classify_retryable,
    _has_search_intent,
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

    def test_reports_configured_providers(self, client) -> None:
        """`models: 11` is a static registry count and says nothing about reachability.

        A first-run user needs to see whether their key actually landed, without
        having to send a billable request to find out.
        """
        data = client.get("/health").json()
        assert data["providers"]["anthropic"] is True
        assert data["providers"]["gemini"] is True
        assert data["providers"]["openrouter"] is False
        assert data["providers"]["vertex"] is False

    def test_reports_missing_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True), TestClient(app) as c:
            data = c.get("/health").json()
        assert data["providers"]["anthropic"] is False
        assert data["providers"]["gemini"] is False

    def test_reports_absent_trust_db(self, client) -> None:
        data = client.get("/health").json()
        assert data["trust_db"]["present"] is False

    def test_reports_present_trust_db(self, tmp_path) -> None:
        import sqlite3

        db = tmp_path / "trust.db"
        conn = sqlite3.connect(db)
        conn.executescript("CREATE TABLE model_trust (model_id TEXT, weight REAL);")
        conn.close()
        with (
            patch.dict(os.environ, {"PROXY_TRUST_DB": str(db)}),
            TestClient(app) as c,
        ):
            data = c.get("/health").json()
        assert data["trust_db"]["present"] is True
        assert data["trust_db"]["readable"] is True

    def test_reports_unreadable_trust_db(self, tmp_path) -> None:
        """A file that exists but is not a database must not read as healthy."""
        db = tmp_path / "trust.db"
        db.write_text("this is not sqlite")
        with (
            patch.dict(os.environ, {"PROXY_TRUST_DB": str(db)}),
            TestClient(app) as c,
        ):
            data = c.get("/health").json()
        assert data["trust_db"]["present"] is True
        assert data["trust_db"]["readable"] is False

    def test_reports_routing_mode(self, client) -> None:
        assert client.get("/health").json()["routing_mode"] == "cascade"


def _trust_db(path, *, trust=(), bandit=()):
    """Build a trust DB using exactly the schema the README documents."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE model_trust (model_id TEXT, weight REAL);"
        "CREATE TABLE bandit_state (model_id TEXT, mu REAL, sigma REAL,"
        " n_obs INTEGER, sum_reward REAL, sum_sq_reward REAL,"
        " effective_n REAL, effective_sum REAL);"
    )
    for model_id, weight in trust:
        conn.execute("INSERT INTO model_trust VALUES (?,?)", (model_id, weight))
    for row in bandit:
        conn.execute("INSERT INTO bandit_state VALUES (?,?,?,?,?,?,?,?)", row)
    conn.commit()
    conn.close()
    return path


class TestCacheColdStart:
    """The first read must actually read.

    Both caches primed _last_check to time.monotonic() in __init__, so the
    5-second poll throttle short-circuited the very first call and returned the
    empty initial value. A ROUTING_MODE=bandit deployment therefore ran the
    plain cascade until 5 seconds had elapsed, with no signal that it was doing
    so, and the learned layer looked switched on while doing nothing.
    """

    def test_trust_weights_available_on_first_call(self, tmp_path) -> None:
        db = _trust_db(tmp_path / "t.db", trust=[("claude-opus-4", 0.9)])
        cache = TrustCache(str(db), ttl=300)
        assert cache.get_weights(), "first call returned nothing; cold start is broken"

    def test_bandit_states_available_on_first_call(self, tmp_path) -> None:
        db = _trust_db(
            tmp_path / "t.db",
            bandit=[("claude-opus-4", 0.98, 0.005, 900, 880.0, 870.0, 900.0, 882.0)],
        )
        cache = _proxy.BanditCache(str(db), ttl=300)
        assert cache.get_states(), "first call returned nothing; bandit can never engage"


class TestCacheFailureVisibility:
    """Absent is supported and quiet; present-but-broken is a user error and loud.

    The old code swallowed every exception at log.debug, invisible at the
    default level, so a bandit_state table missing two columns downgraded the
    user to the static cascade forever with zero output. Verified against the
    live proxy: a malformed schema produced HTTP 200 and not one log line.
    """

    def test_absent_db_is_quiet(self, tmp_path, caplog) -> None:
        cache = TrustCache(str(tmp_path / "nope.db"), ttl=300)
        with caplog.at_level(logging.WARNING, logger="pdp_router._proxy"):
            assert cache.get_weights() == {}
        assert not caplog.records, "an absent trust DB is a supported setup, not a warning"

    def test_unreadable_db_warns(self, tmp_path, caplog) -> None:
        bad = tmp_path / "bad.db"
        bad.write_text("this is not sqlite")
        cache = TrustCache(str(bad), ttl=300)
        with caplog.at_level(logging.WARNING, logger="pdp_router._proxy"):
            cache.get_weights()
        assert [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_wrong_schema_warns(self, tmp_path, caplog) -> None:
        """The realistic user error: followed the README but dropped two columns."""
        import sqlite3

        bad = tmp_path / "partial.db"
        conn = sqlite3.connect(bad)
        conn.executescript(
            "CREATE TABLE bandit_state (model_id TEXT, mu REAL, sigma REAL,"
            " n_obs INTEGER, sum_reward REAL, sum_sq_reward REAL);"
        )
        conn.commit()
        conn.close()
        cache = _proxy.BanditCache(str(bad), ttl=300)
        with caplog.at_level(logging.WARNING, logger="pdp_router._proxy"):
            cache.get_states()
        assert "bandit" in caplog.text.lower()

    def test_warning_does_not_repeat_every_poll(self, tmp_path, caplog) -> None:
        """One warning per failure episode, not one per request forever."""
        bad = tmp_path / "bad.db"
        bad.write_text("not sqlite")
        cache = TrustCache(str(bad), ttl=300)
        with caplog.at_level(logging.WARNING, logger="pdp_router._proxy"):
            for _ in range(5):
                cache._last_poll = float("-inf")  # force each poll past the throttle
                cache.get_weights()
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"


class TestRoutingModeProvenance:
    """The logged routing_mode must be the policy that actually made the pick.

    Every _make_routing_row call site passed a hardcoded literal, so a
    ROUTING_MODE=bandit deployment recorded every Thompson-sampled decision as
    "cascade". routing_mode is the field you group by to compare policies, so a
    wrong value silently invalidates the outcome analysis the whole design rests
    on. Verified against the live proxy: bandit posteriors with opus mu=0.98 and
    no trust weights sent a trivial prompt to Opus 3/3 while every row said
    cascade.
    """

    def _rows(self, tmp_path) -> list[dict]:
        import json as _json

        rows = []
        for path in sorted(tmp_path.glob("*.jsonl")):
            rows += [_json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return rows

    def _post(self, tmp_path, env: dict, body: dict) -> list[dict]:
        mock = MagicMock()
        mock.complete_multi.return_value = _mock_completion()
        mock.complete.return_value = _mock_completion(text="1 0")
        base = {
            "ANTHROPIC_API_KEY": "sk-test",
            "GEMINI_API_KEY": "gm-test",
            "PROXY_ROUTING_INBOX_DIR": str(tmp_path),
        }
        base.update(env)
        with (
            patch.dict(os.environ, base),
            patch.object(_proxy, "get_client", return_value=mock),
            TestClient(app) as c,
        ):
            c.post("/v1/chat/completions", json=body)
        return self._rows(tmp_path)

    def test_cascade_mode_records_cascade(self, tmp_path) -> None:
        rows = self._post(
            tmp_path,
            {"ROUTING_MODE": "cascade", "PROXY_EXPLORE_RATE": "0"},
            {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert [r["routing_mode"] for r in rows] == ["cascade"]

    def test_bandit_mode_records_bandit(self, tmp_path) -> None:
        """The executed policy was Thompson Sampling; the row must say so."""
        db = _trust_db(
            tmp_path / "trust.db",
            bandit=[("claude-opus-4", 0.98, 0.005, 900, 880.0, 870.0, 900.0, 882.0)],
        )
        rows = self._post(
            tmp_path,
            {
                "ROUTING_MODE": "bandit",
                "PROXY_EXPLORE_RATE": "0",
                "PROXY_TRUST_DB": str(db),
            },
            {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert [r["routing_mode"] for r in rows] == ["bandit"]

    def test_bandit_mode_without_posteriors_records_cascade(self, tmp_path) -> None:
        """Configured bandit but no usable posteriors executes the cascade, so log cascade."""
        rows = self._post(
            tmp_path,
            {
                "ROUTING_MODE": "bandit",
                "PROXY_EXPLORE_RATE": "0",
                "PROXY_TRUST_DB": str(tmp_path / "absent.db"),
            },
            {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert [r["routing_mode"] for r in rows] == ["cascade"]

    def test_pinned_model_is_not_a_cascade_decision(self, tmp_path) -> None:
        """An explicit model pin bypasses routing entirely; calling it cascade is a lie."""
        rows = self._post(
            tmp_path,
            {"PROXY_EXPLORE_RATE": "0"},
            {"model": OPUS, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert [r["routing_mode"] for r in rows] == ["explicit"]

    def test_explore_branch_is_marked(self, tmp_path) -> None:
        """cascade_explored exists in the drain schema and was never populated.

        Uniform-random explore picks must be excludable from agreement-rate
        analytics, which is exactly what the column is documented for.
        """
        rows = self._post(
            tmp_path,
            {"ROUTING_MODE": "cascade", "PROXY_EXPLORE_RATE": "1.0"},
            {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert rows[0]["cascade_explored"] is True

    def test_threshold_pick_is_not_marked_explored(self, tmp_path) -> None:
        rows = self._post(
            tmp_path,
            {"ROUTING_MODE": "cascade", "PROXY_EXPLORE_RATE": "0"},
            {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert rows[0]["cascade_explored"] is False

    def test_row_keys_stay_within_the_drain_signature(self, tmp_path) -> None:
        """pdp_record_routing_decision takes explicit kwargs; unknown keys would TypeError."""
        accepted = {
            "alert_id",
            "model_selected",
            "context_json",
            "context_bucket",
            "confidence",
            "domain",
            "severity",
            "agreement_level",
            "routing_mode",
            "prediction_id",
            "shadow_model_selected",
            "cascade_pick_mu",
            "shadow_pick_mu",
            "cascade_pick_n_obs",
            "shadow_pick_n_obs",
            "cascade_explored",
        }
        rows = self._post(
            tmp_path,
            {"PROXY_EXPLORE_RATE": "0"},
            {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert set(rows[0]) <= accepted, f"unknown keys: {set(rows[0]) - accepted}"

    def test_correlation_id_matches_the_response_header(self, tmp_path) -> None:
        """context_json.chat_request_id is the documented join key for outcomes."""
        import json as _json

        mock = MagicMock()
        mock.complete_multi.return_value = _mock_completion()
        mock.complete.return_value = _mock_completion(text="1 0")
        with (
            patch.dict(
                os.environ,
                {
                    "ANTHROPIC_API_KEY": "sk-test",
                    "GEMINI_API_KEY": "gm-test",
                    "PROXY_ROUTING_INBOX_DIR": str(tmp_path),
                    "PROXY_EXPLORE_RATE": "0",
                },
            ),
            patch.object(_proxy, "get_client", return_value=mock),
            TestClient(app) as c,
        ):
            resp = c.post(
                "/v1/chat/completions",
                json={"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
            )
        header_id = resp.headers["X-PDP-Prediction-Id"]
        row = self._rows(tmp_path)[0]
        assert _json.loads(row["context_json"])["chat_request_id"] == header_id
        assert row["alert_id"] == f"chat-{header_id}"
        assert row["prediction_id"] == 0


class TestErrorResponseShape:
    """Errors must be OpenAI-compatible and valid JSON.

    FastAPI wraps HTTPException detail under `detail`, and the previous code
    passed str(model_dump()), producing a Python repr with single quotes nested
    inside a JSON string. An OpenAI SDK client cannot parse that. The streaming
    surface already emits {"error": {...}} frames, so this aligns the two.
    """

    def test_unknown_model_is_openai_shaped(self, client) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert "detail" not in body
        assert isinstance(body["error"]["message"], str)
        assert "no-such-model" in body["error"]["message"]
        assert isinstance(body["error"]["type"], str)

    def test_client_creation_failure_is_openai_shaped(self, client) -> None:
        with patch.object(_proxy, "get_client", side_effect=RuntimeError("no credentials")):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": OPUS, "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 503
        body = resp.json()
        assert "detail" not in body
        assert "no credentials" in body["error"]["message"]
        assert body["error"]["type"] == "server_error"

    def test_credit_exhaustion_is_openai_shaped(self, client) -> None:
        mock = MagicMock()
        mock.complete_multi.side_effect = CreditExhaustionError("out of credits")
        mock.complete.side_effect = CreditExhaustionError("out of credits")
        with patch.object(_proxy, "get_client", return_value=mock):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": OPUS, "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 402
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "billing_error"
        assert "out of credits" in body["error"]["message"]

    def test_validation_errors_are_openai_shaped_too(self, client) -> None:
        """422s must carry the same {"error": {...}} envelope as every other error.

        The HTTPException handler covered raise sites but not pydantic
        validation, so a malformed request still returned FastAPI's
        {"detail": [...]}. That left the surface inconsistent: an OpenAI client
        could parse a routing error and not a bad-request error. Real OpenAI
        returns the error envelope for both.
        """
        resp = client.post(
            "/openai/v1/chat/completions",
            json={"model": "pdp-auto", "messages": [{"role": "user", "content": None}]},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "invalid_request_error"
        assert isinstance(body["error"]["message"], str) and body["error"]["message"]

    def test_validation_error_message_keeps_the_field_path(self, client) -> None:
        """Losing which field failed would make the envelope useless for debugging."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "pdp-auto", "messages": [{"role": "user"}]},
        )
        assert resp.status_code == 422
        assert "content" in resp.json()["error"]["message"]

    def test_error_body_has_no_python_repr(self, client) -> None:
        """Regression guard: the old shape embedded "{'error': {'message': ...}}"."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert "'" not in resp.text or '"error"' in resp.text
        assert not resp.text.startswith('{"detail"')


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
        # default for any OpenAI-compat client that wants routed model selection
        # rather than forcing a specific model.
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert "pdp-auto" in ids

    def test_models_owned_by_router_for_pdp_auto(self, client) -> None:
        resp = client.get("/v1/models")
        pdp_auto = next(m for m in resp.json()["data"] if m["id"] == "pdp-auto")
        assert pdp_auto["owned_by"] == "pdp-router"
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

    def test_panel_transcript_derives_from_inbox_when_env_unset(self, monkeypatch) -> None:
        """GAP-2: with no PROXY_PANEL_TRANSCRIPT_DIR, the transcript dir is a sibling of
        the (env-overridden) inbox, so the two cannot drift to different bases."""
        monkeypatch.delenv("PROXY_PANEL_TRANSCRIPT_DIR", raising=False)
        monkeypatch.setenv("PROXY_ROUTING_INBOX_DIR", "/home/x/.pdp-router/inbox")
        config = ProxyConfig()
        assert str(config.panel_transcript_dir) == "/home/x/.pdp-router/panel-transcripts"

    def test_panel_transcript_env_override_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("PROXY_PANEL_TRANSCRIPT_DIR", "/tmp/custom/transcripts")
        monkeypatch.setenv("PROXY_ROUTING_INBOX_DIR", "/home/x/.pdp-router/inbox")
        config = ProxyConfig()
        assert str(config.panel_transcript_dir) == "/tmp/custom/transcripts"


class TestFlagEnvFallback:
    """When clawflag is absent (a standalone build), the flag helpers honor
    PROXY_*_ENABLED env vars so the flag-gated features are reachable. With
    clawflag present, clawflag wins and the env vars are not consulted."""

    def test_autopanel_env_fallback_enables(self, monkeypatch) -> None:
        monkeypatch.setattr(_proxy, "_clawflag", None)
        monkeypatch.setenv("PROXY_AUTOPANEL_ENABLED", "1")
        assert _proxy._autopanel_enabled() is True

    def test_autopanel_env_fallback_defaults_off(self, monkeypatch) -> None:
        monkeypatch.setattr(_proxy, "_clawflag", None)
        monkeypatch.delenv("PROXY_AUTOPANEL_ENABLED", raising=False)
        assert _proxy._autopanel_enabled() is False

    def test_panel_streaming_env_fallback_defaults_on(self, monkeypatch) -> None:
        monkeypatch.setattr(_proxy, "_clawflag", None)
        monkeypatch.delenv("PROXY_PANEL_STREAMING_ENABLED", raising=False)
        assert _proxy._panel_streaming_enabled() is True

    def test_panel_streaming_env_fallback_can_disable(self, monkeypatch) -> None:
        monkeypatch.setattr(_proxy, "_clawflag", None)
        monkeypatch.setenv("PROXY_PANEL_STREAMING_ENABLED", "false")
        assert _proxy._panel_streaming_enabled() is False

    def test_clawflag_present_takes_precedence_over_env(self, monkeypatch) -> None:
        class _FakeFlag:
            @staticmethod
            def get_bool(key: str, default: bool = False) -> bool:
                return False

        monkeypatch.setattr(_proxy, "_clawflag", _FakeFlag())
        monkeypatch.setenv("PROXY_AUTOPANEL_ENABLED", "1")
        assert _proxy._autopanel_enabled() is False


class TestEffortRouting:
    """_handle_chat computes the deterministic effort level from the score and
    threads it to the model call -- only for pdp-auto picks, only when the flag is
    on, only for effort-capable arms. _route_request is patched so the pick + score
    are fixed (no classifier call, no epsilon-greedy explore randomness)."""

    _ROUTE = (
        "claude-opus-4-8",  # model_name (effort-capable arm)
        0.15,  # confidence
        5,  # score -> high effort
        0,  # panel_score
        False,  # search_intent
        "",  # system
        [ChatMessage(role="user", content="hard")],  # non_system
        _proxy._RouteProvenance(mode="cascade", explored=False),  # provenance
    )

    def _mock_client(self) -> MagicMock:
        m = MagicMock()
        m.complete.return_value = _mock_completion("ok")
        return m

    @patch("pdp_router._proxy._effort_routing_enabled", return_value=True)
    @patch("pdp_router._proxy._route_request", return_value=_ROUTE)
    @patch("pdp_router._proxy.get_client")
    def test_effort_threaded_when_flag_on(self, mock_get_client, _route, _flag, client) -> None:
        mock_llm = self._mock_client()
        mock_get_client.return_value = mock_llm
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "pdp-auto", "messages": [{"role": "user", "content": "hard"}]},
        )
        assert resp.status_code == 200
        assert mock_llm.complete.call_args.kwargs["effort"] == "high"

    @patch("pdp_router._proxy._effort_routing_enabled", return_value=False)
    @patch("pdp_router._proxy._route_request", return_value=_ROUTE)
    @patch("pdp_router._proxy.get_client")
    def test_effort_none_when_flag_off(self, mock_get_client, _route, _flag, client) -> None:
        mock_llm = self._mock_client()
        mock_get_client.return_value = mock_llm
        client.post(
            "/v1/chat/completions",
            json={"model": "pdp-auto", "messages": [{"role": "user", "content": "hard"}]},
        )
        assert mock_llm.complete.call_args.kwargs["effort"] is None

    @patch("pdp_router._proxy._effort_routing_enabled", return_value=True)
    @patch("pdp_router._proxy.get_client")
    def test_effort_none_for_explicit_model(self, mock_get_client, _flag, client) -> None:
        """Explicit-model requests skip classification (score=0) and must not get an
        auto-effort even with the flag on -- gated on request.model == pdp-auto."""
        mock_llm = self._mock_client()
        mock_get_client.return_value = mock_llm
        resp = client.post(
            "/v1/chat/completions",
            json={"model": OPUS, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert mock_llm.complete.call_args.kwargs["effort"] is None

    @patch("pdp_router._proxy._effort_routing_enabled", return_value=True)
    @patch(
        "pdp_router._proxy._route_request",
        return_value=(
            "claude-haiku-4-5-20251001",  # no-dial arm
            0.95,
            1,
            0,
            False,
            "",
            [ChatMessage(role="user", content="easy")],
            _proxy._RouteProvenance(mode="cascade", explored=False),
        ),
    )
    @patch("pdp_router._proxy.get_client")
    def test_no_dial_model_gets_no_effort_when_flag_on(
        self, mock_get_client, _route, _flag, client
    ) -> None:
        """supports_effort excludes Haiku/Gemini/DeepSeek/Llama at the proxy layer:
        even with the flag on and a real score, a no-dial pick gets effort=None."""
        mock_llm = self._mock_client()
        mock_get_client.return_value = mock_llm
        client.post(
            "/v1/chat/completions",
            json={"model": "pdp-auto", "messages": [{"role": "user", "content": "easy"}]},
        )
        assert mock_llm.complete.call_args.kwargs["effort"] is None


class TestClassifyRequest:
    @patch("pdp_router._proxy.get_client")
    def test_returns_confidence_and_score(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion(text="3")
        mock_get_client.return_value = mock_client

        config = ProxyConfig(gemini_api_key="gk-test")
        from pdp_router._proxy import ChatMessage

        messages = [ChatMessage(role="user", content="Explain recursion")]
        _confidence, score, _panel_score = _classify_request(messages, config)

        assert score == 3
        assert _confidence == 0.55

    @patch("pdp_router._proxy.get_client")
    def test_classifier_failure_falls_back_to_3(self, mock_get_client) -> None:
        mock_get_client.side_effect = Exception("API down")

        config = ProxyConfig(gemini_api_key="gk-test")
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

        config = ProxyConfig(classify_fallback_model="", gemini_api_key="gk-test")
        from pdp_router._proxy import ChatMessage

        messages = [ChatMessage(role="user", content="test")]
        _confidence, score, _panel_score = _classify_request(messages, config)

        assert score == 3

    @patch("pdp_router._proxy.get_client")
    def test_retries_transient_then_succeeds(self, mock_get_client) -> None:
        """A transient classifier 503 is retried; the second call's score is used."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = [
            RuntimeError("503 UNAVAILABLE: model is overloaded"),
            _mock_completion(text="4 8"),
        ]
        mock_get_client.return_value = mock_client

        config = ProxyConfig(gemini_api_key="gk-test")
        messages = [ChatMessage(role="user", content="compare A vs B")]
        with patch.dict(os.environ, {"PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"}):
            confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (4, 8)
        assert confidence == _SCORE_TO_CONFIDENCE[4]
        assert mock_client.complete.call_count == 2

    @patch("pdp_router._proxy.get_client")
    def test_persistent_transient_falls_back_after_retries(self, mock_get_client) -> None:
        """A classifier that keeps 503ing falls back to (3, 0) after the retry budget."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("503 UNAVAILABLE")
        mock_get_client.return_value = mock_client

        config = ProxyConfig(classify_fallback_model="", gemini_api_key="gk-test")
        messages = [ChatMessage(role="user", content="test")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "2", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (3, 0)
        assert mock_client.complete.call_count == 3  # initial + 2 retries

    @patch("pdp_router._proxy.get_client")
    def test_non_transient_error_does_not_retry(self, mock_get_client) -> None:
        """A non-transient error (bad key) falls back immediately, no wasted retries."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = ValueError("invalid api key")
        mock_get_client.return_value = mock_client

        config = ProxyConfig(classify_fallback_model="", gemini_api_key="gk-test")
        messages = [ChatMessage(role="user", content="test")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "2", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (3, 0)
        assert mock_client.complete.call_count == 1


class TestClassifyFallback:
    """Cross-lineage fallback classifier: rescues the (3, 0) collapse that
    silently disables the auto-panel when the primary classifier is down."""

    def test_missing_primary_credentials_skips_the_doomed_attempt(self, caplog) -> None:
        """A single-provider user must not see a traceback on the happy path.

        The default classifier is gemini-2.5-flash-lite, so an Anthropic-only
        user had every request construct a Gemini client that could only raise,
        logging a full ValueError traceback before recovering. The README says
        one key is enough; the logs said otherwise.
        """
        fallback = MagicMock()
        fallback.complete.return_value = _mock_completion(text="4 8")
        config = ProxyConfig(anthropic_api_key="sk-test", gemini_api_key="")
        messages = [ChatMessage(role="user", content="compare A vs B")]

        with (
            patch("pdp_router._proxy.get_client", return_value=fallback) as mock_get_client,
            caplog.at_level(logging.DEBUG, logger="pdp_router._proxy"),
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (4, 8)
        # The uncredentialed primary is never constructed at all.
        assert mock_get_client.call_count == 1
        assert mock_get_client.call_args.args[0] == config.classify_fallback_model
        assert "Traceback" not in caplog.text
        assert not [r for r in caplog.records if r.exc_info]

    def test_genuine_primary_failure_still_logs_the_traceback(self, caplog) -> None:
        """Diagnostics are preserved: a credentialed classifier that breaks is loud."""
        primary = MagicMock()
        primary.complete.side_effect = RuntimeError("503 UNAVAILABLE")
        fallback = MagicMock()
        fallback.complete.return_value = _mock_completion(text="2 0")
        config = ProxyConfig(anthropic_api_key="sk-test", gemini_api_key="gk-test")
        messages = [ChatMessage(role="user", content="test")]

        with (
            patch("pdp_router._proxy.get_client", side_effect=[primary, fallback]),
            patch.dict(
                os.environ,
                {"PROXY_CLASSIFY_RETRIES": "0", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
            ),
            caplog.at_level(logging.WARNING, logger="pdp_router._proxy"),
        ):
            _classify_request(messages, config)

        assert [r for r in caplog.records if r.exc_info], "real failures must keep exc_info"

    def test_no_credentials_at_all_collapses_quietly(self) -> None:
        """Neither model is reachable: collapse to (3, 0) without constructing clients."""
        config = ProxyConfig(anthropic_api_key="", gemini_api_key="")
        messages = [ChatMessage(role="user", content="test")]
        with patch("pdp_router._proxy.get_client") as mock_get_client:
            _confidence, score, panel_score = _classify_request(messages, config)
        assert (score, panel_score) == (3, 0)
        mock_get_client.assert_not_called()

    def _config(self, **overrides) -> ProxyConfig:
        kwargs = {
            "anthropic_api_key": "sk-test",
            "gemini_api_key": "gk-test",
        }
        kwargs.update(overrides)
        return ProxyConfig(**kwargs)

    @patch("pdp_router._proxy.get_client")
    def test_fallback_rescues_exhausted_primary(self, mock_get_client) -> None:
        primary = MagicMock()
        primary.complete.side_effect = RuntimeError("503 UNAVAILABLE")
        fallback = MagicMock()
        fallback.complete.return_value = _mock_completion(text="4 8")
        mock_get_client.side_effect = [primary, fallback]

        config = self._config()
        messages = [ChatMessage(role="user", content="compare A vs B")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "0", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (4, 8)
        assert confidence == _SCORE_TO_CONFIDENCE[4]
        assert mock_get_client.call_count == 2
        primary_call, fallback_call = mock_get_client.call_args_list
        # Primary is gemini-*: gets the Gemini key, not anthropic-or-gemini soup.
        assert primary_call.args[0] == config.classify_model
        assert primary_call.kwargs["api_key"] == "gk-test"
        # Fallback is claude-*: gets the Anthropic key.
        assert fallback_call.args[0] == config.classify_fallback_model
        assert fallback_call.args[0].startswith("claude-")
        assert fallback_call.kwargs["api_key"] == "sk-test"

    @patch("pdp_router._proxy.get_client")
    def test_fallback_covers_non_retryable_primary_error(self, mock_get_client) -> None:
        primary = MagicMock()
        primary.complete.side_effect = ValueError("400 bad request")
        fallback = MagicMock()
        fallback.complete.return_value = _mock_completion(text="2 0")
        mock_get_client.side_effect = [primary, fallback]

        config = self._config()
        messages = [ChatMessage(role="user", content="test")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "2", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (2, 0)
        assert primary.complete.call_count == 1  # non-retryable: no wasted retries

    @patch("pdp_router._proxy.get_client")
    def test_fallback_skipped_without_creds(self, mock_get_client) -> None:
        primary = MagicMock()
        primary.complete.side_effect = RuntimeError("503 UNAVAILABLE")
        mock_get_client.return_value = primary

        config = self._config(anthropic_api_key="")
        messages = [ChatMessage(role="user", content="test")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "0", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (3, 0)
        assert mock_get_client.call_count == 1

    @patch("pdp_router._proxy.get_client")
    def test_fallback_failure_collapses_to_default(self, mock_get_client) -> None:
        broken = MagicMock()
        broken.complete.side_effect = RuntimeError("503 UNAVAILABLE")
        mock_get_client.return_value = broken

        config = self._config()
        messages = [ChatMessage(role="user", content="test")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "0", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (3, 0)
        assert mock_get_client.call_count == 2

    @patch("pdp_router._proxy.get_client")
    def test_fallback_rescues_unparseable_primary_reply(self, mock_get_client) -> None:
        """An HTTP-200 reply the parser cannot read is a classifier failure:
        it must reach the fallback, not silently collapse to (3, 0)."""
        primary = MagicMock()
        primary.complete.return_value = _mock_completion(text="Complexity: 4, Panel: 8")
        fallback = MagicMock()
        fallback.complete.return_value = _mock_completion(text="4 8")
        mock_get_client.side_effect = [primary, fallback]

        config = self._config()
        messages = [ChatMessage(role="user", content="compare A vs B")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "0", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (4, 8)
        assert mock_get_client.call_count == 2

    @patch("pdp_router._proxy.get_client")
    def test_same_model_fallback_skipped(self, mock_get_client) -> None:
        """fallback == primary is a doomed retry against the same dead model;
        the guard must skip it and collapse directly."""
        broken = MagicMock()
        broken.complete.side_effect = RuntimeError("503 UNAVAILABLE")
        mock_get_client.return_value = broken

        config = self._config(
            classify_model="gemini-2.5-flash-lite",
            classify_fallback_model="gemini-2.5-flash-lite",
        )
        messages = [ChatMessage(role="user", content="test")]
        with patch.dict(
            os.environ,
            {"PROXY_CLASSIFY_RETRIES": "0", "PROXY_CLASSIFY_RETRY_BACKOFF_S": "0"},
        ):
            _confidence, score, panel_score = _classify_request(messages, config)

        assert (score, panel_score) == (3, 0)
        assert mock_get_client.call_count == 1

    @patch("pdp_router._proxy.get_client")
    def test_claude_primary_gets_anthropic_key(self, mock_get_client) -> None:
        """Regression: a claude-* classify model used to receive the Gemini key
        whenever both keys were set (gemini_api_key or anthropic_api_key)."""
        mock_client = MagicMock()
        mock_client.complete.return_value = _mock_completion(text="3")
        mock_get_client.return_value = mock_client

        config = self._config(classify_model="claude-haiku-4-5-20251001")
        messages = [ChatMessage(role="user", content="test")]
        _classify_request(messages, config)

        assert mock_get_client.call_args.kwargs["api_key"] == "sk-test"


class TestClassifyRetryable:
    def test_status_code_5xx_and_429_are_retryable(self) -> None:
        for code, expected in [
            (503, True),
            (429, True),
            (500, True),
            (502, True),
            (400, False),
            (404, False),
            (403, False),
        ]:
            e = RuntimeError("boom")
            e.code = code  # type: ignore[attr-defined]
            assert _classify_retryable(e) is expected, code

    def test_message_signals_are_retryable(self) -> None:
        assert _classify_retryable(RuntimeError("503 UNAVAILABLE")) is True
        assert _classify_retryable(RuntimeError("RESOURCE_EXHAUSTED")) is True
        assert _classify_retryable(Exception("deadline exceeded")) is True
        assert _classify_retryable(Exception("429 too many requests")) is True

    def test_non_transient_messages_are_not_retryable(self) -> None:
        assert _classify_retryable(ValueError("invalid api key")) is False
        assert _classify_retryable(ValueError("malformed request")) is False
        assert _classify_retryable(Exception("")) is False


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
    """The panel composer pre-selects models then calls the proxy with
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

    def test_first_poll_is_not_throttled(self) -> None:
        """The poll clock starts un-primed so the first read actually happens.

        This previously asserted _last_check == monotonic(), which pinned the
        cold-start bug in place: priming the clock to "now" made the 5-second
        throttle swallow the first call and return the empty initial value.
        """
        cache = TrustCache("/tmp/nonexistent.db")
        assert cache._last_poll == float("-inf")
        assert cache._last_read == float("-inf")

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

    def test_garbage_returns_none(self) -> None:
        # None signals parse failure so _classify_request can route it to the
        # cross-lineage fallback instead of silently collapsing to (3, 0).
        assert _parse_classifier("moderate") is None
        assert _parse_classifier("") is None
        assert _parse_classifier("no clue") is None
        assert _parse_classifier("4/8") is None

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


@pytest.fixture()
def transcript_dir(client, tmp_path):
    """Per-test override of _config.panel_transcript_dir (mirrors inbox_dir)."""
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
    import json as _json

    files = list(transcript_dir.glob("panel-*.jsonl"))
    assert len(files) == 1, f"expected 1 transcript file, got {len(files)}"
    return [_json.loads(line) for line in files[0].read_text().splitlines() if line]


class TestPanelTranscriptCapture:
    """proxy_panel_transcript_enabled -- persist full panel turns for the chat-quality eval."""

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_nonstream_panel_writes_transcript(
        self,
        mock_get_client,
        mock_compose,
        _classify,
        _autopanel,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        mock_compose.return_value = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]

        def make_client(*_a, **_k):
            m = MagicMock()
            m.complete.return_value = _mock_completion("panelist answer")
            return m

        mock_get_client.side_effect = make_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare lock-free vs mutex"}],
                "max_tokens": 500,
            },
        )
        assert resp.status_code == 200
        rows = _read_transcript_rows(transcript_dir)
        assert len(rows) == 1
        rec = rows[0]
        assert rec["surface"] == "nonstream"
        assert rec["panel_score"] == 9
        assert rec["score"] == 3
        assert rec["prompt"] == "compare lock-free vs mutex"
        assert [m["model_id"] for m in rec["members"]] == [
            "claude-opus-4-7",
            "gemini-2.5-pro",
            "deepseek-chat",
        ]
        assert all(m["text"] == "panelist answer" for m in rec["members"])
        # chair runs through the same mock .complete, so the synthesis is non-empty.
        assert rec["synthesis_text"] == "panelist answer"
        assert rec["synthesis_status"] == "complete"
        assert rec["messages"] == [{"role": "user", "content": "compare lock-free vs mutex"}]

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=False)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_nonstream_panel_no_transcript_when_flag_off(
        self,
        mock_get_client,
        mock_compose,
        _classify,
        _autopanel,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        mock_compose.return_value = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]

        def make_client(*_a, **_k):
            m = MagicMock()
            m.complete.return_value = _mock_completion("panelist answer")
            return m

        mock_get_client.side_effect = make_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare lock-free vs mutex"}],
                "max_tokens": 500,
            },
        )
        assert resp.status_code == 200
        # The panel actually ran (so "no file" is meaningful, not vacuous)...
        assert resp.json()["model"].startswith("pdp-panel-")
        # ...but the flag is off, so nothing was persisted.
        assert not transcript_dir.exists() or not list(transcript_dir.glob("panel-*.jsonl"))

    @patch("pdp_router._proxy._panel_transcript_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_nonstream_chair_empty_records_empty_synthesis_not_fallback(
        self,
        mock_get_client,
        mock_compose,
        _classify,
        _autopanel,
        _transcript,
        transcript_dir,
        client,
    ) -> None:
        # Chair (claude-sonnet-4-6) returns empty -> the proxy substitutes the
        # first-survivor text into the client response, but the transcript must record
        # synthesis_text='' (chair.text), status chair_empty -- NOT the survivor text.
        mock_compose.return_value = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]

        def make_client(model_id, *_a, **_k):
            m = MagicMock()
            m.complete.return_value = _mock_completion(
                "" if model_id == "claude-sonnet-4-6" else "member text"
            )
            return m

        mock_get_client.side_effect = make_client
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "q"}],
                "max_tokens": 500,
            },
        )
        assert resp.status_code == 200
        # Client got the first-survivor fallback + the chair_fallback relabel.
        assert resp.json()["choices"][0]["message"]["content"] == "member text"
        assert "chair_fallback" in resp.json()["model"]
        rec = _read_transcript_rows(transcript_dir)[0]
        assert rec["synthesis_text"] == ""
        assert rec["synthesis_status"] == "chair_empty"
        assert "member text" not in rec["synthesis_text"]


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
    def test_errored_member_excluded_from_routing_rows(
        self,
        mock_get_client,
        mock_compose,
        _mock_classify,
        _mock_flag,
        client,
        inbox_dir,
    ) -> None:
        """An errored panel member must not get a panel_member routing row.

        Regression for the results-vs-survivors bug: errored arms carry no
        signal and must never reach the JSONL inbox the bandit drain consumes.
        """
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

        rows = _read_inbox_rows(inbox_dir)
        member_rows = [r for r in rows if r["routing_mode"] == "panel"]
        chair_rows = [r for r in rows if r["routing_mode"] == "panel_chair"]
        member_models = {r["model_selected"] for r in member_rows}

        # The errored deepseek arm gets no row; only the 2 survivors do.
        assert "deepseek-chat" not in member_models
        assert member_models == {"claude-opus-4-7", "gemini-2.5-pro"}
        assert len(member_rows) == 2
        assert len(chair_rows) == 1

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


class TestSearchIntentGateAndFloor:
    """Search-intent routing: detect explicit web-search intent, skip the
    search-free auto-panel, and floor non-searcher cascade picks to Sonnet so
    the attached web_search tool actually fires (Haiku/meta/openai/qwen do not)."""

    @pytest.mark.parametrize(
        "text",
        [
            "search the web for the best framework",
            "do a web search for Y",
            "browse the web for Z",
            "look it up online",
            "google it",
            "google the release date of X",
            "what is the latest news on AI",
            "the latest AI news",
            "current headlines about the vote",
            "today's news",
            "as of today, what changed",
        ],
    )
    def test_has_search_intent_positive(self, text) -> None:
        assert _has_search_intent([ChatMessage(role="user", content=text)]) is True

    @pytest.mark.parametrize(
        "text",
        [
            # Routine coding phrasing must NOT trip the gate (heavy-coding-user
            # precision; these are the adversarial-review false positives).
            "write a function to search a list",
            "binary search for this sorted array",
            "search for the user in the database",
            "explain binary search",
            "implement a hashtable lookup",
            "look up the value in the hashmap",
            "port the service to google cloud run",
            "from google.cloud import storage",
            "get_weather(city)",
            "stock price moving average from this CSV",
            "config url is https://example.com",
            "bump to the latest release of pytest",
            "fix the latest test results",
            "who won the merge conflict",
            "refactor the current code",
            "what is the latest version of the code I wrote",
            "compare Postgres and MySQL",
        ],
    )
    def test_has_search_intent_negative(self, text) -> None:
        assert _has_search_intent([ChatMessage(role="user", content=text)]) is False

    def test_has_search_intent_uses_latest_user_message(self) -> None:
        msgs = [
            ChatMessage(role="user", content="search for the latest X"),
            ChatMessage(role="assistant", content="..."),
            ChatMessage(role="user", content="now refactor it"),
        ]
        assert _has_search_intent(msgs) is False

    def test_has_search_intent_empty(self) -> None:
        assert _has_search_intent([]) is False

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_floor_haiku_to_sonnet(self, mock_get_client, _mock_cascade, _mock_ws, client) -> None:
        """Search intent + flag on + cascade picked Haiku -> floored to Sonnet."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search for the latest news"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-sonnet-4-6"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch(
        "pdp_router._proxy.confidence_cascade",
        return_value=("meta/llama-4-scout-17b-16e-instruct-maas", False),
    )
    @patch("pdp_router._proxy.get_client")
    def test_floor_meta_llama_to_sonnet(
        self, mock_get_client, _mock_cascade, _mock_ws, client
    ) -> None:
        """A non-searcher pick (meta/llama rejects the tool) is floored to Sonnet.
        meta/llama is one of the models the ~10% epsilon-greedy explore branch can
        emit; the floor sits after confidence_cascade so it catches those too."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "look it up online"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-sonnet-4-6"

    @patch("pdp_router._proxy._web_search_enabled", return_value=False)
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_no_floor_when_web_search_off(
        self, mock_get_client, _mock_cascade, _mock_ws, client
    ) -> None:
        """Flag off -> no floor even on a search-intent query (zero behavior change)."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search for the latest news"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-haiku-4-5-20251001"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy.confidence_cascade", return_value=("gemini-2.5-pro", False))
    @patch("pdp_router._proxy.get_client")
    def test_no_floor_when_already_searcher(
        self, mock_get_client, _mock_cascade, _mock_ws, client
    ) -> None:
        """A reliable-searcher cascade pick (gemini-2.5-pro) is left untouched."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search for the latest news"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "gemini-2.5-pro"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_search_intent_skips_panel_and_floors(
        self, mock_get_client, _mock_cascade, _mock_classify, _mock_panel, _mock_ws, client
    ) -> None:
        """Search intent wins over panel_score>=7: skip the search-free panel,
        stay on cascade, and floor Haiku -> Sonnet with web search attached."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search for the latest AI news"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pdp-panel" not in data["model"]
        assert data["model"] == "claude-sonnet-4-6"
        mock_llm.complete.assert_called_once()
        assert mock_llm.complete.call_args.kwargs.get("enable_web_search") is True

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9))
    @patch("pdp_router._proxy.compose_panel")
    @patch("pdp_router._proxy.get_client")
    def test_panel_still_fires_without_search_intent(
        self, mock_get_client, mock_compose, _mock_classify, _mock_panel, _mock_ws, client
    ) -> None:
        """Control: panel_score=9 + no search intent -> panel still fires
        (the skip is search-intent-gated, not a blanket disable)."""
        mock_compose.return_value = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]

        def make_client(*args, **kwargs):
            m = MagicMock()
            m.complete.return_value = _mock_completion("panelist answer")
            return m

        mock_get_client.side_effect = make_client

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "compare lock-free vs mutex queues"}],
                "max_tokens": 500,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"].startswith("pdp-panel-")

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy.get_client")
    def test_explicit_model_never_floored(self, mock_get_client, _mock_ws, client) -> None:
        """Explicit model selection bypasses the floor (search_intent=False)."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-haiku-4-5-20251001",
                "messages": [{"role": "user", "content": "search for the latest news"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-haiku-4-5-20251001"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_routing_row_records_search_intent(
        self, mock_get_client, _mock_cascade, _mock_ws, client, inbox_dir
    ) -> None:
        """The cascade routing-decision row records search_intent for the drain."""
        import json as _json

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search for the latest news"}],
            },
        )
        assert resp.status_code == 200
        rows = _read_inbox_rows(inbox_dir)
        assert len(rows) == 1
        assert _json.loads(rows[0]["context_json"])["search_intent"] is True

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy.confidence_cascade", return_value=("gemini-2.5-flash-lite", False))
    @patch("pdp_router._proxy.get_client")
    def test_floor_flash_lite_to_sonnet(
        self, mock_get_client, _mock_cascade, _mock_ws, client
    ) -> None:
        """Flash-Lite is deliberately excluded from reliable searchers -> floored."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search the web for X"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-sonnet-4-6"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy.confidence_cascade", return_value=("claude-opus-4-8", False))
    @patch("pdp_router._proxy.get_client")
    def test_no_floor_when_opus(self, mock_get_client, _mock_cascade, _mock_ws, client) -> None:
        """A low-confidence Opus cascade pick already searches -> left untouched."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search the web for X"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-opus-4-8"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.95, 1, 3))
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_floor_without_panel_when_low_panel_score(
        self, mock_get_client, _mock_cascade, _mock_classify, _mock_panel, _mock_ws, client
    ) -> None:
        """Search intent + low panel_score (no panel) -> plain cascade floored to Sonnet."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "search the web for X"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pdp-panel" not in data["model"]
        assert data["model"] == "claude-sonnet-4-6"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0))
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_streaming_search_intent_floors_and_searches(
        self, mock_get_client, _mock_cascade, _mock_classify, _mock_streaming, _mock_ws, client
    ) -> None:
        """A streaming search-intent request whose cascade pick is a non-searcher is
        floored to Sonnet and still streams with web search enabled."""
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
                "messages": [{"role": "user", "content": "search the web for today's headlines"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        _ = resp.text  # drain so the stream generator runs
        assert mock_get_client.call_args.args[0] == "claude-sonnet-4-6"
        assert captured.get("enable_web_search") is True
        assert "claude-sonnet-4-6" in resp.text

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_routing_row_records_search_intent_false(
        self, mock_get_client, _mock_cascade, _mock_ws, client, inbox_dir
    ) -> None:
        """A non-search query records search_intent=False in the routing row."""
        import json as _json

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ans")
        mock_get_client.return_value = mock_llm
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto",
                "messages": [{"role": "user", "content": "explain how a hashmap works"}],
            },
        )
        assert resp.status_code == 200
        rows = _read_inbox_rows(inbox_dir)
        assert len(rows) == 1
        assert _json.loads(rows[0]["context_json"])["search_intent"] is False


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
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "search for the latest on X"},
                ],
            },
        )

        assert resp.status_code == 200
        kwargs = mock_llm.complete.call_args.kwargs
        assert kwargs.get("enable_web_search") is True
        assert "web_search tool" in kwargs.get("system", "")
        assert "You are a helpful assistant." in kwargs.get("system", "")

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
