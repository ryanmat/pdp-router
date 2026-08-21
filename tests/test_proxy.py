# Description: Tests for the PDP Router Proxy FastAPI endpoints.
# Description: Uses TestClient to validate routing, classification, and error handling.

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
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
from pdp_router._tools import ToolCall, ToolTranslationError


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

    def test_health_reports_the_real_package_version(self, client) -> None:
        """The deploy gate reads /health's version to prove which build is
        serving, so it must be the installed package metadata, never a
        hardcoded literal that survives a release bump."""
        import importlib.metadata

        resp = client.get("/health")
        assert resp.json()["version"] == importlib.metadata.version("pdp-router")
        assert _proxy.app.version == importlib.metadata.version("pdp-router")

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

    def test_tool_passthrough_defaults_off(self, monkeypatch) -> None:
        """Default OFF is the whole safety story for the tool-passthrough work:
        until it is deliberately enabled, /openai/v1 behaves exactly as it does
        today for every existing client."""
        monkeypatch.setattr(_proxy, "_clawflag", None)
        monkeypatch.delenv("PROXY_TOOL_PASSTHROUGH_ENABLED", raising=False)
        assert _proxy._tool_passthrough_enabled() is False

    def test_tool_passthrough_env_fallback_enables(self, monkeypatch) -> None:
        monkeypatch.setattr(_proxy, "_clawflag", None)
        monkeypatch.setenv("PROXY_TOOL_PASSTHROUGH_ENABLED", "1")
        assert _proxy._tool_passthrough_enabled() is True

    def test_tool_passthrough_reads_its_own_clawflag_key(self, monkeypatch) -> None:
        """Guards against a copy-paste of a neighbouring helper's flag key, which
        would silently tie tool passthrough to another feature's kill-switch."""
        seen: list[str] = []

        class _RecordingFlag:
            @staticmethod
            def get_bool(key: str, default: bool = False) -> bool:
                seen.append(key)
                return True

        monkeypatch.setattr(_proxy, "_clawflag", _RecordingFlag())
        assert _proxy._tool_passthrough_enabled() is True
        assert seen == ["pipeline.proxy_tool_passthrough_enabled"]


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
        "general",  # task_category
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
            "general",
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
        _confidence, score, _panel_score, _task_category = _classify_request(messages, config)

        assert score == 3
        assert _confidence == 0.55

    @patch("pdp_router._proxy.get_client")
    def test_classifier_failure_falls_back_to_3(self, mock_get_client) -> None:
        mock_get_client.side_effect = Exception("API down")

        config = ProxyConfig(gemini_api_key="gk-test")
        from pdp_router._proxy import ChatMessage

        messages = [ChatMessage(role="user", content="test")]
        _confidence, score, _panel_score, _task_category = _classify_request(messages, config)

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
        _confidence, score, _panel_score, _task_category = _classify_request(messages, config)

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
            confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)
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
            confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
            _confidence, score, panel_score, _task_category = _classify_request(messages, config)

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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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

_A_CALL = ToolCall(id="call_abc", name="run", arguments='{"cmd": "ls -la"}')


def _tool_completion(
    text: str = "",
    tool_calls: tuple[ToolCall, ...] = (_A_CALL,),
    finish_reason: str = "tool_calls",
) -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=50,
        output_tokens=20,
        model="claude-sonnet-4-6",
        estimated_cost_usd=0.0001,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


class TestToolPassthroughNonStream:
    """Prompt 6: /openai/v1 takes tools and returns assistant tool_calls.

    The flag is patched explicitly in every test rather than left to its
    default. _flag_enabled reads the developer's real flags.toml, so a test
    that relies on the default passes for the wrong reason today and flips the
    day the flag is turned on in production.
    """

    @staticmethod
    def _route(non_system: list | None = None) -> tuple:
        """The fixed 8-tuple. Patching _route_request rather than the classifier
        keeps the pick deterministic (the cascade's explore step is random) and
        keeps the classifier's network call out of the suite."""
        return (
            "claude-sonnet-4-6",
            0.5,
            4,
            0,
            "general",
            False,
            "",
            non_system
            if non_system is not None
            else [ChatMessage(role="user", content="list files")],
            _proxy._RouteProvenance(mode="cascade", explored=False),
        )

    @staticmethod
    def _mock(result: CompletionResult | None = None) -> MagicMock:
        m = MagicMock()
        m.complete_with_tools.return_value = result or _tool_completion()
        m.complete.return_value = _mock_completion("plain")
        m.complete_multi.return_value = _mock_completion("plain")
        return m

    def _body(self, **overrides) -> dict:
        body: dict = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "list files"}],
            "tools": _TOOLS_PARAM,
        }
        body.update(overrides)
        return body

    # -- behavior: the with-tools branch --

    def test_tool_calls_reach_the_response_in_openai_shape(self, client) -> None:
        """Asserted on the TOP-LEVEL response body on purpose. Pydantic
        serializes through the ANNOTATED type, so a ToolResponseMessage sitting
        in an un-narrowed slot drops tool_calls with no error; dumping the
        message alone or the choice alone both pass regardless."""
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._body())

        assert resp.status_code == 200
        message = resp.json()["choices"][0]["message"]
        assert message["tool_calls"] == [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "run", "arguments": '{"cmd": "ls -la"}'},
            }
        ]
        assert message["content"] is None
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
        assert resp.json()["usage"]["total_tokens"] == 70
        assert resp.headers["X-PDP-Prediction-Id"]

    def test_text_and_tool_calls_both_survive(self, client) -> None:
        mock_llm = self._mock(_tool_completion(text="Let me look."))
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._body())

        message = resp.json()["choices"][0]["message"]
        assert message["content"] == "Let me look."
        assert len(message["tool_calls"]) == 1

    def test_tools_ride_verbatim_to_the_client(self, client) -> None:
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            client.post(
                "/openai/v1/chat/completions",
                json=self._body(tool_choice="auto", parallel_tool_calls=False),
            )

        kwargs = mock_llm.complete_with_tools.call_args.kwargs
        assert kwargs["tools"] == _TOOLS_PARAM
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["parallel_tool_calls"] is False

    def test_second_round_transcript_reaches_the_client(self, client) -> None:
        """The round trip that makes tool calling work at all: the assistant
        turn goes back out carrying tool_calls and NO content key (omitted, not
        null -- that is the shape OpenAI clients send), and the result turn
        keeps the tool_call_id that binds it to its call."""
        history = [
            _proxy.ToolChatMessage(role="user", content="list files"),
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
        mock_llm = self._mock(_tool_completion(text="Done.", tool_calls=(), finish_reason="stop"))
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route(history)),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._body())

        assert resp.status_code == 200
        sent = mock_llm.complete_with_tools.call_args.kwargs["messages"]
        assert sent[0] == {"role": "user", "content": "list files"}
        assert "content" not in sent[1]
        assert sent[1]["tool_calls"][0]["id"] == "call_abc"
        assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"cmd": "ls"}'
        assert sent[2] == {"role": "tool", "content": "a.txt", "tool_call_id": "call_abc"}

    def test_no_empty_content_warning_on_a_tool_only_turn(self, client, caplog) -> None:
        """A tool-only turn legitimately has no text. The plain path's warning
        exists to flag a safety-filtered empty answer and would fire on every
        single tool call if it were reused here."""
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
            caplog.at_level(logging.WARNING),
        ):
            client.post("/openai/v1/chat/completions", json=self._body())

        assert "Empty content" not in caplog.text

    def test_empty_stop_turn_still_warns(self, client, caplog) -> None:
        """The counterpart: a turn that ended with "stop" and no text is the
        condition the warning was written for, and it must survive the move."""
        mock_llm = self._mock(_tool_completion(text="", tool_calls=(), finish_reason="stop"))
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
            caplog.at_level(logging.WARNING),
        ):
            client.post("/openai/v1/chat/completions", json=self._body())

        assert "Empty content" in caplog.text

    def test_tool_request_writes_one_routing_row(self, client) -> None:
        """The router exists to learn from its picks; a branch that skips the
        row would silently stop feeding the bandit for every tool request."""
        inbox = Path(os.environ["PROXY_ROUTING_INBOX_DIR"])
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            client.post("/openai/v1/chat/completions", json=self._body())

        rows = [
            json.loads(line)
            for path in sorted(inbox.glob("*.jsonl"))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["model_selected"] == "claude-sonnet-4-6"
        assert rows[0]["routing_mode"] == "cascade"

    def test_streaming_tool_request_writes_exactly_one_routing_row(self, client) -> None:
        """A streaming tool turn is a real exposure and has to feed the bandit
        exactly once, like its non-streaming sibling above.

        This replaces the Prompt 6 interim pair (a 501 for the streaming leg, and
        no row written for that refusal). The wire shape now served is pinned in
        test_proxy_streaming.py::TestToolStreaming; what matters here is that
        adding the branch neither skipped the row nor doubled it.
        """
        inbox = Path(os.environ["PROXY_ROUTING_INBOX_DIR"])
        mock_llm = self._mock()

        async def _stream(*_args: object, **_kwargs: object):
            yield "the answer"

        mock_llm.stream_with_tools = _stream
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._streaming_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._body(stream=True))

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        rows = [
            json.loads(line)
            for path in sorted(inbox.glob("*.jsonl"))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["model_selected"] == "claude-sonnet-4-6"

    def test_plain_request_still_takes_the_legacy_path(self, client) -> None:
        """Flag on but no tools and no tool history: nothing changes."""
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_count == 0
        assert mock_llm.complete.call_count == 1

    # -- guards: the flag-off contract, which the annotation switch endangers --

    def test_flag_off_rejects_tool_calls_with_null_content(self, client) -> None:
        """Ruling 2026-07-27. Widening the endpoint annotation is itself what
        makes this shape parse: ToolChatMessage defaults content to None, so
        without an explicit re-reject a transcript that 422s today would start
        flowing down the legacy path with its tool_calls silently dropped.
        """
        mock_route = MagicMock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False),
            patch("pdp_router._proxy._route_request", mock_route),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                },
            )

        assert resp.status_code == 422
        assert "detail" not in resp.json()
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert "content" in resp.json()["error"]["message"]
        assert mock_route.call_count == 0

    def test_flag_off_omitted_content_with_tool_calls_is_also_rejected(self, client) -> None:
        """The omitted-key form is what real OpenAI clients send, and it reaches
        the same defaulted None as an explicit null."""
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": "{}"},
                                }
                            ],
                        }
                    ],
                },
            )

        assert resp.status_code == 422

    def test_flag_off_tool_role_message_still_reaches_the_provider(self, client) -> None:
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "tool", "tool_call_id": "call_abc", "content": "a.txt"},
                    ],
                },
            )

        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_count == 0

    def test_flag_off_assistant_tool_calls_with_text_proceeds_without_tools(self, client) -> None:
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "thinking",
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": "{}"},
                                }
                            ],
                        }
                    ],
                },
            )

        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_count == 0

    def test_flag_off_bare_tools_param_is_stripped(self, client) -> None:
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False),
            patch("pdp_router._proxy._route_request", return_value=self._route()),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=self._body())

        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_count == 0
        assert mock_llm.complete.call_count == 1

    def test_content_null_without_tool_calls_is_still_422(self, client) -> None:
        """Shape 1 of the flag-off contract, and the one existing assertion the
        annotation switch could quietly change."""
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=False):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={"model": "pdp-auto", "messages": [{"role": "user", "content": None}]},
            )

        assert resp.status_code == 422
        assert "content" in resp.json()["error"]["message"]

    def test_v1_never_grows_a_tool_surface(self, client) -> None:
        """/v1 keeps the original models forever: tool_calls with null content
        stays a 422 there whatever the flag says, because the annotation is
        never widened."""
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "run", "arguments": "{}"},
                                }
                            ],
                        }
                    ],
                },
            )

        assert resp.status_code == 422


_USER_TURN = {"role": "user", "content": "hi"}
_GOOD_FUNCTION = {"name": "run", "arguments": "{}"}


def _assistant_with_call(call: dict) -> dict:
    """An assistant turn whose content is a real string, so the turn itself is
    never the shape-5 rejection -- only the tool call inside it is malformed."""
    return {"role": "assistant", "content": "thinking", "tool_calls": [call]}


# Payloads whose tool-shaped fields are malformed while role and content stay
# valid strings. Every one parses under the legacy models, because the malformed
# part sits in a field ChatCompletionRequest drops as an extra.
_MALFORMED_TOOL_SHAPES: dict[str, dict] = {
    "call_type_custom_without_function": {
        "messages": [_USER_TURN, _assistant_with_call({"id": "call_1", "type": "custom"})]
    },
    "call_omits_arguments": {
        "messages": [
            _USER_TURN,
            _assistant_with_call({"id": "call_1", "type": "function", "function": {"name": "run"}}),
        ]
    },
    "arguments_sent_as_an_object": {
        "messages": [
            _USER_TURN,
            _assistant_with_call(
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run", "arguments": {"cmd": "ls"}},
                }
            ),
        ]
    },
    "call_id_null": {
        "messages": [
            _USER_TURN,
            _assistant_with_call({"id": None, "type": "function", "function": _GOOD_FUNCTION}),
        ]
    },
    "call_id_missing": {
        "messages": [
            _USER_TURN,
            _assistant_with_call({"type": "function", "function": _GOOD_FUNCTION}),
        ]
    },
    "call_id_int": {
        "messages": [
            _USER_TURN,
            _assistant_with_call({"id": 7, "type": "function", "function": _GOOD_FUNCTION}),
        ]
    },
    "tool_calls_sent_as_an_object": {
        "messages": [
            _USER_TURN,
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": {"id": "call_1", "type": "function", "function": _GOOD_FUNCTION},
            },
        ]
    },
    "tool_call_id_non_string": {
        "messages": [_USER_TURN, {"role": "tool", "content": "a.txt", "tool_call_id": 7}]
    },
    "message_name_non_string": {"messages": [{"role": "user", "content": "hi", "name": 7}]},
    "tools_sent_as_an_object": {
        "messages": [_USER_TURN],
        "tools": {"type": "function", "function": {"name": "run"}},
    },
    "tool_choice_sent_as_a_bool": {
        "messages": [_USER_TURN],
        "tools": _TOOLS_PARAM,
        "tool_choice": True,
    },
    "parallel_tool_calls_non_bool": {
        "messages": [_USER_TURN],
        "tools": _TOOLS_PARAM,
        # Not "yes"/"no": pydantic reads those as booleans, so they would prove
        # nothing about a field that rejects non-bools.
        "parallel_tool_calls": "maybe",
    },
}

_MALFORMED_PARAMS = [pytest.param(body, id=name) for name, body in _MALFORMED_TOOL_SHAPES.items()]


class TestFlagOffMalformedToolShapes:
    """Malformed tool-shaped payloads keep the answer they have always had.

    Validating the tool schema at the endpoint annotation puts it in FastAPI's
    dependency layer, which runs before the flag is ever read: a payload that has
    always been parsed-and-dropped-and-served would start answering 422 with
    passthrough OFF, where the contract promises behavior identical to today.
    The strictness lives behind the flag instead, and applies to exactly the
    requests that carry tools to a provider.

    None of these are shapes a real client sends -- Crush and the OpenAI SDKs
    send canonical function calls, which is why every passthrough test above
    misses this and why the suite stayed green through the break.
    """

    _MODEL = "claude-haiku-4-5-20251001"

    def _post(self, client, shape: dict, *, flag: bool, path: str = "/openai/v1/chat/completions"):
        """Post a shape with the flag forced. The model is explicit so the real
        _route_request runs and skips the classifier, leaving the provider client
        as the only mock."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion()
        mock_llm.complete_multi.return_value = _mock_completion()
        mock_llm.complete_with_tools.return_value = _tool_completion()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=flag),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(path, json={"model": self._MODEL, **shape})
        return resp, mock_llm

    @pytest.mark.parametrize("shape", _MALFORMED_PARAMS)
    def test_flag_off_serves_every_malformed_shape(self, shape, client) -> None:
        resp, mock_llm = self._post(client, shape, flag=False)

        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_count == 0

    @pytest.mark.parametrize("shape", _MALFORMED_PARAMS)
    def test_flag_on_refuses_every_malformed_shape(self, shape, client) -> None:
        """The other half of the contract: turning the flag on must not turn the
        strict schema off. These are the 422s the annotation used to produce."""
        resp, _ = self._post(client, shape, flag=True)

        assert resp.status_code == 422

    @pytest.mark.parametrize("shape", _MALFORMED_PARAMS)
    def test_legacy_models_parse_every_malformed_shape(self, shape) -> None:
        """What makes "identical to today" measurable rather than asserted.

        Today's flag-off answer IS whatever ChatCompletionRequest does with the
        payload, and /v1 keeps that model forever (invariant 1). A clean parse
        here is why the 200 above is the right expectation; if one of these ever
        stops parsing, its flag-off expectation has to move with it.
        """
        request = _proxy.ChatCompletionRequest.model_validate({"model": self._MODEL, **shape})

        assert request.model_extra is None

    @pytest.mark.parametrize("shape", _MALFORMED_PARAMS)
    def test_v1_serves_every_malformed_shape_whatever_the_flag(self, shape, client) -> None:
        resp, _ = self._post(client, shape, flag=True, path="/v1/chat/completions")

        assert resp.status_code == 200

    @pytest.mark.parametrize("shape", _MALFORMED_PARAMS)
    def test_v1_serves_every_malformed_shape_with_the_flag_off(self, shape, client) -> None:
        """The other flag state: /v1 is provably flag-independent in both
        directions (invariant 1's regression wall)."""
        resp, _ = self._post(client, shape, flag=False, path="/v1/chat/completions")

        assert resp.status_code == 200

    def test_flag_on_rejection_is_openai_shaped_and_keeps_the_field_path(self, client) -> None:
        """Moving validation out of the dependency layer must not change the
        body a client reads: same envelope, same folded field path, no `detail`."""
        resp, _ = self._post(
            client, _MALFORMED_TOOL_SHAPES["arguments_sent_as_an_object"], flag=True
        )

        assert resp.status_code == 422
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "invalid_request_error"
        # startswith and single-line, not a substring match: raw str(ValidationError)
        # also CONTAINS the path, so a substring assertion passes even if the fold
        # is dropped and pydantic's multi-line dump reaches the client verbatim.
        message = body["error"]["message"]
        assert message.startswith("messages.1.tool_calls.0.function.arguments: ")
        assert "\n" not in message

    def test_flag_on_rejection_precedes_routing(self, client) -> None:
        """No classifier spend on a payload that never reaches a provider -- the
        ordering _reject_tool_shaped_messages already holds on the flag-off side."""
        mock_route = MagicMock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", mock_route),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={"model": "pdp-auto", **_MALFORMED_TOOL_SHAPES["arguments_sent_as_an_object"]},
            )

        assert resp.status_code == 422
        assert mock_route.call_count == 0

    def test_flag_on_canonical_second_round_reaches_the_provider(self, client) -> None:
        """The payload the whole feature exists for, posted as a real client sends it.

        An OpenAI client's second round omits the content key on the tool_calls
        turn rather than sending null. Every other tool test either sends content
        or injects the history through a mocked _route_request, so without this
        one nothing POSTs the primary shape through the endpoint with the flag on.

        The call omits `type` to make the rebind observable: ToolCallSpec defaults
        it to "function", so the provider sees that default only if the strictly
        validated request is what reaches the tools path. Asserting on names alone
        cannot tell a parsed call from the raw dict, because both dump the same.
        """
        shape = {
            # Overrides the class's haiku: the Prompt 8 driver floor answers an
            # explicit non-driver + tools with a 400, and this test is about the
            # rebind reaching the provider, so it names a driver. Still explicit
            # (real _route_request, no classifier), still one mock.
            "model": "claude-sonnet-4-6",
            "messages": [
                _USER_TURN,
                {"role": "assistant", "tool_calls": [{"id": "call_1", "function": _GOOD_FUNCTION}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
            ],
            "tools": _TOOLS_PARAM,
            "tool_choice": "auto",
        }
        resp, mock_llm = self._post(client, shape, flag=True)

        assert resp.status_code == 200
        kwargs = mock_llm.complete_with_tools.call_args.kwargs
        assert kwargs["tools"] == _TOOLS_PARAM
        assert kwargs["tool_choice"] == "auto"
        assistant = kwargs["messages"][1]
        assert "content" not in assistant
        assert assistant["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
        ]
        assert kwargs["messages"][2]["tool_call_id"] == "call_1"


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

    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    def test_three_int_format(self) -> None:
        assert _parse_classifier("4 8 2") == (4, 8, "codegen")
        assert _parse_classifier("1 0 1") == (1, 0, "general")
        assert _parse_classifier("5 10 6") == (5, 10, "writing")

    def test_two_int_back_compat(self) -> None:
        assert _parse_classifier("4 8") == (4, 8, "general")
        assert _parse_classifier("1 0") == (1, 0, "general")
        assert _parse_classifier("5 10") == (5, 10, "general")

    def test_single_int_back_compat(self) -> None:
        assert _parse_classifier("4") == (4, 0, "general")
        assert _parse_classifier("1") == (1, 0, "general")

    def test_garbage_returns_none(self) -> None:
        # None signals parse failure so _classify_request can route it to the
        # cross-lineage fallback instead of silently collapsing to (3, 0).
        assert _parse_classifier("moderate") is None
        assert _parse_classifier("") is None
        assert _parse_classifier("no clue") is None
        assert _parse_classifier("4/8") is None

    def test_bad_category_never_fails_the_parse(self) -> None:
        # The first two fields carry routing weight; the third is telemetry.
        # A junk or out-of-range category must collapse to "general", not
        # return None -- None would re-bill the classifier via retry/fallback.
        assert _parse_classifier("4 8 zebra") == (4, 8, "general")
        assert _parse_classifier("4 8 99") == (4, 8, "general")
        assert _parse_classifier("4 8 0") == (4, 8, "general")
        assert _parse_classifier("4 8 -3") == (4, 8, "general")

    def test_clamps_both_axes(self) -> None:
        assert _parse_classifier("9 99") == (5, 10, "general")
        assert _parse_classifier("0 -5") == (1, 0, "general")
        assert _parse_classifier("7 -1") == (5, 0, "general")

    def test_strips_markdown_fences(self) -> None:
        assert _parse_classifier("```\n4 8 3\n```") == (4, 8, "debugging")

    def test_every_category_code_maps(self) -> None:
        expected = ["general", "codegen", "debugging", "planning", "ops", "writing"]
        got = [_parse_classifier(f"3 0 {code}")[2] for code in range(1, 7)]
        assert got == expected


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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
        # Chair (claude-sonnet-5) returns empty -> the proxy substitutes the
        # first-survivor text into the client response, but the transcript must record
        # synthesis_text='' (chair.text), status chair_empty -- NOT the survivor text.
        mock_compose.return_value = ["claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"]

        def make_client(model_id, *_a, **_k):
            m = MagicMock()
            m.complete.return_value = _mock_completion(
                "" if model_id == "claude-sonnet-5" else "member text"
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 4, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
        assert "claude-sonnet-5" in data["model"]
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
        # Always-present telemetry key (search_intent precedent): a two-int
        # classifier reply still yields the "general" default here.
        assert json.loads(rows[0]["context_json"])["task_category"] == "general"

    @patch("pdp_router._proxy._autopanel_enabled", return_value=False)
    @patch(
        "pdp_router._proxy._classify_request",
        return_value=(0.55, 3, 0, "debugging"),
    )
    @patch("pdp_router._proxy.get_client")
    def test_cascade_row_carries_task_category(
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
                "messages": [{"role": "user", "content": "why does this crash"}],
            },
        )

        assert resp.status_code == 200
        rows = _read_inbox_rows(inbox_dir)
        assert json.loads(rows[0]["context_json"])["task_category"] == "debugging"

    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
        assert resp.json()["model"] == "claude-sonnet-5"

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
        assert resp.json()["model"] == "claude-sonnet-5"

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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
        assert data["model"] == "claude-sonnet-5"
        mock_llm.complete.assert_called_once()
        assert mock_llm.complete.call_args.kwargs.get("enable_web_search") is True

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._autopanel_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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
        assert resp.json()["model"] == "claude-sonnet-5"

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
    @patch("pdp_router._proxy._classify_request", return_value=(0.95, 1, 3, "general"))
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
        assert data["model"] == "claude-sonnet-5"

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._streaming_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
        assert mock_get_client.call_args.args[0] == "claude-sonnet-5"
        assert captured.get("enable_web_search") is True
        assert "claude-sonnet-5" in resp.text

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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
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
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 9, "general"))
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


# -- Prompt 8: tool-driver floor, pin, and refusals --


@contextlib.contextmanager
def _credential_overrides(**fields):
    """Temporarily override credential fields on the live frozen config.

    The inbox_dir fixture's object.__setattr__ idiom, generalized: _config is a
    frozen dataclass built during lifespan, so per-test credential states go
    through the freeze bypass with a guaranteed restore.
    """
    assert _proxy._config is not None
    originals = {name: getattr(_proxy._config, name) for name in fields}
    for name, value in fields.items():
        object.__setattr__(_proxy._config, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            object.__setattr__(_proxy._config, name, value)


class TestToolDriverModelUnit:
    """_tool_driver_model: sha256 pin over the FULL driver tuple + usability walk.

    Expected drivers are hardcoded from the formula (utf-8 encode, hexdigest,
    int base 16, mod len(_TOOL_DRIVERS)) computed at authoring time, so these
    pin the exact arithmetic. The tuple is 6 long, so: "open the pod bay
    doors" -> 2, "what is the capital of france" -> 1, "run the test
    suite" -> 0, "list the files in the repo" -> 5, "" -> 1. Growing the tuple
    remaps every pin -- that is stateless and harmless, but it does mean these
    constants have to be recomputed whenever a driver is added or removed.
    """

    @staticmethod
    def _cfg(anthropic: str = "", openrouter: str = "") -> ProxyConfig:
        return ProxyConfig(
            anthropic_api_key=anthropic,
            gemini_api_key="",
            openrouter_api_key=openrouter,
        )

    def test_pin_lands_on_the_hashed_index_when_every_driver_is_usable(self) -> None:
        cfg = self._cfg(anthropic="sk-a", openrouter="or-b")
        assert _proxy._tool_driver_model("open the pod bay doors", cfg) == "claude-sonnet-4-6"
        assert _proxy._tool_driver_model("run the test suite", cfg) == "claude-sonnet-5"

    def test_walk_skips_credential_less_drivers(self) -> None:
        """Pin index 2 with no Anthropic key: past both Anthropic arms to gpt-5.5."""
        cfg = self._cfg(anthropic="", openrouter="or-b")
        assert _proxy._tool_driver_model("open the pod bay doors", cfg) == "openai/gpt-5.5"

    def test_walk_wraps_past_the_end_of_the_tuple(self) -> None:
        """Pin index 5 (qwen) with no OpenRouter key wraps to index 0 (sonnet 5)."""
        cfg = self._cfg(anthropic="sk-a", openrouter="")
        assert _proxy._tool_driver_model("list the files in the repo", cfg) == "claude-sonnet-5"

    def test_empty_pin_key_still_returns_a_driver(self) -> None:
        """The empty string is the documented no-user-message key (spec:
        system-only requests must route, not 500)."""
        cfg = self._cfg(anthropic="sk-a")
        assert _proxy._tool_driver_model("", cfg) == "claude-opus-5"

    def test_returns_none_when_no_driver_is_usable(self) -> None:
        assert _proxy._tool_driver_model("open the pod bay doors", self._cfg()) is None

    def test_retired_driver_is_skipped_by_the_walk(self) -> None:
        """The registry kill switch (available=False) retires an arm from the
        tool floor even while its credential is still set -- the walk predicate
        composes BOTH notions, not credentials alone. ModelCapability is a
        frozen dataclass, so the flip goes through the object.__setattr__
        bypass with a guaranteed restore."""
        from pdp_router._router import DEFAULT_REGISTRY

        cap = DEFAULT_REGISTRY.get("claude-sonnet-4-6")
        assert cap is not None and cap.available is True
        object.__setattr__(cap, "available", False)
        try:
            cfg = self._cfg(anthropic="sk-a")
            # Pin index 2 ("open the pod bay doors") starts on the retired
            # sonnet 4.6; the walk must move on despite the live key.
            assert _proxy._tool_driver_model("open the pod bay doors", cfg) == "claude-opus-4-8"
        finally:
            object.__setattr__(cap, "available", True)

    def test_first_user_text_takes_the_first_user_turn_stripped(self) -> None:
        """The pin key convention: FIRST user turn (stable per conversation),
        not the latest (the search gate's convention), stripped."""
        messages = [
            ChatMessage(role="user", content="  what is the capital of france  "),
            ChatMessage(role="assistant", content="Paris."),
            ChatMessage(role="user", content="hello"),
        ]
        assert _proxy._first_user_text(messages) == "what is the capital of france"
        assert _proxy._first_user_text([ChatMessage(role="assistant", content="x")]) == ""
        assert _proxy._first_user_text([]) == ""


class TestToolDriverFloorAndPin:
    """Prompt 8 endpoint policy: floor, pin, override, and the two refusals.

    _route_request is patched with the 8-tuple to keep picks deterministic; the
    floor runs after it in the handler, so the dispatched model is read off
    get_client. The client fixture holds Anthropic+Gemini keys only, so the
    usable drivers are {sonnet, opus} unless a test grants the OpenRouter key.
    """

    _mock = staticmethod(TestToolPassthroughNonStream._mock)

    @staticmethod
    def _route(
        model: str = "claude-haiku-4-5-20251001",
        non_system: list | None = None,
        score: int = 4,
        panel_score: int = 0,
    ) -> tuple:
        return (
            model,
            0.5,
            score,
            panel_score,
            "general",
            False,
            "",
            non_system
            if non_system is not None
            else [ChatMessage(role="user", content="what is the capital of france")],
            _proxy._RouteProvenance(mode="cascade", explored=False),
        )

    def _post_auto(self, client, *, route: tuple, mock_llm: MagicMock | None = None):
        """POST a canonical tools request on /openai/v1 with the flag on."""
        mock_llm = mock_llm or self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=route),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_get_client,
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "what is the capital of france"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        return resp, mock_llm, mock_get_client

    # -- the floor --

    def test_non_driver_pick_is_floored_to_the_pinned_driver(self, client) -> None:
        """Haiku pick + tools: the dispatched model is the pin for the first
        user turn ("what is the capital of france" -> opus), not the pick."""
        resp, mock_llm, mock_gc = self._post_auto(client, route=self._route())
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-opus-5"
        mock_llm.complete_with_tools.assert_called_once()

    def test_driver_pick_is_kept(self, client) -> None:
        """Sonnet pick stays sonnet even though the pin for this turn is opus:
        the walk only runs for picks outside the driver set."""
        resp, _, mock_gc = self._post_auto(client, route=self._route(model="claude-sonnet-4-6"))
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-sonnet-4-6"

    def test_same_first_user_message_pins_the_same_driver(self, client) -> None:
        picks = []
        for _ in range(2):
            _, _, mock_gc = self._post_auto(client, route=self._route())
            picks.append(mock_gc.call_args[0][0])
        assert picks == ["claude-opus-5", "claude-opus-5"]

    def test_different_first_user_messages_can_pin_different_drivers(self, client) -> None:
        """Deterministic spread, no random nonces: "hello" pins index 0
        (sonnet) and "what is the capital of france" pins index 1 (opus)."""
        _, _, gc_a = self._post_auto(
            client,
            route=self._route(non_system=[ChatMessage(role="user", content="hello")]),
        )
        _, _, gc_b = self._post_auto(client, route=self._route())
        assert gc_a.call_args[0][0] == "claude-sonnet-5"
        assert gc_b.call_args[0][0] == "claude-opus-5"

    def test_pin_keys_on_the_first_user_turn_not_the_latest(self, client) -> None:
        """A later user turn that would pin differently ("hello" -> sonnet)
        must not move the driver; only first-turn rewrites re-pin (spec's
        accepted history-compaction edge)."""
        non_system = [
            ChatMessage(role="user", content="what is the capital of france"),
            ChatMessage(role="assistant", content="Paris."),
            ChatMessage(role="user", content="hello"),
        ]
        _, _, mock_gc = self._post_auto(client, route=self._route(non_system=non_system))
        assert mock_gc.call_args[0][0] == "claude-opus-5"

    def test_system_only_request_routes_on_the_empty_pin_key(self, client) -> None:
        """No user message anywhere: pin key "" (-> opus), 200, not a 500."""
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=self._route(non_system=[])),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_gc,
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "system", "content": "You are a tool runner."}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-opus-5"

    def test_no_usable_driver_refuses_rather_than_serving_the_pick(self, client, caplog) -> None:
        """Every key empty: the floor refuses with a router-level 503 before any
        dispatch, rather than leaving a non-driver pick that a claude-* client
        would serve (invariant 5). The degraded state is logged loudly and no
        client is built."""
        with (
            caplog.at_level(logging.WARNING, logger="pdp_router._proxy"),
            _credential_overrides(anthropic_api_key="", gemini_api_key="", openrouter_api_key=""),
        ):
            resp, _, mock_gc = self._post_auto(client, route=self._route())
        assert resp.status_code == 503
        assert resp.json()["error"]["message"] == "No tool-capable model is currently available"
        mock_gc.assert_not_called()
        assert "no tool driver is usable" in caplog.text

    # -- PROXY_TOOL_MODEL --

    def test_env_override_beats_the_pin_and_a_driver_pick(self, client, monkeypatch) -> None:
        monkeypatch.setenv("PROXY_TOOL_MODEL", "claude-opus-4-8")
        resp, _, mock_gc = self._post_auto(
            client,
            route=self._route(
                model="claude-sonnet-4-6",
                non_system=[ChatMessage(role="user", content="hello")],
            ),
        )
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-opus-4-8"

    @pytest.mark.parametrize(
        "override",
        [
            pytest.param("gemini-2.5-pro", id="non_driver"),
            pytest.param("openai/gpt-5.5", id="credential_less_driver"),
        ],
    )
    def test_unusable_env_override_degrades_to_the_pin_walk(
        self, override, client, monkeypatch, caplog
    ) -> None:
        """A bad operator value must not fail requests: warn and pin-walk."""
        monkeypatch.setenv("PROXY_TOOL_MODEL", override)
        with caplog.at_level(logging.WARNING, logger="pdp_router._proxy"):
            resp, _, mock_gc = self._post_auto(client, route=self._route())
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-opus-5"
        assert any("PROXY_TOOL_MODEL" in r.message for r in caplog.records)

    # -- the explicit-model contract --

    def test_explicit_driver_model_is_honored_and_beats_the_env_override(
        self, client, monkeypatch
    ) -> None:
        """The caller's named driver survives both the floor and the operator
        override; the real _route_request runs (explicit path, classifier-free)."""
        monkeypatch.setenv("PROXY_TOOL_MODEL", "claude-opus-4-8")
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_gc,
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "run it"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-sonnet-4-6"

    def test_explicit_non_driver_model_with_tools_is_a_400(self, client, inbox_dir) -> None:
        """Registry-valid but outside the driver set: refuse with the client
        wording, in the OpenAI envelope, before any client build or routing row."""
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_gc,
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "messages": [{"role": "user", "content": "run it"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 400
        body = resp.json()
        assert "detail" not in body
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"] == (
            "claude-haiku-4-5-20251001 does not support tool calls on this proxy"
        )
        assert resp.headers["X-PDP-Prediction-Id"]
        mock_gc.assert_not_called()
        assert list(inbox_dir.glob("proxy-*.jsonl")) == []

    def test_explicit_non_driver_400_holds_for_stream_requests(self, client) -> None:
        """stream:true changes nothing: the refusal is a JSON 400, not SSE --
        it fires before any stream dispatch."""
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "stream": True,
                    "messages": [{"role": "user", "content": "run it"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["message"] == (
            "claude-haiku-4-5-20251001 does not support tool calls on this proxy"
        )

    def test_explicit_driver_is_honored_on_membership_not_credentials(self, client) -> None:
        """Honored-if-in-the-driver-set is membership, not usability: an
        explicit OpenRouter driver with no OpenRouter key still dispatches
        verbatim, so the provider-side failure stays the caller's faithful
        error surface instead of a proxy 400."""
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_gc,
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "openai/gpt-5.5",
                    "messages": [{"role": "user", "content": "run it"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "openai/gpt-5.5"

    # -- tool history without tools --

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param(
                [
                    {"role": "user", "content": "run it"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "done"},
                ],
                id="tool_result_turn",
            ),
            pytest.param(
                [
                    {"role": "user", "content": "run it"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{}"},
                            }
                        ],
                    },
                ],
                id="assistant_tool_calls_turn",
            ),
        ],
    )
    def test_tool_history_without_tools_is_a_422_before_routing(
        self, history, client, inbox_dir
    ) -> None:
        """Anthropic rejects tool blocks without a tools param and the legacy
        path may never see tool-shaped messages, so fail loud -- and ahead of
        _route_request, so the refusal costs no classifier call."""
        mock_route = MagicMock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", mock_route),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={"model": "pdp-auto", "messages": history},
            )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["message"] == "tool history requires tools"
        assert body["error"]["type"] == "invalid_request_error"
        assert resp.headers["X-PDP-Prediction-Id"]
        assert mock_route.call_count == 0
        assert list(inbox_dir.glob("proxy-*.jsonl")) == []

    def test_empty_tools_array_with_tool_history_is_still_a_422(self, client) -> None:
        """tools: [] counts as absent: Anthropic rejects tool blocks without a
        usable tools param, so an explicit empty array refuses the same way."""
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "tools": [],
                    "messages": [
                        {"role": "user", "content": "run it"},
                        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
                    ],
                },
            )
        assert resp.status_code == 422
        assert resp.json()["error"]["message"] == "tool history requires tools"

    def test_tool_history_422_holds_for_stream_requests(self, client) -> None:
        """stream:true changes nothing: the refusal is a JSON 422, not SSE."""
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "run it"},
                        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
                    ],
                },
            )
        assert resp.status_code == 422
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"]["message"] == "tool history requires tools"

    # -- structure pins: panel skip + web-search suppression (Prompt 8 ruling 1:
    # the tool branch returns before the panel guard and the web-search binding,
    # so these pin that position instead of asserting dead guard terms) --

    def test_panel_worthy_tools_request_takes_the_single_model_tool_path(self, client) -> None:
        """panel_score over the threshold + autopanel on: tools still pre-empt
        the panel. Pins the tool branch's position above the panel guard."""
        panel_spy = MagicMock()
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._autopanel_enabled", return_value=True),
            patch("pdp_router._proxy._execute_panel_with_synth", panel_spy),
            patch("pdp_router._proxy._route_request", return_value=self._route(panel_score=9)),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "what is the capital of france"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 200
        assert "pdp-panel" not in resp.json()["model"]
        panel_spy.assert_not_called()
        mock_llm.complete_with_tools.assert_called_once()

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_tools_suppress_web_search_even_on_a_search_intent_turn(
        self, mock_get_client, _mock_cascade, _mock_classify, _mock_ws, client
    ) -> None:
        """Real _route_request with a haiku pick: the tool dispatch carries no
        web-search kwarg at all (attaching it would replace the caller's own
        tools). The dispatched model is the tool-driver floor's pick, and the
        search floor is suppressed for tools (see the pin-preservation test)."""
        mock_llm = self._mock()
        mock_get_client.return_value = mock_llm
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [
                        {"role": "user", "content": "search the web for the latest AI news"}
                    ],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 200
        mock_llm.complete.assert_not_called()
        mock_llm.complete_multi.assert_not_called()
        mock_llm.complete_with_tools.assert_called_once()
        assert "enable_web_search" not in mock_llm.complete_with_tools.call_args.kwargs

    @patch("pdp_router._proxy._web_search_enabled", return_value=True)
    @patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general"))
    @patch(
        "pdp_router._proxy.confidence_cascade", return_value=("claude-haiku-4-5-20251001", False)
    )
    @patch("pdp_router._proxy.get_client")
    def test_search_intent_does_not_override_the_tool_pin(
        self, mock_get_client, _mock_cascade, _mock_classify, _mock_ws, client
    ) -> None:
        """The search floor is suppressed for a tools request, so the driver pin
        wins. "look that up online now" has search intent AND pins to opus; the
        old behavior flipped the haiku pick to the searcher Sonnet, overriding
        the pin. Asserting opus proves the pin is honored, not the search floor."""
        mock_llm = self._mock()
        mock_get_client.return_value = mock_llm
        with patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": "look that up online now"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 200
        assert mock_get_client.call_args[0][0] == "claude-opus-5"
        mock_llm.complete_with_tools.assert_called_once()

    # -- effort + rows --

    def test_effort_recomputes_for_the_floored_driver(self, client) -> None:
        """The dial ran against the haiku pick (None); the floored opus supports
        a level, so score 4 must arrive as "high", not None."""
        mock_llm = self._mock()
        with patch("pdp_router._proxy._effort_routing_enabled", return_value=True):
            resp, _, _ = self._post_auto(client, route=self._route(score=4), mock_llm=mock_llm)
        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_args.kwargs["effort"] == "high"

    def test_effort_recomputes_when_the_override_changes_the_model(
        self, client, monkeypatch
    ) -> None:
        """The other half of the recompute contract: PROXY_TOOL_MODEL replacing
        a no-dial pick must also re-run the dial, not just the pin walk."""
        monkeypatch.setenv("PROXY_TOOL_MODEL", "claude-opus-4-8")
        mock_llm = self._mock()
        with patch("pdp_router._proxy._effort_routing_enabled", return_value=True):
            resp, _, mock_gc = self._post_auto(
                client, route=self._route(score=4), mock_llm=mock_llm
            )
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-opus-4-8"
        assert mock_llm.complete_with_tools.call_args.kwargs["effort"] == "high"

    def test_floored_request_carries_no_effort_when_the_dial_is_off(self, client) -> None:
        """The recompute stays behind the effort flag: floor fires, dial off ->
        the driver receives effort=None, never an uninvited level."""
        mock_llm = self._mock()
        with patch("pdp_router._proxy._effort_routing_enabled", return_value=False):
            resp, _, mock_gc = self._post_auto(
                client, route=self._route(score=4), mock_llm=mock_llm
            )
        assert resp.status_code == 200
        assert mock_gc.call_args[0][0] == "claude-opus-5"
        assert mock_llm.complete_with_tools.call_args.kwargs["effort"] is None

    def test_floored_row_records_the_driver_without_touching_routing_mode(
        self, client, inbox_dir
    ) -> None:
        """model_selected holds what ran; routing_mode stays the policy
        ("cascade") -- the floor is a constraint, not a policy (spec.md)."""
        resp, _, _ = self._post_auto(client, route=self._route())
        assert resp.status_code == 200
        rows = _read_inbox_rows(inbox_dir)
        assert len(rows) == 1
        assert rows[0]["model_selected"] == "claude-opus-5"
        assert rows[0]["routing_mode"] == "cascade"


class TestStickyDriver:
    """Fully sticky tool conversations behind pipeline.proxy_sticky_driver_enabled.

    Precedence under the flag: PROXY_TOOL_MODEL override > conversation
    incumbent > cascade-pick-if-driver > pin walk. Flag off must be
    byte-identical to the floor-and-pin behavior above.
    """

    _mock = staticmethod(TestToolPassthroughNonStream._mock)
    _route = staticmethod(TestToolDriverFloorAndPin._route)

    _CONV_TEXT = "what is the capital of france"

    def _post_auto(self, client, *, route: tuple, sticky: bool = True, env: dict | None = None):
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._sticky_driver_enabled", return_value=sticky),
            patch("pdp_router._proxy._route_request", return_value=route),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_get_client,
            patch.dict(os.environ, env or {}),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": self._CONV_TEXT}],
                    "tools": _TOOLS_PARAM,
                },
            )
        return resp, mock_get_client

    def test_incumbent_beats_a_usable_driver_pick(self, client) -> None:
        """Turn 1 floors Haiku to the pin (opus-5); turn 2's cascade pick is
        itself a usable driver (sonnet-4-6) but the incumbent must serve."""
        _, gc1 = self._post_auto(client, route=self._route())
        assert gc1.call_args[0][0] == "claude-opus-5"
        _, gc2 = self._post_auto(client, route=self._route(model="claude-sonnet-4-6"))
        assert gc2.call_args[0][0] == "claude-opus-5"

    def test_flag_off_keeps_per_turn_routing(self, client) -> None:
        _, gc1 = self._post_auto(client, route=self._route(), sticky=False)
        assert gc1.call_args[0][0] == "claude-opus-5"
        _, gc2 = self._post_auto(
            client, route=self._route(model="claude-sonnet-4-6"), sticky=False
        )
        assert gc2.call_args[0][0] == "claude-sonnet-4-6"

    def test_unusable_incumbent_falls_through_to_the_pin_walk(self, client) -> None:
        """An incumbent whose credentials are gone (gpt-5.5, no OpenRouter key
        in the fixture) is walked past, and the served driver replaces it."""
        digest = _proxy._tool_pin_digest(self._CONV_TEXT)
        assert _proxy._conversation_cache is not None
        _proxy._conversation_cache.get(digest).driver = "openai/gpt-5.5"
        _, gc = self._post_auto(client, route=self._route())
        assert gc.call_args[0][0] == "claude-opus-5"
        assert _proxy._conversation_cache.get(digest).driver == "claude-opus-5"

    def test_override_beats_the_incumbent(self, client) -> None:
        _, gc1 = self._post_auto(client, route=self._route())
        assert gc1.call_args[0][0] == "claude-opus-5"
        _, gc2 = self._post_auto(
            client,
            route=self._route(),
            env={"PROXY_TOOL_MODEL": "claude-sonnet-4-6"},
        )
        assert gc2.call_args[0][0] == "claude-sonnet-4-6"

    def test_explicit_model_never_records_an_incumbent(self, client) -> None:
        mock_llm = self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._sticky_driver_enabled", return_value=True),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "claude-sonnet-5",
                    "messages": [{"role": "user", "content": self._CONV_TEXT}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 200
        digest = _proxy._tool_pin_digest(self._CONV_TEXT)
        assert _proxy._conversation_cache is not None
        # Spend accounting may touch the entry; the sticky invariant is that
        # no INCUMBENT is recorded for a caller-pinned model.
        state = _proxy._conversation_cache.peek(digest)
        assert state is None or state.driver is None

    def test_sticky_row_carries_the_flag_and_conversation_key(self, client, inbox_dir) -> None:
        self._post_auto(client, route=self._route())
        resp, _ = self._post_auto(client, route=self._route(model="claude-sonnet-4-6"))
        assert resp.status_code == 200
        # Filter to the exposure rows: a deliberately-enabled feedback flag
        # would interleave its own rows and shift bare indices.
        rows = [r for r in _read_inbox_rows(inbox_dir) if r["routing_mode"] == "cascade"]
        first, second = json.loads(rows[0]["context_json"]), json.loads(rows[1]["context_json"])
        assert "tool_sticky" not in first
        assert second["tool_sticky"] is True
        digest = _proxy._tool_pin_digest(self._CONV_TEXT)
        assert first["conversation_key"] == digest[:8]
        assert second["conversation_key"] == digest[:8]


class TestAutoModelModes:
    """pdp-auto-cost / pdp-auto-max aliases shift the score->confidence map."""

    def test_parse_auto_model(self) -> None:
        from pdp_router._proxy import _parse_auto_model

        assert _parse_auto_model("pdp-auto") == (True, "balanced")
        assert _parse_auto_model("") == (True, "balanced")
        assert _parse_auto_model("pdp-auto-cost") == (True, "cost")
        assert _parse_auto_model("pdp-auto-max") == (True, "max")
        assert _parse_auto_model("pdp-auto-turbo") == (False, "")
        assert _parse_auto_model("claude-sonnet-5") == (False, "")

    def test_models_endpoints_list_the_aliases(self, client) -> None:
        for path in ("/v1/models", "/openai/v1/models"):
            ids = {m["id"] for m in client.get(path).json()["data"]}
            assert {"pdp-auto", "pdp-auto-cost", "pdp-auto-max"} <= ids

    def _routed_confidence(self, client, model: str, classify: tuple) -> tuple:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ok")
        with (
            patch("pdp_router._proxy._autopanel_enabled", return_value=False),
            patch("pdp_router._proxy._classify_request", return_value=classify),
            patch(
                "pdp_router._proxy._cascade_with_provenance",
                return_value=(
                    "claude-haiku-4-5-20251001",
                    _proxy._RouteProvenance(mode="cascade", explored=False),
                ),
            ) as mock_cascade,
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
            )
        return resp, mock_cascade.call_args.kwargs["confidence"]

    def test_cost_mode_shifts_confidence_up_a_tier(self, client) -> None:
        # Score 2 reads 0.75 balanced; cost mode feeds the cascade 0.95.
        resp, conf = self._routed_confidence(
            client, "pdp-auto-cost", (0.75, 2, 0, "general")
        )
        assert resp.status_code == 200
        assert conf == 0.95

    def test_max_mode_shifts_confidence_down_a_tier(self, client) -> None:
        resp, conf = self._routed_confidence(client, "pdp-auto-max", (0.95, 1, 0, "general"))
        assert resp.status_code == 200
        assert conf == 0.75

    def test_balanced_is_untouched(self, client) -> None:
        resp, conf = self._routed_confidence(client, "pdp-auto", (0.75, 2, 0, "general"))
        assert resp.status_code == 200
        assert conf == 0.75

    def test_mode_recorded_on_the_row_only_for_aliases(self, client, inbox_dir) -> None:
        self._routed_confidence(client, "pdp-auto-cost", (0.75, 2, 0, "general"))
        self._routed_confidence(client, "pdp-auto", (0.75, 2, 0, "general"))
        rows = _read_inbox_rows(inbox_dir)
        contexts = [json.loads(r["context_json"]) for r in rows]
        assert contexts[0]["mode"] == "cost"
        assert "mode" not in contexts[1]

    def test_unknown_auto_alias_is_a_400(self, client) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pdp-auto-turbo",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 400


class TestClassifyImplicitFeedback:
    """Deterministic next-turn grading: markers, exact retry, moved_on default."""

    def test_correction_markers(self) -> None:
        from pdp_router._proxy import _classify_implicit_feedback

        for text in (
            "no, that breaks the tests",
            "That's wrong, look again",
            "that's not what I asked for",
            "undo that change",
            "revert it",
            "use uv instead",
            "try again with the real file",
            "still broken after your fix",
            "don't touch the config",
        ):
            assert _classify_implicit_feedback(text, None) == "correction", text

    def test_moved_on_default(self) -> None:
        from pdp_router._proxy import _classify_implicit_feedback

        for text in (
            "now add the tests",
            "great, next let us deploy it",
            "what does the second function do?",
            "nothing else, thanks",
        ):
            assert _classify_implicit_feedback(text, None) == "moved_on", text

    def test_exact_resend_is_a_retry(self) -> None:
        import hashlib as _hashlib

        from pdp_router._proxy import _classify_implicit_feedback

        digest = _hashlib.sha256(b"list the files").hexdigest()
        assert _classify_implicit_feedback("list the files", digest) == "retry"
        assert _classify_implicit_feedback("list the dirs", digest) == "moved_on"


class TestImplicitFeedbackRows:
    """One feedback row per routed multi-turn exchange, targeting the prior turn."""

    _T1 = "compare these two approaches"

    def _post(self, client, messages: list, *, flag: bool = True, model: str = "pdp-auto"):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ok")
        mock_llm.complete_multi.return_value = _mock_completion("ok")
        with (
            patch("pdp_router._proxy._implicit_feedback_enabled", return_value=flag),
            patch("pdp_router._proxy._autopanel_enabled", return_value=False),
            patch(
                "pdp_router._proxy._classify_request",
                return_value=(0.55, 3, 0, "general"),
            ),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            return client.post(
                "/v1/chat/completions", json={"model": model, "messages": messages}
            )

    def _two_turns(self, client, second_text: str, **kwargs):
        first = self._post(client, [{"role": "user", "content": self._T1}], **kwargs)
        second = self._post(
            client,
            [
                {"role": "user", "content": self._T1},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": second_text},
            ],
            **kwargs,
        )
        return first, second

    def test_second_turn_emits_a_feedback_row_targeting_the_first(
        self, client, inbox_dir
    ) -> None:
        first, second = self._two_turns(client, "now write the tests")
        assert second.status_code == 200
        rows = _read_inbox_rows(inbox_dir)
        fb = [r for r in rows if r["routing_mode"] == "implicit_feedback"]
        assert len(fb) == 1
        ctx = json.loads(fb[0]["context_json"])
        assert ctx["feedback_signal"] == "moved_on"
        assert ctx["target_chat_request_id"] == first.headers["X-PDP-Prediction-Id"]
        assert fb[0]["context_bucket"] == "chat:feedback"
        # Attributed to the model that served the graded turn.
        assert fb[0]["model_selected"] == rows[0]["model_selected"]

    def test_correction_signal_recorded(self, client, inbox_dir) -> None:
        self._two_turns(client, "that's wrong, the cascade never runs there")
        fb = [
            r
            for r in _read_inbox_rows(inbox_dir)
            if r["routing_mode"] == "implicit_feedback"
        ]
        assert json.loads(fb[0]["context_json"])["feedback_signal"] == "correction"

    def test_first_turn_emits_nothing(self, client, inbox_dir) -> None:
        self._post(client, [{"role": "user", "content": self._T1}])
        rows = _read_inbox_rows(inbox_dir)
        assert [r for r in rows if r["routing_mode"] == "implicit_feedback"] == []

    def test_flag_off_emits_nothing(self, client, inbox_dir) -> None:
        self._two_turns(client, "that's wrong", flag=False)
        rows = _read_inbox_rows(inbox_dir)
        assert [r for r in rows if r["routing_mode"] == "implicit_feedback"] == []

    def test_explicit_model_emits_nothing(self, client, inbox_dir) -> None:
        self._two_turns(client, "that's wrong", model="claude-sonnet-5")
        rows = _read_inbox_rows(inbox_dir) if list(inbox_dir.glob("proxy-*.jsonl")) else []
        assert [r for r in rows if r["routing_mode"] == "implicit_feedback"] == []


class TestPromptCachingThreading:
    """The prompt-caching flag rides to the client as a per-call kwarg."""

    def _post_tools(self, client, *, caching: bool):
        mock_llm = TestToolPassthroughNonStream._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._prompt_caching_enabled", return_value=caching),
            patch(
                "pdp_router._proxy._route_request",
                return_value=TestToolDriverFloorAndPin._route(model="claude-sonnet-4-6"),
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
        return resp, mock_llm

    def test_flag_off_threads_false(self, client) -> None:
        resp, mock_llm = self._post_tools(client, caching=False)
        assert resp.status_code == 200
        kwargs = mock_llm.complete_with_tools.call_args.kwargs
        assert kwargs["enable_prompt_caching"] is False

    def test_flag_on_threads_true(self, client) -> None:
        resp, mock_llm = self._post_tools(client, caching=True)
        assert resp.status_code == 200
        kwargs = mock_llm.complete_with_tools.call_args.kwargs
        assert kwargs["enable_prompt_caching"] is True


class TestSpendCap:
    """Per-conversation spend ceiling behind pipeline.proxy_spend_cap_enabled.

    Warn once at budget_warn_usd (footer note + log), refuse with an
    OpenAI-shaped 429 insufficient_quota past budget_max_usd -- before any
    client build or routing row. Flag off is a strict no-op.
    """

    _TEXT = "what is the capital of france"

    def _seed(self, spend: float):
        digest = _proxy._tool_pin_digest(self._TEXT)
        assert _proxy._conversation_cache is not None
        state = _proxy._conversation_cache.get(digest)
        state.spend_usd = spend
        return state

    def _post(self, client, *, cap: bool = True):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = _mock_completion("ok")
        with (
            patch("pdp_router._proxy._spend_cap_enabled", return_value=cap),
            patch("pdp_router._proxy._autopanel_enabled", return_value=False),
            patch(
                "pdp_router._proxy._classify_request",
                return_value=(0.55, 3, 0, "general"),
            ),
            patch("pdp_router._proxy.get_client", return_value=mock_llm) as mock_gc,
        ):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "pdp-auto",
                    "messages": [{"role": "user", "content": self._TEXT}],
                },
            )
        return resp, mock_gc

    def test_over_max_refuses_with_insufficient_quota(self, client, inbox_dir) -> None:
        self._seed(5.50)
        resp, mock_gc = self._post(client)
        assert resp.status_code == 429
        error = resp.json()["error"]
        assert error["type"] == "insufficient_quota"
        assert "$5.50" in error["message"]
        # Refused BEFORE any dispatch or exposure row.
        mock_gc.assert_not_called()
        assert list(inbox_dir.glob("proxy-*.jsonl")) == []

    def test_under_warn_serves_normally(self, client) -> None:
        self._seed(0.10)
        resp, _ = self._post(client)
        assert resp.status_code == 200

    def test_flag_off_never_refuses(self, client) -> None:
        self._seed(99.0)
        resp, _ = self._post(client, cap=False)
        assert resp.status_code == 200

    def test_warn_latch_fires_once(self, client) -> None:
        state = self._seed(1.50)
        assert state.budget_warned is False
        resp, _ = self._post(client)
        assert resp.status_code == 200
        assert state.budget_warned is True

    def test_fresh_conversation_is_never_capped(self, client) -> None:
        # No cache entry at all: peek() must not create one, and the request
        # must serve.
        resp, _ = self._post(client)
        assert resp.status_code == 200


# -- Prompt 9: tools-request routing-row observability --


def _row_context(row: dict) -> dict:
    return json.loads(row["context_json"])


class TestToolRowObservability:
    """Prompt 9: tools-request rows carry the observability fields.

    Everything rides inside context_json, never as a top-level key (the drain
    takes explicit kwargs, so top-level is closed), and every field is
    omit-when-absent. The non-stream row is written after completion so it can
    carry finish_reason/tool_call_count/tool_names; the stream row keeps its
    pre-stream timing and carries request-side fields only.
    """

    _route = staticmethod(TestToolDriverFloorAndPin._route)
    _mock = staticmethod(TestToolPassthroughNonStream._mock)

    def _post_tools(
        self,
        client,
        inbox_dir,
        *,
        route: tuple | None = None,
        mock_llm: MagicMock | None = None,
        body: dict | None = None,
    ):
        mock_llm = mock_llm or self._mock()
        body = body or {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "what is the capital of france"}],
            "tools": _TOOLS_PARAM,
        }
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True)
            )
            stack.enter_context(patch("pdp_router._proxy.get_client", return_value=mock_llm))
            if route is not None:
                stack.enter_context(patch("pdp_router._proxy._route_request", return_value=route))
            resp = client.post("/openai/v1/chat/completions", json=body)
        rows = _read_inbox_rows(inbox_dir) if list(inbox_dir.glob("proxy-*.jsonl")) else []
        return resp, rows, mock_llm

    # -- request-side fields --

    def test_tool_row_carries_the_request_side_fields(self, client, inbox_dir) -> None:
        """The drain-side joins in one place: what policy wanted vs what ran,
        the conversation pin, and the request shape."""
        resp, rows, _ = self._post_tools(client, inbox_dir, route=self._route())
        assert resp.status_code == 200
        assert len(rows) == 1
        assert rows[0]["model_selected"] == "claude-opus-5"
        context = _row_context(rows[0])
        assert context["tools_present"] is True
        assert context["tool_count"] == 1
        assert context["loop_depth"] == 0
        assert context["model_cascade_pick"] == "claude-haiku-4-5-20251001"
        # sha256("what is the capital of france").hexdigest()[:8]
        assert context["tool_pin_key"] == "d832be57"
        assert context["provider_path"] == "anthropic-translated"
        assert "tool_choice" not in context

    def test_tool_choice_is_recorded_in_str_form(self, client, inbox_dir) -> None:
        """spec: tool_choice rides as its str form, whatever shape the caller
        sent -- one column type for the drain."""
        body = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "what is the capital of france"}],
            "tools": _TOOLS_PARAM,
            "tool_choice": "auto",
        }
        _, rows, _ = self._post_tools(client, inbox_dir, route=self._route(), body=body)
        assert _row_context(rows[0])["tool_choice"] == "auto"

    def test_dict_tool_choice_is_still_a_string_in_the_row(self, client, inbox_dir) -> None:
        body = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "what is the capital of france"}],
            "tools": _TOOLS_PARAM,
            "tool_choice": {"type": "function", "function": {"name": "run"}},
        }
        _, rows, _ = self._post_tools(client, inbox_dir, route=self._route(), body=body)
        recorded = _row_context(rows[0])["tool_choice"]
        assert isinstance(recorded, str)
        assert "function" in recorded

    def test_loop_depth_counts_assistant_tool_call_turns(self, client, inbox_dir) -> None:
        """Two completed tool rounds in the incoming transcript -> 2. Explicit
        driver model so the real _route_request builds non_system from the
        posted history, classifier-free."""
        call = {"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
        call2 = {**call, "id": "call_2"}
        body = {
            "model": "claude-sonnet-4-6",
            "tools": _TOOLS_PARAM,
            "messages": [
                {"role": "user", "content": "run it"},
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                {"role": "assistant", "tool_calls": [call2]},
                {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
            ],
        }
        _, rows, _ = self._post_tools(client, inbox_dir, body=body)
        assert _row_context(rows[0])["loop_depth"] == 2

    def test_driver_pick_kept_still_records_pick_and_pin(self, client, inbox_dir) -> None:
        """model_cascade_pick equals model_selected when the floor kept the
        pick -- the drain should not have to infer "kept" from key absence."""
        _, rows, _ = self._post_tools(
            client, inbox_dir, route=self._route(model="claude-sonnet-4-6")
        )
        context = _row_context(rows[0])
        assert context["model_cascade_pick"] == "claude-sonnet-4-6"
        assert rows[0]["model_selected"] == "claude-sonnet-4-6"
        assert context["tool_pin_key"] == "d832be57"

    # -- post-completion fields (non-stream only) --

    def test_post_completion_fields_land_on_the_non_stream_row(self, client, inbox_dir) -> None:
        _, rows, _ = self._post_tools(client, inbox_dir, route=self._route())
        context = _row_context(rows[0])
        assert context["finish_reason"] == "tool_calls"
        assert context["tool_call_count"] == 1
        assert context["tool_names"] == ["run"]

    def test_tool_names_are_capped_at_eight(self, client, inbox_dir) -> None:
        calls = tuple(ToolCall(id=f"c{i}", name=f"t{i}", arguments="{}") for i in range(9))
        mock_llm = self._mock(_tool_completion(tool_calls=calls))
        _, rows, _ = self._post_tools(client, inbox_dir, route=self._route(), mock_llm=mock_llm)
        context = _row_context(rows[0])
        assert context["tool_call_count"] == 9
        assert context["tool_names"] == [f"t{i}" for i in range(8)]

    def test_stream_row_carries_request_side_fields_only(self, client, inbox_dir) -> None:
        """The stream row is written before the body runs, so the outcome
        fields cannot exist on it -- absent, not null."""
        mock_llm = self._mock()

        async def _stream(*_args: object, **_kwargs: object):
            yield "the answer"

        mock_llm.stream_with_tools = _stream
        body = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "what is the capital of france"}],
            "tools": _TOOLS_PARAM,
            "stream": True,
        }
        with patch("pdp_router._proxy._streaming_enabled", return_value=True):
            resp, rows, _ = self._post_tools(
                client, inbox_dir, route=self._route(), mock_llm=mock_llm, body=body
            )
        assert resp.status_code == 200
        assert len(rows) == 1
        context = _row_context(rows[0])
        assert context["tool_pin_key"] == "d832be57"
        assert context["model_cascade_pick"] == "claude-haiku-4-5-20251001"
        assert "finish_reason" not in context
        assert "tool_call_count" not in context
        assert "tool_names" not in context

    def test_failed_tool_completion_still_writes_one_row(self, client, inbox_dir) -> None:
        """A 402 is a real exposure the bandit must see; the failure row keeps
        the request-side fields and carries no outcome it never produced."""
        mock_llm = self._mock()
        mock_llm.complete_with_tools.side_effect = CreditExhaustionError("credit balance too low")
        resp, rows, _ = self._post_tools(client, inbox_dir, route=self._route(), mock_llm=mock_llm)
        assert resp.status_code == 402
        assert len(rows) == 1
        context = _row_context(rows[0])
        assert context["tools_present"] is True
        assert context["model_cascade_pick"] == "claude-haiku-4-5-20251001"
        assert "finish_reason" not in context
        assert "tool_names" not in context

    # -- omit-when-absent boundaries --

    def test_explicit_tool_row_omits_pick_and_pin(self, client, inbox_dir) -> None:
        """No cascade ran and no pin was consulted for a caller-pinned model;
        absent keys, not nulls (the cascade_explored precedent)."""
        body = {
            "model": "claude-sonnet-4-6",
            "tools": _TOOLS_PARAM,
            "messages": [{"role": "user", "content": "run it"}],
        }
        _, rows, _ = self._post_tools(client, inbox_dir, body=body)
        assert rows[0]["routing_mode"] == "explicit"
        context = _row_context(rows[0])
        assert context["tools_present"] is True
        assert "model_cascade_pick" not in context
        assert "tool_pin_key" not in context
        assert context["provider_path"] == "anthropic-translated"

    def test_openrouter_driver_records_openai_native_path(self, client, inbox_dir) -> None:
        body = {
            "model": "openai/gpt-5.5",
            "tools": _TOOLS_PARAM,
            "messages": [{"role": "user", "content": "run it"}],
        }
        _, rows, _ = self._post_tools(client, inbox_dir, body=body)
        assert _row_context(rows[0])["provider_path"] == "openai-native"

    def test_provider_path_omitted_for_a_non_driver_model(self) -> None:
        """provider_path names only the two driver families; a non-driver
        model_selected omits it rather than mislabeling. Pinned at the builder
        level: the floor now refuses before a non-driver is ever served, so
        this defensive branch has no endpoint path to reach."""
        req = _proxy.ToolChatCompletionRequest(
            model="pdp-auto",
            messages=[_proxy.ToolChatMessage(role="user", content="hi")],
            tools=_TOOLS_PARAM,
        )
        context = _proxy._tool_row_context(
            req,
            [ChatMessage(role="user", content="hi")],
            model_selected="gemini-2.5-pro",
            cascade_pick="gemini-2.5-pro",
        )
        assert "provider_path" not in context
        anthropic = _proxy._tool_row_context(
            req,
            [ChatMessage(role="user", content="hi")],
            model_selected="claude-sonnet-4-6",
            cascade_pick=None,
        )
        assert anthropic["provider_path"] == "anthropic-translated"

    def test_non_tool_rows_gain_no_new_keys(self, client, inbox_dir) -> None:
        """The key-set wall: a flag-on request WITHOUT tools takes the legacy
        path and its context keys are exactly the base telemetry set -- none
        of the tool-only keys (tools_present, tool_pin_key, ...) may appear."""
        body = {"model": "pdp-auto", "messages": [{"role": "user", "content": "hi"}]}
        resp, rows, _ = self._post_tools(
            client, inbox_dir, route=self._route(model="claude-sonnet-4-6"), body=body
        )
        assert resp.status_code == 200
        assert len(rows) == 1
        assert set(_row_context(rows[0])) == {
            "chat_request_id",
            "complexity",
            "panel_score",
            "search_intent",
            "task_category",
            "conversation_key",
        }


class TestToolNonStreamHardening:
    """Pre-flip mutation-gap closures for the non-stream tool leg: what the
    branch is responsible for handing the client and the exact response
    envelope, none of which had a discriminating wall."""

    _route = staticmethod(TestToolDriverFloorAndPin._route)
    _mock = staticmethod(TestToolPassthroughNonStream._mock)

    def _post(self, client, *, body: dict, route: tuple, mock_llm: MagicMock | None = None):
        mock_llm = mock_llm or self._mock()
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy._route_request", return_value=route),
            patch("pdp_router._proxy.get_client", return_value=mock_llm),
        ):
            resp = client.post("/openai/v1/chat/completions", json=body)
        return resp, mock_llm

    def test_system_prompt_reaches_the_client(self, client) -> None:
        """A non-empty system turn must be threaded to complete_with_tools;
        the empty-system fixtures elsewhere let a dropped system pass."""
        body = {
            "model": "pdp-auto",
            "messages": [
                {"role": "system", "content": "You are a careful tool runner."},
                {"role": "user", "content": "run it"},
            ],
            "tools": _TOOLS_PARAM,
        }
        base = self._route()
        route = (*base[:6], "You are a careful tool runner.", *base[7:])
        resp, mock_llm = self._post(client, body=body, route=route)
        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_args.kwargs["system"] == (
            "You are a careful tool runner."
        )

    def test_max_tokens_reaches_the_client(self, client) -> None:
        """The token budget is the client's to honor; a dropped max_tokens would
        silently cap output. The stream leg walls this; the non-stream did not."""
        body = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "run it"}],
            "tools": _TOOLS_PARAM,
            "max_tokens": 1234,
        }
        resp, mock_llm = self._post(client, body=body, route=self._route())
        assert resp.status_code == 200
        assert mock_llm.complete_with_tools.call_args.kwargs["max_tokens"] == 1234

    def test_response_envelope_reports_the_run_model_and_object(self, client) -> None:
        """The response must name what ran (the floored driver) and the OpenAI
        object type, so a client is not told a pdp-auto placeholder served it."""
        body = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "what is the capital of france"}],
            "tools": _TOOLS_PARAM,
        }
        resp, _ = self._post(client, body=body, route=self._route())
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "claude-opus-5"

    def test_translation_error_is_a_400_invalid_request(self, client) -> None:
        """A caller payload the provider cannot express is a 400
        invalid_request_error, not a 503 server fault -- the non-stream sibling
        of the stream leg's frame classification."""
        mock_llm = self._mock()
        mock_llm.complete_with_tools.side_effect = ToolTranslationError(
            "tool call c1 has unparseable arguments"
        )
        body = {
            "model": "pdp-auto",
            "messages": [{"role": "user", "content": "run it"}],
            "tools": _TOOLS_PARAM,
        }
        resp, _ = self._post(client, body=body, route=self._route(), mock_llm=mock_llm)
        assert resp.status_code == 400
        body_json = resp.json()
        assert body_json["error"]["type"] == "invalid_request_error"
        assert "unparseable arguments" in body_json["error"]["message"]

    def test_unknown_model_with_tools_wins_over_the_driver_400(self, client) -> None:
        """Ordering: an unknown model + tools hits the registry check inside
        _route_request first, so it is the 'Unknown model' 400, not the
        driver-set 'does not support tool calls' 400."""
        with (
            patch("pdp_router._proxy._tool_passthrough_enabled", return_value=True),
            patch("pdp_router._proxy.get_client") as mock_get_client,
        ):
            resp = client.post(
                "/openai/v1/chat/completions",
                json={
                    "model": "gpt-9-turbo",
                    "messages": [{"role": "user", "content": "run it"}],
                    "tools": _TOOLS_PARAM,
                },
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Unknown model: gpt-9-turbo"
        mock_get_client.assert_not_called()
