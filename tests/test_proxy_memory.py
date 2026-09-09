# Description: Tests for the proxy's memory surface: lifespan wiring, the /health memory
# Description: section, the /v1/memory routes and their fail-closed shapes.

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from pdp_router import _proxy
from pdp_router._clients import CompletionResult
from pdp_router._conversation import ConversationCache
from pdp_router._memory import MemoryModels, MemoryRuntime, MemoryStore
from pdp_router._proxy import app
from tests._memory_fakes import FakeEmbedder, FakeReranker


def _db_path() -> Path:
    return Path(os.environ["PROXY_MEMORY_DB_PATH"])


def _store() -> MemoryStore:
    return MemoryStore(_db_path())


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


@pytest.fixture()
def fakes() -> MemoryModels:
    return MemoryModels(embedder=FakeEmbedder(), reranker=FakeReranker())


def _factory_with(loader):
    def factory(config, **kwargs):
        return MemoryRuntime(config, loader=loader)

    return factory


@pytest.fixture()
def memory_on(monkeypatch, fakes):
    """Master flag on; the runtime the lifespan builds loads the fakes instead
    of the real library, so no test touches onnxruntime or a model dir."""
    monkeypatch.setattr(
        _proxy, "MemoryRuntime", _factory_with(lambda config, *, allow_download: fakes)
    )
    monkeypatch.setattr(_proxy, "_memory_enabled", lambda: True)


@pytest.fixture()
def memory_client(memory_on):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), TestClient(app) as c:
        _proxy._memory_runtime.wait_for_load(timeout=5)
        yield c


@pytest.fixture()
def plain_client():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), TestClient(app) as c:
        yield c


@pytest.fixture()
def no_models_client(monkeypatch):
    """Master flag on but the model load fails: the store works, retrieval and
    writes do not."""

    def loader(config, *, allow_download):
        raise RuntimeError("no model files")

    monkeypatch.setattr(_proxy, "MemoryRuntime", _factory_with(loader))
    monkeypatch.setattr(_proxy, "_memory_enabled", lambda: True)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), TestClient(app) as c:
        _proxy._memory_runtime.wait_for_load(timeout=5)
        yield c


_ROUTES = [
    ("POST", "/v1/memory", {"text": "x"}),
    ("GET", "/v1/memory?q=x", None),
    ("GET", "/v1/memory", None),
    ("GET", "/v1/memory/m_X", None),
    ("DELETE", "/v1/memory/m_X", None),
    ("POST", "/v1/memory/m_X/confirm", None),
]


class TestMasterFlagOff:
    def test_every_route_is_503_disabled_and_nothing_opens(self, plain_client) -> None:
        for method, path, body in _ROUTES:
            resp = plain_client.request(method, path, json=body)
            assert resp.status_code == 503, (method, path)
            err = resp.json()["error"]
            assert err["message"] == "memory is disabled"
            assert err["type"] == "server_error"
        assert _proxy._memory_runtime is None
        assert not _db_path().exists()

    def test_health_memory_section_reads_nothing(self, plain_client) -> None:
        assert plain_client.get("/health").json()["memory"] == {
            "enabled": False,
            "db_present": False,
            "models_loaded": False,
            "load_error": None,
        }
        assert not _db_path().exists()


class TestLifespan:
    def test_flag_on_opens_the_store_and_starts_one_load(self, monkeypatch, fakes) -> None:
        starts: list[int] = []

        class _Spy(MemoryRuntime):
            def start_model_load(self) -> None:
                starts.append(1)
                super().start_model_load()

        def factory(config, **kwargs):
            return _Spy(config, loader=lambda config, *, allow_download: fakes)

        monkeypatch.setattr(_proxy, "MemoryRuntime", factory)
        monkeypatch.setattr(_proxy, "_memory_enabled", lambda: True)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), TestClient(app) as c:
            assert starts == [1]
            assert _db_path().exists()  # the store is opened at startup
            _proxy._memory_runtime.wait_for_load(timeout=5)
            assert c.get("/health").json()["memory"] == {
                "enabled": True,
                "db_present": True,
                "models_loaded": True,
                "load_error": None,
            }

    def test_flag_off_builds_no_runtime(self, monkeypatch) -> None:
        built: list[int] = []

        def factory(config, **kwargs):
            built.append(1)
            return MemoryRuntime(config)

        monkeypatch.setattr(_proxy, "MemoryRuntime", factory)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), TestClient(app):
            assert built == []
            assert _proxy._memory_runtime is None

    def test_flag_flipped_on_after_start_builds_lazily(self, monkeypatch, fakes) -> None:
        """A hot flip needs no restart: the first flagged request builds the
        runtime and starts the load; the request itself is served."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}), TestClient(app) as c:
            assert _proxy._memory_runtime is None
            monkeypatch.setattr(
                _proxy, "MemoryRuntime", _factory_with(lambda config, *, allow_download: fakes)
            )
            monkeypatch.setattr(_proxy, "_memory_enabled", lambda: True)
            assert c.get("/v1/memory").status_code == 200
            assert _proxy._memory_runtime is not None
            _proxy._memory_runtime.wait_for_load(timeout=5)
            assert c.post("/v1/memory", json={"text": "hot flip"}).status_code == 200

    def test_load_error_surfaces_in_health(self, no_models_client) -> None:
        memory = no_models_client.get("/health").json()["memory"]
        assert memory["enabled"] is True
        assert memory["db_present"] is True
        assert memory["models_loaded"] is False
        assert memory["load_error"] == "RuntimeError: no model files"


class TestWrite:
    def test_post_stores_a_fact_with_defaults(self, memory_client) -> None:
        resp = memory_client.post("/v1/memory", json={"text": "  prefers dark mode  "})
        assert resp.status_code == 200
        item = resp.json()
        assert item["object"] == "memory.item"
        assert item["id"].startswith("m_")
        assert item["kind"] == "fact"
        assert item["text"] == "prefers dark mode"
        assert item["score"] == 0.6
        assert item["source"] == "explicit:api"
        assert item["surface"] == "api"
        assert item["observed_at"] == _today()
        assert item["tier"] == "working"
        assert item["uses"] == 0
        assert item["embedding_model"] == "fake/embed"
        assert _store().get_item(item["id"]).text == "prefers dark mode"
        [event] = _store().events("endpoint")
        assert event.details == {
            "route": "/v1/memory",
            "method": "POST",
            "status": 200,
            "item_id": item["id"],
        }

    def test_post_honors_kind_surface_and_observed_at(self, memory_client) -> None:
        resp = memory_client.post(
            "/v1/memory",
            json={
                "text": "asked about the grow calendar",
                "kind": "ask",
                "surface": "cli",
                "observed_at": "2026-09-01",
            },
        )
        assert resp.status_code == 200
        item = resp.json()
        assert item["kind"] == "ask"
        assert item["source"] == "explicit:cli"
        assert item["surface"] == "cli"
        assert item["observed_at"] == "2026-09-01"

    def test_post_unknown_kind_is_422_in_the_envelope(self, memory_client) -> None:
        resp = memory_client.post("/v1/memory", json={"text": "x", "kind": "opinion"})
        assert resp.status_code == 422
        err = resp.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "kind" in err["message"]

    def test_post_bad_values_are_422(self, memory_client) -> None:
        assert memory_client.post("/v1/memory", json={"text": "   "}).status_code == 422
        resp = memory_client.post("/v1/memory", json={"text": "x", "observed_at": "soon"})
        assert resp.status_code == 422
        assert "observed_at" in resp.json()["error"]["message"]

    def test_post_duplicate_is_409_unless_forced(self, memory_client) -> None:
        first = memory_client.post("/v1/memory", json={"text": "prefers dark mode"}).json()
        dup = memory_client.post("/v1/memory", json={"text": "prefers dark mode"})
        assert dup.status_code == 409
        err = dup.json()["error"]
        assert err["message"] == f"duplicate of {first['id']}: prefers dark mode"
        assert err["type"] == "invalid_request_error"
        forced = memory_client.post("/v1/memory", json={"text": "prefers dark mode", "force": True})
        assert forced.status_code == 200
        assert forced.json()["id"] != first["id"]
        # A same-text ask is not a duplicate of a fact: dedup is per kind.
        ask = memory_client.post("/v1/memory", json={"text": "prefers dark mode", "kind": "ask"})
        assert ask.status_code == 200
        statuses = [e.details["status"] for e in _store().events("endpoint")]
        assert sorted(statuses) == [200, 200, 200, 409]

    def test_post_without_models_is_503_and_still_recorded(self, no_models_client) -> None:
        resp = no_models_client.post("/v1/memory", json={"text": "x"})
        assert resp.status_code == 503
        assert resp.json()["error"]["message"] == "memory models are not loaded"
        [event] = _store().events("endpoint")
        assert event.details["status"] == 503
        assert _store().list_active() == []


def _seed(client, text: str, kind: str = "fact") -> dict:
    resp = client.post("/v1/memory", json={"text": text, "kind": kind})
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestSearch:
    def test_search_ranks_by_cross_encoder_and_records_a_retrieve_event(
        self, memory_client
    ) -> None:
        deposit = _seed(memory_client, "the brokerage account received a deposit")
        cat = _seed(memory_client, "the cat prefers the sunny window")
        ask = _seed(memory_client, "asked about brokerage deposits", kind="ask")
        resp = memory_client.get("/v1/memory", params={"q": "brokerage deposit"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        data = body["data"]
        # Token-overlap fake: "deposit" matches the fact, "deposits" does not
        # match the ask, the cat shares nothing (0.0, kept by the inclusive floor).
        assert [d["id"] for d in data] == [deposit["id"], ask["id"], cat["id"]]
        assert all("cosine" in d and "ce_score" in d for d in data)
        assert data[0]["ce_score"] >= data[-1]["ce_score"]
        assert {d["kind"] for d in data} == {"fact", "ask"}
        [event] = _store().events("retrieve")
        assert event.details["origin"] == "endpoint"
        assert event.details["item_ids"] == [d["id"] for d in data]
        # Search is not exposure: nothing gained a use.
        assert all(i.uses == 0 for i in _store().list_active())

    def test_search_limit(self, memory_client) -> None:
        for i in range(4):
            _seed(memory_client, f"fact about topic number {i}")
        resp = memory_client.get("/v1/memory", params={"q": "topic", "limit": 2})
        assert len(resp.json()["data"]) == 2
        assert memory_client.get("/v1/memory", params={"q": "t", "limit": 0}).status_code == 422

    def test_list_without_query_is_newest_first(self, memory_client) -> None:
        a = _seed(memory_client, "first")
        b = _seed(memory_client, "second")
        data = memory_client.get("/v1/memory").json()["data"]
        assert [d["id"] for d in data] == [b["id"], a["id"]]
        assert "cosine" not in data[0]
        assert _store().events("retrieve") == []

    def test_search_without_models_is_503_but_listing_works(self, no_models_client) -> None:
        assert no_models_client.get("/v1/memory").status_code == 200
        resp = no_models_client.get("/v1/memory", params={"q": "x"})
        assert resp.status_code == 503
        assert resp.json()["error"]["message"] == "memory models are not loaded"


class TestItemRoutes:
    def test_get_item_and_404(self, memory_client) -> None:
        item = _seed(memory_client, "a fact")
        assert memory_client.get(f"/v1/memory/{item['id']}").json() == item
        missing = memory_client.get("/v1/memory/m_NOPE")
        assert missing.status_code == 404
        assert missing.json()["error"]["message"] == "memory item not found"
        statuses = [e.details["status"] for e in _store().events("endpoint")]
        assert statuses == [404, 200, 200]  # newest first

    def test_delete_archives_once_with_a_forgotten_outcome(self, memory_client) -> None:
        item = _seed(memory_client, "a fact")
        resp = memory_client.delete(f"/v1/memory/{item['id']}")
        assert resp.status_code == 200
        gone = resp.json()
        assert gone["archived_at"] is not None
        assert gone["archive_reason"] == "forgotten"
        assert gone["last_outcome"] == "forgotten"
        assert gone["score"] == pytest.approx(0.3)
        [outcome] = _store().outcomes(item["id"])
        assert (outcome.signal, outcome.origin) == ("forgotten", "explicit")
        again = memory_client.delete(f"/v1/memory/{item['id']}")
        assert again.status_code == 200
        assert again.json() == gone
        assert len(_store().outcomes(item["id"])) == 1
        assert memory_client.delete("/v1/memory/m_NOPE").status_code == 404

    def test_confirm_moves_score_and_refuses_archived(self, memory_client) -> None:
        item = _seed(memory_client, "a fact")
        resp = memory_client.post(f"/v1/memory/{item['id']}/confirm")
        assert resp.status_code == 200
        assert resp.json()["score"] == pytest.approx(0.8)
        assert resp.json()["last_outcome"] == "confirmed"
        memory_client.delete(f"/v1/memory/{item['id']}")
        refused = memory_client.post(f"/v1/memory/{item['id']}/confirm")
        assert refused.status_code == 409
        assert refused.json()["error"]["message"] == "memory item is archived"
        assert memory_client.post("/v1/memory/m_NOPE/confirm").status_code == 404
        assert [o.signal for o in _store().outcomes(item["id"])] == ["forgotten", "confirmed"]


class TestFailClosed:
    def test_store_failure_is_503_and_logged_once_per_route(
        self, monkeypatch, fakes, caplog
    ) -> None:
        """A store that cannot open (a directory where the file should be, a
        permission slip): every route answers 503 in the envelope, never a 500,
        and the traceback lands once per cause."""

        class _Broken(MemoryRuntime):
            @property
            def store(self) -> MemoryStore:
                raise sqlite3.OperationalError("unable to open database file")

        def factory(config, **kwargs):
            return _Broken(config, loader=lambda config, *, allow_download: fakes)

        monkeypatch.setattr(_proxy, "MemoryRuntime", factory)
        monkeypatch.setattr(_proxy, "_memory_enabled", lambda: True)
        with (
            caplog.at_level(logging.WARNING, logger="pdp_router._proxy"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}),
            TestClient(app) as c,
        ):
            _proxy._memory_runtime.wait_for_load(timeout=5)
            for _ in range(2):
                resp = c.get("/v1/memory")
                assert resp.status_code == 503
                assert resp.json()["error"]["message"] == "memory store unavailable"
            resp = c.post("/v1/memory", json={"text": "x"})
            assert resp.status_code == 503
            assert c.get("/health").status_code == 200
        route_records = [
            r for r in caplog.records if "unable to open database file" in r.message
        ]
        contexts = sorted(r.message.split("(")[1].split(")")[0] for r in route_records)
        assert contexts == ["GET /v1/memory", "POST /v1/memory", "startup"]
        assert all(r.exc_info is not None for r in route_records)


# -- Step 6: shadow retrieval on conversation start --


def _shadow_dir() -> Path:
    return _db_path().parent / "memory-shadow"


def _shadow_lines() -> list[dict]:
    files = sorted(_shadow_dir().glob("shadow-*.jsonl")) if _shadow_dir().exists() else []
    lines: list[dict] = []
    for f in files:
        lines.extend(json.loads(line) for line in f.read_text().splitlines() if line)
    return lines


def _inbox_rows() -> list[dict]:
    inbox = Path(os.environ["PROXY_ROUTING_INBOX_DIR"])
    rows: list[dict] = []
    for f in sorted(inbox.glob("proxy-*.jsonl")):
        rows.extend(json.loads(line) for line in f.read_text().splitlines() if line)
    return rows


def _mock_completion(text: str = "ok") -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=50,
        output_tokens=20,
        model="gemini-2.5-flash",
        estimated_cost_usd=0.0001,
    )


def _chat(client, messages: list[dict], *, path: str = "/v1/chat/completions"):
    """One non-streaming routed turn with the provider and classifier mocked;
    returns (response, mock client) so the provider payload can be inspected."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = _mock_completion()
    mock_llm.complete_multi.return_value = _mock_completion()
    with (
        patch("pdp_router._proxy._autopanel_enabled", return_value=False),
        patch("pdp_router._proxy._streaming_enabled", return_value=False),
        patch("pdp_router._proxy._web_search_enabled", return_value=False),
        patch("pdp_router._proxy._classify_request", return_value=(0.55, 3, 0, "general")),
        patch("pdp_router._proxy.get_client", return_value=mock_llm),
    ):
        resp = client.post(path, json={"model": "pdp-auto", "messages": messages})
    return resp, mock_llm


_T1 = "how much did I deposit into the brokerage account"
_T2 = "and when was that"


def _turn1(client, **kw):
    return _chat(client, [{"role": "user", "content": _T1}], **kw)


def _turn2(client, **kw):
    return _chat(
        client,
        [
            {"role": "user", "content": _T1},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": _T2},
        ],
        **kw,
    )


@pytest.fixture()
def shadow_client(memory_client, monkeypatch):
    """Master + shadow on, fakes loaded, and a cross-encoder floor the fake's
    [0, 1] overlap score can actually fail (the real reranker goes negative
    on irrelevant text; the fake bottoms out at exactly 0.0)."""
    monkeypatch.setattr(_proxy, "_memory_shadow_enabled", lambda: True)
    object.__setattr__(_proxy._config, "memory_min_ce_score", 0.01)
    return memory_client


class TestShadowRetrieval:
    def test_two_turns_write_one_shadow_line_one_pin_one_event(self, shadow_client, fakes) -> None:
        deposit = _seed(shadow_client, "deposited money into the brokerage account")
        _seed(shadow_client, "cat likes windows")
        fakes.embedder.calls.clear()
        fakes.reranker.calls.clear()

        first, _ = _turn1(shadow_client)
        assert first.status_code == 200
        assert len(fakes.embedder.calls) == 1
        assert fakes.embedder.calls[0] == [_T1]
        second, _ = _turn2(shadow_client)
        assert second.status_code == 200
        # The second turn resolved from the in-process mirror: no model work.
        assert len(fakes.embedder.calls) == 1
        assert len(fakes.reranker.calls) == 1

        [line] = _shadow_lines()
        digest = _proxy._tool_pin_digest(_T1)
        assert line["surface"] == "v1"
        assert line["query"] == _T1
        assert line["item_ids"] == [deposit["id"]]
        assert line["block"].startswith("[memory] Context as of ")
        assert "deposited money into the brokerage account" in line["block"]
        assert "cat" not in line["block"]
        assert line["conversation_key8"] == digest[:8]
        routed = [r for r in _inbox_rows() if r["routing_mode"] != "implicit_feedback"]
        assert {json.loads(r["context_json"])["conversation_key"] for r in routed} == {digest[:8]}

        pin = _store().get_pin(digest)
        assert pin is not None
        assert pin.block == line["block"]
        assert pin.item_ids == [deposit["id"]]
        assert _proxy._conversation_cache.peek(digest).memory_block == pin.block

        [event] = _store().events("retrieve")
        assert event.details["origin"] == "shadow"
        assert event.details["surface"] == "v1"
        assert event.details["item_ids"] == [deposit["id"]]
        assert event.details["chat_request_id"] == first.headers["X-PDP-Prediction-Id"]
        # Shadow is not exposure: no model saw the block, so uses stays 0.
        assert _store().get_item(deposit["id"]).uses == 0

    def test_openai_v1_surface_is_labeled(self, shadow_client) -> None:
        _seed(shadow_client, "deposited money into the brokerage account")
        resp, _ = _turn1(shadow_client, path="/openai/v1/chat/completions")
        assert resp.status_code == 200
        [line] = _shadow_lines()
        assert line["surface"] == "openai_v1"

    def test_empty_retrieval_still_pins_the_date_line(self, shadow_client, fakes) -> None:
        """Nothing stored: the block is the date line alone, pinned so later
        turns do not retry, and the shadow line records the empty result."""
        _turn1(shadow_client)
        _turn2(shadow_client)
        [line] = _shadow_lines()
        assert line["item_ids"] == []
        assert line["block"].endswith("(UTC).\n[/memory]")
        assert len(fakes.embedder.calls) == 1
        assert _store().get_pin(_proxy._tool_pin_digest(_T1)) is not None

    def test_pin_survives_a_state_cache_reset(self, shadow_client, fakes) -> None:
        """A restart or an eviction loses the in-process mirror; the pin in
        memory.db means the block is re-read, never re-retrieved."""
        _seed(shadow_client, "deposited money into the brokerage account")
        _turn1(shadow_client)
        fakes.embedder.calls.clear()
        _proxy._conversation_cache = ConversationCache(max_entries=512, ttl_s=7200.0)
        resp, _ = _turn2(shadow_client)
        assert resp.status_code == 200
        assert fakes.embedder.calls == []
        assert len(_shadow_lines()) == 1
        digest = _proxy._tool_pin_digest(_T1)
        assert (
            _proxy._conversation_cache.peek(digest).memory_block == _store().get_pin(digest).block
        )

    def test_pin_expiry_retrieves_again(self, shadow_client, fakes) -> None:
        _seed(shadow_client, "deposited money into the brokerage account")
        _turn1(shadow_client)
        digest = _proxy._tool_pin_digest(_T1)
        pin = _store().get_pin(digest)
        # Expire it (ttl 0 -> expires_at == created_at) and drop the mirror,
        # the state a day-idle conversation is in.
        _store().set_pin(digest, pin.block, pin.item_ids, ttl_s=0)
        _proxy._conversation_cache = ConversationCache(max_entries=512, ttl_s=7200.0)
        fakes.embedder.calls.clear()
        _turn2(shadow_client)
        assert len(fakes.embedder.calls) == 1
        assert len(_shadow_lines()) == 2
        assert _store().get_pin(digest) is not None

    def test_shadow_off_never_resolves(self, memory_client, fakes, monkeypatch) -> None:
        calls: list[int] = []
        real = _proxy._memory_shadow_turn

        async def spy(*args, **kwargs):
            calls.append(1)
            return await real(*args, **kwargs)

        monkeypatch.setattr(_proxy, "_memory_shadow_turn", spy)
        _seed(memory_client, "deposited money into the brokerage account")
        fakes.embedder.calls.clear()
        _turn1(memory_client)
        _turn2(memory_client)
        assert calls == []
        assert fakes.embedder.calls == []
        assert not _shadow_dir().exists()
        assert _store().get_pin(_proxy._tool_pin_digest(_T1)) is None
        assert _store().events("retrieve") == []

    def test_master_off_never_resolves(self, plain_client, monkeypatch) -> None:
        monkeypatch.setattr(_proxy, "_memory_shadow_enabled", lambda: True)
        resp, _ = _turn1(plain_client)
        assert resp.status_code == 200
        assert not _db_path().exists()
        assert not _shadow_dir().exists()

    def test_models_not_loaded_serves_and_leaves_no_pin(
        self, no_models_client, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(_proxy, "_memory_shadow_enabled", lambda: True)
        with caplog.at_level(logging.INFO, logger="pdp_router._proxy"):
            resp, _ = _turn1(no_models_client)
        assert resp.status_code == 200
        assert _store().get_pin(_proxy._tool_pin_digest(_T1)) is None
        assert not _shadow_dir().exists()
        assert any("models not loaded" in r.message for r in caplog.records)

    def test_provider_payload_is_untouched_by_shadow(self, shadow_client) -> None:
        """Shadow injects nothing: the provider sees the request's own text,
        no block anywhere, on the single-turn and the multi-turn path."""
        _seed(shadow_client, "deposited money into the brokerage account")
        _, llm1 = _turn1(shadow_client)
        args1 = llm1.complete.call_args
        assert args1 is not None
        assert "[memory]" not in json.dumps(
            [str(a) for a in args1.args] + [str(v) for v in args1.kwargs.values()]
        )
        assert _T1 in [str(a) for a in args1.args] + [str(v) for v in args1.kwargs.values()]
        _, llm2 = _turn2(shadow_client)
        args2 = llm2.complete_multi.call_args
        assert args2 is not None
        sent = [
            m
            for a in list(args2.args) + list(args2.kwargs.values())
            if isinstance(a, list)
            for m in a
        ]
        contents = [
            m["content"] if isinstance(m, dict) else getattr(m, "content", None) for m in sent
        ]
        assert contents == [_T1, "answer", _T2]

    def test_store_failure_serves_the_request_and_logs_once(
        self, monkeypatch, fakes, caplog
    ) -> None:
        class _Broken(MemoryRuntime):
            @property
            def store(self) -> MemoryStore:
                raise sqlite3.OperationalError("unable to open database file")

        def factory(config, **kwargs):
            return _Broken(config, loader=lambda config, *, allow_download: fakes)

        monkeypatch.setattr(_proxy, "MemoryRuntime", factory)
        monkeypatch.setattr(_proxy, "_memory_enabled", lambda: True)
        monkeypatch.setattr(_proxy, "_memory_shadow_enabled", lambda: True)
        with (
            caplog.at_level(logging.WARNING, logger="pdp_router._proxy"),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}),
            TestClient(app) as c,
        ):
            _proxy._memory_runtime.wait_for_load(timeout=5)
            first, llm = _turn1(c)
            second, _ = _turn2(c)
        assert first.status_code == 200 and second.status_code == 200
        assert llm.complete.call_args is not None
        shadow_records = [r for r in caplog.records if "(shadow retrieval)" in r.message]
        assert len(shadow_records) == 1
        assert shadow_records[0].exc_info is not None
        assert not _shadow_dir().exists()

    def test_spend_capped_conversation_never_retrieves(self, shadow_client, fakes) -> None:
        """The resolver sits after the spend cap and before the forks: a 429
        conversation pays no retrieval and leaves no pin."""
        _seed(shadow_client, "deposited money into the brokerage account")
        fakes.embedder.calls.clear()
        digest = _proxy._tool_pin_digest(_T1)
        _proxy._conversation_cache.get(digest).spend_usd = 99.0
        with patch("pdp_router._proxy._spend_cap_enabled", return_value=True):
            resp, _ = _turn1(shadow_client)
        assert resp.status_code == 429
        assert fakes.embedder.calls == []
        assert _store().get_pin(digest) is None
        assert not _shadow_dir().exists()

    def test_implicit_feedback_still_grades_the_raw_text(self, shadow_client) -> None:
        """Retrieval runs after the feedback digest is taken, and the block
        never enters messages, so the feedback row is what it was before."""
        _seed(shadow_client, "deposited money into the brokerage account")
        with patch("pdp_router._proxy._implicit_feedback_enabled", return_value=True):
            first, _ = _turn1(shadow_client)
            _turn2(shadow_client)
        fb = [r for r in _inbox_rows() if r["routing_mode"] == "implicit_feedback"]
        assert len(fb) == 1
        ctx = json.loads(fb[0]["context_json"])
        assert ctx["feedback_signal"] == "moved_on"
        assert ctx["target_chat_request_id"] == first.headers["X-PDP-Prediction-Id"]
        assert len(_shadow_lines()) == 1
