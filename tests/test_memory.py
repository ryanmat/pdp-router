# Description: Tests for the proxy memory sidecar: import discipline, store, retrieval,
# Description: block assembly, consolidation. Real-model tests live in test_memory_models.py.

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

pytest.importorskip("fastapi")

from pdp_router import _memory
from pdp_router._memory import (
    CANDIDATES_PER_KIND,
    KINDS,
    ORIGINS,
    OUTCOME_DELTAS,
    QUERY_MAX_CHARS,
    SCHEMA_VERSION,
    SIGNALS,
    TIERS,
    TOP_ASKS,
    TOP_FACTS,
    MemoryItem,
    MemoryModels,
    MemoryRuntime,
    MemoryStore,
    Retrieval,
    UnknownMemoryItemError,
    append_shadow_jsonl,
    assemble_block,
    consolidate,
    cosine_top_k,
    decode_embedding,
    encode_embedding,
    find_duplicate,
    new_item_id,
    retrieve,
)
from pdp_router._proxy_config import ProxyConfig
from tests._memory_fakes import (
    FakeEmbedder,
    FakeReranker,
    ScoreTableReranker,
    VectorTableEmbedder,
)

NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)
_CROCKFORD = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


class TestLazyImports:
    def test_importing_the_proxy_loads_no_embedding_stack(self) -> None:
        """A flag-off proxy must never pay for onnxruntime. Importing the app
        module leaves the whole embedding stack out of sys.modules; the check
        runs in a subprocess so imports made earlier in this test process
        cannot mask a regression."""
        code = (
            "import sys, json; import pdp_router._proxy; "
            "print(json.dumps(sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('fastembed', 'onnxruntime', 'numpy'))))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout) == []

    def test_store_module_needs_no_numeric_stack(self) -> None:
        """The store (used by the stats CLI and a future tick check) must open
        and migrate without numpy: only the embedding codec touches it."""
        code = (
            "import sys, json, tempfile, os; from pdp_router._memory import MemoryStore; "
            "d = tempfile.mkdtemp(); MemoryStore(os.path.join(d, 'm.db')).migrate(); "
            "print(json.dumps(sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('fastembed', 'onnxruntime', 'numpy'))))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout) == []


class TestVocabulary:
    def test_constants_match_the_spec(self) -> None:
        assert SCHEMA_VERSION == 1
        assert KINDS == ("fact", "ask")
        assert TIERS == ("working", "history", "patterns")
        assert SIGNALS == ("worked", "failed", "partial", "unknown", "confirmed", "forgotten")
        assert ORIGINS == ("sidecar", "implicit", "explicit")
        assert OUTCOME_DELTAS == {
            "worked": (0.20, 1.0),
            "failed": (-0.30, 0.0),
            "partial": (0.05, 0.5),
            "unknown": (-0.05, 0.25),
            "confirmed": (0.20, 1.0),
            "forgotten": (-0.30, 0.0),
        }


class TestItemIds:
    def test_shape(self) -> None:
        item_id = new_item_id()
        assert item_id.startswith("m_")
        assert len(item_id) == 28
        assert set(item_id[2:]) <= _CROCKFORD

    def test_time_ordered(self) -> None:
        """Ids sort by creation time (a ULID): the time part is a fixed-width
        base32 of the millisecond clock, so a later id always compares greater."""
        earlier = new_item_id(now_ms=1_700_000_000_000)
        later = new_item_id(now_ms=1_700_000_000_001)
        assert earlier < later
        assert earlier[:12] != later[:12]

    def test_distinct_within_one_millisecond(self) -> None:
        ids = {new_item_id(now_ms=1_700_000_000_000) for _ in range(50)}
        assert len(ids) == 50


class TestEmbeddingCodec:
    def test_round_trip_is_float32_little_endian(self) -> None:
        blob = encode_embedding([0.5, -1.0, 2.0])
        assert blob == struct.pack("<3f", 0.5, -1.0, 2.0)
        vec = decode_embedding(blob)
        assert vec.dtype == np.dtype("<f4")
        assert vec.tolist() == [0.5, -1.0, 2.0]

    def test_accepts_numpy_input(self) -> None:
        blob = encode_embedding(np.array([1.0, 2.0], dtype=np.float64))
        assert decode_embedding(blob).tolist() == [1.0, 2.0]

    def test_rejects_a_truncated_blob(self) -> None:
        with pytest.raises(ValueError, match="multiple of 4"):
            decode_embedding(b"\x00\x00\x00\x00\x00")


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    s = MemoryStore(tmp_path / "memory.db")
    s.migrate()
    return s


def _fact(store: MemoryStore, text: str = "prefers dark mode", **kw) -> MemoryItem:
    kw.setdefault("kind", "fact")
    kw.setdefault("embedding", encode_embedding([1.0, 0.0]))
    kw.setdefault("embedding_model", "test/embed")
    kw.setdefault("observed_at", "2026-09-01")
    kw.setdefault("source", "explicit:test")
    kw.setdefault("now", NOW)
    return store.add_item(text=text, **kw)


class TestMigrate:
    def test_creates_every_table_and_records_the_version(self, tmp_path) -> None:
        path = tmp_path / "nested" / "dir" / "memory.db"
        MemoryStore(path).migrate()
        conn = sqlite3.connect(path)
        names = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_version")]
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert {
            "memory_items",
            "memory_outcomes",
            "memory_pins",
            "memory_events",
            "schema_version",
        } <= names
        assert versions == [SCHEMA_VERSION]
        assert journal == "wal"

    def test_migrate_is_idempotent(self, tmp_path) -> None:
        s = MemoryStore(tmp_path / "memory.db")
        s.migrate()
        s.migrate()
        conn = sqlite3.connect(tmp_path / "memory.db")
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_version")]
        conn.close()
        assert versions == [SCHEMA_VERSION]

    def test_connections_carry_the_busy_timeout(self, tmp_path) -> None:
        """Two writers (the proxy and the nightly CLI) share this file; without a
        busy timeout the second one fails instantly on SQLITE_BUSY."""
        s = MemoryStore(tmp_path / "memory.db", busy_timeout_ms=1234)
        s.migrate()
        with s._connect() as conn:
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1234


class TestItems:
    def test_add_item_defaults_and_round_trip(self, store) -> None:
        item = _fact(store, surface="cli", conversation_key8="abcd1234", chat_request_id="req-1")
        assert item.id.startswith("m_")
        assert item.kind == "fact"
        assert item.text == "prefers dark mode"
        assert item.embedding_model == "test/embed"
        assert item.created_at == "2026-09-07T12:00:00Z"
        assert item.observed_at == "2026-09-01"
        assert item.source == "explicit:test"
        assert item.surface == "cli"
        assert item.conversation_key8 == "abcd1234"
        assert item.chat_request_id == "req-1"
        assert item.tier == "working"
        assert item.score == 0.5
        assert item.uses == 0
        assert item.success == 0.0
        assert item.last_outcome is None
        assert item.archived_at is None
        assert item.archive_reason is None
        assert store.get_item(item.id) == item

    def test_add_item_honors_an_explicit_id_and_score(self, store) -> None:
        item = _fact(store, item_id="m_FIXED", score=0.6)
        assert item.id == "m_FIXED"
        assert item.score == 0.6

    def test_add_item_strips_text(self, store) -> None:
        assert _fact(store, text="  spaced out  ").text == "spaced out"

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("kind", "opinion", "kind"),
            ("text", "   ", "empty"),
            ("observed_at", "yesterday", "observed_at"),
        ],
    )
    def test_add_item_rejects_bad_input(self, store, field, value, match) -> None:
        with pytest.raises(ValueError, match=match):
            _fact(store, **{field: value})

    def test_get_item_unknown_is_none(self, store) -> None:
        assert store.get_item("m_NOPE") is None

    def test_list_active_filters_kind_and_excludes_archived(self, store) -> None:
        older = _fact(store, "a", now=NOW)
        newer = _fact(store, "b", now=NOW + timedelta(seconds=1))
        ask = _fact(store, "asked about c", kind="ask")
        gone = _fact(store, "d")
        store.archive_item(gone.id, "forgotten", now=NOW)
        # Newest first by created_at; ties (older and ask share NOW) break on
        # the time-ordered id, so the item created later in wall-clock wins.
        assert [i.id for i in store.list_active()] == [newer.id, ask.id, older.id]
        assert [i.id for i in store.list_active("fact")] == [newer.id, older.id]
        assert [i.id for i in store.list_active("ask")] == [ask.id]
        assert [i.id for i in store.list_active("fact", limit=1)] == [newer.id]

    def test_active_embeddings_scopes_to_kind_model_and_liveness(self, store) -> None:
        kept = _fact(store, "kept", embedding=encode_embedding([1.0, 0.0]))
        _fact(store, "an ask", kind="ask")
        other = _fact(store, "other space", embedding_model="other/embed")
        _fact(store, "no vector", embedding=None, embedding_model=None)
        gone = _fact(store, "gone")
        store.archive_item(gone.id, "forgotten")
        rows = store.active_embeddings("fact", embedding_model="test/embed")
        assert rows == [(kept.id, encode_embedding([1.0, 0.0]))]
        # Without a model filter every vector-bearing active fact is returned,
        # whichever space it was embedded in.
        assert {r[0] for r in store.active_embeddings("fact")} == {kept.id, other.id}

    def test_archive_item_is_soft_and_single_shot(self, store) -> None:
        item = _fact(store)
        assert store.archive_item(item.id, "forgotten", now=NOW) is True
        archived = store.get_item(item.id)
        assert archived.archived_at == "2026-09-07T12:00:00Z"
        assert archived.archive_reason == "forgotten"
        assert store.list_active() == []
        assert store.archive_item(item.id, "again") is False
        assert store.get_item(item.id).archive_reason == "forgotten"
        assert store.archive_item("m_NOPE", "forgotten") is False

    def test_mark_exposed_increments_only_the_named_items(self, store) -> None:
        a = _fact(store, "a")
        b = _fact(store, "b")
        c = _fact(store, "c")
        store.mark_exposed([a.id, b.id])
        store.mark_exposed([a.id])
        assert store.get_item(a.id).uses == 2
        assert store.get_item(b.id).uses == 1
        assert store.get_item(c.id).uses == 0
        store.mark_exposed([])  # no-op, no error


class TestOutcomes:
    @pytest.mark.parametrize(
        ("signal", "score", "success"),
        [
            ("worked", 0.7, 1.0),
            ("failed", 0.2, 0.0),
            ("partial", 0.55, 0.5),
            ("unknown", 0.45, 0.25),
            ("confirmed", 0.7, 1.0),
            ("forgotten", 0.2, 0.0),
        ],
    )
    def test_delta_table_moves_score_and_success(self, store, signal, score, success) -> None:
        item = _fact(store)
        updated = store.add_outcome(item_id=item.id, signal=signal, origin="explicit", now=NOW)
        assert updated.score == pytest.approx(score)
        assert updated.success == pytest.approx(success)
        assert updated.last_outcome == signal
        assert store.get_item(item.id) == updated

    def test_score_is_clamped_to_the_unit_interval(self, store) -> None:
        high = _fact(store, "high", score=0.95)
        low = _fact(store, "low", score=0.1)
        assert store.add_outcome(item_id=high.id, signal="worked", origin="sidecar").score == 1.0
        assert store.add_outcome(item_id=low.id, signal="failed", origin="sidecar").score == 0.0

    def test_explicit_delta_overrides_the_table(self, store) -> None:
        item = _fact(store)
        updated = store.add_outcome(
            item_id=item.id, signal="partial", origin="implicit", delta=-0.1, now=NOW
        )
        assert updated.score == pytest.approx(0.4)
        assert updated.success == pytest.approx(0.5)
        [row] = store.outcomes(item.id)
        assert row.delta == pytest.approx(-0.1)

    def test_rows_carry_provenance(self, store) -> None:
        item = _fact(store)
        store.add_outcome(
            item_id=item.id,
            signal="worked",
            origin="sidecar",
            conversation_key8="abcd1234",
            chat_request_id="req-9",
            note="model judged it useful",
            now=NOW,
        )
        store.add_outcome(
            item_id=item.id, signal="failed", origin="explicit", now=NOW + timedelta(seconds=5)
        )
        rows = store.outcomes(item.id)
        assert [r.signal for r in rows] == ["failed", "worked"]
        first = rows[1]
        assert first.ts == "2026-09-07T12:00:00Z"
        assert first.origin == "sidecar"
        assert first.delta == pytest.approx(0.2)
        assert first.conversation_key8 == "abcd1234"
        assert first.chat_request_id == "req-9"
        assert first.note == "model judged it useful"

    def test_rejects_unknown_vocabulary_and_items(self, store) -> None:
        item = _fact(store)
        with pytest.raises(ValueError, match="signal"):
            store.add_outcome(item_id=item.id, signal="great", origin="explicit")
        with pytest.raises(ValueError, match="origin"):
            store.add_outcome(item_id=item.id, signal="worked", origin="telepathy")
        with pytest.raises(UnknownMemoryItemError):
            store.add_outcome(item_id="m_NOPE", signal="worked", origin="explicit")
        assert store.outcomes(item.id) == []


class TestPins:
    def test_round_trip(self, store) -> None:
        pin = store.set_pin("k" * 64, "[memory] block", ["m_A", "m_B"], ttl_s=60, now=NOW)
        assert pin.conversation_key == "k" * 64
        assert pin.block == "[memory] block"
        assert pin.item_ids == ["m_A", "m_B"]
        assert pin.created_at == "2026-09-07T12:00:00Z"
        assert pin.expires_at == "2026-09-07T12:01:00Z"
        assert store.get_pin("k" * 64, now=NOW + timedelta(seconds=59)) == pin

    def test_expired_reads_as_absent(self, store) -> None:
        store.set_pin("k", "block", [], ttl_s=60, now=NOW)
        assert store.get_pin("k", now=NOW + timedelta(seconds=60)) is None
        assert store.get_pin("k", now=NOW + timedelta(days=3)) is None

    def test_unknown_is_none_and_set_replaces(self, store) -> None:
        assert store.get_pin("missing", now=NOW) is None
        store.set_pin("k", "first", ["m_A"], ttl_s=60, now=NOW)
        store.set_pin("k", "second", [], ttl_s=60, now=NOW + timedelta(seconds=1))
        pin = store.get_pin("k", now=NOW + timedelta(seconds=2))
        assert pin.block == "second"
        assert pin.item_ids == []
        assert pin.created_at == "2026-09-07T12:00:01Z"


class TestEvents:
    def test_details_round_trip_and_ordering(self, store) -> None:
        store.add_event("retrieve", {"ids": ["m_A"], "n": 1}, now=NOW)
        store.add_event("consolidate", {"archived": 2}, now=NOW + timedelta(seconds=1))
        store.add_event("retrieve", {"ids": [], "n": 0}, now=NOW + timedelta(seconds=2))
        events = store.events()
        assert [e.event for e in events] == ["retrieve", "consolidate", "retrieve"]
        assert events[0].ts == "2026-09-07T12:00:02Z"
        assert events[-1].details == {"ids": ["m_A"], "n": 1}

    def test_filters(self, store) -> None:
        store.add_event("retrieve", {"n": 1}, now=NOW)
        store.add_event("retrieve", {"n": 2}, now=NOW + timedelta(hours=1))
        store.add_event("endpoint", {"route": "/v1/memory"}, now=NOW + timedelta(hours=2))
        assert [e.details["n"] for e in store.events("retrieve")] == [2, 1]
        assert [e.details["n"] for e in store.events("retrieve", limit=1)] == [2]
        assert [e.event for e in store.events(since=NOW + timedelta(minutes=30))] == [
            "endpoint",
            "retrieve",
        ]
        assert store.events("grade") == []


class TestStats:
    def test_shape(self, store, tmp_path) -> None:
        a = _fact(store, "a")
        _fact(store, "b", kind="ask")
        gone = _fact(store, "c")
        store.archive_item(gone.id, "forgotten", now=NOW)
        store.add_outcome(item_id=a.id, signal="worked", origin="sidecar", now=NOW)
        store.set_pin("live", "block", [a.id], ttl_s=3600, now=NOW)
        store.set_pin("dead", "block", [], ttl_s=1, now=NOW - timedelta(days=1))
        store.add_event("consolidate", {"archived": 0}, now=NOW - timedelta(hours=30))
        store.add_event("retrieve", {"n": 1}, now=NOW - timedelta(hours=1))
        store.add_event("retrieve", {"n": 0}, now=NOW - timedelta(hours=25))
        stats = store.stats(now=NOW)
        assert stats["path"] == str(tmp_path / "memory.db")
        assert stats["schema_version"] == SCHEMA_VERSION
        assert stats["items"]["active"] == 2
        assert stats["items"]["archived"] == 1
        assert stats["items"]["by_kind_tier"] == {"fact": {"working": 1}, "ask": {"working": 1}}
        assert stats["outcomes"] == 1
        assert stats["pins"] == {"active": 1, "expired": 1}
        assert stats["last_consolidate_ts"] == "2026-09-06T06:00:00Z"
        assert stats["events_24h"] == {"retrieve": 1}

    def test_empty_store(self, store) -> None:
        stats = store.stats(now=NOW)
        assert stats["items"] == {"active": 0, "archived": 0, "by_kind_tier": {}}
        assert stats["last_consolidate_ts"] is None
        assert stats["events_24h"] == {}


class TestConcurrency:
    def test_two_handles_interleave_on_one_file(self, tmp_path) -> None:
        """The proxy and the nightly CLI open the same file as separate
        processes; each call opens its own connection so neither holds a lock
        across calls."""
        path = tmp_path / "memory.db"
        proxy = MemoryStore(path)
        proxy.migrate()
        nightly = MemoryStore(path)
        a = _fact(proxy, "from the proxy")
        b = _fact(nightly, "from the nightly job")
        nightly.archive_item(a.id, "consolidate:test")
        assert proxy.get_item(a.id).archived_at is not None
        assert [i.id for i in proxy.list_active()] == [b.id]

    def test_threaded_writers_all_land(self, tmp_path) -> None:
        """asyncio.to_thread runs retrieval beside request-path writes; WAL plus
        the busy timeout serializes them instead of raising SQLITE_BUSY."""
        path = tmp_path / "memory.db"
        MemoryStore(path).migrate()
        errors: list[BaseException] = []

        def writer(n: int) -> None:
            s = MemoryStore(path)
            try:
                for i in range(25):
                    _fact(s, f"thread {n} item {i}")
            except BaseException as exc:  # collected for the assertion below
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(MemoryStore(path).list_active()) == 100


# -- Step 3: models, runtime, lazy loading --


class TestFakesSatisfyTheProtocols:
    """The fakes are the double every retrieval test runs on, so their SHAPE
    is a claim about the real wrappers; test_memory_models.py pins the real
    side of that claim (dim, dtype, norm, ordering) against fastembed."""

    def test_embedder(self) -> None:
        from pdp_router._memory import Embedder

        fake = FakeEmbedder(dim=8)
        assert isinstance(fake, Embedder)
        [a, b, a2] = fake.embed(["the cat sat", "quarterly revenue rose", "the cat sat"])
        assert a.dtype == np.float32 and a.shape == (8,)
        assert float(np.linalg.norm(a)) == pytest.approx(1.0)
        assert a.tolist() == a2.tolist()
        assert float(a @ b) < float(a @ a2)

    def test_reranker(self) -> None:
        from pdp_router._memory import Reranker

        fake = FakeReranker()
        assert isinstance(fake, Reranker)
        scores = fake.score("cat on the mat", ["the cat sat on the mat", "stock prices"])
        assert scores[0] > scores[1]
        assert fake.score("q", []) == []


def _runtime_config(monkeypatch, tmp_path) -> ProxyConfig:
    monkeypatch.setenv("PROXY_MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("PROXY_MEMORY_MODEL_DIR", str(tmp_path / "models"))
    return ProxyConfig()


def _fake_models() -> MemoryModels:
    return MemoryModels(embedder=FakeEmbedder(), reranker=FakeReranker())


class TestRuntime:
    def test_store_opens_lazily_and_migrates_once(self, monkeypatch, tmp_path) -> None:
        rt = MemoryRuntime(_runtime_config(monkeypatch, tmp_path))
        assert not (tmp_path / "memory.db").exists()
        store = rt.store
        assert (tmp_path / "memory.db").exists()
        assert store.stats()["schema_version"] == SCHEMA_VERSION
        assert rt.store is store

    def test_store_failure_raises_to_the_caller(self, monkeypatch, tmp_path) -> None:
        """A directory where the db file should be: the runtime does not hide
        it, the caller (route guard, shadow hook) is what fails closed."""
        (tmp_path / "memory.db").mkdir()
        rt = MemoryRuntime(_runtime_config(monkeypatch, tmp_path))
        with pytest.raises(sqlite3.OperationalError):
            _ = rt.store

    def test_blocking_load_returns_and_caches_the_models(self, monkeypatch, tmp_path) -> None:
        calls: list[bool] = []

        def loader(config, *, allow_download):
            calls.append(allow_download)
            return _fake_models()

        rt = MemoryRuntime(_runtime_config(monkeypatch, tmp_path), loader=loader)
        assert rt.models is None and rt.models_ready is False
        models = rt.load_models_blocking(allow_download=True)
        assert rt.models is models and rt.models_ready is True
        assert rt.load_error is None
        rt.start_model_load()
        rt.wait_for_load(timeout=5)
        assert calls == [True]

    def test_background_load_runs_once_and_lands(self, monkeypatch, tmp_path) -> None:
        release = threading.Event()
        calls: list[int] = []

        def loader(config, *, allow_download):
            calls.append(1)
            assert allow_download is False  # the service never downloads
            release.wait(timeout=5)
            return _fake_models()

        rt = MemoryRuntime(_runtime_config(monkeypatch, tmp_path), loader=loader)
        rt.start_model_load()
        rt.start_model_load()
        rt.start_model_load()
        assert rt.models is None
        release.set()
        rt.wait_for_load(timeout=5)
        assert rt.models_ready is True
        assert calls == [1]

    def test_load_failure_is_recorded_and_logged_once(self, monkeypatch, tmp_path, caplog) -> None:
        """Missing extra, missing model files, unreadable cache: the load fails,
        the runtime reports it, and the SAME cause never logs twice."""
        calls: list[int] = []

        def loader(config, *, allow_download):
            calls.append(1)
            raise ImportError("No module named 'fastembed'")

        rt = MemoryRuntime(_runtime_config(monkeypatch, tmp_path), loader=loader)
        with caplog.at_level(logging.WARNING, logger="pdp_router._memory"):
            rt.start_model_load()
            rt.wait_for_load(timeout=5)
            rt.start_model_load()
            rt.wait_for_load(timeout=5)
        assert rt.models is None and rt.models_ready is False
        assert rt.load_error == "ImportError: No module named 'fastembed'"
        assert calls == [1]  # inside the retry cooldown: no second attempt
        records = [r for r in caplog.records if "Memory models failed to load" in r.message]
        assert len(records) == 1
        assert records[0].exc_info is not None

    def test_failed_load_retries_after_the_cooldown_and_logs_new_causes(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        clock = {"now": 1000.0}
        monkeypatch.setattr(_memory.time, "monotonic", lambda: clock["now"])
        errors = iter(
            [
                ValueError("Could not load model a"),
                ValueError("Could not load model a"),
                ValueError("Could not load model b"),
            ]
        )

        def loader(config, *, allow_download):
            raise next(errors)

        rt = MemoryRuntime(_runtime_config(monkeypatch, tmp_path), loader=loader)
        with caplog.at_level(logging.WARNING, logger="pdp_router._memory"):
            for _ in range(3):
                rt.start_model_load()
                rt.wait_for_load(timeout=5)
                clock["now"] += _memory.LOAD_RETRY_S + 1
        messages = [r.message for r in caplog.records if "Memory models failed" in r.message]
        assert len(messages) == 2  # a, a (suppressed), b
        assert "model b" in rt.load_error

    def test_blocking_load_raises_and_records(self, monkeypatch, tmp_path) -> None:
        def loader(config, *, allow_download):
            raise RuntimeError("boom")

        rt = MemoryRuntime(_runtime_config(monkeypatch, tmp_path), loader=loader)
        with pytest.raises(RuntimeError, match="boom"):
            rt.load_models_blocking()
        assert rt.load_error == "RuntimeError: boom"
        assert rt.models is None


# -- Step 4: retrieval, block assembly, dedup, consolidation, shadow writer --


def _unit(v: list[float]) -> list[float]:
    arr = np.asarray(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


class TestStoreBatchReads:
    def test_get_items_returns_only_known_ids_keyed(self, store) -> None:
        a = _fact(store, "a")
        b = _fact(store, "b")
        found = store.get_items([b.id, "m_NOPE", a.id])
        assert set(found) == {a.id, b.id}
        assert found[a.id] == a
        assert store.get_items([]) == {}

    def test_delete_expired_pins(self, store) -> None:
        store.set_pin("live", "b", [], ttl_s=3600, now=NOW)
        store.set_pin("dead1", "b", [], ttl_s=1, now=NOW - timedelta(days=1))
        store.set_pin("dead2", "b", [], ttl_s=1, now=NOW - timedelta(days=2))
        assert store.delete_expired_pins(now=NOW) == 2
        assert store.get_pin("live", now=NOW) is not None
        assert store.stats(now=NOW)["pins"] == {"active": 1, "expired": 0}


class TestCosineTopK:
    def test_orders_by_cosine_and_cuts_at_k(self) -> None:
        q = np.asarray(_unit([1.0, 0.0, 0.0]), dtype=np.float32)
        rows = [
            ("far", encode_embedding(_unit([0.0, 1.0, 0.0]))),
            ("near", encode_embedding(_unit([1.0, 0.1, 0.0]))),
            ("exact", encode_embedding(_unit([1.0, 0.0, 0.0]))),
            ("mid", encode_embedding(_unit([1.0, 1.0, 0.0]))),
        ]
        ranked = cosine_top_k(q, rows, 3)
        assert [r[0] for r in ranked] == ["exact", "near", "mid"]
        assert ranked[0][1] == pytest.approx(1.0)
        assert ranked[2][1] == pytest.approx(0.7071, abs=1e-3)
        assert cosine_top_k(q, rows, 10)[-1][0] == "far"

    def test_empty_rows_and_zero_k(self) -> None:
        q = np.asarray([1.0, 0.0], dtype=np.float32)
        assert cosine_top_k(q, [], 5) == []
        assert cosine_top_k(q, [("a", encode_embedding([1.0, 0.0]))], 0) == []

    def test_skips_rows_of_another_width(self, caplog) -> None:
        """A blob from a different embedding space must never be compared; it
        is skipped and named, not raised, so one bad row cannot take retrieval
        down."""
        q = np.asarray([1.0, 0.0], dtype=np.float32)
        rows = [("ok", encode_embedding([1.0, 0.0])), ("bad", encode_embedding([1.0, 0.0, 0.0]))]
        with caplog.at_level(logging.WARNING, logger="pdp_router._memory"):
            assert [r[0] for r in cosine_top_k(q, rows, 5)] == ["ok"]
        assert any("1 embedding" in r.message and "width" in r.message for r in caplog.records)


def _table_models(table: dict[str, list[float]], scores: dict[str, float] | None = None, dim=3):
    return MemoryModels(
        embedder=VectorTableEmbedder(table, dim=dim), reranker=ScoreTableReranker(scores)
    )


def _seed(store, models, kind, text, vec, **kw) -> MemoryItem:
    return store.add_item(
        kind=kind,
        text=text,
        embedding=encode_embedding(_unit(vec)),
        embedding_model=models.embedder.model_name,
        observed_at=kw.pop("observed_at", "2026-09-01"),
        source=kw.pop("source", "explicit:test"),
        **kw,
    )


class TestRetrieve:
    def test_candidate_cut_then_rerank_then_gate_then_top_n(self, store) -> None:
        """45 facts; only the 40 nearest by cosine reach the reranker; the CE
        gate drops scores below the floor; the final order is CE order, which
        here disagrees with cosine order."""
        table = {"q": [1.0, 0.0, 0.0]}
        scores = {}
        for i in range(45):
            # cosine decreases with i: fact 0 is nearest, fact 44 farthest.
            table[f"fact {i}"] = [1.0, 0.02 * i, 0.0]
        # CE prefers the FARTHER of the survivors: 39 > 38 > ... ; 0 and 1 are gated out.
        for i in range(2, 40):
            scores[f"fact {i}"] = i / 100
        scores["fact 0"] = -0.5
        scores["fact 1"] = -0.1
        models = _table_models(table, scores)
        for i in range(45):
            _seed(store, models, "fact", f"fact {i}", table[f"fact {i}"])
        result = retrieve(store, models, "q", min_ce_score=0.0)
        [(_, sent)] = models.reranker.calls
        assert len(sent) == CANDIDATES_PER_KIND
        assert "fact 44" not in sent and "fact 40" not in sent and "fact 39" in sent
        assert [r.item.text for r in result.facts] == ["fact 39", "fact 38", "fact 37", "fact 36"]
        assert result.facts[0].ce_score == pytest.approx(0.39)
        assert result.facts[0].cosine < result.facts[3].cosine
        assert result.asks == []
        assert result.query == "q"
        assert result.item_ids == [r.item.id for r in result.facts]

    def test_gate_floor_is_inclusive_and_configurable(self, store) -> None:
        table = {"q": [1.0, 0.0, 0.0], "at floor": [1.0, 0.0, 0.0], "below": [1.0, 0.1, 0.0]}
        models = _table_models(table, {"at floor": 0.5, "below": 0.49})
        _seed(store, models, "fact", "at floor", table["at floor"])
        _seed(store, models, "fact", "below", table["below"])
        result = retrieve(store, models, "q", min_ce_score=0.5)
        assert [r.item.text for r in result.facts] == ["at floor"]

    def test_asks_are_ranked_separately_and_capped(self, store) -> None:
        table = {"q": [1.0, 0.0, 0.0]}
        scores = {}
        for i in range(5):
            table[f"ask {i}"] = [1.0, 0.01 * i, 0.0]
            scores[f"ask {i}"] = 0.9 - 0.1 * i
        table["a fact"] = [1.0, 0.0, 0.0]
        scores["a fact"] = 0.1
        models = _table_models(table, scores)
        for i in range(5):
            _seed(store, models, "ask", f"ask {i}", table[f"ask {i}"], surface="cli")
        _seed(store, models, "fact", "a fact", table["a fact"])
        result = retrieve(store, models, "q", min_ce_score=0.0)
        assert [r.item.text for r in result.asks] == ["ask 0", "ask 1", "ask 2"]
        assert len(result.asks) == TOP_ASKS
        assert [r.item.text for r in result.facts] == ["a fact"]
        assert result.item_ids == [result.facts[0].item.id] + [r.item.id for r in result.asks]
        # One reranker call per kind that had candidates.
        assert len(models.reranker.calls) == 2

    def test_top_facts_cap(self, store) -> None:
        table = {"q": [1.0, 0.0, 0.0]}
        for i in range(6):
            table[f"f{i}"] = [1.0, 0.0, 0.0]
        models = _table_models(table, {f"f{i}": 0.5 for i in range(6)})
        for i in range(6):
            _seed(store, models, "fact", f"f{i}", table[f"f{i}"])
        assert len(retrieve(store, models, "q", min_ce_score=0.0).facts) == TOP_FACTS

    def test_query_is_stripped_and_truncated_before_embedding(self, store) -> None:
        long_query = "x" * (QUERY_MAX_CHARS + 500)
        models = _table_models({long_query[:QUERY_MAX_CHARS]: [1.0, 0.0, 0.0]})
        retrieve(store, models, "  " + long_query + "  ", min_ce_score=0.0)
        assert models.embedder.calls == [[long_query[:QUERY_MAX_CHARS]]]

    def test_empty_query_and_empty_store_do_no_work(self, store) -> None:
        models = _table_models({"q": [1.0, 0.0, 0.0]})
        empty = retrieve(store, models, "   ", min_ce_score=0.0)
        assert empty == Retrieval(query="", facts=[], asks=[])
        assert models.embedder.calls == []
        nothing = retrieve(store, models, "q", min_ce_score=0.0)
        assert nothing.facts == [] and nothing.asks == []
        assert models.reranker.calls == []

    def test_archived_and_other_space_items_never_surface(self, store) -> None:
        table = {"q": [1.0, 0.0, 0.0], "gone": [1.0, 0.0, 0.0], "other": [1.0, 0.0, 0.0]}
        models = _table_models(table, {"gone": 0.9, "other": 0.9})
        gone = _seed(store, models, "fact", "gone", table["gone"])
        store.archive_item(gone.id, "forgotten")
        store.add_item(
            kind="fact",
            text="other",
            embedding=encode_embedding(_unit(table["other"])),
            embedding_model="someone/else",
            observed_at="2026-09-01",
            source="explicit:test",
        )
        assert retrieve(store, models, "q", min_ce_score=0.0).facts == []


def _retrieval(store, models, facts, asks) -> Retrieval:
    """Build a Retrieval directly from seeded items (no ranking involved)."""
    from pdp_router._memory import RetrievedItem

    return Retrieval(
        query="q",
        facts=[RetrievedItem(item=i, cosine=0.9, ce_score=0.5) for i in facts],
        asks=[RetrievedItem(item=i, cosine=0.8, ce_score=0.4) for i in asks],
    )


class TestAssembleBlock:
    TODAY = datetime(2026, 9, 7, tzinfo=UTC).date()

    def test_exact_shape(self, store) -> None:
        models = _table_models({})
        f1 = _seed(
            store,
            models,
            "fact",
            "deposited $250 into the brokerage account to get started.",
            [1, 0, 0],
            observed_at="2026-09-03",
            item_id="m_01J8AAAAAAAAAAAAAAAAAAAAAA",
        )
        f2 = _seed(
            store,
            models,
            "fact",
            "grow calendar entered; trim in progress belongs to the prior run.",
            [1, 0, 0],
            observed_at="2026-08-05",
            item_id="m_01J7BBBBBBBBBBBBBBBBBBBBBB",
        )
        a1 = _seed(
            store,
            models,
            "ask",
            "asked which starters to bench for the Week 1 sweep.",
            [1, 0, 0],
            observed_at="2026-09-05",
            surface="crush",
        )
        block = assemble_block(
            _retrieval(store, models, [f1, f2], [a1]), today=self.TODAY, max_chars=1500
        )
        assert block == (
            "[memory] Context as of 2026-09-07 (UTC). Facts the user stated in earlier "
            "conversations, most relevant first; ids are for reference only.\n"
            "- (m_01J8AAAAAAAAAAAAAAAAAAAAAA) 2026-09-03: deposited $250 into the brokerage "
            "account to get started.\n"
            "- (m_01J7BBBBBBBBBBBBBBBBBBBBBB) 2026-08-05: grow calendar entered; trim in "
            "progress belongs to the prior run.\n"
            "Earlier asks:\n"
            "- 2026-09-05 (crush): asked which starters to bench for the Week 1 sweep.\n"
            "[/memory]"
        )

    def test_date_line_alone_when_nothing_survives(self) -> None:
        block = assemble_block(
            Retrieval(query="q", facts=[], asks=[]), today=self.TODAY, max_chars=1500
        )
        assert block == "[memory] Context as of 2026-09-07 (UTC).\n[/memory]"

    def test_asks_only_and_surface_absent(self, store) -> None:
        models = _table_models({})
        a1 = _seed(store, models, "ask", "asked about x.", [1, 0, 0], observed_at="2026-09-05")
        block = assemble_block(
            _retrieval(store, models, [], [a1]), today=self.TODAY, max_chars=1500
        )
        assert block == (
            "[memory] Context as of 2026-09-07 (UTC).\n"
            "Earlier asks:\n"
            "- 2026-09-05: asked about x.\n"
            "[/memory]"
        )

    def test_cap_drops_whole_items_asks_first_then_lowest_facts(self, store) -> None:
        models = _table_models({})
        facts = [
            _seed(
                store, models, "fact", f"fact number {i} " + "y" * 40, [1, 0, 0], item_id=f"m_F{i}"
            )
            for i in range(3)
        ]
        asks = [
            _seed(store, models, "ask", f"ask number {i} " + "z" * 40, [1, 0, 0], item_id=f"m_A{i}")
            for i in range(2)
        ]
        full = assemble_block(
            _retrieval(store, models, facts, asks), today=self.TODAY, max_chars=10_000
        )
        assert "m_F2" in full and "ask number 1" in full
        # Just too small for everything: the lowest-ranked ask goes first.
        cap = len(full) - 1
        trimmed = assemble_block(
            _retrieval(store, models, facts, asks), today=self.TODAY, max_chars=cap
        )
        assert len(trimmed) <= cap
        assert "ask number 1" not in trimmed and "ask number 0" in trimmed and "m_F2" in trimmed
        # Smaller still: asks gone, then facts from the bottom of the ranking.
        tight = assemble_block(
            _retrieval(store, models, facts, asks), today=self.TODAY, max_chars=260
        )
        assert len(tight) <= 260
        assert "Earlier asks" not in tight
        assert "m_F0" in tight and "m_F2" not in tight
        assert tight.endswith("[/memory]")
        for line in tight.splitlines():
            assert not line.startswith("- ") or line.endswith("y" * 40)  # never mid-item

    def test_header_always_fits(self, store) -> None:
        models = _table_models({})
        f1 = _seed(store, models, "fact", "x", [1, 0, 0])
        block = assemble_block(_retrieval(store, models, [f1], []), today=self.TODAY, max_chars=5)
        assert block == "[memory] Context as of 2026-09-07 (UTC).\n[/memory]"


class TestFindDuplicate:
    def test_threshold_is_inclusive_and_scoped(self, store) -> None:
        table = {"new": [1.0, 0.0, 0.0]}
        models = _table_models(table)
        near = _seed(store, models, "fact", "near", [1.0, 0.1, 0.0])  # cosine ~0.995
        _seed(store, models, "fact", "far", [0.0, 1.0, 0.0])
        _seed(store, models, "ask", "same but an ask", [1.0, 0.0, 0.0])
        gone = _seed(store, models, "fact", "identical but archived", [1.0, 0.0, 0.0])
        store.archive_item(gone.id, "forgotten")
        vec = models.embedder.embed(["new"])[0]
        assert find_duplicate(store, models, "fact", vec, threshold=0.99) == near
        assert find_duplicate(store, models, "fact", vec, threshold=0.999) is None
        assert find_duplicate(store, models, "ask", vec, threshold=0.5).text == "same but an ask"

    def test_empty_store(self, store) -> None:
        models = _table_models({})
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        assert find_duplicate(store, models, "fact", vec, threshold=0.9) is None


class TestConsolidate:
    def test_near_duplicates_keep_the_used_or_older_item(self, store) -> None:
        models = _table_models({})
        older = _seed(store, models, "fact", "older", [1.0, 0.0, 0.0], now=NOW)
        newer = _seed(
            store, models, "fact", "newer twin", [1.0, 0.01, 0.0], now=NOW + timedelta(hours=1)
        )
        used_new = _seed(
            store, models, "fact", "used newer", [0.0, 1.0, 0.0], now=NOW + timedelta(hours=2)
        )
        unused_old = _seed(store, models, "fact", "unused older", [0.0, 1.0, 0.01], now=NOW)
        store.mark_exposed([used_new.id])
        lone = _seed(store, models, "fact", "lone", [0.0, 0.0, 1.0], now=NOW)
        same_vec_other_kind = _seed(store, models, "ask", "an ask", [1.0, 0.0, 0.0], now=NOW)
        counts = consolidate(
            store, dedup_sim=0.99, working_ttl_days=90, now=NOW + timedelta(days=1)
        )
        assert counts["near_duplicates_archived"] == 2
        assert store.get_item(newer.id).archive_reason == "consolidate:near_duplicate"
        assert store.get_item(older.id).archived_at is None
        assert store.get_item(unused_old.id).archive_reason == "consolidate:near_duplicate"
        assert store.get_item(used_new.id).archived_at is None
        assert store.get_item(lone.id).archived_at is None
        assert store.get_item(same_vec_other_kind.id).archived_at is None
        [event] = store.events("consolidate")
        assert event.details["near_duplicates_archived"] == 2
        assert sorted(event.details["near_duplicate_pairs"]) == sorted(
            [[newer.id, older.id], [unused_old.id, used_new.id]]
        )

    def test_chains_archive_each_duplicate_once(self, store) -> None:
        models = _table_models({})
        a = _seed(store, models, "fact", "a", [1.0, 0.0, 0.0], now=NOW)
        b = _seed(store, models, "fact", "b", [1.0, 0.005, 0.0], now=NOW + timedelta(hours=1))
        c = _seed(store, models, "fact", "c", [1.0, 0.01, 0.0], now=NOW + timedelta(hours=2))
        counts = consolidate(
            store, dedup_sim=0.99, working_ttl_days=90, now=NOW + timedelta(days=1)
        )
        assert counts["near_duplicates_archived"] == 2
        assert store.get_item(a.id).archived_at is None
        assert store.get_item(b.id).archived_at is not None
        assert store.get_item(c.id).archived_at is not None

    def test_working_ttl_archives_only_unused_working_items(self, store) -> None:
        models = _table_models({})
        stale = _seed(store, models, "fact", "stale", [1.0, 0.0, 0.0], now=NOW - timedelta(days=91))
        used = _seed(store, models, "fact", "used", [0.0, 1.0, 0.0], now=NOW - timedelta(days=91))
        store.mark_exposed([used.id])
        fresh = _seed(store, models, "fact", "fresh", [0.0, 0.0, 1.0], now=NOW - timedelta(days=89))
        counts = consolidate(store, dedup_sim=0.99, working_ttl_days=90, now=NOW)
        assert counts["working_ttl_archived"] == 1
        assert store.get_item(stale.id).archive_reason == "consolidate:working_ttl"
        assert store.get_item(used.id).archived_at is None
        assert store.get_item(fresh.id).archived_at is None

    def test_other_tiers_and_pins_and_counts(self, store) -> None:
        models = _table_models({})
        old = _seed(
            store, models, "fact", "promoted", [1.0, 0.0, 0.0], now=NOW - timedelta(days=200)
        )
        with store._connect() as conn:
            conn.execute("UPDATE memory_items SET tier = 'history' WHERE id = ?", (old.id,))
        store.set_pin("dead", "b", [], ttl_s=1, now=NOW - timedelta(days=2))
        store.set_pin("live", "b", [], ttl_s=3600, now=NOW)
        counts = consolidate(store, dedup_sim=0.99, working_ttl_days=90, now=NOW)
        assert store.get_item(old.id).archived_at is None
        assert counts == {
            "near_duplicates_archived": 0,
            "working_ttl_archived": 0,
            "pins_pruned": 1,
            "active_after": 1,
        }
        assert store.get_pin("live", now=NOW) is not None

    def test_different_embedding_spaces_are_never_compared(self, store) -> None:
        models = _table_models({})
        a = _seed(store, models, "fact", "a", [1.0, 0.0, 0.0], now=NOW)
        store.add_item(
            kind="fact",
            text="same vector, other model",
            embedding=encode_embedding(_unit([1.0, 0.0, 0.0])),
            embedding_model="someone/else",
            observed_at="2026-09-01",
            source="explicit:test",
            now=NOW + timedelta(hours=1),
        )
        counts = consolidate(
            store, dedup_sim=0.99, working_ttl_days=90, now=NOW + timedelta(days=1)
        )
        assert counts["near_duplicates_archived"] == 0
        assert store.get_item(a.id).archived_at is None


class TestShadowWriter:
    def test_writes_one_line_per_call_dated_by_utc(self, tmp_path) -> None:
        shadow = tmp_path / "memory-shadow"
        append_shadow_jsonl(
            shadow,
            conversation_key8="abcd1234",
            surface="v1",
            query="what did I say",
            block="[memory] ...",
            item_ids=["m_A"],
            now=NOW,
        )
        append_shadow_jsonl(
            shadow,
            conversation_key8="ef",
            surface="openai_v1",
            query="q2",
            block="b",
            item_ids=[],
            now=NOW,
        )
        [path] = list(shadow.iterdir())
        assert path.name == "shadow-20260907.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert lines[0] == {
            "ts": "2026-09-07T12:00:00Z",
            "conversation_key8": "abcd1234",
            "surface": "v1",
            "query": "what did I say",
            "block": "[memory] ...",
            "item_ids": ["m_A"],
        }
        assert lines[1]["item_ids"] == []

    def test_unwritable_dir_logs_and_never_raises(self, tmp_path, caplog) -> None:
        blocker = tmp_path / "file-not-dir"
        blocker.write_text("x")
        with caplog.at_level(logging.WARNING, logger="pdp_router._memory"):
            append_shadow_jsonl(
                blocker / "memory-shadow",
                conversation_key8="k",
                surface="v1",
                query="q",
                block="b",
                item_ids=[],
                now=NOW,
            )
        [record] = [r for r in caplog.records if "shadow" in r.message.lower()]
        assert record.exc_info is not None
