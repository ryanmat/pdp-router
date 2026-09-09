# Description: Tests for `python -m pdp_router.memory {warmup,consolidate,grade,stats}`.
# Description: Drives main() directly with scripted answers; no model files, no network.

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dotenv")

from pdp_router._memory import MemoryModels, MemoryStore, encode_embedding
from pdp_router.memory.__main__ import main
from tests._memory_fakes import FakeEmbedder, FakeReranker

NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)


def _db_path() -> Path:
    return Path(os.environ["PROXY_MEMORY_DB_PATH"])


def _run(argv: list[str], *, answers: list[str] | None = None, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    script = iter(answers or [])

    def ask(prompt: str) -> str:
        out.write(prompt)
        try:
            return next(script)
        except StopIteration:
            raise AssertionError(f"grade asked more than scripted: {prompt!r}") from None

    code = main(argv, ask=ask, out=out, err=err, **kwargs)
    return code, out.getvalue(), err.getvalue()


def _unit(v):
    arr = np.asarray(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _seed(store: MemoryStore, text: str, vec, **kw):
    kw.setdefault("kind", "fact")
    kw.setdefault("now", NOW)
    return store.add_item(
        text=text,
        embedding=encode_embedding(_unit(vec)),
        embedding_model="fake/embed",
        observed_at="2026-09-01",
        source="explicit:test",
        **kw,
    )


class TestStats:
    def test_absent_store_is_not_an_error(self) -> None:
        code, out, err = _run(["stats"])
        assert code == 0
        assert json.loads(out) == {"present": False, "path": str(_db_path())}
        assert err == ""
        assert not _db_path().exists()  # read-only: never creates the file

    def test_present_store_prints_the_shared_stats_shape(self) -> None:
        store = MemoryStore(_db_path())
        store.migrate()
        _seed(store, "a fact", [1, 0, 0])
        code, out, _ = _run(["stats"])
        assert code == 0
        stats = json.loads(out)
        assert stats["present"] is True
        assert stats["items"]["active"] == 1
        assert stats["schema_version"] == 1

    def test_dot_env_in_the_working_directory_is_read(self, tmp_path, monkeypatch) -> None:
        """The service unit sets PROXY_ROUTING_INBOX_DIR through .env; the CLI
        must derive the same memory.db from it or it reads the wrong store."""
        monkeypatch.delenv("PROXY_MEMORY_DB_PATH", raising=False)
        monkeypatch.delenv("PROXY_ROUTING_INBOX_DIR", raising=False)
        (tmp_path / ".env").write_text(f"PROXY_ROUTING_INBOX_DIR={tmp_path}/state/inbox\n")
        monkeypatch.chdir(tmp_path)
        code, out, _ = _run(["stats"])
        assert code == 0
        assert json.loads(out)["path"] == str(tmp_path / "state" / "memory.db")


class TestConsolidate:
    def test_runs_the_sweep_and_prints_counts(self) -> None:
        store = MemoryStore(_db_path())
        store.migrate()
        _seed(store, "older", [1, 0, 0])
        _seed(store, "newer twin", [1, 0.01, 0], now=NOW.replace(hour=13))
        code, out, err = _run(["consolidate"])
        assert code == 0
        counts = json.loads(out)
        assert counts["near_duplicates_archived"] == 1
        assert counts["active_after"] == 1
        assert err == ""
        assert len(store.events("consolidate")) == 1

    def test_absent_store_is_a_no_op(self) -> None:
        code, out, _ = _run(["consolidate"])
        assert code == 0
        assert json.loads(out) == {"present": False, "path": str(_db_path())}
        assert not _db_path().exists()

    def test_store_failure_exits_one_with_a_traceback(self) -> None:
        _db_path().mkdir(parents=True)  # a directory where the file should be
        code, out, err = _run(["consolidate"])
        assert code == 1
        assert out == ""
        assert "Traceback" in err and "unable to open database file" in err


def _write_shadow(lines: list[dict]) -> Path:
    shadow_dir = _db_path().parent / "memory-shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    path = shadow_dir / "shadow-20260907.jsonl"
    with path.open("a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def _shadow_line(
    ts: str, key8: str, block: str, item_ids: list[str], query="what did I say"
) -> dict:
    return {
        "ts": ts,
        "conversation_key8": key8,
        "surface": "v1",
        "query": query,
        "block": block,
        "item_ids": item_ids,
    }


class TestGrade:
    def test_records_one_event_per_line_with_item_grades(self) -> None:
        store = MemoryStore(_db_path())
        store.migrate()
        a = _seed(store, "fact a", [1, 0, 0])
        b = _seed(store, "fact b", [0, 1, 0])
        _write_shadow(
            [
                _shadow_line(
                    "2026-09-07T10:00:00Z", "aaaa0001", "[memory] block one", [a.id, b.id]
                ),
                _shadow_line("2026-09-07T11:00:00Z", "bbbb0002", "[memory] empty", []),
            ]
        )
        code, out, _ = _run(["grade"], answers=["h", "c", "s", "n"])
        assert code == 0
        assert "what did I say" in out and "[memory] block one" in out
        assert "graded=2 skipped=0" in out
        events = store.events("grade")
        assert len(events) == 2
        by_key = {e.details["conversation_key8"]: e.details for e in events}
        assert by_key["aaaa0001"] == {
            "shadow_ts": "2026-09-07T10:00:00Z",
            "conversation_key8": "aaaa0001",
            "block": "helpful",
            "items": {a.id: "correct", b.id: "stale"},
        }
        assert by_key["bbbb0002"]["block"] == "neutral"
        assert by_key["bbbb0002"]["items"] == {}

    def test_full_words_and_reprompt_on_junk(self) -> None:
        store = MemoryStore(_db_path())
        store.migrate()
        a = _seed(store, "fact a", [1, 0, 0])
        _write_shadow([_shadow_line("2026-09-07T10:00:00Z", "aaaa0001", "b", [a.id])])
        code, out, _ = _run(["grade"], answers=["maybe", "harmful", "", "incorrect"])
        assert code == 0
        [event] = store.events("grade")
        assert event.details["block"] == "harmful"
        assert event.details["items"] == {a.id: "incorrect"}
        assert out.count("block [") == 2  # re-asked once
        assert out.count(f"{a.id} [") == 2

    def test_already_graded_lines_are_skipped(self) -> None:
        store = MemoryStore(_db_path())
        store.migrate()
        _write_shadow([_shadow_line("2026-09-07T10:00:00Z", "aaaa0001", "b", [])])
        assert _run(["grade"], answers=["h"])[0] == 0
        code, out, _ = _run(["grade"], answers=[])
        assert code == 0
        assert "graded=0 skipped=1" in out
        assert len(store.events("grade")) == 1

    def test_quit_records_nothing_for_the_open_line(self) -> None:
        store = MemoryStore(_db_path())
        store.migrate()
        _write_shadow(
            [
                _shadow_line("2026-09-07T10:00:00Z", "aaaa0001", "b1", []),
                _shadow_line("2026-09-07T11:00:00Z", "bbbb0002", "b2", []),
            ]
        )
        code, out, _ = _run(["grade"], answers=["h", "q"])
        assert code == 0
        assert "graded=1 skipped=0 remaining=1" in out
        assert len(store.events("grade")) == 1

    def test_no_shadow_lines(self) -> None:
        code, out, _ = _run(["grade"], answers=[])
        assert code == 0
        assert "no shadow lines" in out
        assert not _db_path().exists()

    def test_explicit_shadow_dir(self, tmp_path) -> None:
        store = MemoryStore(_db_path())
        store.migrate()
        other = tmp_path / "elsewhere"
        other.mkdir()
        (other / "shadow-20260901.jsonl").write_text(
            json.dumps(_shadow_line("2026-09-01T10:00:00Z", "cccc0003", "b", [])) + "\n"
        )
        code, _, _ = _run(["grade", "--shadow-dir", str(other)], answers=["n"])
        assert code == 0
        assert store.events("grade")[0].details["conversation_key8"] == "cccc0003"


class TestWarmup:
    def test_missing_library_exits_two_with_the_install_hint(self) -> None:
        def load(config):
            raise ImportError("No module named 'fastembed'")

        code, out, err = _run(["warmup"], load=load)
        assert code == 2
        assert out == ""
        assert "uv sync --extra memory" in err

    def test_loads_and_smokes_the_models(self) -> None:
        fakes = MemoryModels(embedder=FakeEmbedder(), reranker=FakeReranker())
        code, out, err = _run(["warmup"], load=lambda config: fakes)
        assert code == 0
        report = json.loads(out)
        assert report["model_dir"] == os.environ["PROXY_MEMORY_MODEL_DIR"]
        assert report["embed_model"] == "fake/embed"
        assert report["dim"] == 8
        assert report["rerank_model"] == "fake/rerank"
        assert report["embed_ms"] >= 0 and report["rerank_ms"] >= 0
        assert fakes.embedder.calls and fakes.reranker.calls
        assert err == ""

    def test_load_failure_exits_one_with_a_traceback(self) -> None:
        def load(config):
            raise ValueError("Could not load model x from any source.")

        code, _, err = _run(["warmup"], load=load)
        assert code == 1
        assert "Traceback" in err and "Could not load model" in err


class TestUsage:
    def test_unknown_or_missing_command_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            _run(["frobnicate"])
        assert exc.value.code == 2
        with pytest.raises(SystemExit) as exc:
            _run([])
        assert exc.value.code == 2
