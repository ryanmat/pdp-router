# Description: Operator CLI for the proxy memory store: warmup (download and smoke the models),
# Description: consolidate (nightly sweep), grade (shadow-log grading), stats (the tick's read).

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from pdp_router.__main__ import _load_env_file
from pdp_router._memory import MemoryModels, MemoryRuntime, MemoryStore, consolidate
from pdp_router._proxy_config import ProxyConfig

# Single-keystroke grading keeps the eval gate cheap: a graded line costs
# one key for the block and one per item. Full words are accepted too.
BLOCK_GRADES = {"h": "helpful", "n": "neutral", "f": "harmful"}
ITEM_GRADES = {"c": "correct", "i": "incorrect", "s": "stale"}
_QUIT = "q"


def _default_load(config: ProxyConfig) -> MemoryModels:
    # The one caller that downloads: warmup runs with network before the
    # service (which loads offline) is restarted.
    return MemoryRuntime(config).load_models_blocking(allow_download=True)


def _absent(config: ProxyConfig, out: TextIO) -> int:
    out.write(json.dumps({"present": False, "path": str(config.memory_db_path)}) + "\n")
    return 0


def cmd_stats(config: ProxyConfig, out: TextIO) -> int:
    """The read a health check shares: counts, last consolidation, 24 h events.
    Read-only -- an absent store is reported, never created."""
    if not config.memory_db_path.exists():
        return _absent(config, out)
    stats = MemoryStore(config.memory_db_path).stats()
    out.write(json.dumps({"present": True, **stats}, indent=2) + "\n")
    return 0


def cmd_consolidate(config: ProxyConfig, out: TextIO) -> int:
    """The nightly sweep; prints the counts it recorded in the consolidate event."""
    if not config.memory_db_path.exists():
        return _absent(config, out)
    store = MemoryStore(config.memory_db_path)
    store.migrate()
    counts = consolidate(
        store,
        dedup_sim=config.memory_dedup_sim,
        working_ttl_days=config.memory_working_ttl_days,
    )
    out.write(json.dumps(counts) + "\n")
    return 0


def _answer(ask: Callable[[str], str], prompt: str, choices: dict[str, str]) -> str | None:
    """The full grade word, or None on quit; junk re-asks."""
    while True:
        raw = ask(prompt).strip().lower()
        if raw == _QUIT:
            return None
        if raw in choices:
            return choices[raw]
        if raw in choices.values():
            return raw


def _read_shadow_lines(shadow_dir: Path) -> list[dict[str, Any]]:
    if not shadow_dir.exists():
        return []
    lines: list[dict[str, Any]] = []
    for path in sorted(shadow_dir.glob("shadow-*.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw:
                lines.append(json.loads(raw))
    return lines


def cmd_grade(
    config: ProxyConfig, out: TextIO, ask: Callable[[str], str], shadow_dir: Path | None
) -> int:
    """Walk the shadow log oldest first, show each query with the block that
    would have been injected, and record one grade event per line: the
    block's verdict plus a verdict per item. Lines already graded are
    skipped, so grading is incremental; `q` stops before the open line is
    recorded."""
    shadow_dir = shadow_dir or config.memory_shadow_dir
    lines = _read_shadow_lines(shadow_dir)
    if not lines:
        out.write(f"no shadow lines under {shadow_dir}\n")
        return 0
    store = MemoryStore(config.memory_db_path)
    store.migrate()
    graded = {
        (e.details.get("shadow_ts"), e.details.get("conversation_key8"))
        for e in store.events("grade", limit=100_000)
    }
    n_graded = n_skipped = 0
    for idx, line in enumerate(lines):
        key = (line.get("ts"), line.get("conversation_key8"))
        if key in graded:
            n_skipped += 1
            continue
        out.write(
            f"\n--- {line.get('ts')} key={line.get('conversation_key8')} "
            f"surface={line.get('surface')} ---\n"
            f"Q: {line.get('query')}\n{line.get('block')}\n"
        )
        block = _answer(ask, "block [h]elpful/[n]eutral/harm[f]ul/[q]uit: ", BLOCK_GRADES)
        items: dict[str, str] = {}
        quit_now = block is None
        for item_id in [] if quit_now else line.get("item_ids", []):
            grade = _answer(ask, f"{item_id} [c]orrect/[i]ncorrect/[s]tale/[q]uit: ", ITEM_GRADES)
            if grade is None:
                quit_now = True
                break
            items[item_id] = grade
        if quit_now:
            remaining = sum(
                1
                for later in lines[idx:]
                if (later.get("ts"), later.get("conversation_key8")) not in graded
            )
            out.write(f"graded={n_graded} skipped={n_skipped} remaining={remaining}\n")
            return 0
        store.add_event(
            "grade",
            {
                "shadow_ts": line.get("ts"),
                "conversation_key8": line.get("conversation_key8"),
                "block": block,
                "items": items,
            },
        )
        graded.add(key)
        n_graded += 1
    out.write(f"graded={n_graded} skipped={n_skipped} remaining=0\n")
    return 0


def cmd_warmup(
    config: ProxyConfig, out: TextIO, err: TextIO, load: Callable[[ProxyConfig], MemoryModels]
) -> int:
    """Download (if needed) and load both models into the configured model
    dir, then embed and rerank once so a broken cache fails here, not in
    the service."""
    try:
        models = load(config)
    except ImportError as exc:
        err.write(
            f"{exc}\nThe memory extra is not installed. Install it with:\n"
            "  uv sync --extra memory\nthen re-run warmup.\n"
        )
        return 2
    t0 = time.perf_counter()
    models.embedder.embed(["warmup sentence one", "warmup sentence two"])
    t1 = time.perf_counter()
    models.reranker.score("warmup query", ["a relevant document", "an unrelated one"])
    t2 = time.perf_counter()
    out.write(
        json.dumps(
            {
                "model_dir": str(config.memory_model_dir),
                "embed_model": models.embedder.model_name,
                "dim": models.embedder.dim,
                "rerank_model": models.reranker.model_name,
                "embed_ms": round((t1 - t0) * 1000, 1),
                "rerank_ms": round((t2 - t1) * 1000, 1),
            }
        )
        + "\n"
    )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    ask: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
    load: Callable[[ProxyConfig], MemoryModels] = _default_load,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pdp_router.memory",
        description="Operator commands for the proxy memory store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("warmup", help="download and smoke-test the embedding and rerank models")
    sub.add_parser("consolidate", help="run the nightly dedup / TTL / pin sweep")
    grade = sub.add_parser("grade", help="grade shadow-log lines interactively")
    grade.add_argument("--shadow-dir", type=Path, default=None, help="override the shadow dir")
    sub.add_parser("stats", help="print store statistics as JSON")
    args = parser.parse_args(argv)

    # Same .env discipline as the proxy entry point: the working directory's
    # .env supplies the inbox path memory.db is derived from.
    _load_env_file()
    config = ProxyConfig()
    try:
        if args.command == "stats":
            return cmd_stats(config, out)
        if args.command == "consolidate":
            return cmd_consolidate(config, out)
        if args.command == "grade":
            return cmd_grade(config, out, ask, args.shadow_dir)
        return cmd_warmup(config, out, err, load)
    except Exception:
        err.write(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
