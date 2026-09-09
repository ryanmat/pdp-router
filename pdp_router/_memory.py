# Description: Persistent user-memory store for the proxy: SQLite memory.db holding dated
# Description: atomic facts, outcomes, per-conversation pins and events; float32 embedding blobs.

from __future__ import annotations

import json
import logging
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pdp_router._proxy_config import ProxyConfig

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
KINDS = ("fact", "ask")
TIERS = ("working", "history", "patterns")
SIGNALS = ("worked", "failed", "partial", "unknown", "confirmed", "forgotten")
ORIGINS = ("sidecar", "implicit", "explicit")

# (score delta, success credit) per outcome signal. worked / failed / partial /
# unknown are the sidecar and implicit-feedback vocabulary; confirmed and
# forgotten are the explicit endpoints and carry the worked / failed weights.
# Scores move items between tiers and archive them; they never rank retrieval.
OUTCOME_DELTAS: dict[str, tuple[float, float]] = {
    "worked": (0.20, 1.0),
    "failed": (-0.30, 0.0),
    "partial": (0.05, 0.5),
    "unknown": (-0.05, 0.25),
    "confirmed": (0.20, 1.0),
    "forgotten": (-0.30, 0.0),
}

# A failed background model load is retried no sooner than this, so a model
# dir filled by warmup AFTER the service started is picked up within a minute
# while a persistent failure (no extra installed) never spins.
LOAD_RETRY_S = 60.0

# Retrieval shape: the query is truncated like the classifier's, each kind
# takes its nearest candidates by cosine to the cross-encoder, and the block
# keeps the top few by cross-encoder order. Scores never enter the ranking.
QUERY_MAX_CHARS = 2000
CANDIDATES_PER_KIND = 40
TOP_FACTS = 4
TOP_ASKS = 3

_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class UnknownMemoryItemError(KeyError):
    """Raised when an operation names an item id the store does not hold."""


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    return now if now.tzinfo is not None else now.replace(tzinfo=UTC)


def _iso(now: datetime | None) -> str:
    """Second-precision UTC timestamp; one format everywhere so strings compare."""
    return _now(now).astimezone(UTC).strftime(_TS_FORMAT)


def _crockford(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def new_item_id(now_ms: int | None = None) -> str:
    """Return a new item id: `m_` plus a 26-character ULID.

    Ten characters of fixed-width Crockford base32 millisecond time followed
    by sixteen of randomness, so ids sort by creation time and never collide
    within a millisecond. The alphabet is ASCII-ascending, which is what makes
    plain string comparison equal time order.
    """
    ms = int(time.time() * 1000) if now_ms is None else now_ms
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    return "m_" + _crockford(ms, 10) + _crockford(rand, 16)


def _np():
    # Lazy: the store, the stats CLI and a tick check must work without the
    # numeric stack; only the embedding codec and the retrieval math need it.
    import numpy

    return numpy


def encode_embedding(vec: Sequence[float] | Any) -> bytes:
    """float32 little-endian bytes, the on-disk shape of every embedding."""
    return _np().asarray(vec, dtype="<f4").tobytes()


def decode_embedding(blob: bytes) -> Any:
    """Inverse of encode_embedding; a read-only float32 array."""
    if len(blob) % 4:
        raise ValueError(f"embedding blob length {len(blob)} is not a multiple of 4")
    return _np().frombuffer(blob, dtype="<f4")


@dataclass(frozen=True)
class MemoryItem:
    id: str
    kind: str
    text: str
    embedding_model: str | None
    created_at: str
    observed_at: str
    source: str
    surface: str | None
    conversation_key8: str | None
    chat_request_id: str | None
    tier: str
    score: float
    uses: int
    success: float
    last_outcome: str | None
    archived_at: str | None
    archive_reason: str | None


@dataclass(frozen=True)
class MemoryOutcome:
    id: int
    item_id: str
    ts: str
    signal: str
    origin: str
    delta: float | None
    conversation_key8: str | None
    chat_request_id: str | None
    note: str | None


@dataclass(frozen=True)
class MemoryPin:
    conversation_key: str
    block: str
    item_ids: list[str]
    created_at: str
    expires_at: str


@dataclass(frozen=True)
class MemoryEvent:
    ts: str
    event: str
    details: dict[str, Any]


_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('fact', 'ask')),
    text TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT,
    created_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL,
    surface TEXT,
    conversation_key8 TEXT,
    chat_request_id TEXT,
    tier TEXT NOT NULL DEFAULT 'working'
        CHECK (tier IN ('working', 'history', 'patterns')),
    score REAL NOT NULL DEFAULT 0.5,
    uses INTEGER NOT NULL DEFAULT 0,
    success REAL NOT NULL DEFAULT 0.0,
    last_outcome TEXT,
    archived_at TEXT,
    archive_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_items_active ON memory_items (kind, archived_at);
CREATE TABLE IF NOT EXISTS memory_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES memory_items (id),
    ts TEXT NOT NULL,
    signal TEXT NOT NULL,
    origin TEXT NOT NULL,
    delta REAL,
    conversation_key8 TEXT,
    chat_request_id TEXT,
    note TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_outcomes_item ON memory_outcomes (item_id, ts);
CREATE TABLE IF NOT EXISTS memory_pins (
    conversation_key TEXT PRIMARY KEY,
    block TEXT NOT NULL,
    item_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    details_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_events_event_ts ON memory_events (event, ts);
"""

# Additive only: a version's script may create tables, columns and indexes,
# never rewrite or drop. schema_version records each applied version.
_MIGRATIONS: dict[int, str] = {1: _SCHEMA_V1}

_ITEM_COLUMNS = (
    "id, kind, text, embedding_model, created_at, observed_at, source, surface, "
    "conversation_key8, chat_request_id, tier, score, uses, success, last_outcome, "
    "archived_at, archive_reason"
)


def _item_from_row(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        kind=row["kind"],
        text=row["text"],
        embedding_model=row["embedding_model"],
        created_at=row["created_at"],
        observed_at=row["observed_at"],
        source=row["source"],
        surface=row["surface"],
        conversation_key8=row["conversation_key8"],
        chat_request_id=row["chat_request_id"],
        tier=row["tier"],
        score=row["score"],
        uses=row["uses"],
        success=row["success"],
        last_outcome=row["last_outcome"],
        archived_at=row["archived_at"],
        archive_reason=row["archive_reason"],
    )


class MemoryStore:
    """memory.db access. One short-lived connection per call.

    Opening a connection per operation (WAL, busy timeout) keeps the store
    safe to call from asyncio.to_thread beside request-path writes and lets a
    second process (the nightly consolidation) share the file: no connection
    is ever held across calls, so no lock is either. SQLite connect costs
    well under a millisecond, and every operation here is a handful of rows.
    """

    def __init__(self, path: Path | str, *, busy_timeout_ms: int = 5000) -> None:
        self._path = Path(path)
        self._busy_timeout_ms = busy_timeout_ms

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._path), timeout=self._busy_timeout_ms / 1000)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        """Create the file and apply every schema version not yet recorded."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            # WAL is a property of the file, set once here; readers and the
            # nightly writer inherit it.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = row[0] or 0
            for version in sorted(_MIGRATIONS):
                if version > current:
                    conn.executescript(_MIGRATIONS[version])
                    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    log.info("memory.db migrated to schema version %d at %s", version, self._path)

    # -- items --

    def add_item(
        self,
        *,
        kind: str,
        text: str,
        embedding: bytes | None,
        embedding_model: str | None,
        observed_at: str,
        source: str,
        surface: str | None = None,
        conversation_key8: str | None = None,
        chat_request_id: str | None = None,
        score: float = 0.5,
        item_id: str | None = None,
        now: datetime | None = None,
    ) -> MemoryItem:
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind!r}; expected one of {KINDS}")
        text = text.strip()
        if not text:
            raise ValueError("memory item text is empty")
        try:
            date.fromisoformat(observed_at[:10])
        except ValueError as e:
            raise ValueError(f"observed_at must start with YYYY-MM-DD, got {observed_at!r}") from e
        item_id = item_id or new_item_id()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_items (id, kind, text, embedding, embedding_model, "
                "created_at, observed_at, source, surface, conversation_key8, "
                "chat_request_id, score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    kind,
                    text,
                    embedding,
                    embedding_model,
                    _iso(now),
                    observed_at,
                    source,
                    surface,
                    conversation_key8,
                    chat_request_id,
                    score,
                ),
            )
            row = conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
        return _item_from_row(row)

    def get_item(self, item_id: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
        return _item_from_row(row) if row is not None else None

    def get_items(self, item_ids: Sequence[str]) -> dict[str, MemoryItem]:
        """Items by id (unknown ids simply absent), one query."""
        if not item_ids:
            return {}
        placeholders = ", ".join("?" for _ in item_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM memory_items WHERE id IN ({placeholders})",
                list(item_ids),
            ).fetchall()
        return {r["id"]: _item_from_row(r) for r in rows}

    def list_active(self, kind: str | None = None, *, limit: int | None = None) -> list[MemoryItem]:
        """Active (unarchived) items, newest first; optionally one kind."""
        sql = f"SELECT {_ITEM_COLUMNS} FROM memory_items WHERE archived_at IS NULL"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_item_from_row(r) for r in rows]

    def active_embeddings(
        self, kind: str, *, embedding_model: str | None = None
    ) -> list[tuple[str, bytes]]:
        """(id, blob) for every active item of the kind that carries a vector.

        Pass embedding_model to stay inside one embedding space: vectors from
        different models are not comparable, so retrieval names the model it
        is querying with and items embedded under another are invisible to it
        until re-embedded.
        """
        sql = (
            "SELECT id, embedding FROM memory_items "
            "WHERE archived_at IS NULL AND kind = ? AND embedding IS NOT NULL"
        )
        params: list[Any] = [kind]
        if embedding_model is not None:
            sql += " AND embedding_model = ?"
            params.append(embedding_model)
        sql += " ORDER BY created_at, id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [(r["id"], bytes(r["embedding"])) for r in rows]

    def archive_item(self, item_id: str, reason: str, *, now: datetime | None = None) -> bool:
        """Soft-archive an active item. False when unknown or already archived,
        so the first reason recorded is the one that stands."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE memory_items SET archived_at = ?, archive_reason = ? "
                "WHERE id = ? AND archived_at IS NULL",
                (_iso(now), reason, item_id),
            )
        return cur.rowcount == 1

    def mark_exposed(self, item_ids: Sequence[str]) -> None:
        """uses += 1 for items a model actually received in its prompt."""
        if not item_ids:
            return
        placeholders = ", ".join("?" for _ in item_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE memory_items SET uses = uses + 1 WHERE id IN ({placeholders})",
                list(item_ids),
            )

    # -- outcomes --

    def add_outcome(
        self,
        *,
        item_id: str,
        signal: str,
        origin: str,
        delta: float | None = None,
        conversation_key8: str | None = None,
        chat_request_id: str | None = None,
        note: str | None = None,
        now: datetime | None = None,
    ) -> MemoryItem:
        """Record an outcome and apply it: score by the signal's delta (or the
        explicit one), clamped to [0, 1]; success by the signal's credit."""
        if signal not in SIGNALS:
            raise ValueError(f"unknown outcome signal {signal!r}; expected one of {SIGNALS}")
        if origin not in ORIGINS:
            raise ValueError(f"unknown outcome origin {origin!r}; expected one of {ORIGINS}")
        score_delta, success_credit = OUTCOME_DELTAS[signal]
        if delta is not None:
            score_delta = delta
        with self._connect() as conn:
            row = conn.execute(
                "SELECT score, success FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                raise UnknownMemoryItemError(item_id)
            new_score = min(1.0, max(0.0, row["score"] + score_delta))
            conn.execute(
                "INSERT INTO memory_outcomes (item_id, ts, signal, origin, delta, "
                "conversation_key8, chat_request_id, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    _iso(now),
                    signal,
                    origin,
                    score_delta,
                    conversation_key8,
                    chat_request_id,
                    note,
                ),
            )
            conn.execute(
                "UPDATE memory_items SET score = ?, success = success + ?, last_outcome = ? "
                "WHERE id = ?",
                (new_score, success_credit, signal, item_id),
            )
            updated = conn.execute(
                f"SELECT {_ITEM_COLUMNS} FROM memory_items WHERE id = ?", (item_id,)
            ).fetchone()
        return _item_from_row(updated)

    def outcomes(self, item_id: str) -> list[MemoryOutcome]:
        """Outcome rows for one item, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, item_id, ts, signal, origin, delta, conversation_key8, "
                "chat_request_id, note FROM memory_outcomes WHERE item_id = ? "
                "ORDER BY ts DESC, id DESC",
                (item_id,),
            ).fetchall()
        return [
            MemoryOutcome(
                id=r["id"],
                item_id=r["item_id"],
                ts=r["ts"],
                signal=r["signal"],
                origin=r["origin"],
                delta=r["delta"],
                conversation_key8=r["conversation_key8"],
                chat_request_id=r["chat_request_id"],
                note=r["note"],
            )
            for r in rows
        ]

    # -- pins --

    def set_pin(
        self,
        conversation_key: str,
        block: str,
        item_ids: Sequence[str],
        *,
        ttl_s: int,
        now: datetime | None = None,
    ) -> MemoryPin:
        """Persist the block a conversation was resolved to, replacing any prior pin."""
        created = _now(now)
        pin = MemoryPin(
            conversation_key=conversation_key,
            block=block,
            item_ids=list(item_ids),
            created_at=_iso(created),
            expires_at=_iso(created + timedelta(seconds=ttl_s)),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory_pins (conversation_key, block, item_ids_json, "
                "created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                (
                    pin.conversation_key,
                    pin.block,
                    json.dumps(pin.item_ids),
                    pin.created_at,
                    pin.expires_at,
                ),
            )
        return pin

    def get_pin(self, conversation_key: str, *, now: datetime | None = None) -> MemoryPin | None:
        """The live pin for a conversation; an expired or absent pin reads as None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT conversation_key, block, item_ids_json, created_at, expires_at "
                "FROM memory_pins WHERE conversation_key = ?",
                (conversation_key,),
            ).fetchone()
        if row is None or row["expires_at"] <= _iso(now):
            return None
        return MemoryPin(
            conversation_key=row["conversation_key"],
            block=row["block"],
            item_ids=json.loads(row["item_ids_json"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    def delete_expired_pins(self, *, now: datetime | None = None) -> int:
        """Drop pins past their TTL; returns how many went."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memory_pins WHERE expires_at <= ?", (_iso(now),))
        return cur.rowcount

    # -- events --

    def add_event(
        self, event: str, details: dict[str, Any], *, now: datetime | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_events (ts, event, details_json) VALUES (?, ?, ?)",
                (_iso(now), event, json.dumps(details)),
            )

    def events(
        self,
        event: str | None = None,
        *,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Events newest first, optionally one name and/or at or after `since`."""
        sql = "SELECT ts, event, details_json FROM memory_events WHERE 1 = 1"
        params: list[Any] = []
        if event is not None:
            sql += " AND event = ?"
            params.append(event)
        if since is not None:
            sql += " AND ts >= ?"
            params.append(_iso(since))
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            MemoryEvent(ts=r["ts"], event=r["event"], details=json.loads(r["details_json"]))
            for r in rows
        ]

    # -- stats --

    def stats(self, *, now: datetime | None = None) -> dict[str, Any]:
        """The read a health check and the stats CLI share: counts, the last
        consolidation, and the last day's event mix."""
        now_iso = _iso(now)
        day_ago = _iso(_now(now) - timedelta(hours=24))
        with self._connect() as conn:
            version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE archived_at IS NULL"
            ).fetchone()[0]
            archived = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE archived_at IS NOT NULL"
            ).fetchone()[0]
            by_kind_tier: dict[str, dict[str, int]] = {}
            for r in conn.execute(
                "SELECT kind, tier, COUNT(*) AS n FROM memory_items "
                "WHERE archived_at IS NULL GROUP BY kind, tier"
            ):
                by_kind_tier.setdefault(r["kind"], {})[r["tier"]] = r["n"]
            outcomes = conn.execute("SELECT COUNT(*) FROM memory_outcomes").fetchone()[0]
            pins_active = conn.execute(
                "SELECT COUNT(*) FROM memory_pins WHERE expires_at > ?", (now_iso,)
            ).fetchone()[0]
            pins_expired = conn.execute(
                "SELECT COUNT(*) FROM memory_pins WHERE expires_at <= ?", (now_iso,)
            ).fetchone()[0]
            last_consolidate = conn.execute(
                "SELECT MAX(ts) FROM memory_events WHERE event = 'consolidate'"
            ).fetchone()[0]
            events_24h = {
                r["event"]: r["n"]
                for r in conn.execute(
                    "SELECT event, COUNT(*) AS n FROM memory_events WHERE ts >= ? "
                    "GROUP BY event",
                    (day_ago,),
                )
            }
        return {
            "path": str(self._path),
            "schema_version": version,
            "items": {"active": active, "archived": archived, "by_kind_tier": by_kind_tier},
            "outcomes": outcomes,
            "pins": {"active": pins_active, "expired": pins_expired},
            "last_consolidate_ts": last_consolidate,
            "events_24h": events_24h,
        }


# -- embedding and rerank models --


@runtime_checkable
class Embedder(Protocol):
    """Text -> unit-normalized float32 vectors of width `dim`."""

    model_name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[Any]: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder relevance of each document to the query; higher is better."""

    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True)
class MemoryModels:
    embedder: Embedder
    reranker: Reranker


class FastembedEmbedder:
    """ONNX embedder on CPU behind the Embedder protocol.

    The library is imported here, not at module import, so a proxy with the
    memory flags off never loads onnxruntime. cache_dir is always passed:
    the library's own fallback (an env var or a tmp dir) is not something a
    service may depend on. With allow_download=False a missing model raises
    at once instead of pulling files in a request-adjacent thread; the
    warmup CLI is the one caller that downloads. Output is normalized here
    regardless of what the model does, so cosine is a dot product downstream.
    """

    def __init__(self, model_name: str, cache_dir: Path, *, allow_download: bool) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            local_files_only=not allow_download,
        )
        self.dim = int(self._model.embedding_size)

    def embed(self, texts: Sequence[str]) -> list[Any]:
        texts = list(texts)
        if not texts:
            return []
        np = _np()
        out = []
        for vec in self._model.embed(texts):
            v = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            out.append(v / norm if norm else v)
        return out


class FastembedReranker:
    """ONNX cross-encoder on CPU behind the Reranker protocol (same lazy-import
    and cache_dir discipline as FastembedEmbedder)."""

    def __init__(self, model_name: str, cache_dir: Path, *, allow_download: bool) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir),
            local_files_only=not allow_download,
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        if not documents:
            return []
        return [float(s) for s in self._model.rerank(query, documents)]


def load_models(config: ProxyConfig, *, allow_download: bool = False) -> MemoryModels:
    """Build both models from the configured names and model dir. Raises on
    a missing library, missing files (when downloads are forbidden) or a
    broken cache; the runtime is what turns that into off-for-this-request."""
    model_dir = Path(config.memory_model_dir)
    if allow_download:
        model_dir.mkdir(parents=True, exist_ok=True)
    embedder = FastembedEmbedder(
        config.memory_embed_model, model_dir, allow_download=allow_download
    )
    reranker = FastembedReranker(
        config.memory_rerank_model, model_dir, allow_download=allow_download
    )
    return MemoryModels(embedder=embedder, reranker=reranker)


class MemoryRuntime:
    """Process-wide holder for the store and the models.

    The store opens (and migrates) lazily on first use and raises to the
    caller when unusable: the callers are the route guard and the shadow
    hook, and failing closed is their job. The models load in a background
    thread that never raises: a failure is recorded on load_error, logged
    ONCE per distinct cause with a traceback, and retried no sooner than
    LOAD_RETRY_S. Until models_ready is True every consumer serves without
    memory.
    """

    def __init__(
        self,
        config: ProxyConfig,
        *,
        loader: Callable[..., MemoryModels] = load_models,
    ) -> None:
        self._config = config
        self._loader = loader
        self._lock = threading.Lock()
        self._store: MemoryStore | None = None
        self._models: MemoryModels | None = None
        self._load_error: str | None = None
        self._thread: threading.Thread | None = None
        self._last_attempt: float | None = None
        self._logged_causes: set[str] = set()

    @property
    def config(self) -> ProxyConfig:
        return self._config

    @property
    def store(self) -> MemoryStore:
        with self._lock:
            if self._store is None:
                store = MemoryStore(self._config.memory_db_path)
                store.migrate()
                self._store = store
            return self._store

    @property
    def models(self) -> MemoryModels | None:
        return self._models

    @property
    def models_ready(self) -> bool:
        return self._models is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def start_model_load(self) -> None:
        """Kick off a background load unless one is done, running, or failed
        inside the retry cooldown. Cheap and idempotent: every request path
        may call it."""
        with self._lock:
            if self._models is not None:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            now = time.monotonic()
            if self._last_attempt is not None and now - self._last_attempt < LOAD_RETRY_S:
                return
            self._last_attempt = now
            self._thread = threading.Thread(
                target=self._load_in_background, name="memory-model-load", daemon=True
            )
            self._thread.start()

    def wait_for_load(self, timeout: float | None = None) -> None:
        """Block until the current background load (if any) finishes."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def load_models_blocking(self, *, allow_download: bool = False) -> MemoryModels:
        """Load in the calling thread; raises on failure (the CLI's path)."""
        return self._attempt(allow_download=allow_download)

    def _attempt(self, *, allow_download: bool) -> MemoryModels:
        try:
            models = self._loader(self._config, allow_download=allow_download)
        except Exception as exc:
            with self._lock:
                self._load_error = f"{type(exc).__name__}: {exc}"
            raise
        with self._lock:
            self._models = models
            self._load_error = None
        log.info(
            "Memory models loaded: embed=%s (dim %d) rerank=%s",
            models.embedder.model_name,
            models.embedder.dim,
            models.reranker.model_name,
        )
        return models

    def _load_in_background(self) -> None:
        try:
            self._attempt(allow_download=False)
        except Exception as exc:
            cause = f"{type(exc).__name__}: {exc}"
            with self._lock:
                first = cause not in self._logged_causes
                self._logged_causes.add(cause)
            if first:
                log.warning(
                    "Memory models failed to load (memory stays off until they do): %s",
                    cause,
                    exc_info=True,
                )


# -- retrieval, block assembly, dedup, consolidation --


@dataclass(frozen=True)
class RetrievedItem:
    item: MemoryItem
    cosine: float
    ce_score: float


@dataclass(frozen=True)
class Retrieval:
    """What one query surfaced, per kind, in final (cross-encoder) order."""

    query: str
    facts: list[RetrievedItem]
    asks: list[RetrievedItem]

    @property
    def item_ids(self) -> list[str]:
        return [r.item.id for r in self.facts] + [r.item.id for r in self.asks]


def cosine_top_k(
    query_vec: Any, rows: Sequence[tuple[str, bytes]], k: int
) -> list[tuple[str, float]]:
    """The k rows nearest the query by cosine, best first, as (id, cosine).

    Vectors are unit-normalized at embed time, so cosine is a dot product.
    A row whose width differs from the query's comes from another embedding
    space (or is corrupt); it is skipped and counted, never compared and never
    fatal, so one bad row cannot take retrieval down.
    """
    if k <= 0 or not rows:
        return []
    np = _np()
    query = np.asarray(query_vec, dtype=np.float32)
    ids: list[str] = []
    vectors: list[Any] = []
    skipped = 0
    for item_id, blob in rows:
        vec = decode_embedding(blob)
        if vec.shape != query.shape:
            skipped += 1
            continue
        ids.append(item_id)
        vectors.append(vec)
    if skipped:
        log.warning(
            "Skipped %d embedding(s) whose width differs from the query's (%d)",
            skipped,
            int(query.shape[0]),
        )
    if not vectors:
        return []
    sims = np.stack(vectors) @ query
    order = np.argsort(-sims, kind="stable")[:k]
    return [(ids[i], float(sims[i])) for i in order]


def _retrieve_kind(
    store: MemoryStore,
    models: MemoryModels,
    query: str,
    query_vec: Any,
    kind: str,
    *,
    min_ce_score: float,
    candidates: int,
    top_n: int,
) -> list[RetrievedItem]:
    rows = store.active_embeddings(kind, embedding_model=models.embedder.model_name)
    ranked = cosine_top_k(query_vec, rows, candidates)
    if not ranked:
        return []
    items = store.get_items([item_id for item_id, _ in ranked])
    ordered = [(items[item_id], cosine) for item_id, cosine in ranked if item_id in items]
    scores = models.reranker.score(query, [item.text for item, _ in ordered])
    kept = [
        RetrievedItem(item=item, cosine=cosine, ce_score=score)
        for (item, cosine), score in zip(ordered, scores, strict=True)
        if score >= min_ce_score
    ]
    kept.sort(key=lambda r: (-r.ce_score, -r.cosine))
    return kept[:top_n]


def retrieve(
    store: MemoryStore,
    models: MemoryModels,
    query: str,
    *,
    min_ce_score: float,
    candidates: int = CANDIDATES_PER_KIND,
    top_facts: int = TOP_FACTS,
    top_asks: int = TOP_ASKS,
) -> Retrieval:
    """Cosine candidates, then cross-encoder order, then the quality gate,
    then the per-kind cut. Pure read: no uses change, no event; the caller
    decides what exposure means on its path."""
    query = query.strip()[:QUERY_MAX_CHARS]
    if not query:
        return Retrieval(query="", facts=[], asks=[])
    query_vec = models.embedder.embed([query])[0]
    facts = _retrieve_kind(
        store,
        models,
        query,
        query_vec,
        "fact",
        min_ce_score=min_ce_score,
        candidates=candidates,
        top_n=top_facts,
    )
    asks = _retrieve_kind(
        store,
        models,
        query,
        query_vec,
        "ask",
        min_ce_score=min_ce_score,
        candidates=candidates,
        top_n=top_asks,
    )
    return Retrieval(query=query, facts=facts, asks=asks)


_BLOCK_FACTS_INTRO = (
    " Facts the user stated in earlier conversations, most relevant first; "
    "ids are for reference only."
)


def _fact_line(item: MemoryItem) -> str:
    return f"- ({item.id}) {item.observed_at[:10]}: {item.text}"


def _ask_line(item: MemoryItem) -> str:
    surface = f" ({item.surface})" if item.surface else ""
    return f"- {item.observed_at[:10]}{surface}: {item.text}"


def _render_block(date_line: str, facts: list[str], asks: list[str]) -> str:
    lines = [date_line + (_BLOCK_FACTS_INTRO if facts else "")]
    lines.extend(facts)
    if asks:
        lines.append("Earlier asks:")
        lines.extend(asks)
    lines.append("[/memory]")
    return "\n".join(lines)


def assemble_block(retrieval: Retrieval, *, today: date, max_chars: int) -> str:
    """Render the block: a date line always, then facts, then asks.

    Over the cap, whole items are dropped from the bottom of the ranking --
    asks first (the cheaper context), then facts -- never a truncated line.
    The date line is never dropped, so a tiny cap still yields a valid block.
    The delimiters are framing for the model only; no code strips on them.
    """
    date_line = f"[memory] Context as of {today.isoformat()} (UTC)."
    facts = [_fact_line(r.item) for r in retrieval.facts]
    asks = [_ask_line(r.item) for r in retrieval.asks]
    n_facts, n_asks = len(facts), len(asks)
    while True:
        block = _render_block(date_line, facts[:n_facts], asks[:n_asks])
        if len(block) <= max_chars or (n_facts == 0 and n_asks == 0):
            return block
        if n_asks > 0:
            n_asks -= 1
        else:
            n_facts -= 1


def find_duplicate(
    store: MemoryStore, models: MemoryModels, kind: str, vec: Any, *, threshold: float
) -> MemoryItem | None:
    """The nearest active same-kind item in the current embedding space when
    its cosine reaches the threshold, else None."""
    rows = store.active_embeddings(kind, embedding_model=models.embedder.model_name)
    top = cosine_top_k(vec, rows, 1)
    if top and top[0][1] >= threshold:
        return store.get_item(top[0][0])
    return None


def _near_duplicate_pairs(
    rows: Sequence[tuple[str, bytes]], threshold: float
) -> list[tuple[float, str, str]]:
    """Every pair of rows at or above the cosine threshold, most similar first."""
    if len(rows) < 2:
        return []
    np = _np()
    ids: list[str] = []
    vectors: list[Any] = []
    for item_id, blob in rows:
        vec = decode_embedding(blob)
        if vectors and vec.shape != vectors[0].shape:
            continue
        ids.append(item_id)
        vectors.append(vec)
    if len(vectors) < 2:
        return []
    mat = np.stack(vectors)
    sims = mat @ mat.T
    i_idx, j_idx = np.triu_indices(len(ids), k=1)
    hits = np.nonzero(sims[i_idx, j_idx] >= threshold)[0]
    pairs = [(float(sims[i_idx[h], j_idx[h]]), ids[i_idx[h]], ids[j_idx[h]]) for h in hits]
    pairs.sort(key=lambda p: -p[0])
    return pairs


def _keep_and_drop(a: MemoryItem, b: MemoryItem) -> tuple[MemoryItem, MemoryItem]:
    """The item with more uses survives; on a tie the older one does."""
    if a.uses != b.uses:
        return (a, b) if a.uses > b.uses else (b, a)
    if (a.created_at, a.id) <= (b.created_at, b.id):
        return a, b
    return b, a


def consolidate(
    store: MemoryStore,
    *,
    dedup_sim: float,
    working_ttl_days: int,
    now: datetime | None = None,
) -> dict[str, int]:
    """Nightly, deterministic, no model calls: archive near-duplicates (per
    kind, within one embedding space), archive never-used working-tier items
    past the TTL, drop expired pins, and record one consolidate event with
    the counts. Consolidation by dedup only -- never by summarization."""
    now_dt = _now(now)
    pairs: list[list[str]] = []
    for kind in KINDS:
        items = {item.id: item for item in store.list_active(kind)}
        models_seen = sorted({i.embedding_model for i in items.values() if i.embedding_model})
        for model_name in models_seen:
            rows = [
                (item_id, blob)
                for item_id, blob in store.active_embeddings(kind, embedding_model=model_name)
                if item_id in items
            ]
            gone: set[str] = set()
            for _, id_a, id_b in _near_duplicate_pairs(rows, dedup_sim):
                if id_a in gone or id_b in gone:
                    continue
                keep, drop = _keep_and_drop(items[id_a], items[id_b])
                if store.archive_item(drop.id, "consolidate:near_duplicate", now=now_dt):
                    gone.add(drop.id)
                    pairs.append([drop.id, keep.id])
    cutoff = _iso(now_dt - timedelta(days=working_ttl_days))
    ttl_archived = 0
    for item in store.list_active():
        if (
            item.tier == "working"
            and item.uses == 0
            and item.created_at <= cutoff
            and store.archive_item(item.id, "consolidate:working_ttl", now=now_dt)
        ):
            ttl_archived += 1
    pins_pruned = store.delete_expired_pins(now=now_dt)
    counts = {
        "near_duplicates_archived": len(pairs),
        "working_ttl_archived": ttl_archived,
        "pins_pruned": pins_pruned,
        "active_after": len(store.list_active()),
    }
    store.add_event("consolidate", {**counts, "near_duplicate_pairs": pairs}, now=now_dt)
    log.info("Memory consolidate: %s", counts)
    return counts


# -- shadow log --


def append_shadow_jsonl(
    shadow_dir: Path,
    *,
    conversation_key8: str,
    surface: str,
    query: str,
    block: str,
    item_ids: Sequence[str],
    now: datetime | None = None,
) -> None:
    """Append what WOULD have been injected for one conversation start to
    {shadow_dir}/shadow-{YYYYMMDD}.jsonl.

    The only place a first user turn is written beside a block, flag-gated
    and local-only, with the same trust boundary as the panel transcripts.
    Fire-and-forget like that writer: any failure is logged with a traceback
    and never reaches the request path.
    """
    now_dt = _now(now)
    record = {
        "ts": _iso(now_dt),
        "conversation_key8": conversation_key8,
        "surface": surface,
        "query": query,
        "block": block,
        "item_ids": list(item_ids),
    }
    try:
        line = json.dumps(record) + "\n"
        shadow_dir.mkdir(parents=True, exist_ok=True)
        path = shadow_dir / f"shadow-{now_dt.astimezone(UTC).strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        log.warning("Failed to write memory shadow JSONL to %s", shadow_dir, exc_info=True)
