# Description: Deterministic embedder and reranker fakes for the memory tests.
# Description: Built on the pdp_router._memory Protocols; the real-model tests pin the contract.

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class FakeEmbedder:
    """Hashed bag-of-words into `dim` buckets, unit-normalized float32.

    Same text -> same vector; shared tokens -> positive cosine; disjoint
    tokens -> cosine near zero. Dim is deliberately NOT 384 so nothing
    downstream can hardcode the real model's width. `calls` records every
    embed() input so tests can assert what was (and was not) embedded.
    """

    model_name = "fake/embed"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        texts = list(texts)
        self.calls.append(texts)
        out = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for tok in _tokens(text):
                bucket = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % self.dim
                vec[bucket] += 1.0
            norm = float(np.linalg.norm(vec))
            out.append(vec / norm if norm else vec)
        return out


class FakeReranker:
    """Token-overlap ratio between query and document, in [0, 1].

    Deterministic and monotone in shared vocabulary, so a test can build a
    case where cross-encoder order disagrees with cosine order. `calls`
    records (query, documents) per score() call.
    """

    model_name = "fake/rerank"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        self.calls.append((query, documents))
        q = set(_tokens(query))
        scores = []
        for doc in documents:
            d = set(_tokens(doc))
            scores.append(len(q & d) / len(q | d) if q | d else 0.0)
        return scores


class VectorTableEmbedder:
    """Exact vectors per text, unit-normalized: for tests that need precise
    cosine values (candidate cuts, tie-breaks). Unknown text raises, so a
    test cannot silently embed something it did not set up."""

    model_name = "fake/table"

    def __init__(self, table: dict[str, Sequence[float]], dim: int) -> None:
        self.dim = dim
        self.table = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        texts = list(texts)
        self.calls.append(texts)
        out = []
        for text in texts:
            vec = self.table[text]
            norm = float(np.linalg.norm(vec))
            out.append(vec / norm if norm else vec)
        return out


class ScoreTableReranker:
    """Fixed cross-encoder score per document text (default 0.0)."""

    model_name = "fake/score-table"

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = dict(scores or {})
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        self.calls.append((query, documents))
        return [self.scores.get(d, 0.0) for d in documents]
