# Description: Real-model tests for the memory embedder and reranker wrappers (fastembed, CPU).
# Description: requires_models loads ~150 MB from the host model dir; the offline test needs none.

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pdp_router._memory import (
    FastembedEmbedder,
    MemoryRuntime,
    load_models,
)
from pdp_router._proxy_config import ProxyConfig
from tests.conftest import HOST_MODEL_DIR

# The host's model dir, captured before the autouse isolation shadowed the
# env. Package default when unset: a contributor running these on purpose
# gets one download into ~/.pdp-router/models.
MODEL_DIR = Path(HOST_MODEL_DIR or "~/.pdp-router/models").expanduser()


def _host_config() -> ProxyConfig:
    with patch.dict(os.environ, {"PROXY_MEMORY_MODEL_DIR": str(MODEL_DIR)}):
        return ProxyConfig()


class TestOfflineLoadFailsFast:
    def test_missing_files_raise_without_downloading(self, tmp_path) -> None:
        """The service loads with downloads forbidden: an empty model dir must
        raise promptly (the runtime turns that into off-for-this-request), not
        pull 150 MB in a request-adjacent thread."""
        pytest.importorskip("fastembed")
        with pytest.raises(ValueError, match="Could not load model"):
            FastembedEmbedder("BAAI/bge-small-en-v1.5", tmp_path / "empty", allow_download=False)
        # The library may create empty cache directories; it must not fetch files.
        assert not any(p.is_file() for p in (tmp_path / "empty").rglob("*"))


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("fastembed")
    return load_models(_host_config(), allow_download=True)


@pytest.mark.requires_models
class TestRealEmbedder:
    def test_embed_returns_unit_float32_vectors_of_the_model_width(self, models) -> None:
        vecs = models.embedder.embed(["the cat sat on the mat", "quarterly revenue rose"])
        assert len(vecs) == 2
        assert models.embedder.dim == 384
        for v in vecs:
            assert isinstance(v, np.ndarray)
            assert v.dtype == np.float32
            assert v.shape == (384,)
            assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)
        assert models.embedder.model_name == "BAAI/bge-small-en-v1.5"

    def test_embed_is_deterministic_and_semantic(self, models) -> None:
        sentence = "deposited money into the brokerage account"
        [a, b, a2] = models.embedder.embed([sentence, "a cat slept", sentence])
        assert float(a @ a2) == pytest.approx(1.0, abs=1e-5)
        assert float(a @ b) < 0.8

    def test_embed_empty_input(self, models) -> None:
        assert models.embedder.embed([]) == []


@pytest.mark.requires_models
class TestRealReranker:
    def test_relevant_document_scores_above_irrelevant(self, models) -> None:
        scores = models.reranker.score(
            "what is the capital of France",
            ["Paris is the capital of France.", "Bananas are yellow when ripe."],
        )
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] > scores[1]
        assert models.reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"

    def test_score_empty_input(self, models) -> None:
        assert models.reranker.score("q", []) == []


@pytest.mark.requires_models
class TestRealLoad:
    def test_model_files_live_under_the_configured_dir(self, models) -> None:
        """cache_dir must be honored: a service that leaned on the library's
        tmp-dir fallback would lose its models on reboot."""
        files = [p for p in MODEL_DIR.rglob("*") if p.is_file()]
        assert files, f"no model files under {MODEL_DIR}"
        assert sum(p.stat().st_size for p in files) > 50 * 1024 * 1024

    def test_runtime_blocking_load_is_offline_once_warmed(self, models) -> None:
        rt = MemoryRuntime(_host_config())
        loaded = rt.load_models_blocking(allow_download=False)
        assert rt.models is loaded
        assert loaded.embedder.dim == 384
        assert rt.load_error is None


@pytest.mark.requires_models
class TestRealWarmup:
    def test_warmup_cli_end_to_end(self) -> None:
        """The operator's one download step: exit 0 and a report naming the
        configured dir, run against the host model dir (already warmed by the
        fixture above, so this is the fast re-run path)."""
        import io

        from pdp_router.memory.__main__ import main

        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"PROXY_MEMORY_MODEL_DIR": str(MODEL_DIR)}):
            code = main(["warmup"], out=out, err=err)
        assert code == 0, err.getvalue()
        report = json.loads(out.getvalue())
        assert report["model_dir"] == str(MODEL_DIR)
        assert report["dim"] == 384
