# Description: Tests for the Forward-Evidence Registry skeleton (Sprint 7 Phase 1).
# Description: Verifies protocol contract, stub adapter shapes, registry deadline + error tolerance.

from __future__ import annotations

import time

from pdp_router._bandit import RoutingContext
from pdp_router._forward_models import (
    BanditPosteriorAdapter,
    ChronosForecastAdapter,
    ForwardEvidence,
    ForwardModel,
    ForwardModelRegistry,
    PrecursorXDecAdapter,
    default_registry,
)


def _make_context() -> RoutingContext:
    """Minimal RoutingContext for tests; relies on dataclass defaults."""
    return RoutingContext()


class TestProtocolConformance:
    def test_precursor_adapter_conforms(self) -> None:
        assert isinstance(PrecursorXDecAdapter(), ForwardModel)

    def test_chronos_adapter_conforms(self) -> None:
        assert isinstance(ChronosForecastAdapter(), ForwardModel)

    def test_bandit_adapter_conforms(self) -> None:
        assert isinstance(BanditPosteriorAdapter(), ForwardModel)


class TestStubAdapterShapes:
    def test_precursor_emits_x_dec_cluster_evidence(self) -> None:
        adapter = PrecursorXDecAdapter()
        evidence = adapter.predict(_make_context())
        assert isinstance(evidence, ForwardEvidence)
        assert evidence.model_name == "precursor-x-dec"
        assert evidence.evidence_type == "x_dec_cluster"
        assert evidence.payload == {}
        assert evidence.latency_ms >= 0.0

    def test_chronos_emits_forecast_evidence(self) -> None:
        evidence = ChronosForecastAdapter().predict(_make_context())
        assert evidence is not None
        assert evidence.model_name == "chronos-bolt"
        assert evidence.evidence_type == "chronos_forecast"

    def test_bandit_emits_posterior_evidence(self) -> None:
        evidence = BanditPosteriorAdapter().predict(_make_context())
        assert evidence is not None
        assert evidence.model_name == "bandit-posterior"
        assert evidence.evidence_type == "bandit_posterior"


class TestRegistryCollection:
    def test_empty_registry_returns_empty_list(self) -> None:
        registry = ForwardModelRegistry()
        assert registry.collect(_make_context()) == []

    def test_default_registry_collects_three_evidences(self) -> None:
        registry = default_registry()
        results = registry.collect(_make_context())
        assert len(results) == 3
        types = {e.evidence_type for e in results}
        assert types == {"x_dec_cluster", "chronos_forecast", "bandit_posterior"}

    def test_register_appends_adapter(self) -> None:
        registry = ForwardModelRegistry()
        registry.register(PrecursorXDecAdapter())
        registry.register(ChronosForecastAdapter())
        results = registry.collect(_make_context())
        assert len(results) == 2


class TestRegistryFailureTolerance:
    def test_failing_adapter_is_skipped(self) -> None:
        class FailingAdapter:
            name = "failing"

            def predict(self, context: RoutingContext) -> ForwardEvidence | None:
                raise RuntimeError("boom")

        registry = ForwardModelRegistry(
            models=[FailingAdapter(), PrecursorXDecAdapter()],
        )
        results = registry.collect(_make_context())
        # Only the working adapter contributes evidence.
        assert len(results) == 1
        assert results[0].model_name == "precursor-x-dec"

    def test_adapter_returning_none_is_dropped(self) -> None:
        class NullAdapter:
            name = "null"

            def predict(self, context: RoutingContext) -> ForwardEvidence | None:
                return None

        registry = ForwardModelRegistry(
            models=[NullAdapter(), ChronosForecastAdapter()],
        )
        results = registry.collect(_make_context())
        assert len(results) == 1
        assert results[0].model_name == "chronos-bolt"


class TestRegistryDeadline:
    def test_slow_adapter_dropped_at_deadline(self) -> None:
        class SlowAdapter:
            name = "slow"

            def predict(self, context: RoutingContext) -> ForwardEvidence | None:
                time.sleep(0.5)  # 500ms -- well past the 50ms deadline
                return ForwardEvidence(
                    model_name="slow",
                    evidence_type="slow_evidence",
                    payload={"too": "late"},
                )

        registry = ForwardModelRegistry(
            models=[SlowAdapter(), PrecursorXDecAdapter()],
            deadline_ms=50.0,
        )
        start = time.perf_counter()
        results = registry.collect(_make_context())
        elapsed = time.perf_counter() - start
        # Should return within ~deadline + small overhead, NOT 500ms
        assert elapsed < 0.3
        # Fast adapter should still contribute
        names = {e.model_name for e in results}
        assert "precursor-x-dec" in names
        assert "slow" not in names
