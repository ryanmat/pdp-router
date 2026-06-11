# Description: Forward-Evidence Registry skeleton for the router.
# Description: Composes learned predictors into structured evidence; soft-injection, not Dyna-Q.

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pdp_router._bandit import RoutingContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForwardEvidence:
    """Evidence emitted by a single forward model adapter.

    Composed by the router (or chair prompt builder) into a structured
    "Routing Evidence" block. This is observability + soft-evidence injection,
    not a queryable forward simulator.
    """

    model_name: str
    evidence_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@runtime_checkable
class ForwardModel(Protocol):
    """Adapter interface for a forward-evidence source.

    Phase 1 skeleton: stubs return empty payloads. Phase 2 wires real
    predictors (Precursor X-DEC via mcporter, Chronos via metric-forecast,
    bandit posterior via pdp-tracker MCP).
    """

    name: str

    def predict(self, context: RoutingContext) -> ForwardEvidence | None: ...


class _BaseAdapter:
    """Common scaffolding so adapter stubs measure their own latency."""

    name: str = "base"
    evidence_type: str = "base"

    def predict(self, context: RoutingContext) -> ForwardEvidence | None:
        start = time.perf_counter()
        payload = self._payload(context)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if payload is None:
            return None
        return ForwardEvidence(
            model_name=self.name,
            evidence_type=self.evidence_type,
            payload=payload,
            latency_ms=elapsed_ms,
        )

    def _payload(self, context: RoutingContext) -> dict[str, Any] | None:
        raise NotImplementedError


class PrecursorXDecAdapter(_BaseAdapter):
    """Stub. Phase 2 dispatches to precursor-bridge MCP for X-DEC cluster + risk."""

    name = "precursor-x-dec"
    evidence_type = "x_dec_cluster"

    def _payload(self, context: RoutingContext) -> dict[str, Any] | None:
        # Phase 2: call mcporter precursor.predict with context features and
        # return cluster_id + risk_band + confidence. Skeleton: empty.
        return {}


class ChronosForecastAdapter(_BaseAdapter):
    """Stub. Phase 2 dispatches to metric-forecast pipeline for short-horizon forecasts."""

    name = "chronos-bolt"
    evidence_type = "chronos_forecast"

    def _payload(self, context: RoutingContext) -> dict[str, Any] | None:
        # Phase 2: pull recent forecasts for the alert's resource and emit
        # next-N-step trajectory + breach probability. Skeleton: empty.
        return {}


class BanditPosteriorAdapter(_BaseAdapter):
    """Stub. Phase 2 reads the bandit posterior for each candidate model in the routing context."""

    name = "bandit-posterior"
    evidence_type = "bandit_posterior"

    def _payload(self, context: RoutingContext) -> dict[str, Any] | None:
        # Phase 2: query pdp-tracker MCP for current posteriors keyed on
        # (model_id, domain) for each candidate. Skeleton: empty.
        return {}


class ForwardModelRegistry:
    """Collects evidence from registered forward models within a deadline budget.

    Adapters that exceed the deadline or raise are skipped; partial evidence
    is preferred over a failed routing decision. Phase 2 may switch the
    executor model (asyncio, process pool) once latency targets are clearer.
    """

    def __init__(
        self,
        models: list[ForwardModel] | None = None,
        deadline_ms: float = 100.0,
    ) -> None:
        self._models = list(models) if models is not None else []
        self._deadline_ms = deadline_ms

    def register(self, model: ForwardModel) -> None:
        self._models.append(model)

    def collect(self, context: RoutingContext) -> list[ForwardEvidence]:
        if not self._models:
            return []

        results: list[ForwardEvidence] = []
        results_lock = threading.Lock()
        deadline_sec = self._deadline_ms / 1000.0

        def _worker(model: ForwardModel) -> None:
            try:
                evidence = model.predict(context)
            except Exception as exc:
                log.warning(
                    "ForwardModel %s raised: %s",
                    getattr(model, "name", type(model).__name__),
                    exc,
                )
                return
            if evidence is None:
                return
            with results_lock:
                results.append(evidence)

        # Daemon threads: stragglers past the deadline keep running in the
        # background but do not block program exit or this function's return.
        threads: list[tuple[threading.Thread, ForwardModel]] = []
        for model in self._models:
            thread = threading.Thread(target=_worker, args=(model,), daemon=True)
            thread.start()
            threads.append((thread, model))

        deadline_at = time.monotonic() + deadline_sec
        for thread, model in threads:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                if thread.is_alive():
                    log.warning(
                        "ForwardModel %s exceeded deadline %.0fms; skipping",
                        getattr(model, "name", type(model).__name__),
                        self._deadline_ms,
                    )
                continue
            thread.join(timeout=remaining)
            if thread.is_alive():
                log.warning(
                    "ForwardModel %s exceeded deadline %.0fms; skipping",
                    getattr(model, "name", type(model).__name__),
                    self._deadline_ms,
                )

        with results_lock:
            return list(results)


def default_registry(deadline_ms: float = 100.0) -> ForwardModelRegistry:
    """Construct a registry with the three Phase 1 stubs registered."""
    return ForwardModelRegistry(
        models=[
            PrecursorXDecAdapter(),
            ChronosForecastAdapter(),
            BanditPosteriorAdapter(),
        ],
        deadline_ms=deadline_ms,
    )
