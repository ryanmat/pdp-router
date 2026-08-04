# Description: Confidence-based LLM routing with model registry and fallback chains.
# Description: Layer 3 (Intelligence Cascade) of the three-layer routing architecture.

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from pdp_router._models import (
    DEEPSEEK,
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    GEMINI_PRO,
    GPT_5_5,
    HAIKU,
    LLAMA_MAVERICK,
    LLAMA_SCOUT,
    OPUS,
    OPUS_5,
    QWEN_3_7_PLUS,
    SONNET,
    SONNET_5,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapability:
    """Registry entry describing a model's properties."""

    name: str
    tier: int  # 1=frontier, 2=standard, 3=fast, 4=budget, 5=local
    cost_per_mtok_in: float
    cost_per_mtok_out: float
    provider: str  # "anthropic" | "gemini" | "vertex" | "ollama" | "deepseek" | "openrouter"
    available: bool


@dataclass(frozen=True)
class ModelSelection:
    """Result of a routing decision."""

    model: str
    tier: int
    reason: str


class ModelRegistry:
    """Registry of available models with tier-based lookup."""

    def __init__(self, models: dict[str, ModelCapability] | None = None) -> None:
        self._models = dict(models) if models else {}

    def get(self, name: str) -> ModelCapability | None:
        """Look up a model by name."""
        return self._models.get(name)

    def by_tier(self, tier: int) -> list[ModelCapability]:
        """Return all models in a given tier."""
        return [m for m in self._models.values() if m.tier == tier]

    def available_models(self) -> list[ModelCapability]:
        """Return all models marked as available."""
        return [m for m in self._models.values() if m.available]


DEFAULT_REGISTRY = ModelRegistry(
    {
        # Tier 1: Frontier
        OPUS_5: ModelCapability(
            name=OPUS_5,
            tier=1,
            cost_per_mtok_in=5.0,
            cost_per_mtok_out=25.0,
            provider="anthropic",
            available=True,
        ),
        OPUS: ModelCapability(
            name=OPUS,
            tier=1,
            cost_per_mtok_in=5.0,
            cost_per_mtok_out=25.0,
            provider="anthropic",
            available=True,
        ),
        # Tier 2: Standard
        GEMINI_PRO: ModelCapability(
            name=GEMINI_PRO,
            tier=2,
            cost_per_mtok_in=1.25,
            cost_per_mtok_out=10.0,
            provider="gemini",
            available=True,
        ),
        SONNET_5: ModelCapability(
            name=SONNET_5,
            tier=2,
            # List rates. Introductory pricing of $2/$10 per MTok runs through
            # 2026-08-31; list is carried so cost-efficiency weighting does not
            # over-favour this arm once the promotion lapses.
            cost_per_mtok_in=3.0,
            cost_per_mtok_out=15.0,
            provider="anthropic",
            available=True,
        ),
        SONNET: ModelCapability(
            name=SONNET,
            tier=2,
            cost_per_mtok_in=3.0,
            cost_per_mtok_out=15.0,
            provider="anthropic",
            available=True,
        ),
        # Tier 3: Fast
        GEMINI_FLASH: ModelCapability(
            name=GEMINI_FLASH,
            tier=3,
            cost_per_mtok_in=0.30,
            cost_per_mtok_out=2.50,
            provider="gemini",
            available=True,
        ),
        LLAMA_SCOUT: ModelCapability(
            name=LLAMA_SCOUT,
            tier=3,
            cost_per_mtok_in=0.25,
            cost_per_mtok_out=0.70,
            provider="vertex",
            available=True,
        ),
        HAIKU: ModelCapability(
            name=HAIKU,
            tier=3,
            cost_per_mtok_in=0.25,
            cost_per_mtok_out=1.25,
            provider="anthropic",
            available=True,
        ),
        # Tier 4: Budget
        GEMINI_FLASH_LITE: ModelCapability(
            name=GEMINI_FLASH_LITE,
            tier=4,
            cost_per_mtok_in=0.10,
            cost_per_mtok_out=0.40,
            provider="gemini",
            available=True,
        ),
        LLAMA_MAVERICK: ModelCapability(
            name=LLAMA_MAVERICK,
            tier=4,
            cost_per_mtok_in=0.35,
            cost_per_mtok_out=1.15,
            provider="vertex",
            available=True,
        ),
        # Tier 4: Budget -- DeepSeek (Chinese training lineage, genuinely decorrelated)
        DEEPSEEK: ModelCapability(
            name=DEEPSEEK,
            tier=4,
            cost_per_mtok_in=0.28,
            cost_per_mtok_out=0.42,
            provider="deepseek",
            available=True,
        ),
        # OpenRouter arms -- lineage diversity (OpenAI + Qwen/Alibaba), API-only,
        # reachable via the panel composer and the cascade explore branch. Credentials
        # come from OPENROUTER_API_KEY in the proxy env; set available=False to retire
        # an arm (kill switch).
        GPT_5_5: ModelCapability(
            name=GPT_5_5,
            tier=2,
            cost_per_mtok_in=5.0,
            cost_per_mtok_out=30.0,
            provider="openrouter",
            available=True,
        ),
        QWEN_3_7_PLUS: ModelCapability(
            name=QWEN_3_7_PLUS,
            tier=3,
            cost_per_mtok_in=0.32,
            cost_per_mtok_out=1.28,
            provider="openrouter",
            available=True,
        ),
    }
)


def bandit_branch_active(routing_mode: str, bandit_states: dict | None) -> bool:
    """Whether Thompson Sampling, not the threshold cascade, drives the pick.

    The single source of truth for the bandit gate. confidence_cascade branches
    on this, and callers that record the executed policy read the same predicate
    rather than restating the condition. Restating it is how the routing rows
    came to claim "cascade" for every Thompson-sampled decision: configuring
    ROUTING_MODE=bandit is not the same as the bandit actually running, because
    an absent or unreadable posterior store silently falls through to the
    cascade.
    """
    return routing_mode == "bandit" and bool(bandit_states)


# Trust-based threshold adjustment constants.
_DEFAULT_TRUST = 0.5
_TRUST_SCALE = 0.20  # max +-0.10 threshold shift from +-0.50 trust delta


# Disagreement-triggered exploration: when signals agree, exploit; when they
# disagree, explore. agreement_level is the count of True signals out of 9 in
# the enrichment signal dict. Absolute rates, not multipliers -- the table
# supersedes the caller's explore_rate parameter when an int agreement_level
# is provided. Sorted by threshold descending; first match wins.
_EXPLORE_RATE_BY_AGREEMENT: tuple[tuple[int, float], ...] = (
    (7, 0.02),  # 7+/9 agree -> exploit aggressively
    (5, 0.05),
    (3, 0.10),  # baseline (matches pre-S54 flat rate)
    (0, 0.25),  # <3/9 agree -> explore aggressively
)


def _adjust_threshold(base: float, trust: float) -> float:
    """Shift a routing threshold based on model trust.

    High trust (>0.5) lowers the threshold, allowing the model to be selected at
    lower confidence. Low trust (<0.5) raises the threshold, requiring higher
    confidence before routing to the model. Maximum adjustment is +-0.10.
    """
    delta = (trust - _DEFAULT_TRUST) * _TRUST_SCALE
    return max(0.0, min(1.0, base - delta))


def _dynamic_explore_rate(agreement_level: int) -> float:
    """Map a signal agreement count to an effective exploration rate.

    Walks _EXPLORE_RATE_BY_AGREEMENT top-down; the first threshold satisfied
    by agreement_level wins. Values above the top band still match the top
    band. The table MUST contain a catch-all entry at threshold 0 so that
    every non-negative input resolves to a rate. Negative values raise
    ValueError because they indicate a caller bug (agreement_level is a
    count, cannot be < 0).
    """
    if agreement_level < 0:
        raise ValueError(f"agreement_level must be >= 0, got {agreement_level}")
    for threshold, rate in _EXPLORE_RATE_BY_AGREEMENT:
        if agreement_level >= threshold:
            return rate
    raise ValueError(
        "_EXPLORE_RATE_BY_AGREEMENT has no catch-all entry at threshold 0; "
        f"cannot resolve agreement_level={agreement_level}"
    )


def apply_cost_efficiency(
    trust_weights: dict[str, float],
    registry: ModelRegistry | None = None,
) -> dict[str, float]:
    """Adjust trust weights by model cost efficiency.

    Cheap models get a bonus: adjusted = raw_trust * (cheapest_total / model_total).
    The cheapest available model scores 1.0; expensive models score proportionally less.
    Models not in registry pass through unchanged.

    Args:
        trust_weights: Raw trust weights from pdp-tracker (model_name -> 0.0-1.0).
        registry: Model registry for cost lookup. Defaults to DEFAULT_REGISTRY.

    Returns:
        Cost-adjusted trust weights dict.
    """
    if not trust_weights:
        return {}

    reg = registry or DEFAULT_REGISTRY
    available = reg.available_models()
    if not available:
        return dict(trust_weights)

    totals = {m.name: m.cost_per_mtok_in + m.cost_per_mtok_out for m in available}
    cheapest = min(totals.values())
    if cheapest <= 0:
        return dict(trust_weights)

    adjusted = {}
    for model, raw_trust in trust_weights.items():
        model_total = totals.get(model)
        if model_total is None or model_total <= 0:
            adjusted[model] = raw_trust
        else:
            efficiency = cheapest / model_total
            adjusted[model] = raw_trust * efficiency
    return adjusted


def route_request(
    confidence: float,
    domain: str = "infrastructure",
    agreement_level: int = 0,
    registry: ModelRegistry | None = None,
    trust_weights: dict[str, float] | None = None,
) -> ModelSelection:
    """Select a model tier based on confidence score and signal agreement.

    Cascade thresholds (strict >, adjusted by model trust weights):
        conf > 0.85 -> Haiku (synthesis skip recommended)
        conf > 0.70 AND agreement >= 4 -> Haiku
        conf > 0.40 -> Sonnet
        conf <= 0.40 -> Opus (frontier)

    Args:
        confidence: Pre-calculated confidence score (0.10 to 0.95).
        domain: Alert domain (infrastructure, trading, lab, general).
        agreement_level: Count of True signals from enrichment.
        registry: Model registry (defaults to DEFAULT_REGISTRY).
        trust_weights: Per-model trust weights from pdp-tracker. Keys are model
            name strings, values are 0.0-1.0. None uses default thresholds.

    Returns:
        ModelSelection with chosen model, tier, and reason.
    """
    tw = trust_weights or {}
    haiku_trust = tw.get(HAIKU, _DEFAULT_TRUST)
    # Reads the trust of the arm this threshold actually gates -- the standard
    # tier dispatches to SONNET_5, so SONNET's history must not move it.
    sonnet_trust = tw.get(SONNET_5, _DEFAULT_TRUST)

    skip_thresh = _adjust_threshold(0.85, haiku_trust)
    if confidence > skip_thresh:
        return ModelSelection(
            model=HAIKU,
            tier=3,
            reason=(
                f"conf={confidence:.2f}>{skip_thresh:.2f}: synthesis skip recommended, fast tier"
            ),
        )

    agree_thresh = _adjust_threshold(0.70, haiku_trust)
    if confidence > agree_thresh and agreement_level >= 4:
        return ModelSelection(
            model=HAIKU,
            tier=3,
            reason=(
                f"conf={confidence:.2f}>{agree_thresh:.2f},"
                f" agreement={agreement_level}>=4: fast tier"
            ),
        )

    standard_thresh = _adjust_threshold(0.40, sonnet_trust)
    if confidence > standard_thresh:
        return ModelSelection(
            model=SONNET_5,
            tier=2,
            reason=f"conf={confidence:.2f}>{standard_thresh:.2f}: standard tier",
        )

    return ModelSelection(
        model=OPUS_5,
        tier=1,
        reason=f"conf={confidence:.2f}<={standard_thresh:.2f}: frontier tier",
    )


def route_with_fallback(
    confidence: float,
    domain: str = "infrastructure",
    agreement_level: int = 0,
    registry: ModelRegistry | None = None,
    trust_weights: dict[str, float] | None = None,
) -> ModelSelection:
    """Route with automatic fallback when the selected model is unavailable.

    Falls back to the cheapest available model (highest tier number first).

    Args:
        Same as route_request.

    Returns:
        ModelSelection with an available model.

    Raises:
        RuntimeError: If no models are available in the registry.
    """
    reg = registry or DEFAULT_REGISTRY
    selection = route_request(confidence, domain, agreement_level, registry, trust_weights)

    cap = reg.get(selection.model)
    if cap is not None and cap.available:
        return selection

    available = reg.available_models()
    if not available:
        raise RuntimeError("No available models in registry")

    # Prefer cheapest (highest tier number = cheapest).
    available_sorted = sorted(available, key=lambda m: -m.tier)
    fallback = available_sorted[0]

    return ModelSelection(
        model=fallback.name,
        tier=fallback.tier,
        reason=f"fallback from {selection.model} (unavailable) to {fallback.name}",
    )


def confidence_cascade(
    confidence: float,
    agreement_level: int | None = None,
    domain: str = "infrastructure",
    budget_action: str | None = None,
    registry: ModelRegistry | None = None,
    trust_weights: dict[str, float] | None = None,
    explore_rate: float = 0.0,
    cost_adjusted: bool = False,
    _rng: random.Random | None = None,
    routing_mode: str = "cascade",
    bandit_states: dict | None = None,
    routing_context: object | None = None,
    context_bandit_states: dict | None = None,
    return_debug: bool = False,
) -> str | tuple[str, bool]:
    """Full routing pipeline: cascade + fallback + budget override -> model name.

    Composes route_with_fallback() and apply_budget_override() into a single
    call. This is the public entry point for the orchestrator.

    Args:
        confidence: Pre-calculated confidence score (0.10 to 0.95).
        agreement_level: Count of True signals from enrichment. Two modes:
            - None (default): legacy flat-rate exploration using `explore_rate`.
            - int: disagreement-triggered exploration. The rate is derived
              from _EXPLORE_RATE_BY_AGREEMENT (absolute values, not a
              multiplier of `explore_rate`). `explore_rate > 0` acts as the
              kill switch -- set to 0 to disable exploration entirely even
              when an int agreement_level is provided.
        domain: Alert domain (infrastructure, trading, lab, general).
        budget_action: Budget override action ("downgrade", "skip", or None).
        registry: Model registry (defaults to DEFAULT_REGISTRY).
        trust_weights: Per-model trust weights from pdp-tracker.
        explore_rate: Gate for epsilon-greedy exploration. When
            agreement_level is None, this IS the exploration probability
            (0.0-1.0). When agreement_level is an int, this acts purely as
            a feature toggle: 0.0 disables, any positive value enables the
            dynamic rate from the agreement table.
        cost_adjusted: When True, adjust trust_weights by cost efficiency
            before routing. Cheap models with similar accuracy win.
        _rng: Optional seeded Random instance for deterministic testing.
        routing_mode: "cascade" (default, threshold-based) or "bandit"
            (Thompson Sampling from posteriors). Epsilon-greedy exploration
            only runs in cascade mode; bandit mode's posterior sampling is
            its own exploration mechanism.
        bandit_states: Mapping of model name to BanditState posteriors.
            Required when routing_mode="bandit". Ignored in cascade mode.
        routing_context: Optional RoutingContext for contextual Thompson Sampling.
            When provided alongside context_bandit_states, uses context-specific
            posteriors with fallback to domain-level.
        context_bandit_states: {model_name: {context_key: BanditState}} for
            contextual routing. Ignored if routing_context is None.
        return_debug: When True, returns ``(model_name, explored)`` instead
            of just the model name. ``explored`` is True iff the cascade hit
            its epsilon-greedy explore branch (uniform-random pick rather
            than threshold-driven). The Concern 24 shadow path uses this to
            mark rows that should be excluded from agreement_rate analytics.
            Bandit-mode picks are not "exploration" in this sense (the
            posterior sampling is the bandit's exploration mechanism), so
            ``explored=False`` in bandit mode.

    Returns:
        Model name string ready to pass to get_client(), OR a
        ``(model_name, explored: bool)`` tuple when ``return_debug=True``.

    Raises:
        RuntimeError: If no models are available in the registry.
    """
    reg = registry or DEFAULT_REGISTRY

    # Thompson Sampling: sample from posteriors, pick highest draw.
    if bandit_branch_active(routing_mode, bandit_states):
        from pdp_router._bandit import (
            RoutingContext,
            apply_cost_to_bandit,
            contextual_thompson_sample,
            thompson_sample,
        )

        # Contextual path: use context-aware posteriors with fallback.
        if (
            routing_context is not None
            and isinstance(routing_context, RoutingContext)
            and context_bandit_states is not None
        ):
            # Apply cost adjustment to domain-level states before fallback.
            domain_states = bandit_states
            if cost_adjusted:
                costs = {
                    m.name: m.cost_per_mtok_in + m.cost_per_mtok_out for m in reg.available_models()
                }
                domain_states = apply_cost_to_bandit(domain_states, costs)

            pick = contextual_thompson_sample(
                context_states=context_bandit_states,
                domain_states=domain_states,
                context=routing_context,
                rng=_rng,
            )
            log.info(
                "Contextual Thompson pick: %s (ctx=%s, mode=bandit)",
                pick,
                routing_context.context_key(),
            )
            picked_state = domain_states.get(pick)
            if picked_state and picked_state.mu < 0.10:
                log.warning(
                    "Low-trust model selected: %s mu=%.3f n=%d",
                    pick,
                    picked_state.mu,
                    picked_state.n_obs,
                )
        else:
            # Domain-only bandit path (backward compatible).
            states = bandit_states
            if cost_adjusted:
                costs = {
                    m.name: m.cost_per_mtok_in + m.cost_per_mtok_out for m in reg.available_models()
                }
                states = apply_cost_to_bandit(states, costs)

            pick = thompson_sample(states, rng=_rng)
            log.info("Thompson pick: %s (mode=bandit)", pick)
            picked_state = states.get(pick)
            if picked_state and picked_state.mu < 0.10:
                log.warning(
                    "Low-trust model selected: %s mu=%.3f n=%d",
                    pick,
                    picked_state.mu,
                    picked_state.n_obs,
                )

        if budget_action:
            cap = reg.get(pick)
            tier = cap.tier if cap else 3
            sel = ModelSelection(model=pick, tier=tier, reason="bandit")
            sel = apply_budget_override(sel, budget_action, registry)
            return (sel.model, False) if return_debug else sel.model
        return (pick, False) if return_debug else pick

    # Epsilon-greedy: bypass cascade with probability effective_rate.
    # When agreement_level is provided, use the disagreement-triggered
    # table; otherwise fall back to the caller's flat explore_rate.
    if explore_rate > 0.0:
        effective_rate = (
            _dynamic_explore_rate(agreement_level) if agreement_level is not None else explore_rate
        )
        roll = _rng.random() if _rng else random.random()
        if roll < effective_rate:
            available = reg.available_models()
            if available:
                pick = _rng.choice(available) if _rng else random.choice(available)
                log.info(
                    "Exploration pick: %s (bypassed cascade, roll=%.3f, agreement=%s, rate=%.3f)",
                    pick.name,
                    roll,
                    agreement_level,
                    effective_rate,
                )
                return (pick.name, True) if return_debug else pick.name

    # Cost-efficiency adjustment before routing.
    if cost_adjusted and trust_weights:
        trust_weights = apply_cost_efficiency(trust_weights, reg)

    # Coerce None to 0 for downstream route_request(), which expects an int.
    agreement_int = agreement_level if agreement_level is not None else 0
    selection = route_with_fallback(confidence, domain, agreement_int, registry, trust_weights)
    if budget_action:
        selection = apply_budget_override(selection, budget_action, registry)
    return (selection.model, False) if return_debug else selection.model


def next_cheaper(model: str, registry: ModelRegistry | None = None) -> str:
    """Return the model one tier cheaper, or the same model if already cheapest.

    Registry-driven: walks available models sorted by tier to find the next
    higher tier number (cheaper). Falls back to DEFAULT_REGISTRY if none provided.

    Args:
        model: Current model name.
        registry: Model registry to search. Defaults to DEFAULT_REGISTRY.

    Returns:
        Next cheaper model name, or same model if at cheapest tier or unknown.
    """
    reg = registry or DEFAULT_REGISTRY
    cap = reg.get(model)
    if cap is None:
        return model

    available = sorted(reg.available_models(), key=lambda m: m.tier)
    # Find models in a higher tier number (cheaper).
    cheaper = [m for m in available if m.tier > cap.tier]
    if not cheaper:
        return model
    return cheaper[0].name


def apply_budget_override(
    selection: ModelSelection,
    budget_action: str,
    registry: ModelRegistry | None = None,
) -> ModelSelection:
    """Override model selection when budget is exceeded.

    Picks the cheapest available model from the registry instead of
    hardcoding a specific model.

    Args:
        selection: Original routing decision.
        budget_action: Action from config ("downgrade" or "skip").
        registry: Model registry for finding cheapest model.

    Returns:
        Overridden ModelSelection if downgrade, original otherwise.
    """
    if budget_action == "downgrade":
        reg = registry or DEFAULT_REGISTRY
        available = reg.available_models()
        if not available:
            return selection
        cheapest = sorted(available, key=lambda m: -m.tier)[0]
        return ModelSelection(
            model=cheapest.name,
            tier=cheapest.tier,
            reason=f"budget override: downgraded from {selection.model} to {cheapest.name}",
        )
    return selection


# ---------------------------------------------------------------------------
# Concern 24: shadow A/B helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingPicks:
    """Paired cascade/bandit routing picks for shadow A/B logging.

    The cascade pick is what the caller actually uses to dispatch the LLM
    call. The shadow pick is what the bandit's Thompson sampling would have
    picked given the same context -- never called, only logged. mu / n_obs
    come from the bandit_states dict that was passed in; they are the
    posterior view of each pick at decision time.
    """

    cascade_pick: str
    shadow_pick: str | None
    cascade_pick_mu: float | None
    shadow_pick_mu: float | None
    cascade_pick_n_obs: int | None
    shadow_pick_n_obs: int | None
    cascade_explored: bool


def _lookup_pick_state(
    pick: str | None,
    states: dict | None,
) -> tuple[float | None, int | None]:
    """Pull mu and n_obs for a pick from a bandit_states dict.

    Returns (None, None) when the pick is None, the dict is None/empty, or
    the dict has no entry for this pick. The dict can hold either
    BanditState dataclasses or plain dict rows; we accept both.
    """
    if pick is None or not states:
        return (None, None)
    state = states.get(pick)
    if state is None:
        return (None, None)
    mu = getattr(state, "mu", None)
    n_obs = getattr(state, "n_obs", None)
    if mu is None and isinstance(state, dict):
        mu = state.get("mu")
        n_obs = state.get("n_obs")
    return (mu, n_obs)


def compute_routing_picks(
    confidence: float,
    agreement_level: int,
    domain: str,
    routing_context: object,
    trust_weights: dict[str, float] | None,
    bandit_states: dict | None,
    context_bandit_states: dict | None,
    explore_rate: float,
    cost_adjusted: bool = True,
    budget_action: str | None = None,
    registry: ModelRegistry | None = None,
    _rng: random.Random | None = None,
) -> RoutingPicks:
    """Concern 24: compute both cascade and bandit picks for shadow A/B logging.

    The cascade pick is what the orchestrator actually calls. The shadow pick
    is what the bandit's Thompson sampling would have picked given the same
    bandit_states. Both posterior mu's and n_obs are looked up against the
    same bandit_states dict; cold-start arms (no entry) return None.

    When ``bandit_states`` is None or empty, ``shadow_pick`` is None and the
    shadow fields are all None -- nothing to log against.

    Internally calls confidence_cascade twice. Both calls share the same
    routing_context and trust_weights. The cascade call uses
    ``return_debug=True`` so we know whether the cascade pick was an
    epsilon-greedy explore branch (excluded from agreement_rate analytics).
    The bandit call always uses ``routing_mode="bandit"`` regardless of
    config. Cost adjustment, budget override, and rng are honored on both
    paths so the picks are comparable apples-to-apples.
    """
    cascade_result = confidence_cascade(
        confidence=confidence,
        agreement_level=agreement_level,
        domain=domain,
        budget_action=budget_action,
        registry=registry,
        trust_weights=trust_weights,
        explore_rate=explore_rate,
        cost_adjusted=cost_adjusted,
        _rng=_rng,
        routing_mode="cascade",
        bandit_states=None,
        routing_context=routing_context,
        context_bandit_states=None,
        return_debug=True,
    )
    cascade_pick, cascade_explored = cascade_result  # type: ignore[misc]

    shadow_pick: str | None = None
    if bandit_states:
        shadow_pick = confidence_cascade(  # type: ignore[assignment]
            confidence=confidence,
            agreement_level=agreement_level,
            domain=domain,
            budget_action=budget_action,
            registry=registry,
            trust_weights=trust_weights,
            explore_rate=0.0,  # bandit's posterior sampling is its exploration
            cost_adjusted=cost_adjusted,
            _rng=_rng,
            routing_mode="bandit",
            bandit_states=bandit_states,
            routing_context=routing_context,
            context_bandit_states=context_bandit_states,
            return_debug=False,
        )

    cascade_mu, cascade_n_obs = _lookup_pick_state(cascade_pick, bandit_states)
    shadow_mu, shadow_n_obs = _lookup_pick_state(shadow_pick, bandit_states)

    return RoutingPicks(
        cascade_pick=cascade_pick,
        shadow_pick=shadow_pick,
        cascade_pick_mu=cascade_mu,
        shadow_pick_mu=shadow_mu,
        cascade_pick_n_obs=cascade_n_obs,
        shadow_pick_n_obs=shadow_n_obs,
        cascade_explored=cascade_explored,
    )
