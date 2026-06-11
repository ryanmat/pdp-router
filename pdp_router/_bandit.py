# Description: Thompson Sampling multi-armed bandit for Sibyl model routing.
# Description: Pure algorithm module -- no DB, no side effects, stdlib only.

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BanditState:
    """Posterior state for one bandit arm (model).

    mu and sigma define a Normal posterior. n_obs, sum_reward, and
    sum_sq_reward enable incremental updates without replaying history.
    effective_n and effective_sum track discounted counts for sliding-window
    behavior (gamma < 1.0). When effective_n is 0.0, legacy mode uses n_obs.
    """

    mu: float = 0.5
    sigma: float = 0.25
    n_obs: int = 0
    sum_reward: float = 0.0
    sum_sq_reward: float = 0.0
    effective_n: float = 0.0
    effective_sum: float = 0.0


# Minimum sigma to prevent posterior collapse and preserve exploration.
_SIGMA_FLOOR = 0.01

# Prior sigma used to scale posterior contraction: sigma = PRIOR_SIGMA / sqrt(n).
_PRIOR_SIGMA = 0.25

# Default discount factor for sliding-window posterior updates.
# gamma=0.95 gives a half-life of ~14 observations: weight halves every 14 new obs.
# gamma=1.0 disables discounting (equivalent to pre-S47 behavior).
_DEFAULT_GAMMA = 0.95


def thompson_sample(
    states: dict[str, BanditState],
    rng: random.Random | None = None,
) -> str:
    """Sample from each model's posterior, return the model with the highest draw.

    Args:
        states: Mapping of model name to BanditState posterior.
        rng: Optional seeded Random for deterministic testing.

    Returns:
        Name of the model with the highest Thompson sample.

    Raises:
        ValueError: If states is empty.
    """
    if not states:
        raise ValueError("Cannot sample from empty states dict")

    if len(states) == 1:
        return next(iter(states))

    _gauss = rng.gauss if rng else random.gauss
    best_name = ""
    best_draw = float("-inf")

    for name, state in states.items():
        draw = _gauss(state.mu, state.sigma)
        if draw > best_draw:
            best_draw = draw
            best_name = name

    return best_name


def update_posterior(
    state: BanditState,
    reward: float,
    gamma: float = _DEFAULT_GAMMA,
) -> BanditState:
    """Apply a single reward observation to the posterior with exponential discount.

    Recent observations are weighted more heavily than old ones. The discount
    factor gamma controls the decay rate: gamma=0.95 halves old weight every
    ~14 observations. gamma=1.0 disables discounting (equal weighting).

    Args:
        state: Current posterior state.
        reward: Observed reward (typically accuracy_score in [0, 1]).
        gamma: Discount factor in (0, 1]. Lower = faster forgetting.

    Returns:
        New BanditState with updated posterior.
    """
    # Raw counters for diagnostics (unweighted totals).
    new_n = state.n_obs + 1
    new_sum = state.sum_reward + reward
    new_sum_sq = state.sum_sq_reward + reward * reward

    # Discounted counters drive the posterior.
    # Legacy states (effective_n == 0.0) bootstrap from raw counters.
    old_eff_n = state.effective_n if state.effective_n > 0 else float(state.n_obs)
    old_eff_sum = state.effective_sum if state.effective_n > 0 else state.sum_reward
    new_eff_n = gamma * old_eff_n + 1.0
    new_eff_sum = gamma * old_eff_sum + reward

    new_mu = new_eff_sum / new_eff_n
    new_sigma = max(_SIGMA_FLOOR, _PRIOR_SIGMA / math.sqrt(new_eff_n))

    return BanditState(
        mu=new_mu,
        sigma=new_sigma,
        n_obs=new_n,
        sum_reward=new_sum,
        sum_sq_reward=new_sum_sq,
        effective_n=new_eff_n,
        effective_sum=new_eff_sum,
    )


def apply_cost_to_bandit(
    states: dict[str, BanditState],
    registry_costs: dict[str, float],
) -> dict[str, BanditState]:
    """Adjust bandit mu values by cost efficiency before sampling.

    Cheap models keep their full mu; expensive models get mu scaled down
    proportionally. Sigma is unchanged so expensive models still get explored.

    Args:
        states: Mapping of model name to BanditState.
        registry_costs: Mapping of model name to total cost (in + out per Mtok).

    Returns:
        New dict with cost-adjusted BanditState objects.
    """
    if not states or not registry_costs:
        return dict(states)

    costs_for_models = {
        name: registry_costs[name]
        for name in states
        if name in registry_costs and registry_costs[name] > 0
    }
    if not costs_for_models:
        return dict(states)

    cheapest = min(costs_for_models.values())

    adjusted = {}
    for name, state in states.items():
        cost = costs_for_models.get(name)
        if cost is None or cost <= 0:
            adjusted[name] = state
        else:
            ratio = cheapest / cost
            adjusted[name] = BanditState(
                mu=state.mu * ratio,
                sigma=state.sigma,
                n_obs=state.n_obs,
                sum_reward=state.sum_reward,
                sum_sq_reward=state.sum_sq_reward,
                effective_n=state.effective_n,
                effective_sum=state.effective_sum,
            )
    return adjusted


def default_priors(model_names: list[str]) -> dict[str, BanditState]:
    """Create uninformative priors for a list of models.

    Args:
        model_names: List of model name strings.

    Returns:
        Dict mapping each name to a default BanditState (mu=0.5, sigma=0.25).
    """
    return {name: BanditState() for name in model_names}


# ---------------------------------------------------------------------------
# Contextual Thompson Sampling
# ---------------------------------------------------------------------------

# Minimum observations before trusting a context-specific posterior.
_CONTEXT_MIN_OBS = 5


@dataclass(frozen=True)
class RoutingContext:
    """18-dimension context vector for contextual Thompson Sampling.

    Buckets the full context into a tractable key via signal_density,
    severity_band, time_regime, and precursor_risk_band -- yielding
    7 x 3 x 3 x 2 x 3 = 378 buckets.
    """

    # 9 boolean signals from enrichment pipeline
    correlated_alerts: bool = False
    change_correlation: bool = False
    metric_trend: bool = False
    blast_radius: bool = False
    not_in_maintenance: bool = True
    sustained_alert: bool = True
    precursor_warning: bool = False
    ttm_forecast: bool = False
    tspulse_anomaly: bool = False
    # Derived / scalar dimensions
    agreement_level: int = 0
    confidence_score: float = 0.5
    domain: str = "infrastructure"
    budget_action: str = ""
    severity: float = 0.5
    hour_of_day: int = 12
    # Continuous Precursor signals (extracted from predict_lookup response)
    precursor_confidence: float = 0.0
    precursor_cluster_id: int = 0
    precursor_priority: int = 0
    # Raw continuous signals from Precursor forecast/drift/anomaly endpoints.
    # Carried for logging, LangSmith, and downstream learning. Not folded into
    # context_key() -- bucket cardinality stays at 378.
    forecast_slope: float | None = None
    forecast_uncertainty: float | None = None
    drift_psi: float | None = None
    anomaly_score: float | None = None

    def signal_density(self) -> str:
        """Collapse 9 booleans into low/mid/high (0-2/3-5/6-9)."""
        count = sum(
            [
                self.correlated_alerts,
                self.change_correlation,
                self.metric_trend,
                self.blast_radius,
                self.not_in_maintenance,
                self.sustained_alert,
                self.precursor_warning,
                self.ttm_forecast,
                self.tspulse_anomaly,
            ]
        )
        if count <= 2:
            return "low"
        if count <= 5:
            return "mid"
        return "high"

    def severity_band(self) -> str:
        """Collapse severity float into routine/moderate/elevated."""
        if self.severity <= 0.3:
            return "routine"
        if self.severity <= 0.6:
            return "moderate"
        return "elevated"

    def time_regime(self) -> str:
        """Collapse hour into business(8-20)/offhours."""
        if 8 <= self.hour_of_day <= 20:
            return "business"
        return "offhours"

    def precursor_risk_band(self) -> str:
        """Collapse Precursor cluster into none/watch/critical risk bands.

        Cluster IDs: 0=NORMAL, 1=PRE_SCALE, 2=PRE_FAILURE,
        3=ACTIVE_DEGRADATION, 4=ANOMALY.
        """
        if self.precursor_cluster_id in (2, 3):
            return "critical"
        if self.precursor_cluster_id in (1, 4):
            return "watch"
        return "none"

    def context_key(self) -> str:
        """Deterministic bucket key for posterior lookup."""
        return (
            f"{self.domain}:{self.signal_density()}"
            f":{self.severity_band()}:{self.time_regime()}"
            f":{self.precursor_risk_band()}"
        )


def contextual_thompson_sample(
    context_states: dict[str, dict[str, BanditState]],
    domain_states: dict[str, BanditState],
    context: RoutingContext,
    rng: random.Random | None = None,
) -> str:
    """Thompson Sampling with context-aware posterior fallback.

    Three-level fallback hierarchy:
      1. Context-specific bucket if n_obs >= _CONTEXT_MIN_OBS
      2. Domain-level posterior (existing bandit_state data)
      3. Default uninformative prior (mu=0.5, sigma=0.25)

    Args:
        context_states: {model_name: {context_key: BanditState}}.
        domain_states: {model_name: BanditState} -- fallback posteriors.
        context: RoutingContext for the current alert.
        rng: Optional seeded Random for deterministic testing.

    Returns:
        Name of the model with the highest Thompson sample.

    Raises:
        ValueError: If both context_states and domain_states are empty.
    """
    all_models = set(domain_states) | set(context_states)
    if not all_models:
        raise ValueError("Cannot sample: no models in context_states or domain_states")

    key = context.context_key()
    merged: dict[str, BanditState] = {}

    for model in all_models:
        ctx_per_key = context_states.get(model, {})
        ctx_state = ctx_per_key.get(key)

        if ctx_state is not None and ctx_state.n_obs >= _CONTEXT_MIN_OBS:
            merged[model] = ctx_state
        elif model in domain_states:
            merged[model] = domain_states[model]
        else:
            merged[model] = BanditState()

    return thompson_sample(merged, rng=rng)
