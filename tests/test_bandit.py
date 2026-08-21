# Description: Tests for the Thompson Sampling bandit algorithm module.
# Description: Covers posterior updates, sampling, cost adjustment, and edge cases.

from __future__ import annotations

import random

import pytest

from pdp_router._bandit import (
    _CONTEXT_MIN_OBS,
    _DEFAULT_GAMMA,
    _SIGMA_FLOOR,
    BanditState,
    RoutingContext,
    apply_cost_to_bandit,
    contextual_thompson_sample,
    default_priors,
    thompson_sample,
    update_posterior,
)


class TestBanditState:
    def test_default_values(self) -> None:
        state = BanditState()
        assert state.mu == 0.5
        assert state.sigma == 0.25
        assert state.n_obs == 0
        assert state.sum_reward == 0.0
        assert state.sum_sq_reward == 0.0
        assert state.effective_n == 0.0
        assert state.effective_sum == 0.0

    def test_frozen(self) -> None:
        state = BanditState()
        with pytest.raises(AttributeError):
            state.mu = 0.9  # type: ignore[misc]


class TestUpdatePosterior:
    def test_single_update_shifts_mu(self) -> None:
        state = BanditState()
        updated = update_posterior(state, reward=1.0)
        assert updated.mu == 1.0
        assert updated.n_obs == 1
        assert updated.sum_reward == 1.0

    def test_sigma_shrinks_after_two_observations(self) -> None:
        state = BanditState()
        # After 1 obs: effective_n=1.0, sigma = 0.25/sqrt(1) = 0.25
        state = update_posterior(state, reward=0.8)
        assert state.sigma == pytest.approx(0.25)
        # After 2 obs: effective_n = 0.95*1 + 1 = 1.95, sigma shrinks
        state = update_posterior(state, reward=0.7)
        assert state.sigma < 0.25
        import math

        expected_eff_n = _DEFAULT_GAMMA * 1.0 + 1.0  # 1.95
        assert state.sigma == pytest.approx(0.25 / math.sqrt(expected_eff_n), abs=0.001)

    def test_zero_reward_pulls_mu_down(self) -> None:
        state = BanditState()
        updated = update_posterior(state, reward=0.0)
        assert updated.mu == 0.0
        assert updated.n_obs == 1

    def test_many_updates_converge(self) -> None:
        state = BanditState()
        for _ in range(100):
            state = update_posterior(state, reward=0.8)
        assert state.mu == pytest.approx(0.8, abs=0.01)
        assert state.n_obs == 100

    def test_sigma_floor(self) -> None:
        state = BanditState()
        for _ in range(10000):
            state = update_posterior(state, reward=0.7)
        assert state.sigma >= _SIGMA_FLOOR

    def test_running_sums_accumulate(self) -> None:
        state = BanditState()
        state = update_posterior(state, reward=0.6)
        state = update_posterior(state, reward=0.8)
        assert state.sum_reward == pytest.approx(1.4)
        assert state.sum_sq_reward == pytest.approx(0.36 + 0.64)
        assert state.n_obs == 2
        # With gamma=0.95: effective_sum = 0.95*0.6 + 0.8 = 1.37
        #                   effective_n  = 0.95*1.0 + 1.0 = 1.95
        #                   mu = 1.37/1.95 ~ 0.70256
        assert state.mu == pytest.approx(
            (_DEFAULT_GAMMA * 0.6 + 0.8) / (_DEFAULT_GAMMA * 1.0 + 1.0),
            abs=0.001,
        )

    def test_sigma_decreases_with_observations(self) -> None:
        state = BanditState()
        sigmas = [state.sigma]
        for _ in range(20):
            state = update_posterior(state, reward=0.5)
            sigmas.append(state.sigma)
        # Sigma should monotonically decrease
        for i in range(1, len(sigmas)):
            assert sigmas[i] <= sigmas[i - 1]


class TestThompsonSample:
    def test_deterministic_with_seed(self) -> None:
        states = default_priors(["a", "b", "c"])
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert thompson_sample(states, rng=rng1) == thompson_sample(states, rng=rng2)

    def test_high_mu_wins_most(self) -> None:
        states = {
            "strong": BanditState(mu=0.9, sigma=0.01),
            "weak": BanditState(mu=0.5, sigma=0.25),
        }
        rng = random.Random(123)
        wins = sum(1 for _ in range(1000) if thompson_sample(states, rng=rng) == "strong")
        assert wins > 900

    def test_high_uncertainty_sometimes_wins(self) -> None:
        states = {
            "certain": BanditState(mu=0.6, sigma=0.01),
            "uncertain": BanditState(mu=0.3, sigma=0.5),
        }
        rng = random.Random(456)
        uncertain_wins = sum(
            1 for _ in range(1000) if thompson_sample(states, rng=rng) == "uncertain"
        )
        # Wide sigma should win sometimes (exploration), but not most of the time
        assert 50 < uncertain_wins < 500

    def test_empty_states_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            thompson_sample({})

    def test_single_model_returns_it(self) -> None:
        states = {"only": BanditState()}
        assert thompson_sample(states) == "only"

    def test_all_models_sampled_over_many_draws(self) -> None:
        names = ["a", "b", "c", "d"]
        states = default_priors(names)
        rng = random.Random(789)
        seen = {thompson_sample(states, rng=rng) for _ in range(1000)}
        assert seen == set(names)


class TestApplyCostToBandit:
    def test_cheap_model_unchanged(self) -> None:
        states = {
            "cheap": BanditState(mu=0.7),
            "expensive": BanditState(mu=0.7),
        }
        costs = {"cheap": 1.0, "expensive": 10.0}
        adjusted = apply_cost_to_bandit(states, costs)
        # Cheapest model keeps full mu
        assert adjusted["cheap"].mu == 0.7
        # Expensive model gets mu scaled down by 1/10
        assert adjusted["expensive"].mu == pytest.approx(0.07)

    def test_sigma_preserved(self) -> None:
        states = {"a": BanditState(mu=0.5, sigma=0.15)}
        costs = {"a": 5.0}
        adjusted = apply_cost_to_bandit(states, costs)
        assert adjusted["a"].sigma == 0.15

    def test_empty_states(self) -> None:
        assert apply_cost_to_bandit({}, {"a": 1.0}) == {}

    def test_empty_costs(self) -> None:
        states = {"a": BanditState()}
        result = apply_cost_to_bandit(states, {})
        assert result["a"].mu == 0.5

    def test_model_not_in_costs_passthrough(self) -> None:
        states = {"known": BanditState(mu=0.8), "unknown": BanditState(mu=0.6)}
        costs = {"known": 2.0}
        adjusted = apply_cost_to_bandit(states, costs)
        # Only model in costs is cheapest by default
        assert adjusted["known"].mu == 0.8  # cheapest (only one)
        assert adjusted["unknown"].mu == 0.6  # passthrough


class TestDefaultPriors:
    def test_creates_priors(self) -> None:
        priors = default_priors(["a", "b"])
        assert len(priors) == 2
        assert priors["a"].mu == 0.5
        assert priors["b"].sigma == 0.25

    def test_empty_list(self) -> None:
        assert default_priors([]) == {}


class TestRoutingContext:
    def test_defaults(self) -> None:
        ctx = RoutingContext()
        assert ctx.domain == "infrastructure"
        assert ctx.severity == 0.5
        assert ctx.hour_of_day == 12

    def test_frozen(self) -> None:
        ctx = RoutingContext()
        with pytest.raises(AttributeError):
            ctx.domain = "network"  # type: ignore[misc]

    def test_context_key_deterministic(self) -> None:
        ctx = RoutingContext(domain="kubernetes", severity=0.8, hour_of_day=14)
        assert ctx.context_key() == ctx.context_key()

    def test_context_key_format(self) -> None:
        ctx = RoutingContext(
            domain="network",
            severity=0.8,
            hour_of_day=14,
            correlated_alerts=True,
            change_correlation=True,
            metric_trend=True,
            blast_radius=True,
            not_in_maintenance=True,
            sustained_alert=True,
        )
        key = ctx.context_key()
        parts = key.split(":")
        assert len(parts) == 5
        assert parts[0] == "network"
        assert parts[4] == "none"  # default precursor_risk_band

    # signal_density boundaries
    def test_signal_density_low_0(self) -> None:
        ctx = RoutingContext(
            correlated_alerts=False,
            change_correlation=False,
            metric_trend=False,
            blast_radius=False,
            not_in_maintenance=False,
            sustained_alert=False,
            precursor_warning=False,
            ttm_forecast=False,
            tspulse_anomaly=False,
        )
        assert ctx.signal_density() == "low"

    def test_signal_density_low_2(self) -> None:
        # Explicitly set all to False, then enable 2
        ctx = RoutingContext(
            correlated_alerts=True,
            change_correlation=True,
            not_in_maintenance=False,
            sustained_alert=False,
        )
        assert ctx.signal_density() == "low"

    def test_signal_density_mid_3(self) -> None:
        ctx = RoutingContext(
            correlated_alerts=True,
            change_correlation=True,
            metric_trend=True,
            not_in_maintenance=False,
            sustained_alert=False,
        )
        assert ctx.signal_density() == "mid"

    def test_signal_density_mid_5(self) -> None:
        ctx = RoutingContext(
            correlated_alerts=True,
            change_correlation=True,
            metric_trend=True,
            blast_radius=True,
            precursor_warning=True,
            not_in_maintenance=False,
            sustained_alert=False,
        )
        assert ctx.signal_density() == "mid"

    def test_signal_density_high_6(self) -> None:
        ctx = RoutingContext(
            correlated_alerts=True,
            change_correlation=True,
            metric_trend=True,
            blast_radius=True,
            not_in_maintenance=True,
            sustained_alert=True,
        )
        assert ctx.signal_density() == "high"

    def test_signal_density_high_9(self) -> None:
        ctx = RoutingContext(
            correlated_alerts=True,
            change_correlation=True,
            metric_trend=True,
            blast_radius=True,
            not_in_maintenance=True,
            sustained_alert=True,
            precursor_warning=True,
            ttm_forecast=True,
            tspulse_anomaly=True,
        )
        assert ctx.signal_density() == "high"

    # severity_band boundaries
    def test_severity_routine_at_boundary(self) -> None:
        assert RoutingContext(severity=0.3).severity_band() == "routine"

    def test_severity_moderate_above_boundary(self) -> None:
        assert RoutingContext(severity=0.31).severity_band() == "moderate"

    def test_severity_moderate_at_upper(self) -> None:
        assert RoutingContext(severity=0.6).severity_band() == "moderate"

    def test_severity_elevated_above(self) -> None:
        assert RoutingContext(severity=0.61).severity_band() == "elevated"

    # time_regime boundaries
    def test_time_offhours_7(self) -> None:
        assert RoutingContext(hour_of_day=7).time_regime() == "offhours"

    def test_time_business_8(self) -> None:
        assert RoutingContext(hour_of_day=8).time_regime() == "business"

    def test_time_business_20(self) -> None:
        assert RoutingContext(hour_of_day=20).time_regime() == "business"

    def test_time_offhours_21(self) -> None:
        assert RoutingContext(hour_of_day=21).time_regime() == "offhours"


class TestContextualThompsonSample:
    def test_uses_context_state_when_sufficient_obs(self) -> None:
        ctx = RoutingContext(domain="compute", severity=0.8, hour_of_day=10)
        key = ctx.context_key()
        context_states = {
            "strong": {key: BanditState(mu=0.95, sigma=0.01, n_obs=10)},
            "weak": {key: BanditState(mu=0.1, sigma=0.01, n_obs=10)},
        }
        domain_states = default_priors(["strong", "weak"])
        rng = random.Random(42)
        wins = sum(
            1
            for _ in range(100)
            if contextual_thompson_sample(context_states, domain_states, ctx, rng=rng) == "strong"
        )
        assert wins > 90

    def test_falls_back_to_domain_when_insufficient_obs(self) -> None:
        ctx = RoutingContext(domain="network")
        key = ctx.context_key()
        # Context state has too few observations
        context_states = {
            "a": {key: BanditState(mu=0.1, sigma=0.01, n_obs=_CONTEXT_MIN_OBS - 1)},
        }
        domain_states = {"a": BanditState(mu=0.9, sigma=0.01, n_obs=50)}
        result = contextual_thompson_sample(context_states, domain_states, ctx)
        # Should use domain state (mu=0.9) not context state (mu=0.1)
        assert result == "a"

    def test_falls_back_to_default_prior(self) -> None:
        ctx = RoutingContext(domain="storage")
        context_states: dict[str, dict[str, BanditState]] = {"a": {}}
        domain_states: dict[str, BanditState] = {}
        # Model "a" in context_states but no matching key and no domain state
        result = contextual_thompson_sample(context_states, domain_states, ctx)
        assert result == "a"

    def test_deterministic_with_seed(self) -> None:
        ctx = RoutingContext(domain="cloud", severity=0.5)
        domain_states = default_priors(["x", "y", "z"])
        context_states: dict[str, dict[str, BanditState]] = {}
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        r1 = contextual_thompson_sample(context_states, domain_states, ctx, rng=rng1)
        r2 = contextual_thompson_sample(context_states, domain_states, ctx, rng=rng2)
        assert r1 == r2

    def test_empty_both_raises(self) -> None:
        ctx = RoutingContext()
        with pytest.raises(ValueError, match="no models"):
            contextual_thompson_sample({}, {}, ctx)

    def test_models_from_both_dicts_merged(self) -> None:
        ctx = RoutingContext(domain="application")
        context_states = {"ctx_only": {}}
        domain_states = {"dom_only": BanditState(mu=0.8, sigma=0.01)}
        # Both models should be considered
        rng = random.Random(42)
        seen = set()
        for _ in range(200):
            seen.add(contextual_thompson_sample(context_states, domain_states, ctx, rng=rng))
        assert "dom_only" in seen
        assert "ctx_only" in seen

    def test_context_key_mismatch_falls_back(self) -> None:
        ctx = RoutingContext(domain="compute", severity=0.8)
        wrong_key = "network:low:routine:offhours:none"
        context_states = {
            "a": {wrong_key: BanditState(mu=0.1, sigma=0.01, n_obs=100)},
        }
        domain_states = {"a": BanditState(mu=0.9, sigma=0.01, n_obs=50)}
        # Wrong key should not match, falls back to domain state
        result = contextual_thompson_sample(context_states, domain_states, ctx)
        assert result == "a"


class TestDiscountedPosterior:
    """Tests for sliding-window TS with gamma discount factor."""

    def test_discount_reduces_old_influence(self) -> None:
        """10 high rewards then 10 low: mu should be pulled toward recent data."""
        state = BanditState()
        for _ in range(10):
            state = update_posterior(state, reward=0.9)
        for _ in range(10):
            state = update_posterior(state, reward=0.1)
        # With discount, mu should be closer to 0.1 than the midpoint 0.5
        assert state.mu < 0.4

    def test_gamma_one_matches_legacy_mu(self) -> None:
        """gamma=1.0 should produce the same mu as simple mean."""
        state = BanditState()
        rewards = [0.6, 0.8, 0.4, 0.9, 0.7]
        for r in rewards:
            state = update_posterior(state, r, gamma=1.0)
        expected_mu = sum(rewards) / len(rewards)
        assert state.mu == pytest.approx(expected_mu, abs=0.001)

    def test_effective_n_tracks_discounted_count(self) -> None:
        """effective_n should be less than n_obs when gamma < 1.0."""
        state = BanditState()
        for _ in range(20):
            state = update_posterior(state, reward=0.5)
        assert state.n_obs == 20
        assert state.effective_n < 20.0
        assert state.effective_n > 0.0

    def test_sigma_uses_effective_n(self) -> None:
        """Sigma should contract based on effective_n, not raw n_obs."""
        import math

        state = BanditState()
        for _ in range(10):
            state = update_posterior(state, reward=0.5)
        expected_sigma = max(_SIGMA_FLOOR, 0.25 / math.sqrt(state.effective_n))
        assert state.sigma == pytest.approx(expected_sigma)

    def test_backwards_compat_zero_effective(self) -> None:
        """Legacy BanditState (effective_n=0) bootstraps from n_obs on first update."""
        legacy = BanditState(mu=0.7, sigma=0.15, n_obs=10, sum_reward=7.0)
        updated = update_posterior(legacy, reward=0.8)
        # Should bootstrap: old_eff_n=10.0 (from n_obs), old_eff_sum=7.0
        expected_eff_n = _DEFAULT_GAMMA * 10.0 + 1.0
        expected_eff_sum = _DEFAULT_GAMMA * 7.0 + 0.8
        assert updated.effective_n == pytest.approx(expected_eff_n)
        assert updated.effective_sum == pytest.approx(expected_eff_sum)


class TestPrecursorRiskBand:
    """Tests for the precursor_risk_band bucket dimension."""

    def test_normal_returns_none(self) -> None:
        ctx = RoutingContext(precursor_cluster_id=0)
        assert ctx.precursor_risk_band() == "none"

    def test_pre_scale_returns_watch(self) -> None:
        ctx = RoutingContext(precursor_cluster_id=1)
        assert ctx.precursor_risk_band() == "watch"

    def test_pre_failure_returns_critical(self) -> None:
        ctx = RoutingContext(precursor_cluster_id=2)
        assert ctx.precursor_risk_band() == "critical"

    def test_active_degradation_returns_critical(self) -> None:
        ctx = RoutingContext(precursor_cluster_id=3)
        assert ctx.precursor_risk_band() == "critical"

    def test_anomaly_returns_watch(self) -> None:
        ctx = RoutingContext(precursor_cluster_id=4)
        assert ctx.precursor_risk_band() == "watch"

    def test_context_key_includes_risk_band(self) -> None:
        ctx = RoutingContext(precursor_cluster_id=2)
        assert ctx.context_key().endswith(":critical")

    def test_default_precursor_fields(self) -> None:
        ctx = RoutingContext()
        assert ctx.precursor_confidence == 0.0
        assert ctx.precursor_cluster_id == 0
        assert ctx.precursor_priority == 0

    def test_signal_density_unchanged_by_precursor_fields(self) -> None:
        """Continuous precursor fields should not affect the 9-boolean signal_density."""
        base = RoutingContext(not_in_maintenance=False, sustained_alert=False)
        with_precursor = RoutingContext(
            not_in_maintenance=False,
            sustained_alert=False,
            precursor_confidence=0.95,
            precursor_cluster_id=3,
            precursor_priority=5,
        )
        assert base.signal_density() == with_precursor.signal_density()


class TestPrecursorRawSignals:
    """Tests for forecast_slope, forecast_uncertainty, drift_psi, anomaly_score.

    These fields ride along on RoutingContext for logging and downstream
    learning but must NOT affect the bucket key or signal_density count.
    """

    def test_new_fields_default_to_none(self) -> None:
        ctx = RoutingContext()
        assert ctx.forecast_slope is None
        assert ctx.forecast_uncertainty is None
        assert ctx.drift_psi is None
        assert ctx.anomaly_score is None

    def test_accepts_float_values_without_typeerror(self) -> None:
        ctx = RoutingContext(
            forecast_slope=0.04,
            forecast_uncertainty=0.22,
            drift_psi=0.31,
            anomaly_score=0.78,
        )
        assert ctx.forecast_slope == 0.04
        assert ctx.drift_psi == 0.31
        assert ctx.anomaly_score == 0.78

    def test_context_key_ignores_new_float_fields(self) -> None:
        """Two contexts differing only in new floats must produce same bucket key."""
        base = RoutingContext(domain="infrastructure", severity=0.5, hour_of_day=14)
        with_floats = RoutingContext(
            domain="infrastructure",
            severity=0.5,
            hour_of_day=14,
            drift_psi=0.9,
            anomaly_score=0.85,
            forecast_slope=-0.5,
            forecast_uncertainty=0.95,
        )
        assert base.context_key() == with_floats.context_key()

    def test_context_key_still_five_parts(self) -> None:
        """Bucket key format is domain:density:severity:time:risk -- 4 colons."""
        ctx = RoutingContext(drift_psi=0.3, anomaly_score=0.7)
        assert ctx.context_key().count(":") == 4

    def test_signal_density_ignores_new_fields(self) -> None:
        base = RoutingContext(not_in_maintenance=False, sustained_alert=False)
        with_floats = RoutingContext(
            not_in_maintenance=False,
            sustained_alert=False,
            drift_psi=0.9,
            anomaly_score=0.85,
            forecast_slope=0.5,
            forecast_uncertainty=0.3,
        )
        assert base.signal_density() == with_floats.signal_density()


class TestProbBeats:
    def test_symmetry(self) -> None:
        from pdp_router._bandit import prob_beats

        a = BanditState(mu=0.7, sigma=0.05, n_obs=50)
        b = BanditState(mu=0.6, sigma=0.05, n_obs=50)
        assert prob_beats(a, b) + prob_beats(b, a) == pytest.approx(1.0)
        assert prob_beats(a, b) > 0.5

    def test_equal_posteriors_are_a_coin_flip(self) -> None:
        from pdp_router._bandit import prob_beats

        a = BanditState(mu=0.5, sigma=0.1)
        assert prob_beats(a, a) == pytest.approx(0.5)

    def test_wide_sigma_dilutes_a_mu_edge(self) -> None:
        # The same mean gap is less convincing from a noisy posterior.
        from pdp_router._bandit import prob_beats

        baseline = BanditState(mu=0.5, sigma=0.05)
        sharp = BanditState(mu=0.6, sigma=0.02)
        noisy = BanditState(mu=0.6, sigma=0.25)
        assert prob_beats(sharp, baseline) > prob_beats(noisy, baseline)

    def test_zero_variance_falls_back_to_mu_comparison(self) -> None:
        from pdp_router._bandit import prob_beats

        hi = BanditState(mu=0.9, sigma=0.0)
        lo = BanditState(mu=0.1, sigma=0.0)
        assert prob_beats(hi, lo) == 1.0
        assert prob_beats(lo, hi) == 0.0
        assert prob_beats(hi, hi) == 0.5


class TestEligibleArms:
    _BASELINE = "cheap-model"

    def _states(self) -> dict:
        return {
            self._BASELINE: BanditState(mu=0.60, sigma=0.02, n_obs=200),
            "clear-winner": BanditState(mu=0.80, sigma=0.02, n_obs=100),
            "marginal": BanditState(mu=0.61, sigma=0.05, n_obs=100),
            "young-star": BanditState(mu=0.95, sigma=0.02, n_obs=3),
            "loser": BanditState(mu=0.40, sigma=0.02, n_obs=100),
        }

    def test_only_demonstrated_uplift_survives(self) -> None:
        from pdp_router._bandit import eligible_arms

        kept = eligible_arms(self._states(), self._BASELINE)
        assert set(kept) == {self._BASELINE, "clear-winner"}

    def test_baseline_always_retained(self) -> None:
        from pdp_router._bandit import eligible_arms

        states = {
            self._BASELINE: BanditState(mu=0.9, sigma=0.02, n_obs=200),
            "loser": BanditState(mu=0.1, sigma=0.02, n_obs=100),
        }
        kept = eligible_arms(states, self._BASELINE)
        assert set(kept) == {self._BASELINE}

    def test_min_obs_blocks_a_lucky_prior(self) -> None:
        from pdp_router._bandit import eligible_arms

        kept = eligible_arms(self._states(), self._BASELINE, min_obs=10)
        assert "young-star" not in kept
        kept_low_bar = eligible_arms(self._states(), self._BASELINE, min_obs=2)
        assert "young-star" in kept_low_bar

    def test_missing_baseline_returns_states_unchanged(self) -> None:
        from pdp_router._bandit import eligible_arms

        states = self._states()
        kept = eligible_arms(states, "not-a-model")
        assert kept == states
        assert kept is not states  # copy, not the caller's dict

    def test_min_prob_is_the_bar(self) -> None:
        from pdp_router._bandit import eligible_arms, prob_beats

        states = self._states()
        p = prob_beats(states["marginal"], states[self._BASELINE])
        assert "marginal" in eligible_arms(states, self._BASELINE, min_prob=p - 0.01)
        assert "marginal" not in eligible_arms(states, self._BASELINE, min_prob=p + 0.01)
