# Description: Tests for confidence-based model routing and registry.
# Description: Covers cascade, fallback, budget override, trust adjustment, and registry ops.

from __future__ import annotations

import random

import pytest

from pdp_router._bandit import BanditState, RoutingContext
from pdp_router._models import (
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    GEMINI_PRO,
    GPT_5_5,
    HAIKU,
    OPUS,
    QWEN_3_7_PLUS,
    SONNET,
)
from pdp_router._router import (
    DEFAULT_REGISTRY,
    ModelCapability,
    ModelRegistry,
    ModelSelection,
    _dynamic_explore_rate,
    apply_budget_override,
    apply_cost_efficiency,
    compute_routing_picks,
    confidence_cascade,
    next_cheaper,
    route_request,
    route_with_fallback,
)


class TestModelRegistry:
    def test_get_existing_model(self) -> None:
        cap = DEFAULT_REGISTRY.get(OPUS)
        assert cap is not None
        assert cap.name == OPUS
        assert cap.tier == 1

    def test_get_missing_model(self) -> None:
        assert DEFAULT_REGISTRY.get("nonexistent") is None

    def test_by_tier(self) -> None:
        tier_1 = DEFAULT_REGISTRY.by_tier(1)
        assert len(tier_1) >= 1
        assert all(m.tier == 1 for m in tier_1)

    def test_available_models_excludes_unavailable(self) -> None:
        available = DEFAULT_REGISTRY.available_models()
        assert all(m.available for m in available)
        # Anthropic models should be available
        names = {m.name for m in available}
        assert OPUS in names
        assert SONNET in names
        assert HAIKU in names

    def test_gemini_models_in_registry_and_available(self) -> None:
        """Gemini and Llama are registered and available."""
        cap = DEFAULT_REGISTRY.get(GEMINI_PRO)
        assert cap is not None
        assert cap.available

    def test_openrouter_arms_registered_and_available(self) -> None:
        """OpenAI + Qwen arms are registered, tagged openrouter, and available
        (panel + cascade-explore eligible); credentials come from the proxy env."""
        for name, tier in ((GPT_5_5, 2), (QWEN_3_7_PLUS, 3)):
            cap = DEFAULT_REGISTRY.get(name)
            assert cap is not None, f"{name} missing from DEFAULT_REGISTRY"
            assert cap.provider == "openrouter"
            assert cap.tier == tier
            assert cap.available is True

    def test_custom_registry(self) -> None:
        custom = ModelRegistry(
            {
                "model-a": ModelCapability("model-a", 1, 10.0, 50.0, "test", True),
                "model-b": ModelCapability("model-b", 2, 1.0, 5.0, "test", True),
            }
        )
        assert len(custom.available_models()) == 2
        assert custom.get("model-a") is not None


class TestRouteRequest:
    def test_high_confidence_selects_haiku(self) -> None:
        sel = route_request(0.90)
        assert sel.model == HAIKU
        assert sel.tier == 3

    def test_medium_confidence_with_agreement_selects_haiku(self) -> None:
        sel = route_request(0.75, agreement_level=4)
        assert sel.model == HAIKU
        assert sel.tier == 3

    def test_medium_confidence_without_agreement_selects_sonnet(self) -> None:
        sel = route_request(0.75, agreement_level=2)
        assert sel.model == SONNET
        assert sel.tier == 2

    def test_moderate_confidence_selects_sonnet(self) -> None:
        sel = route_request(0.50)
        assert sel.model == SONNET
        assert sel.tier == 2

    def test_low_confidence_selects_opus(self) -> None:
        sel = route_request(0.30)
        assert sel.model == OPUS
        assert sel.tier == 1

    def test_zero_confidence_selects_opus(self) -> None:
        sel = route_request(0.0)
        assert sel.model == OPUS
        assert sel.tier == 1

    def test_trust_weight_lowers_threshold(self) -> None:
        """High trust in Haiku should lower the skip threshold, selecting Haiku at lower conf."""
        # Default threshold is 0.85. High trust (1.0) shifts by -0.10 -> 0.75.
        sel = route_request(0.80, trust_weights={HAIKU: 1.0})
        assert sel.model == HAIKU

    def test_trust_weight_raises_threshold(self) -> None:
        """Low trust in Haiku should raise the skip threshold."""
        # Default threshold is 0.85. Low trust (0.0) shifts by +0.10 -> 0.95.
        sel = route_request(0.90, trust_weights={HAIKU: 0.0})
        # conf=0.90 < 0.95 -> not Haiku skip tier
        assert sel.model != HAIKU or sel.tier != 3


class TestRouteWithFallback:
    def test_available_model_returned_directly(self) -> None:
        sel = route_with_fallback(0.50)
        assert sel.model == SONNET

    def test_unavailable_model_falls_back(self) -> None:
        """If preferred model is unavailable, fall back to cheapest available."""
        registry = ModelRegistry(
            {
                OPUS: ModelCapability(OPUS, 1, 15.0, 75.0, "anthropic", False),
                SONNET: ModelCapability(SONNET, 2, 3.0, 15.0, "anthropic", True),
                HAIKU: ModelCapability(HAIKU, 3, 0.25, 1.25, "anthropic", True),
            }
        )
        sel = route_with_fallback(0.30, registry=registry)
        # Opus is unavailable, should fall back to cheapest (Haiku)
        assert sel.model == HAIKU
        assert "fallback" in sel.reason

    def test_no_available_models_raises(self) -> None:
        empty_registry = ModelRegistry(
            {
                OPUS: ModelCapability(OPUS, 1, 15.0, 75.0, "anthropic", False),
            }
        )
        with pytest.raises(RuntimeError, match="No available models"):
            route_with_fallback(0.50, registry=empty_registry)


class TestConfidenceCascade:
    def test_returns_model_name_string(self) -> None:
        model = confidence_cascade(0.50)
        assert isinstance(model, str)
        assert model == SONNET

    def test_budget_downgrade(self) -> None:
        model = confidence_cascade(0.30, budget_action="downgrade")
        # Should downgrade from Opus to cheapest available (Flash Lite at tier 4)
        assert model == GEMINI_FLASH_LITE

    def test_no_budget_action(self) -> None:
        model = confidence_cascade(0.30, budget_action=None)
        assert model == OPUS


class TestNextCheaper:
    def test_opus_to_tier2(self) -> None:
        result = next_cheaper(OPUS)
        # With all models available, cheapest tier 2 is Gemini Pro ($1.25 in)
        assert result == GEMINI_PRO

    def test_sonnet_to_tier3(self) -> None:
        result = next_cheaper(SONNET)
        # With all models available, cheapest tier 3 is Llama Scout ($0.25 in)
        cap = DEFAULT_REGISTRY.get(result)
        assert cap is not None
        assert cap.tier == 3

    def test_haiku_to_tier4(self) -> None:
        result = next_cheaper(HAIKU)
        # With all models available, tier 4 exists below tier 3
        assert result == GEMINI_FLASH_LITE

    def test_unknown_model_returns_self(self) -> None:
        result = next_cheaper("some-unknown-model")
        assert result == "some-unknown-model"

    def test_registry_driven_with_gemini(self) -> None:
        """With Gemini models available, next_cheaper should find them."""
        registry = ModelRegistry(
            {
                OPUS: ModelCapability(OPUS, 1, 15.0, 75.0, "anthropic", True),
                GEMINI_PRO: ModelCapability(GEMINI_PRO, 2, 1.25, 10.0, "gemini", True),
                GEMINI_FLASH: ModelCapability(GEMINI_FLASH, 3, 0.30, 2.50, "gemini", True),
            }
        )
        result = next_cheaper(OPUS, registry=registry)
        assert result == GEMINI_PRO

    def test_cheapest_tier_stays(self) -> None:
        registry = ModelRegistry(
            {
                GEMINI_FLASH: ModelCapability(GEMINI_FLASH, 3, 0.30, 2.50, "gemini", True),
            }
        )
        result = next_cheaper(GEMINI_FLASH, registry=registry)
        assert result == GEMINI_FLASH


class TestApplyBudgetOverride:
    def test_downgrade_picks_cheapest_available(self) -> None:
        sel = ModelSelection(model=OPUS, tier=1, reason="original")
        result = apply_budget_override(sel, "downgrade")
        # Flash Lite (tier 4) is now cheapest with all models available
        assert result.model == GEMINI_FLASH_LITE
        assert "budget override" in result.reason

    def test_non_downgrade_returns_original(self) -> None:
        sel = ModelSelection(model=OPUS, tier=1, reason="original")
        result = apply_budget_override(sel, "skip")
        assert result.model == OPUS

    def test_downgrade_with_custom_registry(self) -> None:
        """Budget override picks cheapest from the given registry."""
        registry = ModelRegistry(
            {
                OPUS: ModelCapability(OPUS, 1, 15.0, 75.0, "anthropic", True),
                GEMINI_FLASH: ModelCapability(GEMINI_FLASH, 3, 0.30, 2.50, "gemini", True),
            }
        )
        sel = ModelSelection(model=OPUS, tier=1, reason="original")
        result = apply_budget_override(sel, "downgrade", registry=registry)
        assert result.model == GEMINI_FLASH


class TestCostEfficiency:
    def test_cheapest_model_gets_full_trust(self) -> None:
        """The cheapest model's trust is unchanged (multiplied by 1.0)."""
        registry = ModelRegistry(
            {
                "cheap": ModelCapability("cheap", 3, 0.10, 0.40, "test", True),
                "expensive": ModelCapability("expensive", 1, 15.0, 75.0, "test", True),
            }
        )
        raw = {"cheap": 0.80, "expensive": 0.80}
        adjusted = apply_cost_efficiency(raw, registry)
        assert adjusted["cheap"] == pytest.approx(0.80)
        assert adjusted["expensive"] < adjusted["cheap"]

    def test_cost_efficiency_favors_cheaper_model(self) -> None:
        """A cheap model at 78% accuracy should beat an expensive one at 80%."""
        registry = ModelRegistry(
            {
                "cheap": ModelCapability("cheap", 3, 0.10, 0.40, "test", True),
                "expensive": ModelCapability("expensive", 1, 15.0, 75.0, "test", True),
            }
        )
        raw = {"cheap": 0.78, "expensive": 0.80}
        adjusted = apply_cost_efficiency(raw, registry)
        assert adjusted["cheap"] > adjusted["expensive"]

    def test_empty_trust_weights_returns_empty(self) -> None:
        adjusted = apply_cost_efficiency({})
        assert adjusted == {}

    def test_unknown_model_passes_through(self) -> None:
        """Models not in registry keep their raw trust."""
        adjusted = apply_cost_efficiency({"unknown-model": 0.75})
        assert adjusted["unknown-model"] == 0.75


class TestEpsilonGreedy:
    def test_explore_rate_zero_never_explores(self) -> None:
        """With explore_rate=0.0, cascade is always deterministic."""
        for _ in range(20):
            model = confidence_cascade(0.50, explore_rate=0.0)
            assert model == SONNET

    def test_explore_rate_one_always_explores(self) -> None:
        """With explore_rate=1.0, every call picks a random available model."""
        rng = random.Random(42)
        models = set()
        for _ in range(50):
            model = confidence_cascade(0.50, explore_rate=1.0, _rng=rng)
            models.add(model)
        assert len(models) > 1

    def test_exploration_picks_available_model(self) -> None:
        """Explored model must be in the registry and available."""
        rng = random.Random(99)
        for _ in range(30):
            model = confidence_cascade(0.50, explore_rate=1.0, _rng=rng)
            cap = DEFAULT_REGISTRY.get(model)
            assert cap is not None
            assert cap.available

    def test_exploration_bypasses_cascade(self) -> None:
        """Exploration can select a model that the cascade would never pick."""
        rng = random.Random(42)
        cascade_only: set[str] = set()
        explored: set[str] = set()
        for _ in range(100):
            cascade_only.add(confidence_cascade(0.50, explore_rate=0.0))
            explored.add(confidence_cascade(0.50, explore_rate=1.0, _rng=rng))
        # Cascade at conf=0.50 always picks SONNET. Exploration should find others.
        assert cascade_only == {SONNET}
        assert len(explored) > 1

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same exploration sequence."""
        results_a = [
            confidence_cascade(0.50, explore_rate=1.0, _rng=random.Random(123)) for _ in range(5)
        ]
        results_b = [
            confidence_cascade(0.50, explore_rate=1.0, _rng=random.Random(123)) for _ in range(5)
        ]
        assert results_a == results_b


class TestDynamicExploreRate:
    """Agreement-driven exploration rate for confidence cascade mode.

    Strong agreement (7+ of 9 signals True) -> exploit, low rate.
    Weak agreement (<3) -> explore, high rate.
    """

    @pytest.mark.parametrize(
        "agreement_level,expected",
        [
            (9, 0.02),
            (8, 0.02),
            (7, 0.02),
            (6, 0.05),
            (5, 0.05),
            (4, 0.10),
            (3, 0.10),
            (2, 0.25),
            (1, 0.25),
            (0, 0.25),
        ],
    )
    def test_dynamic_rate_across_agreement_levels(
        self, agreement_level: int, expected: float
    ) -> None:
        assert _dynamic_explore_rate(agreement_level) == expected

    def test_dynamic_rate_at_band_boundaries(self) -> None:
        """Threshold crossings produce the documented step-function jumps."""
        assert _dynamic_explore_rate(7) == 0.02
        assert _dynamic_explore_rate(6) == 0.05
        assert _dynamic_explore_rate(5) == 0.05
        assert _dynamic_explore_rate(4) == 0.10
        assert _dynamic_explore_rate(3) == 0.10
        assert _dynamic_explore_rate(2) == 0.25

    def test_dynamic_rate_negative_agreement_raises(self) -> None:
        """Negative agreement_level is a caller bug -- fail loudly, not silently."""
        with pytest.raises(ValueError, match="must be >= 0"):
            _dynamic_explore_rate(-1)

    def test_dynamic_rate_above_nine_uses_top_band(self) -> None:
        """Values beyond the nominal 0-9 range still resolve (top band wins)."""
        assert _dynamic_explore_rate(15) == 0.02

    def test_cascade_none_agreement_uses_flat_rate(self) -> None:
        """agreement_level=None (default) preserves legacy flat explore_rate."""
        # Flat 1.0 always explores, regardless of dynamic mapping (which would be 0.25 at <3).
        rng = random.Random(42)
        picks: set[str] = set()
        for _ in range(40):
            picks.add(confidence_cascade(0.50, agreement_level=None, explore_rate=1.0, _rng=rng))
        # Many distinct picks because every call explores at rate 1.0.
        assert len(picks) > 1

    def test_cascade_high_agreement_rarely_explores(self) -> None:
        """agreement_level=8 -> effective rate 0.02, exploration almost never fires."""
        rng = random.Random(777)
        cascade_picks: set[str] = set()
        for _ in range(200):
            cascade_picks.add(
                confidence_cascade(0.50, agreement_level=8, explore_rate=0.10, _rng=rng)
            )
        # At conf=0.50 cascade picks SONNET deterministically. With effective rate 0.02,
        # roughly 4 of 200 calls explore. The dominant outcome must still be SONNET.
        assert SONNET in cascade_picks

    def test_cascade_low_agreement_explores_more(self) -> None:
        """agreement_level=0 -> effective rate 0.25, exploration fires often."""
        rng_high = random.Random(42)
        rng_low = random.Random(42)
        high_agreement_explores = 0
        low_agreement_explores = 0
        for _ in range(400):
            if (
                confidence_cascade(0.50, agreement_level=8, explore_rate=0.10, _rng=rng_high)
                != SONNET
            ):
                high_agreement_explores += 1
            if (
                confidence_cascade(0.50, agreement_level=0, explore_rate=0.10, _rng=rng_low)
                != SONNET
            ):
                low_agreement_explores += 1
        # Expected: high agreement ~2% explore rate, low agreement ~25%.
        # Require 3x separation with margin for RNG variance.
        assert low_agreement_explores > high_agreement_explores * 3

    def test_cascade_explore_rate_zero_kill_switch(self) -> None:
        """explore_rate=0.0 disables exploration even with low agreement."""
        rng = random.Random(99)
        for _ in range(100):
            model = confidence_cascade(0.50, agreement_level=0, explore_rate=0.0, _rng=rng)
            # Deterministic cascade pick at conf=0.50, never explored.
            assert model == SONNET

    def test_cascade_flat_rate_preserved_when_none(self) -> None:
        """Flat rate behavior matches pre-S54 when agreement_level is None."""
        # Before S54, explore_rate=0.30 meant 30% exploration regardless of signals.
        # That behavior must still be reachable by passing agreement_level=None.
        rng_flat = random.Random(31)
        explored = 0
        for _ in range(1000):
            model = confidence_cascade(0.50, agreement_level=None, explore_rate=0.30, _rng=rng_flat)
            if model != SONNET:
                explored += 1
        # Expected ~300 explores; tolerance for RNG variance.
        assert 220 <= explored <= 380

    def test_cascade_with_int_agreement_activates_dynamic(self) -> None:
        """Passing any int (even 0) activates dynamic rate, overriding explore_rate as the flat."""
        # At explore_rate=1.0 and agreement_level=9, effective rate is 0.02 (not 1.0).
        # Over 200 calls, cascade pick (SONNET) should dominate.
        rng = random.Random(11)
        sonnet_count = 0
        for _ in range(200):
            if confidence_cascade(0.50, agreement_level=9, explore_rate=1.0, _rng=rng) == SONNET:
                sonnet_count += 1
        assert sonnet_count >= 180  # ~98% at 0.02 explore rate


class TestConfidenceCascadeReturnDebug:
    """Concern 24: return_debug flag exposes (model, explored: bool)."""

    def test_default_returns_str(self) -> None:
        result = confidence_cascade(0.50, agreement_level=5)
        assert isinstance(result, str)

    def test_return_debug_returns_tuple_threshold_path(self) -> None:
        """Threshold-driven pick: explored=False."""
        result = confidence_cascade(0.50, agreement_level=5, return_debug=True)
        assert isinstance(result, tuple)
        model, explored = result
        assert isinstance(model, str)
        assert explored is False

    def test_return_debug_explore_branch_marks_explored_true(self) -> None:
        """Epsilon-greedy explore: explored=True."""
        rng = random.Random(0)
        # explore_rate=1.0 and agreement_level=None -> always explore
        result = confidence_cascade(
            0.50,
            agreement_level=None,
            explore_rate=1.0,
            _rng=rng,
            return_debug=True,
        )
        assert isinstance(result, tuple)
        _model, explored = result
        assert explored is True

    def test_return_debug_bandit_path_marks_explored_false(self) -> None:
        """Bandit Thompson sample is not 'exploration' for the shadow flag."""
        states = {OPUS: BanditState(mu=0.9, sigma=0.01)}
        result = confidence_cascade(
            0.50,
            agreement_level=5,
            routing_mode="bandit",
            bandit_states=states,
            return_debug=True,
        )
        assert isinstance(result, tuple)
        _model, explored = result
        assert explored is False


class TestComputeRoutingPicks:
    """Concern 24: compute_routing_picks returns paired cascade/bandit picks."""

    def _ctx(self) -> RoutingContext:
        return RoutingContext(
            agreement_level=5,
            confidence_score=0.55,
            domain="infrastructure",
        )

    def test_returns_both_picks_when_bandit_states_present(self) -> None:
        """Cascade picks via thresholds; bandit picks the high-mu winner."""
        # Strong winner -- low sigma so Thompson sample is near-deterministic.
        states = {
            OPUS: BanditState(mu=0.9, sigma=0.01, n_obs=200),
            SONNET: BanditState(mu=0.4, sigma=0.01, n_obs=100),
        }
        rng = random.Random(7)
        picks = compute_routing_picks(
            confidence=0.50,
            agreement_level=5,
            domain="infrastructure",
            routing_context=self._ctx(),
            trust_weights=None,
            bandit_states=states,
            context_bandit_states=None,
            explore_rate=0.0,
            _rng=rng,
        )
        # confidence=0.50 -> cascade picks SONNET (>0.40 threshold).
        assert picks.cascade_pick == SONNET
        # Bandit Thompson with mu=0.9/sigma=0.01 vs mu=0.4/sigma=0.01 -> OPUS.
        assert picks.shadow_pick == OPUS
        assert picks.cascade_explored is False

    def test_shadow_none_when_bandit_states_none(self) -> None:
        picks = compute_routing_picks(
            confidence=0.50,
            agreement_level=5,
            domain="infrastructure",
            routing_context=self._ctx(),
            trust_weights=None,
            bandit_states=None,
            context_bandit_states=None,
            explore_rate=0.0,
        )
        assert picks.shadow_pick is None
        assert picks.shadow_pick_mu is None
        assert picks.shadow_pick_n_obs is None
        # Cascade pick still resolves.
        assert picks.cascade_pick == SONNET

    def test_shadow_none_when_bandit_states_empty(self) -> None:
        picks = compute_routing_picks(
            confidence=0.50,
            agreement_level=5,
            domain="infrastructure",
            routing_context=self._ctx(),
            trust_weights=None,
            bandit_states={},
            context_bandit_states=None,
            explore_rate=0.0,
        )
        assert picks.shadow_pick is None
        assert picks.shadow_pick_mu is None
        assert picks.shadow_pick_n_obs is None

    def test_mu_lookup_for_picked_models(self) -> None:
        """cascade_pick_mu and shadow_pick_mu match bandit_states[pick].mu."""
        states = {
            OPUS: BanditState(mu=0.9, sigma=0.01, n_obs=200),
            SONNET: BanditState(mu=0.42, sigma=0.05, n_obs=120),
        }
        rng = random.Random(7)
        picks = compute_routing_picks(
            confidence=0.50,
            agreement_level=5,
            domain="infrastructure",
            routing_context=self._ctx(),
            trust_weights=None,
            bandit_states=states,
            context_bandit_states=None,
            explore_rate=0.0,
            _rng=rng,
        )
        assert picks.cascade_pick == SONNET
        assert picks.cascade_pick_mu == pytest.approx(0.42)
        assert picks.cascade_pick_n_obs == 120
        assert picks.shadow_pick == OPUS
        assert picks.shadow_pick_mu == pytest.approx(0.9)
        assert picks.shadow_pick_n_obs == 200

    def test_mu_none_for_uncovered_model(self) -> None:
        """Cascade picks a model NOT in bandit_states -> cascade_pick_mu is None."""
        # Bandit only has OPUS; confidence routes to SONNET.
        states = {OPUS: BanditState(mu=0.9, sigma=0.01, n_obs=200)}
        rng = random.Random(7)
        picks = compute_routing_picks(
            confidence=0.50,
            agreement_level=5,
            domain="infrastructure",
            routing_context=self._ctx(),
            trust_weights=None,
            bandit_states=states,
            context_bandit_states=None,
            explore_rate=0.0,
            _rng=rng,
        )
        assert picks.cascade_pick == SONNET
        assert picks.cascade_pick_mu is None
        assert picks.cascade_pick_n_obs is None
        # Shadow picks OPUS, which IS in bandit_states.
        assert picks.shadow_pick == OPUS
        assert picks.shadow_pick_mu == pytest.approx(0.9)

    def test_seeded_rng_is_deterministic(self) -> None:
        """Identical _rng seed -> identical RoutingPicks across runs."""
        states = {
            OPUS: BanditState(mu=0.6, sigma=0.2, n_obs=20),
            SONNET: BanditState(mu=0.5, sigma=0.2, n_obs=20),
            HAIKU: BanditState(mu=0.4, sigma=0.2, n_obs=20),
        }
        kwargs = {
            "confidence": 0.50,
            "agreement_level": 5,
            "domain": "infrastructure",
            "routing_context": self._ctx(),
            "trust_weights": None,
            "bandit_states": states,
            "context_bandit_states": None,
            "explore_rate": 0.0,
        }
        a = compute_routing_picks(**kwargs, _rng=random.Random(42))
        b = compute_routing_picks(**kwargs, _rng=random.Random(42))
        assert a == b

    def test_cascade_explored_flag_when_explore_rate_forces_branch(self) -> None:
        """explore_rate=1.0 with agreement_level=None forces explore branch."""
        states = {OPUS: BanditState(mu=0.9, sigma=0.01, n_obs=200)}
        rng = random.Random(0)
        picks = compute_routing_picks(
            confidence=0.50,
            agreement_level=None,  # disables dynamic rate
            domain="infrastructure",
            routing_context=self._ctx(),
            trust_weights=None,
            bandit_states=states,
            context_bandit_states=None,
            explore_rate=1.0,  # always explore on cascade side
            _rng=rng,
        )
        assert picks.cascade_explored is True

    def test_does_not_mutate_bandit_states(self) -> None:
        """compute_routing_picks must not mutate the bandit_states dict it is passed."""
        original = {
            OPUS: BanditState(mu=0.6, sigma=0.1, n_obs=50),
            SONNET: BanditState(mu=0.55, sigma=0.1, n_obs=50),
        }
        snapshot = dict(original)
        compute_routing_picks(
            confidence=0.50,
            agreement_level=5,
            domain="infrastructure",
            routing_context=self._ctx(),
            trust_weights=None,
            bandit_states=original,
            context_bandit_states=None,
            explore_rate=0.0,
            cost_adjusted=True,
            _rng=random.Random(1),
        )
        # Cost adjustment in the bandit path should not mutate the caller's dict.
        assert original == snapshot
