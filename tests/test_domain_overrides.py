# Description: Unit tests for the per-domain routing-mode override helper.
# Description: Covers parse_overrides edge cases and the resolve_routing_mode dispatch.

from __future__ import annotations

import random

from pdp_router._domain_overrides import parse_overrides, resolve_routing_mode


class TestParseOverrides:
    def test_empty_string_returns_empty_dict(self):
        assert parse_overrides("") == {}

    def test_none_returns_empty_dict(self):
        assert parse_overrides(None) == {}

    def test_whitespace_only_returns_empty_dict(self):
        assert parse_overrides("   \n") == {}

    def test_valid_json_object_returned(self):
        raw = '{"kubernetes": {"mode": "bandit", "fraction": 0.10}}'
        result = parse_overrides(raw)
        assert result == {"kubernetes": {"mode": "bandit", "fraction": 0.10}}

    def test_valid_string_override(self):
        result = parse_overrides('{"network": "bandit"}')
        assert result == {"network": "bandit"}

    def test_malformed_json_returns_empty_dict(self):
        assert parse_overrides("{not json") == {}

    def test_json_array_rejected(self):
        assert parse_overrides('["kubernetes"]') == {}

    def test_json_scalar_rejected(self):
        assert parse_overrides('"bandit"') == {}
        assert parse_overrides("42") == {}


class TestResolveRoutingMode:
    def test_no_overrides_returns_default(self):
        assert resolve_routing_mode("cascade", {}, "kubernetes") == "cascade"

    def test_none_domain_returns_default(self):
        overrides = {"kubernetes": "bandit"}
        assert resolve_routing_mode("cascade", overrides, None) == "cascade"

    def test_empty_domain_returns_default(self):
        overrides = {"kubernetes": "bandit"}
        assert resolve_routing_mode("cascade", overrides, "") == "cascade"

    def test_unmatched_domain_returns_default(self):
        overrides = {"kubernetes": "bandit"}
        assert resolve_routing_mode("cascade", overrides, "network") == "cascade"

    def test_full_string_override(self):
        overrides = {"kubernetes": "bandit"}
        assert resolve_routing_mode("cascade", overrides, "kubernetes") == "bandit"

    def test_canary_fraction_one_always_overrides(self):
        overrides = {"kubernetes": {"mode": "bandit", "fraction": 1.0}}
        rng = random.Random(0)
        for _ in range(20):
            assert resolve_routing_mode("cascade", overrides, "kubernetes", rng=rng) == "bandit"

    def test_canary_fraction_zero_never_overrides(self):
        overrides = {"kubernetes": {"mode": "bandit", "fraction": 0.0}}
        rng = random.Random(0)
        for _ in range(20):
            assert (
                resolve_routing_mode("cascade", overrides, "kubernetes", rng=rng) == "cascade"
            )

    def test_canary_fraction_negative_falls_back_to_default(self):
        overrides = {"kubernetes": {"mode": "bandit", "fraction": -0.5}}
        assert resolve_routing_mode("cascade", overrides, "kubernetes") == "cascade"

    def test_canary_fraction_10pct_seeded(self):
        # Seed 0 produces .8444 first then .7579 etc. We construct a rigged
        # rng that yields 0.05 (< 0.10) once and 0.50 (> 0.10) once.
        class StubRng:
            def __init__(self, values: list[float]):
                self._values = list(values)

            def random(self) -> float:
                return self._values.pop(0)

        overrides = {"kubernetes": {"mode": "bandit", "fraction": 0.10}}
        rng_low = StubRng([0.05])
        assert (
            resolve_routing_mode("cascade", overrides, "kubernetes", rng=rng_low) == "bandit"
        )
        rng_high = StubRng([0.50])
        assert (
            resolve_routing_mode("cascade", overrides, "kubernetes", rng=rng_high) == "cascade"
        )

    def test_canary_fraction_string_coerced(self):
        # JSON often serializes fractions as strings in env vars
        overrides = {"kubernetes": {"mode": "bandit", "fraction": "1.0"}}
        assert resolve_routing_mode("cascade", overrides, "kubernetes") == "bandit"

    def test_canary_missing_mode_returns_default(self):
        overrides = {"kubernetes": {"fraction": 0.5}}
        assert resolve_routing_mode("cascade", overrides, "kubernetes") == "cascade"

    def test_canary_non_dict_non_string_returns_default(self):
        overrides = {"kubernetes": 42}
        assert resolve_routing_mode("cascade", overrides, "kubernetes") == "cascade"

    def test_canary_empty_mode_string_returns_default(self):
        overrides = {"kubernetes": {"mode": "", "fraction": 1.0}}
        assert resolve_routing_mode("cascade", overrides, "kubernetes") == "cascade"

    def test_canary_invalid_fraction_string_returns_default(self):
        overrides = {"kubernetes": {"mode": "bandit", "fraction": "abc"}}
        assert resolve_routing_mode("cascade", overrides, "kubernetes") == "cascade"

    def test_realistic_concern24_kubernetes_canary(self):
        overrides = {"kubernetes": {"mode": "bandit", "fraction": 0.10}}
        rng = random.Random(42)
        # With seed 42 over 10000 rolls expect ~10% +/- noise
        bandit_count = sum(
            1
            for _ in range(10000)
            if resolve_routing_mode("cascade", overrides, "kubernetes", rng=rng) == "bandit"
        )
        # Wide tolerance: 8-12% acceptable, anything else means the wiring leaks.
        assert 800 <= bandit_count <= 1200, f"expected ~1000 bandit picks, got {bandit_count}"

    def test_non_canary_domain_unaffected_by_other_overrides(self):
        overrides = {"kubernetes": {"mode": "bandit", "fraction": 1.0}}
        assert resolve_routing_mode("cascade", overrides, "network") == "cascade"
