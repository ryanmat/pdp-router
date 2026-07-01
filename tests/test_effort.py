# Description: Tests for deterministic effort-level mapping (pdp_router._effort).
# Description: level_for_score thresholds, supports_effort gating, provider fragments.

from __future__ import annotations

from pdp_router import _effort


class TestLevelForScore:
    def test_defaults_partition_scores_1_to_5(self) -> None:
        assert [_effort.level_for_score(s) for s in range(1, 6)] == [
            "low",
            "low",
            "medium",
            "high",
            "high",
        ]

    def test_default_band_boundaries(self) -> None:
        assert _effort.level_for_score(2) == _effort.LOW
        assert _effort.level_for_score(3) == _effort.MEDIUM
        assert _effort.level_for_score(4) == _effort.HIGH

    def test_thresholds_are_tunable(self) -> None:
        # Collapse to a two-level on/off split (no medium band): <=1 low, >=2 high.
        assert _effort.level_for_score(1, low_max=1, high_min=2) == _effort.LOW
        assert _effort.level_for_score(2, low_max=1, high_min=2) == _effort.HIGH

    def test_below_and_above_the_band(self) -> None:
        assert _effort.level_for_score(0) == _effort.LOW
        assert _effort.level_for_score(9) == _effort.HIGH

    def test_levels_constant_is_the_lcd_three(self) -> None:
        assert _effort.LEVELS == ("low", "medium", "high")


class TestSupportsEffort:
    def test_anthropic_non_haiku_supported(self) -> None:
        assert _effort.supports_effort("claude-opus-4-8")
        assert _effort.supports_effort("claude-sonnet-4-6")

    def test_haiku_not_supported(self) -> None:
        assert not _effort.supports_effort("claude-haiku-4-5-20251001")

    def test_openrouter_arms_supported(self) -> None:
        assert _effort.supports_effort("openai/gpt-5.5")
        assert _effort.supports_effort("qwen/qwen3.7-plus")

    def test_no_dial_providers_excluded(self) -> None:
        for model in (
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "deepseek-chat",
            "meta/llama-4-scout",
            "meta/llama-4-maverick",
            "ollama/llama3",
        ):
            assert not _effort.supports_effort(model), model


class TestProviderFragments:
    def test_anthropic_output_config(self) -> None:
        assert _effort.anthropic_output_config("high") == {"effort": "high"}
        assert _effort.anthropic_output_config("low") == {"effort": "low"}

    def test_openrouter_reasoning_body(self) -> None:
        assert _effort.openrouter_reasoning_body("medium") == {"reasoning": {"effort": "medium"}}
