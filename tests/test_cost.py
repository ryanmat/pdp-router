# Description: Tests for per-provider cost estimation.
# Description: Covers all model families and edge cases.

from __future__ import annotations

import pytest

from pdp_router._cost import _MODEL_PRICING, estimate_cost
from pdp_router._router import DEFAULT_REGISTRY


class TestEstimateCost:
    """Per-provider cost estimation tests with multi-provider additions."""

    def test_haiku_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("claude-haiku-4-5-20251001", usage)
        assert cost == pytest.approx(0.25 + 1.25)

    def test_sonnet_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("claude-sonnet-4-20250514", usage)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_opus_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("claude-opus-4-7", usage)
        assert cost == pytest.approx(5.00 + 25.00)

    def test_gemini_pro_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("gemini-2.5-pro", usage)
        assert cost == pytest.approx(1.25 + 10.00)

    def test_gemini_flash_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("gemini-2.5-flash", usage)
        # flash-lite also starts with "gemini-2.5-flash", so flash must match first
        assert cost == pytest.approx(0.30 + 2.50)

    def test_gemini_flash_lite_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("gemini-2.5-flash-lite", usage)
        assert cost == pytest.approx(0.10 + 0.40)

    def test_llama_scout_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("meta/llama-4-scout-17b-16e-instruct-maas", usage)
        assert cost == pytest.approx(0.25 + 0.70)

    def test_llama_maverick_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("meta/llama-4-maverick-17b-128e-instruct-maas", usage)
        assert cost == pytest.approx(0.35 + 1.15)

    def test_zero_tokens(self) -> None:
        usage = {"input_tokens": 0, "output_tokens": 0}
        assert estimate_cost("claude-sonnet-4-20250514", usage) == 0.0

    def test_unknown_model_uses_fallback(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("some-unknown-model", usage)
        assert cost == pytest.approx(3.00 + 15.00)  # fallback rates

    def test_small_token_count_precision(self) -> None:
        usage = {"input_tokens": 100, "output_tokens": 50}
        cost = estimate_cost("claude-haiku-4-5-20251001", usage)
        # (100/1M)*0.25 + (50/1M)*1.25 = 0.0000875, rounded to 6 places
        expected = round((100 / 1_000_000) * 0.25 + (50 / 1_000_000) * 1.25, 6)
        assert cost == expected

    def test_deepseek_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("deepseek-chat", usage)
        assert cost == pytest.approx(0.28 + 0.42)

    def test_gpt_5_5_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("openai/gpt-5.5", usage)
        assert cost == pytest.approx(5.00 + 30.00)

    def test_gpt_5_5_dated_slug_matches_prefix(self) -> None:
        # The dated alias must hit the same prefix row, not the fallback.
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("openai/gpt-5.5-20260423", usage)
        assert cost == pytest.approx(5.00 + 30.00)

    def test_qwen_3_7_plus_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        cost = estimate_cost("qwen/qwen3.7-plus", usage)
        assert cost == pytest.approx(0.32 + 1.28)

    def test_missing_keys_default_to_zero(self) -> None:
        assert estimate_cost("claude-opus-4-7", {}) == 0.0

    def test_opus_5_cost(self) -> None:
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        assert estimate_cost("claude-opus-5", usage) == pytest.approx(5.00 + 25.00)

    def test_sonnet_5_cost(self) -> None:
        # List price. Sonnet 5 carries introductory pricing of $2/$10 through
        # 2026-08-31; the table encodes list so it does not silently under-price
        # once the promotion lapses.
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
        assert estimate_cost("claude-sonnet-5", usage) == pytest.approx(3.00 + 15.00)

    def test_new_generation_has_its_own_row_not_a_prefix_coincidence(self) -> None:
        # The 5-generation ids also match the generic "claude-sonnet"/"claude-opus"
        # prefixes, which happen to carry the same rates today. Pinning them to
        # their own rows keeps a future reprice of the 4-generation row from
        # silently dragging the 5-generation arms with it.
        rows = [prefix for prefix, _ in _MODEL_PRICING]
        assert "claude-opus-5" in rows
        assert "claude-sonnet-5" in rows
        assert rows.index("claude-opus-5") < rows.index("claude-opus")
        assert rows.index("claude-sonnet-5") < rows.index("claude-sonnet")


class TestRegistryPricingParity:
    """Every routable model must resolve to a real row, never the fallback."""

    def test_every_registry_model_has_explicit_pricing(self) -> None:
        for capability in DEFAULT_REGISTRY.available_models():
            matched = [p for p, _ in _MODEL_PRICING if capability.name.startswith(p)]
            assert matched, (
                f"{capability.name} has no pricing row and would hit the fallback"
            )
