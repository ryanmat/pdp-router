# Description: Tests for per-provider cost estimation.
# Description: Covers all model families and edge cases.

from __future__ import annotations

import pytest

from pdp_router._cost import estimate_cost


class TestEstimateCost:
    """Port of enrichment-orchestrator TestEstimateCost + multi-provider additions."""

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

    def test_missing_keys_default_to_zero(self) -> None:
        assert estimate_cost("claude-opus-4-7", {}) == 0.0
