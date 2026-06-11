# Description: Tests for PDP model constants, canonicalization, and error types.
# Description: Verifies roster model IDs, canonicalize/expand helpers, and CreditExhaustionError.

from __future__ import annotations

import pytest

from pdp_router._models import (
    DEEPSEEK,
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    GEMINI_PRO,
    HAIKU,
    LIVE_REGISTRY,
    LLAMA_MAVERICK,
    LLAMA_SCOUT,
    OPUS,
    SONNET,
    CreditExhaustionError,
    canonicalize_model_id,
    expand_canonical_to_live,
)


class TestModelConstants:
    def test_anthropic_models_have_claude_prefix(self) -> None:
        for model in (HAIKU, SONNET, OPUS):
            assert model.startswith("claude-"), f"{model} missing claude- prefix"

    def test_gemini_models_have_gemini_prefix(self) -> None:
        for model in (GEMINI_PRO, GEMINI_FLASH, GEMINI_FLASH_LITE):
            assert model.startswith("gemini-"), f"{model} missing gemini- prefix"

    def test_llama_models_have_meta_prefix(self) -> None:
        for model in (LLAMA_SCOUT, LLAMA_MAVERICK):
            assert model.startswith("meta/"), f"{model} missing meta/ prefix"

    def test_all_constants_are_nonempty_strings(self) -> None:
        for model in (
            HAIKU,
            SONNET,
            OPUS,
            GEMINI_PRO,
            GEMINI_FLASH,
            GEMINI_FLASH_LITE,
            LLAMA_SCOUT,
            LLAMA_MAVERICK,
        ):
            assert isinstance(model, str) and len(model) > 0

    def test_no_duplicate_model_ids(self) -> None:
        models = [
            HAIKU,
            SONNET,
            OPUS,
            GEMINI_PRO,
            GEMINI_FLASH,
            GEMINI_FLASH_LITE,
            LLAMA_SCOUT,
            LLAMA_MAVERICK,
        ]
        assert len(models) == len(set(models))

    def test_live_registry_contains_all_constants(self) -> None:
        for model in (
            HAIKU,
            SONNET,
            OPUS,
            DEEPSEEK,
            GEMINI_PRO,
            GEMINI_FLASH,
            GEMINI_FLASH_LITE,
            LLAMA_SCOUT,
            LLAMA_MAVERICK,
        ):
            assert model in LIVE_REGISTRY


class TestCreditExhaustionError:
    def test_is_exception_subclass(self) -> None:
        assert issubclass(CreditExhaustionError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(CreditExhaustionError, match="billing"):
            raise CreditExhaustionError("billing error")


class TestCanonicalizeModelId:
    @pytest.mark.parametrize(
        "raw,canonical",
        [
            # Opus alias bumps + dated + extended-context all collapse to family+major.
            ("claude-opus-4-6", "claude-opus-4"),
            ("claude-opus-4-7", "claude-opus-4"),
            ("claude-opus-4-7-1m", "claude-opus-4"),
            ("claude-opus-4-20251001", "claude-opus-4"),
            ("claude-opus-4-1", "claude-opus-4"),
            # Sonnet: dated and aliased forms both collapse.
            ("claude-sonnet-4-6", "claude-sonnet-4"),
            ("claude-sonnet-4-20250514", "claude-sonnet-4"),
            ("claude-sonnet-4-1", "claude-sonnet-4"),
            # Haiku: dated form collapses.
            ("claude-haiku-4-5-20251001", "claude-haiku-4"),
            ("claude-haiku-4-5", "claude-haiku-4"),
            ("claude-haiku-4-1", "claude-haiku-4"),
            # Future major bumps land on a fresh canonical key.
            ("claude-opus-5-1", "claude-opus-5"),
            ("claude-sonnet-5-20260101", "claude-sonnet-5"),
            ("claude-haiku-5-1", "claude-haiku-5"),
            # Non-Anthropic IDs pass through unchanged.
            ("gemini-2.5-pro", "gemini-2.5-pro"),
            ("gemini-2.5-flash", "gemini-2.5-flash"),
            ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
            ("deepseek-chat", "deepseek-chat"),
            (
                "meta/llama-4-scout-17b-16e-instruct-maas",
                "meta/llama-4-scout-17b-16e-instruct-maas",
            ),
            (
                "meta/llama-4-maverick-17b-128e-instruct-maas",
                "meta/llama-4-maverick-17b-128e-instruct-maas",
            ),
            ("chair", "chair"),
            # Edge cases: empty, malformed, claude-prefix-but-not-canonical-shape.
            ("", ""),
            ("claude-instant", "claude-instant"),  # no major version, pass through
            ("claude-3-opus", "claude-3-opus"),  # legacy v3 shape, pass through
        ],
    )
    def test_canonicalize(self, raw: str, canonical: str) -> None:
        assert canonicalize_model_id(raw) == canonical

    def test_idempotent(self) -> None:
        for raw in ("claude-opus-4-7", "gemini-2.5-pro", "deepseek-chat"):
            once = canonicalize_model_id(raw)
            twice = canonicalize_model_id(once)
            assert once == twice

    def test_current_production_ids_canonicalize_correctly(self) -> None:
        # The OPUS/SONNET/HAIKU constants today must map to family+major.
        assert canonicalize_model_id(OPUS) == "claude-opus-4"
        assert canonicalize_model_id(SONNET) == "claude-sonnet-4"
        assert canonicalize_model_id(HAIKU) == "claude-haiku-4"


class TestExpandCanonicalToLive:
    def test_anthropic_canonical_expands_to_current_alias(self) -> None:
        assert expand_canonical_to_live("claude-opus-4") == [OPUS]
        assert expand_canonical_to_live("claude-sonnet-4") == [SONNET]
        assert expand_canonical_to_live("claude-haiku-4") == [HAIKU]

    def test_anthropic_canonical_expands_to_multiple_when_registry_has_multiple(
        self,
    ) -> None:
        # Custom registry with two opus aliases simulates a transition window.
        custom_registry = ("claude-opus-4-6", "claude-opus-4-7", "gemini-2.5-pro")
        result = expand_canonical_to_live("claude-opus-4", custom_registry)
        assert set(result) == {"claude-opus-4-6", "claude-opus-4-7"}

    def test_passthrough_id_expands_to_itself(self) -> None:
        assert expand_canonical_to_live(GEMINI_PRO) == [GEMINI_PRO]
        assert expand_canonical_to_live(DEEPSEEK) == [DEEPSEEK]
        assert expand_canonical_to_live(LLAMA_SCOUT) == [LLAMA_SCOUT]

    def test_unknown_canonical_returns_self(self) -> None:
        # When canonical isn't represented (retired family), return canonical
        # itself so callers can still surface the historical row.
        assert expand_canonical_to_live("claude-opus-3") == ["claude-opus-3"]
        assert expand_canonical_to_live("retired-model") == ["retired-model"]

    def test_empty_canonical_returns_empty_list(self) -> None:
        assert expand_canonical_to_live("") == []

    def test_default_registry_used_when_omitted(self) -> None:
        # Same as passing LIVE_REGISTRY explicitly.
        assert expand_canonical_to_live("claude-opus-4") == expand_canonical_to_live(
            "claude-opus-4", LIVE_REGISTRY
        )
