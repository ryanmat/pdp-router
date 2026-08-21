# Description: Per-provider token pricing and cost estimation.
# Description: Prefix-match model name against pricing tables for all supported providers.

from __future__ import annotations

# Per-million-token pricing (USD) by model name prefix.
# Order matters: longer prefixes must come before shorter ones to avoid
# false matches (e.g., "gemini-2.5-flash-lite" before "gemini-2.5-flash").
# cache_read / cache_write (Anthropic rows only) are the 5-minute-TTL prompt
# cache rates: read = 0.1x input, write = 1.25x input. A 1-hour TTL would be
# 2x on writes and is deliberately not encoded -- nothing here requests it.
# The registry's cost_per_mtok_in/out table intentionally stays list-rate
# only: bandit cost weighting compares arms, and cache rates would skew that
# comparison toward whichever arm happened to cache.
_MODEL_PRICING: list[tuple[str, dict[str, float]]] = [
    ("claude-haiku", {"input": 0.25, "output": 1.25, "cache_read": 0.025, "cache_write": 0.3125}),
    # The 5-generation rows carry the same list rates as the 4-generation rows
    # below and would resolve correctly through those generic prefixes today.
    # They are pinned explicitly anyway: without them the 5-generation arms are
    # priced by coincidence, and a reprice of the 4-generation row would drag
    # them along silently. Sonnet 5 also has introductory pricing of $2/$10 per
    # MTok through 2026-08-31 -- list is encoded instead, because this table has
    # no notion of a date and an expired promotion would under-price forever,
    # whereas list merely over-estimates until the promotion lapses.
    ("claude-sonnet-5", {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75}),
    ("claude-opus-5", {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25}),
    ("claude-sonnet", {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75}),
    ("claude-opus", {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25}),
    ("deepseek-", {"input": 0.28, "output": 0.42}),
    ("gemini-2.5-flash-lite", {"input": 0.10, "output": 0.40}),
    ("gemini-2.5-flash", {"input": 0.30, "output": 2.50}),
    ("gemini-2.5-pro", {"input": 1.25, "output": 10.00}),
    ("meta/llama-4-scout", {"input": 0.25, "output": 0.70}),
    ("meta/llama-4-maverick", {"input": 0.35, "output": 1.15}),
    ("openai/gpt-5.5", {"input": 5.00, "output": 30.00}),
    ("qwen/qwen3.7-plus", {"input": 0.32, "output": 1.28}),
]

_FALLBACK_PRICING: dict[str, float] = {"input": 3.00, "output": 15.00}


def estimate_cost(model: str, usage: dict[str, int]) -> float:
    """Estimate USD cost from token usage based on model pricing.

    Args:
        model: Model name (e.g. "claude-sonnet-4-20250514" or "gemini-2.5-flash").
        usage: Dict with input_tokens and output_tokens, optionally
            cache_read_input_tokens and cache_creation_input_tokens.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places.

    Anthropic's input_tokens already EXCLUDES the cache-read and cache-creation
    counts, so the four terms sum without double counting. A row without cache
    rates prices cache tokens at the full input rate -- providers that never
    emit the cache keys are unaffected (terms are zero), and a future mispriced
    row over-estimates instead of silently zeroing.
    """
    pricing = _FALLBACK_PRICING
    for prefix, rates in _MODEL_PRICING:
        if model.startswith(prefix):
            pricing = rates
            break

    input_cost = (usage.get("input_tokens", 0) / 1_000_000) * pricing["input"]
    output_cost = (usage.get("output_tokens", 0) / 1_000_000) * pricing["output"]
    cache_read_cost = (usage.get("cache_read_input_tokens", 0) / 1_000_000) * pricing.get(
        "cache_read", pricing["input"]
    )
    cache_write_cost = (usage.get("cache_creation_input_tokens", 0) / 1_000_000) * pricing.get(
        "cache_write", pricing["input"]
    )
    return round(input_cost + output_cost + cache_read_cost + cache_write_cost, 6)
