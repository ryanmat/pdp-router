# Description: Per-provider token pricing and cost estimation.
# Description: Prefix-match model name against pricing tables for all supported providers.

from __future__ import annotations

# Per-million-token pricing (USD) by model name prefix.
# Order matters: longer prefixes must come before shorter ones to avoid
# false matches (e.g., "gemini-2.5-flash-lite" before "gemini-2.5-flash").
_MODEL_PRICING: list[tuple[str, dict[str, float]]] = [
    ("claude-haiku", {"input": 0.25, "output": 1.25}),
    ("claude-sonnet", {"input": 3.00, "output": 15.00}),
    ("claude-opus", {"input": 5.00, "output": 25.00}),
    ("deepseek-", {"input": 0.28, "output": 0.42}),
    ("gemini-2.5-flash-lite", {"input": 0.10, "output": 0.40}),
    ("gemini-2.5-flash", {"input": 0.30, "output": 2.50}),
    ("gemini-2.5-pro", {"input": 1.25, "output": 10.00}),
    ("meta/llama-4-scout", {"input": 0.25, "output": 0.70}),
    ("meta/llama-4-maverick", {"input": 0.35, "output": 1.15}),
]

_FALLBACK_PRICING: dict[str, float] = {"input": 3.00, "output": 15.00}


def estimate_cost(model: str, usage: dict[str, int]) -> float:
    """Estimate USD cost from token usage based on model pricing.

    Args:
        model: Model name (e.g. "claude-sonnet-4-20250514" or "gemini-2.5-flash").
        usage: Dict with input_tokens and output_tokens.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places.
    """
    pricing = _FALLBACK_PRICING
    for prefix, rates in _MODEL_PRICING:
        if model.startswith(prefix):
            pricing = rates
            break

    input_cost = (usage.get("input_tokens", 0) / 1_000_000) * pricing["input"]
    output_cost = (usage.get("output_tokens", 0) / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)
