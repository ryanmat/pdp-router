# Description: Deterministic reasoning-effort mapping for the PDP Router proxy.
# Description: Maps the classifier complexity score to per-provider effort knobs.

from __future__ import annotations

# Cross-provider lowest-common-denominator effort levels. Anthropic also exposes
# "max", but the deterministic dial stays at the three levels every effort-capable
# arm in the roster supports.
LOW = "low"
MEDIUM = "medium"
HIGH = "high"
LEVELS = (LOW, MEDIUM, HIGH)


def level_for_score(score: int, *, low_max: int = 2, high_min: int = 4) -> str:
    """Map a 1-5 classifier complexity score to an effort level.

    score <= low_max -> LOW; score >= high_min -> HIGH; otherwise MEDIUM.
    Defaults give 1-2 -> low, 3 -> medium, 4-5 -> high -- the same complexity
    signal the cascade already uses to pick the tier.
    """
    if score <= low_max:
        return LOW
    if score >= high_min:
        return HIGH
    return MEDIUM


def is_anthropic_effort_model(model_name: str) -> bool:
    """Anthropic models that accept output_config.effort: claude-* except the fast
    (Haiku) tier, which rejects the parameter."""
    return model_name.startswith("claude-") and "haiku" not in model_name


def is_openrouter_effort_model(model_name: str) -> bool:
    """OpenRouter-fronted arms (openai/*, qwen*) that accept the unified
    reasoning.effort parameter."""
    return model_name.startswith("openai/") or model_name.startswith("qwen")


def supports_effort(model_name: str) -> bool:
    """True if v1 threads a reasoning-effort knob to this model's provider call.

    v1 covers the arms with a verified, unambiguous level knob:
      - Anthropic non-fast models via output_config.effort (Haiku rejects it);
      - the OpenRouter arms (openai/*, qwen*) via reasoning.effort.
    Gemini (thinking_budget/level is Gemini-version dependent on the 2.5 roster),
    DeepSeek (reasoning is a separate model, not a request param), Llama/Vertex,
    Haiku, and Ollama have no dial in v1 -- they are routed only when effort does
    not matter, and the proxy passes effort=None for them.
    """
    return is_anthropic_effort_model(model_name) or is_openrouter_effort_model(model_name)


def anthropic_output_config(level: str) -> dict:
    """output_config kwarg for the Anthropic messages API effort dial.

    Verified against anthropic SDK 0.89.0: OutputConfigParam.effort accepts
    Literal['low','medium','high','max']; the proxy passes low/medium/high.
    """
    return {"effort": level}


def openrouter_reasoning_body(level: str) -> dict:
    """Body fragment for OpenRouter's unified reasoning-effort parameter."""
    return {"reasoning": {"effort": level}}
