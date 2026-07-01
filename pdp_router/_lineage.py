# Description: Maps roster model IDs to their lineage (provider family).
# Description: Drives panel composition diversity -- one model per lineage.

from __future__ import annotations

LINEAGE_ANTHROPIC = "anthropic"
LINEAGE_GOOGLE = "google"
LINEAGE_META = "meta"
LINEAGE_DEEPSEEK = "deepseek"
LINEAGE_OLLAMA = "ollama"
LINEAGE_OPENAI = "openai"
LINEAGE_QWEN = "qwen"
LINEAGE_UNKNOWN = "unknown"

ALL_LINEAGES = (
    LINEAGE_ANTHROPIC,
    LINEAGE_GOOGLE,
    LINEAGE_META,
    LINEAGE_DEEPSEEK,
    LINEAGE_OLLAMA,
    LINEAGE_OPENAI,
    LINEAGE_QWEN,
)


def classify_lineage(model_id: str) -> str:
    """Return the lineage tag for a model ID.

    Inline prefix map -- when a new provider lands, add it here. The
    `unknown` fallback should never fire for a model that's actually in
    DEFAULT_REGISTRY; if it does, the registry has a model that isn't
    family-tagged anywhere and the panel composer can't reason about
    diversity for it.
    """
    if not model_id:
        return LINEAGE_UNKNOWN
    if model_id.startswith("claude-"):
        return LINEAGE_ANTHROPIC
    if model_id.startswith("gemini-"):
        return LINEAGE_GOOGLE
    if model_id.startswith("meta/"):
        return LINEAGE_META
    if model_id.startswith("deepseek"):
        return LINEAGE_DEEPSEEK
    if model_id.startswith("ollama"):
        return LINEAGE_OLLAMA
    if model_id.startswith("openai/"):
        return LINEAGE_OPENAI
    if model_id.startswith("qwen"):
        return LINEAGE_QWEN
    return LINEAGE_UNKNOWN


def filter_by_lineage(model_ids: list[str], lineage_filter: str) -> list[str]:
    """Apply a lineage filter to a candidate list.

    `lineage_filter` is one of:
      - "all" (default) -- no filter
      - "anthropic" -- only claude-* models
      - "non_anthropic" -- everything except claude-*
    """
    if lineage_filter == "all":
        return list(model_ids)
    if lineage_filter == LINEAGE_ANTHROPIC:
        return [m for m in model_ids if classify_lineage(m) == LINEAGE_ANTHROPIC]
    if lineage_filter == "non_anthropic":
        return [m for m in model_ids if classify_lineage(m) != LINEAGE_ANTHROPIC]
    msg = f"Unknown lineage_filter: {lineage_filter!r} (use 'all', 'anthropic', or 'non_anthropic')"
    raise ValueError(msg)
