# Description: Central model ID constants for all PDP providers.
# Description: Single source of truth -- update here when models change.

from __future__ import annotations

import re
from collections.abc import Iterable

# Anthropic
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
# OPUS is the live frontier alias the router dispatches to. claude-opus-4-8 and the
# prior claude-opus-4-7 both canonicalize to claude-opus-4, so moving the alias forward
# keeps the shared bandit posterior + trust history intact (no arm fork).
OPUS = "claude-opus-4-8"

# DeepSeek (OpenAI-compatible API)
DEEPSEEK = "deepseek-chat"

# Google Gemini (API key mode)
GEMINI_PRO = "gemini-2.5-pro"
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"

# Meta Llama (Vertex AI partner models -- -maas suffix required for MaaS endpoint)
LLAMA_SCOUT = "meta/llama-4-scout-17b-16e-instruct-maas"
LLAMA_MAVERICK = "meta/llama-4-maverick-17b-128e-instruct-maas"


# Live registry: every model ID the proxy/orchestrator may select. Used by
# expand_canonical_to_live() to fan a canonical posterior back out to every
# live alias that shares the same family + major version.
LIVE_REGISTRY: tuple[str, ...] = (
    HAIKU,
    SONNET,
    OPUS,
    DEEPSEEK,
    GEMINI_PRO,
    GEMINI_FLASH,
    GEMINI_FLASH_LITE,
    LLAMA_SCOUT,
    LLAMA_MAVERICK,
)


# Match Anthropic IDs of the shape "claude-{family}-{major}{trailing}" where
# trailing is the optional minor / dated / extended-context suffix we collapse.
_ANTHROPIC_ID_RE = re.compile(r"^claude-(opus|sonnet|haiku)-(\d+)(?:[-.].*)?$")


class CreditExhaustionError(Exception):
    """Raised when an LLM API indicates credit exhaustion or billing failure."""


def canonicalize_model_id(model_id: str) -> str:
    """Map a raw or dated model alias to a stable canonical key.

    Used to aggregate Thompson Sampling posteriors and trust weights across
    alias bumps within a major version. Anthropic models collapse to
    "claude-{family}-{major}" so claude-opus-4-6 / claude-opus-4-7 /
    claude-opus-4-7-1m / claude-opus-4-20251001 all become "claude-opus-4".
    A future claude-opus-5-x will canonicalize to "claude-opus-5" and start
    a fresh arm.

    Non-Anthropic IDs pass through unchanged: Gemini's pro/flash/flash-lite
    are capability tiers (not version aliases), Llama's full MaaS suffix is
    required by the Vertex endpoint and the cost lookup, and DeepSeek/local
    IDs are already stable.
    """
    if not model_id:
        return model_id
    match = _ANTHROPIC_ID_RE.match(model_id)
    if match:
        family, major = match.group(1), match.group(2)
        return f"claude-{family}-{major}"
    return model_id


def expand_canonical_to_live(
    canonical_id: str,
    registry_models: Iterable[str] = LIVE_REGISTRY,
) -> list[str]:
    """Return every registry model_id that canonicalizes to canonical_id.

    Inverse of canonicalize_model_id. Used by callers that store posteriors
    keyed on the canonical ID but need to expose them under the live model
    IDs that the routing layer / API client actually use.

    For canonical IDs that are themselves live (Gemini, Llama, DeepSeek),
    this is the identity if the ID exists in the registry, else empty.
    """
    if not canonical_id:
        return []
    matches = [live for live in registry_models if canonicalize_model_id(live) == canonical_id]
    if matches:
        return matches
    # Canonical isn't represented in this registry (e.g. retired family).
    # Return the canonical itself so callers can still surface the row.
    return [canonical_id]
