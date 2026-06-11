# Description: Per-domain routing-mode override and canary roll for Concern 24.
# Description: Parses ROUTING_MODE_OVERRIDES JSON and resolves the effective mode per request.

from __future__ import annotations

import json
import logging
import random
from typing import Any

log = logging.getLogger(__name__)


def parse_overrides(raw: str | None) -> dict[str, Any]:
    """Parse ROUTING_MODE_OVERRIDES JSON env-var content into a dict.

    Returns {} on empty/missing input. Logs a warning and returns {} for
    malformed JSON or non-object payloads. Fail-safe default: no overrides.
    """
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("ROUTING_MODE_OVERRIDES malformed JSON; ignoring (%s)", exc)
        return {}
    if not isinstance(data, dict):
        log.warning(
            "ROUTING_MODE_OVERRIDES must be a JSON object mapping domain -> override; ignoring"
        )
        return {}
    return data


def resolve_routing_mode(
    default_mode: str,
    overrides: dict[str, Any],
    domain: str | None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Resolve the effective routing_mode for a request given its domain.

    overrides shape (per-domain key):
      - string value, e.g. ``"bandit"``: full override.
        Every request on this domain takes the override mode.
      - object value with ``mode`` (string) and ``fraction`` (number in (0, 1]):
        canary. On each call ``random()`` is rolled; if below fraction, the
        override mode wins; otherwise the default mode wins.

    Any malformed override (unknown shape, non-string mode, fraction out of
    range) silently falls back to the default mode -- the safe direction.

    ``rng`` is for test seams. Defaults to the ``random`` module's global state.
    """
    if not domain or not overrides or domain not in overrides:
        return default_mode
    ov = overrides[domain]
    if isinstance(ov, str):
        return ov
    if not isinstance(ov, dict):
        return default_mode
    mode = ov.get("mode")
    if not isinstance(mode, str) or not mode:
        return default_mode
    fraction_raw = ov.get("fraction", 1.0)
    try:
        fraction = float(fraction_raw)
    except (TypeError, ValueError):
        return default_mode
    if fraction <= 0.0:
        return default_mode
    if fraction >= 1.0:
        return mode
    roller = rng if rng is not None else random
    return mode if roller.random() < fraction else default_mode
