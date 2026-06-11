# Description: Shared panel composer + chair synthesis + JSONL routing-decisions writer.
# Description: Pure-Python; no FastMCP, no mcporter; consumers wire I/O.

from __future__ import annotations

import json
import logging
import string
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pdp_router._lineage import classify_lineage, filter_by_lineage
from pdp_router._router import DEFAULT_REGISTRY

log = logging.getLogger(__name__)

_DEFAULT_TRUST = 0.5


@dataclass(frozen=True)
class PanelMemberResponse:
    """One panel member's response, wrapped with lineage + measurement metadata.

    `text` may be empty when the member errored or hit a safety filter; in
    that case `error` carries a short reason string. Callers decide whether
    to include the row in the chair-synth survivor list.
    """

    model_id: str
    lineage: str
    text: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    error: str | None


@dataclass(frozen=True)
class ChairSynthResult:
    """Result of a chair-synthesis pass over N panel responses.

    `text` is the synthesized answer. `input_tokens` / `output_tokens` /
    `estimated_cost_usd` reflect the chair call only -- caller aggregates
    panel-member costs separately.
    """

    text: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    chair_model: str
    error: str | None


CHAIR_SYSTEM = (
    "You are the chair of a panel of {n} AI models. Each panelist independently "
    "answered the user's question below. Your job is to produce ONE coherent answer "
    "that synthesizes the panelists' best content.\n\n"
    "Rules:\n"
    "- Use the strongest reasoning and most-accurate facts across panelists.\n"
    "- Resolve direct contradictions explicitly; never silently pick one side.\n"
    "- Drop padding, repetition, and meta-commentary about the question itself.\n"
    "- Do NOT mention 'panelists', 'models', 'sources', or that synthesis happened.\n"
    "- Match the panelists' rough length unless they disagree -- then go shorter.\n"
    "- Plain text, no JSON, no preamble.\n\n"
    "If the original user system prompt is provided above this section, honor it. "
    "The user message you receive contains: (a) the original user prompt, and (b) "
    "the anonymized panelist answers labeled A, B, C..."
)


def compose_panel(
    n: int,
    lineage_filter: str = "all",
    exclude_models: list[str] | None = None,
    *,
    candidates: list[str] | None = None,
    trust_weights: dict[str, float] | None = None,
) -> list[str]:
    """Pick N models for a panel using lineage diversity + trust weights.

    Algorithm:
      1. Build candidate set (defaults to DEFAULT_REGISTRY.available_models()).
      2. Apply lineage_filter ("all", "anthropic", "non_anthropic").
      3. Drop any model in exclude_models.
      4. Group remaining by lineage; sort each lineage internally by
         trust desc (default 0.5 when no trust row exists).
      5. Walk lineages in trust-of-top-member desc order. From each
         lineage, pick its highest-trust unused member. Repeat until
         N is reached or all lineages exhausted.
      6. If still < N, fall back to picking the next-highest-trust
         unused model from any lineage until N reached or pool empty.

    Returns up to N model IDs. Never raises -- if pool is exhausted
    early, returns whatever was picked.
    """
    if n <= 0:
        return []

    if candidates is None:
        candidates = [m.name for m in DEFAULT_REGISTRY.available_models()]

    pool = filter_by_lineage(candidates, lineage_filter)
    if exclude_models:
        excluded = set(exclude_models)
        pool = [m for m in pool if m not in excluded]

    if not pool:
        return []

    weights = trust_weights if trust_weights is not None else {}

    by_lineage: dict[str, list[str]] = {}
    for model_id in pool:
        by_lineage.setdefault(classify_lineage(model_id), []).append(model_id)
    for lineage in by_lineage:
        by_lineage[lineage].sort(
            key=lambda m: weights.get(m, _DEFAULT_TRUST), reverse=True
        )

    lineages_sorted = sorted(
        by_lineage.keys(),
        key=lambda lin: weights.get(by_lineage[lin][0], _DEFAULT_TRUST),
        reverse=True,
    )
    picked: list[str] = []
    cursors: dict[str, int] = dict.fromkeys(by_lineage, 0)
    for lineage in lineages_sorted:
        if len(picked) >= n:
            break
        picked.append(by_lineage[lineage][cursors[lineage]])
        cursors[lineage] += 1

    while len(picked) < n:
        remaining: list[tuple[str, float]] = []
        for lineage in by_lineage:
            members = by_lineage[lineage]
            cur = cursors[lineage]
            if cur < len(members):
                remaining.append((members[cur], weights.get(members[cur], _DEFAULT_TRUST)))
        if not remaining:
            break
        remaining.sort(key=lambda pair: pair[1], reverse=True)
        chosen, _ = remaining[0]
        picked.append(chosen)
        cursors[classify_lineage(chosen)] += 1

    return picked


def synthesize_chair(
    *,
    user_prompt: str,
    system: str,
    panel_responses: list[PanelMemberResponse],
    chair_model_id: str,
    complete_fn: Callable[[str, str, int], Any],
    max_tokens: int = 2048,
) -> ChairSynthResult:
    """Synthesize N panel responses into one coherent answer via the chair model.

    Panelist identities are anonymized to A/B/C/D/... labels before reaching
    the chair to mitigate brand/positional bias. The chair never sees model
    IDs. Survivors are filtered: rows with error != None or empty text are
    excluded before synthesis.

    `complete_fn` must accept (system, user_message, max_tokens) and return
    an object with `text`, `input_tokens`, `output_tokens`, and
    `estimated_cost_usd` attributes (the proxy's CompletionResult).

    Returns a ChairSynthResult; on any exception from `complete_fn`, returns
    a result with `error` populated and zeroed cost/tokens. Never raises.
    """
    survivors = [r for r in panel_responses if r.error is None and r.text.strip()]
    if not survivors:
        return ChairSynthResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            latency_ms=0.0,
            chair_model=chair_model_id,
            error="no_panel_survivors",
        )

    labels = string.ascii_uppercase
    panel_section = "\n\n".join(
        f"PANELIST {labels[i]}:\n{survivors[i].text}" for i in range(len(survivors))
    )
    chair_user = f"USER PROMPT:\n{user_prompt}\n\n---\n\n{panel_section}"
    chair_system_parts: list[str] = []
    if system:
        chair_system_parts.append(system)
    chair_system_parts.append(CHAIR_SYSTEM.format(n=len(survivors)))
    chair_system = "\n\n".join(chair_system_parts)

    t0 = time.monotonic()
    try:
        result = complete_fn(chair_system, chair_user, max_tokens)
    except Exception as e:
        return ChairSynthResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            chair_model=chair_model_id,
            error=str(e)[:200],
        )

    return ChairSynthResult(
        text=getattr(result, "text", "") or "",
        input_tokens=getattr(result, "input_tokens", 0) or 0,
        output_tokens=getattr(result, "output_tokens", 0) or 0,
        estimated_cost_usd=float(getattr(result, "estimated_cost_usd", 0.0) or 0.0),
        latency_ms=(time.monotonic() - t0) * 1000.0,
        chair_model=chair_model_id,
        error=None,
    )


def append_routing_decisions_jsonl(
    *,
    inbox_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Append routing-decision rows to {inbox_dir}/proxy-{YYYYMMDD}.jsonl.

    Fire-and-forget: swallows OSError with a warn log and never raises.
    A downstream drain sidecar reads these files and writes them into the
    pdp-tracker SQLite DB via the canonical record_routing_decision tool.
    """
    if not rows:
        return
    try:
        inbox_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(UTC).strftime("%Y%m%d")
        path = inbox_dir / f"proxy-{today}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    except OSError as e:
        log.warning(
            "Failed to write routing decisions JSONL to %s: %s", inbox_dir, e
        )
