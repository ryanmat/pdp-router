# Description: Shared panel composer + chair synthesis + JSONL routing-decisions and
# Description: panel-transcript writers. Pure-Python; no FastMCP, no mcporter; consumers wire I/O.

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
        by_lineage[lineage].sort(key=lambda m: weights.get(m, _DEFAULT_TRUST), reverse=True)

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


def build_chair_prompt(
    *,
    user_prompt: str,
    system: str,
    survivors: list[PanelMemberResponse],
) -> tuple[str, str]:
    """Build the (chair_system, chair_user) prompt for chair synthesis.

    Panelist identities are anonymized to A/B/C/... labels so the chair never
    sees model IDs (brand / positional bias mitigation). `survivors` must already
    be the error-free, non-empty-text subset -- this helper does not re-filter.
    Shared by synthesize_chair (blocking) and the proxy streaming chair path so
    both build byte-identical prompts.
    """
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
    return chair_system, chair_user


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

    chair_system, chair_user = build_chair_prompt(
        user_prompt=user_prompt, system=system, survivors=survivors
    )

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
        log.warning("Failed to write routing decisions JSONL to %s: %s", inbox_dir, e)


def append_panel_transcript_jsonl(
    *,
    transcript_dir: Path,
    chat_request_id: str,
    surface: str,
    prompt: str,
    messages: list[dict[str, str]],
    system: str,
    members: list[PanelMemberResponse],
    synthesis_text: str,
    synthesis_status: str,
    chair_model: str,
    panel_score: int,
    score: int,
) -> None:
    """Append ONE panel-turn transcript object to {transcript_dir}/panel-{YYYYMMDD}.jsonl.

    Unlike the routing-decisions writer (one metadata row per member + chair), this
    persists the full gradeable artifact for a single panel turn as ONE JSON object:
    the prompt, each member's response text, and the chair synthesis. It is the only
    place the member/synthesis TEXT is written to disk -- the routing inbox keeps
    metadata only, and both panel paths otherwise discard the texts after the chair
    runs. A downstream Ryan-grade / judge-ensemble eval reads one line == one
    gradeable turn (synthesis vs single best member).

    `members` is the survivor list (error-free, non-empty text); an empty list is a
    no-op (nothing to grade). `messages` is the full conversation the members actually
    saw (members answer multi-turn requests with history, so the grader needs more than
    the last turn); `prompt` is the last user message kept as a convenience key.

    `synthesis_status` lets the grader exclude non-clean turns so a truncated answer is
    never scored as a real synthesis (which would bias the eval against synthesis):
      - "complete": the chair finished cleanly with non-empty output.
      - "chair_empty": the chair produced nothing; the caller streamed/returned the
        first-survivor fallback to the client, but synthesis_text stays "".
      - "error": the chair stream raised mid-flight; synthesis_text holds the partial
        tokens emitted before the error and the client saw a failed turn.
      - "disconnect": the client went away mid-stream; synthesis_text holds whatever
        had streamed. The non-streaming path never produces this status.

    Fire-and-forget: swallows ANY exception with a warn-with-traceback log and never
    raises into the request path -- it runs inside the streaming `finally` (around a
    GeneratorExit on client disconnect), so an escape would corrupt the SSE or replace
    the in-flight GeneratorExit. Single-writer assumption: one uvicorn worker serializes
    appends; a multi-worker scale-out would need O_APPEND-atomic or per-turn files since
    these records exceed PIPE_BUF. No rotation -- a long-running enabled flag accrues
    full-text records (~few MB/week at panel rates); prune manually if it grows.
    Local-only sidecar, same trust boundary as the routing inbox.
    """
    if not members:
        return
    now = datetime.now(UTC)
    record = {
        "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chat_request_id": chat_request_id,
        "surface": surface,
        "panel_score": panel_score,
        "score": score,
        "chair_model": chair_model,
        "system": system,
        "prompt": prompt,
        "messages": messages,
        "members": [
            {
                "model_id": m.model_id,
                "lineage": m.lineage,
                "text": m.text,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "estimated_cost_usd": m.estimated_cost_usd,
            }
            for m in members
        ],
        "synthesis_text": synthesis_text,
        "synthesis_status": synthesis_status,
    }
    try:
        # Serialize BEFORE touching the filesystem so a non-serializable field fails
        # without leaving an empty turd file (open("a") would create one first).
        line = json.dumps(record) + "\n"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        path = transcript_dir / f"panel-{now.strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        log.warning("Failed to write panel transcript JSONL to %s", transcript_dir, exc_info=True)
