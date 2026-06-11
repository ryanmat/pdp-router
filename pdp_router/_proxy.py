# Description: PDP Router Proxy -- OpenAI-compatible HTTP endpoint for PDP routing.
# Description: Classifies request complexity, routes to best model via confidence cascade.

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pdp_router._clients import CompletionResult, get_client
from pdp_router._lineage import classify_lineage
from pdp_router._models import (
    CreditExhaustionError,
    expand_canonical_to_live,
)
from pdp_router._panel import (
    PanelMemberResponse,
    append_routing_decisions_jsonl,
    compose_panel,
    synthesize_chair,
)
from pdp_router._proxy_config import ProxyConfig
from pdp_router._router import DEFAULT_REGISTRY, confidence_cascade
from pdp_router._tracing import init_tracing, shutdown_tracing
from pdp_router._utils import strip_markdown_fences

log = logging.getLogger(__name__)

# Soft import for clawflag so test environments without it still work. The
# proxy_streaming feature flag defaults to False, so missing clawflag means
# streaming stays gated off until the operator installs and enables it.
_clawflag: object | None = None
try:
    import clawflag as _clawflag_mod  # type: ignore[import-not-found]

    _clawflag_mod.init()
    _clawflag = _clawflag_mod
except ImportError:
    _clawflag = None


def _streaming_enabled() -> bool:
    """Return True if pipeline.proxy_streaming_enabled is on (default False)."""
    if _clawflag is None:
        return False
    return _clawflag.get_bool(  # type: ignore[attr-defined]
        "pipeline.proxy_streaming_enabled", default=False
    )


def _autopanel_enabled() -> bool:
    """Return True if pipeline.proxy_autopanel_enabled is on (default False).

    The kill-switch flag for Sprint X.K auto-panel decompose+synth on
    /v1/chat/completions. When off, every request follows the single-model
    cascade path (current behavior). When on, complexity-classified
    panel-worthy requests fan out to N=3 lineage-diverse models and a
    chair-synth pass returns one coherent answer.
    """
    if _clawflag is None:
        return False
    return _clawflag.get_bool(  # type: ignore[attr-defined]
        "pipeline.proxy_autopanel_enabled", default=False
    )


def _web_search_enabled() -> bool:
    """Return True if pipeline.proxy_web_search_enabled is on (default False).

    Gates provider-native web search (Anthropic web_search server tool; Gemini
    Google Search grounding) on the cascade path so downstream chat surfaces
    (e.g. a chat bot or CLI) -- which reach models only as plain completions --
    can answer "search for X." Default OFF: each search is billed ($10/1k for
    Anthropic) and inflates input tokens with retrieved content. The panel path
    does not consume this flag (MVP is cascade-only).
    """
    if _clawflag is None:
        return False
    return _clawflag.get_bool(  # type: ignore[attr-defined]
        "pipeline.proxy_web_search_enabled", default=False
    )


# A model with a web_search/grounding tool attached still DECIDES whether to use
# it; without an explicit nudge it often answers from training and claims it
# cannot browse. Appended to the system prompt on the cascade path when web
# search is on so "search for X" reliably triggers a search.
_WEB_SEARCH_SYSTEM_HINT = (
    "You have a web_search tool available. Use it whenever the user asks you to "
    "search or look something up, or when answering accurately requires current, "
    "real-time, or post-training-cutoff information. Cite the sources you use."
)


def _augment_system_for_search(system: str) -> str:
    """Append the web-search capability hint to the caller's system prompt."""
    return f"{system}\n\n{_WEB_SEARCH_SYSTEM_HINT}" if system.strip() else _WEB_SEARCH_SYSTEM_HINT


# -- OpenAI-compatible request/response models --


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "pdp-auto"
    messages: list[ChatMessage]
    max_tokens: int = 4096
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ErrorDetail(BaseModel):
    message: str
    type: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


# -- Classification --

CLASSIFY_SYSTEM = (
    "Rate the user's request on two axes.\n"
    "1) Complexity (1-5): 1=trivial lookup, 5=deep frontier-grade reasoning.\n"
    "2) Panel-worth (0-10): how much would synthesizing 3 model perspectives "
    "improve the answer vs picking one model? 0=identical, 10=clearly multi-angle. "
    "High panel-worth signals: 'compare', 'trade-off', 'pros/cons', multi-part "
    "questions, open-ended design/strategy queries, conflicting-info topics.\n"
    "Reply with EXACTLY two integers separated by a single space (e.g. '4 8')."
)

_SCORE_TO_CONFIDENCE = {1: 0.95, 2: 0.75, 3: 0.55, 4: 0.35, 5: 0.15}


def _parse_classifier(text: str) -> tuple[int, int]:
    """Parse the classifier reply into (complexity, panel_score).

    Handles three shapes gracefully:
      - '4 8'  -> (4, 8)            Sprint X.K two-int format.
      - '4'    -> (4, 0)            Pre-X.K single-int back-compat.
      - garbage -> (3, 0)           Sonnet-tier complexity, no panel.

    Complexity clamped to [1, 5]; panel_score clamped to [0, 10].
    """
    try:
        cleaned = strip_markdown_fences(text).strip()
        parts = cleaned.split()
        if len(parts) >= 2:
            return (max(1, min(5, int(parts[0]))), max(0, min(10, int(parts[1]))))
        if len(parts) == 1:
            return (max(1, min(5, int(parts[0]))), 0)
    except (ValueError, IndexError):
        pass
    return (3, 0)


def _classify_request(messages: list[ChatMessage], config: ProxyConfig) -> tuple[float, int, int]:
    """Classify request complexity, return (confidence, score, panel_score).

    Falls back to (0.55, 3, 0) -- Sonnet-tier with no panel routing -- on
    any classifier failure.
    """
    try:
        # Classify against the LATEST user message only, not the whole history.
        # The classifier is asking "how complex is THIS request?" -- conversation
        # context isn't relevant and joining all user turns then truncating to
        # 2000 chars drops the newest message (which is the one being asked
        # about) when history exceeds the cap. Caught empirically post-X.K
        # ship 2026-05-23: bot multi-turn DMs to a complex query were getting
        # panel_score=0 because the new query had been truncated out.
        user_msgs = [m.content for m in messages if m.role == "user"]
        user_text = (user_msgs[-1] if user_msgs else "")[:2000]
        client = get_client(
            config.classify_model,
            api_key=config.gemini_api_key or config.anthropic_api_key,
            project=config.gcp_project,
            location=config.gcp_location,
        )
        result = client.complete(
            system=CLASSIFY_SYSTEM,
            user_message=user_text,
            max_tokens=config.classify_max_tokens,
        )
        score, panel_score = _parse_classifier(result.text)
    except Exception:
        log.warning("Classifier failed, falling back to (3, 0)", exc_info=True)
        score, panel_score = 3, 0

    return _SCORE_TO_CONFIDENCE.get(score, 0.55), score, panel_score


# -- Trust cache --


class TrustCache:
    """Mtime-cached trust weights from pdp-tracker SQLite DB."""

    def __init__(self, db_path: str, ttl: int = 300) -> None:
        self._db_path = db_path
        self._ttl = ttl
        self._weights: dict[str, float] = {}
        self._last_mtime: float = 0.0
        self._last_check: float = time.monotonic()

    def get_weights(self) -> dict[str, float]:
        now = time.monotonic()
        if now - self._last_check < 5.0:
            return self._weights

        self._last_check = now

        try:
            import os

            mtime = os.path.getmtime(self._db_path)
            if mtime == self._last_mtime and (now - self._last_check) < self._ttl:
                return self._weights

            import sqlite3

            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            try:
                rows = conn.execute("SELECT model_id, weight FROM model_trust").fetchall()
                # Storage keys on canonical model_id (e.g. claude-opus-4); the
                # router consumes weights keyed on live registry IDs (e.g.
                # claude-opus-4-7). Fan each canonical row out to every live
                # alias that canonicalizes to it. See _models.expand_canonical_to_live.
                weights: dict[str, float] = {}
                for canonical, weight in rows:
                    for live_id in expand_canonical_to_live(canonical):
                        weights[live_id] = weight
                self._weights = weights
            finally:
                conn.close()

            self._last_mtime = mtime
        except Exception:
            log.debug("Trust DB read failed, using cached weights", exc_info=True)

        return self._weights


# -- Bandit cache --


class BanditCache:
    """Mtime-cached bandit posteriors from pdp-tracker SQLite DB."""

    def __init__(self, db_path: str, ttl: int = 300) -> None:
        self._db_path = db_path
        self._ttl = ttl
        self._states: dict | None = None
        self._last_mtime: float = 0.0
        self._last_check: float = time.monotonic()

    def get_states(self) -> dict | None:
        """Read bandit_state table, return dict[str, BanditState] or None."""
        now = time.monotonic()
        if now - self._last_check < 5.0:
            return self._states

        self._last_check = now

        try:
            import os

            mtime = os.path.getmtime(self._db_path)
            if mtime == self._last_mtime and (now - self._last_check) < self._ttl:
                return self._states

            import sqlite3

            from pdp_router._bandit import BanditState

            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            try:
                rows = conn.execute(
                    "SELECT model_id, mu, sigma, n_obs, sum_reward, "
                    "sum_sq_reward, "
                    "IFNULL(effective_n, 0.0), IFNULL(effective_sum, 0.0) "
                    "FROM bandit_state"
                ).fetchall()
                # Storage keys on canonical model_id; expand to every live
                # registry alias for the router. Mirrors the MCP-tool
                # expansion in pdp-tracker so cache and MCP return the same
                # ID shape regardless of which path the caller uses.
                states: dict[str, BanditState] = {}
                for row in rows:
                    canonical = row[0]
                    posterior = BanditState(
                        mu=row[1],
                        sigma=row[2],
                        n_obs=row[3],
                        sum_reward=row[4],
                        sum_sq_reward=row[5],
                        effective_n=row[6],
                        effective_sum=row[7],
                    )
                    for live_id in expand_canonical_to_live(canonical):
                        states[live_id] = posterior
                self._states = states
            finally:
                conn.close()

            self._last_mtime = mtime
        except Exception:
            log.debug("Bandit DB read failed, using cached states", exc_info=True)

        return self._states


# -- FastAPI app --

_config: ProxyConfig | None = None
_trust_cache: TrustCache | None = None
_bandit_cache: BanditCache | None = None


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _config, _trust_cache, _bandit_cache
    init_tracing()
    _config = ProxyConfig()
    _trust_cache = TrustCache(str(_config.trust_db_path), _config.trust_cache_ttl)
    _bandit_cache = BanditCache(str(_config.trust_db_path), _config.trust_cache_ttl)

    if not _config.anthropic_api_key and not _config.gemini_api_key:
        log.warning("No API keys configured. Set ANTHROPIC_API_KEY or GEMINI_API_KEY.")

    available = DEFAULT_REGISTRY.available_models()
    log.info(
        "PDP Router Proxy started. %d models available. routing_mode=%s",
        len(available),
        _config.routing_mode,
    )
    try:
        yield
    finally:
        shutdown_tracing()


app = FastAPI(title="PDP Router Proxy", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models": len(DEFAULT_REGISTRY.available_models()),
    }


@app.get("/v1/models")
async def list_models() -> dict:
    """OpenAI-compat model list. Returns `pdp-auto` (the cascade-routing virtual
    model that triggers routing) plus every concrete model in `DEFAULT_REGISTRY`
    that is marked available. Clients like Open WebUI, Pal Chat, OpenCat, and
    any OpenAI-compat SDK hit this endpoint at config / refresh time to
    populate their model dropdowns. Without it those clients see "no models
    available" and refuse to send chat completions even though the
    /v1/chat/completions endpoint works fine. `pdp-auto` is the recommended
    default -- it dispatches to whatever model the cascade + bandit pick
    per request; the concrete IDs are surfaced so callers that want to force
    a specific model (post-cascade analysis, side-by-side experiments) can.
    """
    # Created timestamp is required-shape on the OpenAI side; use a static
    # value rather than a fresh `now()` so the response is cache-friendly and
    # idempotent across calls.
    created = 1715000000  # arbitrary; OpenAI clients only check it for non-null
    entries: list[dict] = [
        {
            "id": "pdp-auto",
            "object": "model",
            "created": created,
            "owned_by": "pdp-router",
        }
    ]
    for model in DEFAULT_REGISTRY.available_models():
        entries.append(
            {
                "id": model.name,
                "object": "model",
                "created": created,
                "owned_by": model.provider,
            }
        )
    return {"object": "list", "data": entries}


def _route_request(
    request: ChatCompletionRequest,
) -> tuple[str, float, int, int, str, list[ChatMessage]]:
    """Apply classification and routing.

    Returns (model_name, confidence, score, panel_score, system, non_system).
    panel_score is 0 for explicit-model requests (caller did selection;
    auto-panel never triggers).

    Raises HTTPException(400) on unknown explicit model.
    """
    assert _config is not None
    assert _trust_cache is not None
    assert _bandit_cache is not None

    system_parts = [m.content for m in request.messages if m.role == "system"]
    system = "\n".join(system_parts) if system_parts else ""
    non_system = [m for m in request.messages if m.role != "system"]

    # Honor explicit model when caller has already done selection. This is
    # the path used by an external panel composer after trust-weighted selection
    # has picked N specific members. The cascade remains the default for
    # "pdp-auto" (or unset) so single delegations still feed the learning
    # system. -1.0 / 0 are log-distinguishability markers, not real values.
    if request.model and request.model != "pdp-auto":
        if DEFAULT_REGISTRY.get(request.model) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}",
            )
        model_name = request.model
        confidence, score, panel_score = -1.0, 0, 0
    else:
        confidence, score, panel_score = _classify_request(non_system, _config)
        trust_weights = _trust_cache.get_weights()
        bandit_states = _bandit_cache.get_states() if _config.routing_mode == "bandit" else None
        model_name = confidence_cascade(
            confidence=confidence,
            trust_weights=trust_weights,
            explore_rate=_config.explore_rate,
            cost_adjusted=True,
            routing_mode=_config.routing_mode,
            bandit_states=bandit_states,
        )

    return model_name, confidence, score, panel_score, system, non_system


def _build_client(model_name: str) -> object:
    """Instantiate the LLM client for a model, raising HTTPException on failure."""
    assert _config is not None
    try:
        return get_client(
            model_name,
            api_key=(
                _config.gemini_api_key
                if model_name.startswith("gemini")
                else _config.anthropic_api_key
            ),
            auth_token="",
            project=_config.gcp_project,
            location=_config.gcp_location,
        )
    except Exception as e:
        log.error("Failed to create client for %s: %s", model_name, e)
        raise HTTPException(
            status_code=503,
            detail=str(
                ErrorResponse(
                    error=ErrorDetail(message=f"Client creation failed: {e}", type="server_error")
                ).model_dump()
            ),
        ) from e


def _make_routing_row(
    *,
    chat_request_id: str,
    model_selected: str,
    routing_mode: str,
    context_bucket: str,
    confidence: float,
    score: int,
    panel_score: int,
    role: str | None = None,
) -> dict[str, Any]:
    """Build one routing-decisions row matching pdp-tracker's record_routing_decision schema.

    pdp-tracker's record_routing_decision tool coerces both `prediction_id=0`
    and `prediction_id=None` to NULL in the DB row, so the proxy writes 0 as
    a stable "no upstream prediction id" sentinel that the inbox drain can read
    without ambiguity (the JSONL preserves the 0; the MCP tool's coercion is
    where the NULL appears). The per-request UUID rides in `context_json` as
    `chat_request_id` so the drain can correlate panel-member rows + chair
    row + the X-PDP-Prediction-Id response header without a schema migration.
    """
    context: dict[str, Any] = {
        "chat_request_id": chat_request_id,
        "complexity": score,
        "panel_score": panel_score,
    }
    if role is not None:
        context["role"] = role
    return {
        "alert_id": f"chat-{chat_request_id}",
        "model_selected": model_selected,
        "context_json": json.dumps(context),
        "context_bucket": context_bucket,
        "confidence": confidence,
        "domain": "chat",
        "severity": 0.5,
        "agreement_level": 0,
        "routing_mode": routing_mode,
        "prediction_id": 0,
    }


def _execute_single(
    *,
    request: ChatCompletionRequest,
    client: object,
    model_name: str,
    confidence: float,
    score: int,
    system: str,
    non_system: list[ChatMessage],
    start: float,
    enable_web_search: bool = False,
) -> ChatCompletionResponse:
    """Single-model cascade execution. Behavior matches the pre-X.K path.

    Extracted from chat_completions so the panel-empty fallback inside
    _execute_panel_with_synth can re-use the same code path without
    duplicating the try/except + log + response-construction logic.

    enable_web_search attaches the provider web-search tool and appends the
    capability hint to the system prompt. The panel-empty fallback leaves it
    False (the panel branch is search-free for the MVP).
    """
    if enable_web_search:
        system = _augment_system_for_search(system)
    try:
        if len(non_system) == 1:
            result: CompletionResult = client.complete(  # type: ignore[attr-defined]
                system=system,
                user_message=non_system[0].content,
                max_tokens=request.max_tokens,
                enable_web_search=enable_web_search,
            )
        else:
            messages = [{"role": m.role, "content": m.content} for m in non_system]
            result = client.complete_multi(  # type: ignore[attr-defined]
                system=system,
                messages=messages,
                max_tokens=request.max_tokens,
                enable_web_search=enable_web_search,
            )
    except CreditExhaustionError as e:
        raise HTTPException(
            status_code=402,
            detail=str(
                ErrorResponse(error=ErrorDetail(message=str(e), type="billing_error")).model_dump()
            ),
        ) from e
    except Exception as e:
        log.error("Completion failed on %s: %s", model_name, e)
        raise HTTPException(
            status_code=503,
            detail=str(
                ErrorResponse(
                    error=ErrorDetail(message=f"Completion failed: {e}", type="server_error")
                ).model_dump()
            ),
        ) from e

    elapsed = time.monotonic() - start
    log.info(
        "Routed: score=%d conf=%.2f model=%s tokens=%d/%d cost=$%.6f latency=%.2fs",
        score,
        confidence,
        model_name,
        result.input_tokens,
        result.output_tokens,
        result.estimated_cost_usd,
        elapsed,
    )

    searches = getattr(result, "web_search_requests", 0)
    if searches:
        log.info(
            "web_search: %d search(es) on %s (billed separately ~$10/1k + result tokens)",
            searches,
            model_name,
        )

    content = result.text if result.text is not None else ""
    if not content.strip():
        log.warning(
            "Empty content from %s (input_tokens=%d, output_tokens=%d) -- "
            "likely safety filter or no-output condition. concerns.md item 27.",
            model_name,
            result.input_tokens,
            result.output_tokens,
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
    )


async def _execute_panel_with_synth(
    *,
    request: ChatCompletionRequest,
    chat_request_id: str,
    confidence: float,
    score: int,
    panel_score: int,
    system: str,
    non_system: list[ChatMessage],
) -> ChatCompletionResponse:
    """Fan out to N lineage-diverse panel members, then synthesize via the chair.

    Behind `pipeline.proxy_autopanel_enabled` clawflag (default off). When
    `compose_panel` returns an empty list (e.g. exclude_models drained the
    pool), gracefully degrades to cascade by re-running confidence_cascade
    and returning a single-model response.
    """
    assert _config is not None
    assert _trust_cache is not None
    assert _bandit_cache is not None

    panel_n = int(os.getenv("PROXY_AUTOPANEL_N", "3"))
    chair_model = os.getenv("PROXY_CHAIR_MODEL", "claude-sonnet-4-6")
    chair_max_tokens = int(os.getenv("PROXY_CHAIR_MAX_TOKENS", "2048"))

    members = compose_panel(
        n=panel_n,
        exclude_models=[chair_model],
        trust_weights=_trust_cache.get_weights(),
    )
    if not members:
        log.warning(
            "compose_panel returned empty (chair=%s excluded); falling back to cascade",
            chair_model,
        )
        trust_weights = _trust_cache.get_weights()
        bandit_states = _bandit_cache.get_states() if _config.routing_mode == "bandit" else None
        model_name = confidence_cascade(
            confidence=confidence,
            trust_weights=trust_weights,
            explore_rate=_config.explore_rate,
            cost_adjusted=True,
            routing_mode=_config.routing_mode,
            bandit_states=bandit_states,
        )
        client = _build_client(model_name)
        append_routing_decisions_jsonl(
            inbox_dir=_config.routing_inbox_dir,
            rows=[
                _make_routing_row(
                    chat_request_id=chat_request_id,
                    model_selected=model_name,
                    routing_mode="cascade_panel_empty_fallback",
                    context_bucket="chat:cascade",
                    confidence=confidence,
                    score=score,
                    panel_score=panel_score,
                )
            ],
        )
        return _execute_single(
            request=request,
            client=client,
            model_name=model_name,
            confidence=confidence,
            score=score,
            system=system,
            non_system=non_system,
            start=time.monotonic(),
        )

    user_text = non_system[-1].content if non_system else ""
    panel_msgs = [{"role": m.role, "content": m.content} for m in non_system]

    async def _one(model_id: str) -> PanelMemberResponse:
        t0 = time.monotonic()
        try:
            cli = get_client(
                model_id,
                api_key=(
                    _config.gemini_api_key
                    if model_id.startswith("gemini")
                    else _config.anthropic_api_key
                ),
                project=_config.gcp_project,
                location=_config.gcp_location,
            )
            if len(non_system) == 1:
                r: CompletionResult = await asyncio.to_thread(
                    cli.complete,
                    system=system,
                    user_message=user_text,
                    max_tokens=request.max_tokens,
                )
            else:
                r = await asyncio.to_thread(
                    cli.complete_multi,
                    system=system,
                    messages=panel_msgs,
                    max_tokens=request.max_tokens,
                )
            return PanelMemberResponse(
                model_id=model_id,
                lineage=classify_lineage(model_id),
                text=r.text or "",
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                estimated_cost_usd=r.estimated_cost_usd,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                error=None,
            )
        except Exception as e:
            log.warning("panel member %s failed: %s", model_id, e)
            return PanelMemberResponse(
                model_id=model_id,
                lineage=classify_lineage(model_id),
                text="",
                input_tokens=0,
                output_tokens=0,
                estimated_cost_usd=0.0,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                error=str(e)[:200],
            )

    results = await asyncio.gather(*[_one(m) for m in members])
    survivors = [r for r in results if r.error is None and r.text.strip()]
    if not survivors:
        raise HTTPException(
            status_code=503,
            detail=str(
                ErrorResponse(
                    error=ErrorDetail(message="all panel members failed", type="server_error")
                ).model_dump()
            ),
        )

    chair_cli = _build_client(chair_model)
    chair = synthesize_chair(
        user_prompt=user_text,
        system=system,
        panel_responses=survivors,
        chair_model_id=chair_model,
        complete_fn=lambda s, u, mt: chair_cli.complete(  # type: ignore[attr-defined]
            system=s, user_message=u, max_tokens=mt
        ),
        max_tokens=chair_max_tokens,
    )

    rows = [
        _make_routing_row(
            chat_request_id=chat_request_id,
            model_selected=r.model_id,
            routing_mode="panel",
            context_bucket="chat:panel",
            confidence=confidence,
            score=score,
            panel_score=panel_score,
            role="panel_member",
        )
        for r in results
    ]
    rows.append(
        _make_routing_row(
            chat_request_id=chat_request_id,
            model_selected=chair_model,
            routing_mode="panel_chair",
            context_bucket="chat:panel",
            confidence=confidence,
            score=score,
            panel_score=panel_score,
            role="chair",
        )
    )
    append_routing_decisions_jsonl(
        inbox_dir=_config.routing_inbox_dir,
        rows=rows,
    )

    total_in = sum(r.input_tokens for r in results) + chair.input_tokens
    total_out = sum(r.output_tokens for r in results) + chair.output_tokens
    total_cost = sum(r.estimated_cost_usd for r in results) + chair.estimated_cost_usd

    log.info(
        "Panel routed: panel_score=%d N=%d survivors=%d chair=%s tokens=%d/%d cost=$%.6f",
        panel_score,
        len(members),
        len(survivors),
        chair_model,
        total_in,
        total_out,
        total_cost,
    )

    content = chair.text if chair.text else ""
    chair_failed = False
    if not content.strip():
        chair_failed = True
        log.warning(
            "Chair returned empty content (chair_model=%s error=%s); falling "
            "back to first-survivor text",
            chair_model,
            chair.error,
        )
        content = survivors[0].text

    # When the chair errored we silently substituted a panelist answer; the
    # response.model field needs to reflect that so the inbox drain + downstream
    # grading don't attribute single-arm output to the synthesis pipeline.
    response_model_id = (
        f"pdp-panel-{len(survivors)}+chair_fallback"
        if chair_failed
        else f"pdp-panel-{len(survivors)}+{chair_model}"
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=response_model_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=total_in,
            completion_tokens=total_out,
            total_tokens=total_in + total_out,
        ),
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
) -> ChatCompletionResponse | StreamingResponse:
    assert _config is not None
    assert _trust_cache is not None
    assert _bandit_cache is not None

    start = time.monotonic()

    # One UUID per /v1/chat/completions request, attached as X-PDP-Prediction-Id
    # on every response (cascade + panel + streaming + error paths) and threaded
    # into the routing_decisions JSONL rows via context_json.chat_request_id so
    # the inbox drain + grading workflows can correlate response -> rows -> outcome.
    # On HTTPException, FastAPI builds a fresh JSONResponse and discards the
    # injected Response.headers, so error paths get the header via the inner
    # except below that re-raises with HTTPException(headers=...) instead.
    chat_request_id = str(uuid.uuid4())
    response.headers["X-PDP-Prediction-Id"] = chat_request_id

    try:
        model_name, confidence, score, panel_score, system, non_system = _route_request(request)

        if (
            request.model in ("pdp-auto", "")
            and _autopanel_enabled()
            and panel_score >= int(os.getenv("PROXY_AUTOPANEL_THRESHOLD", "7"))
            and not request.stream
        ):
            return await _execute_panel_with_synth(
                request=request,
                chat_request_id=chat_request_id,
                confidence=confidence,
                score=score,
                panel_score=panel_score,
                system=system,
                non_system=non_system,
            )

        client = _build_client(model_name)

        append_routing_decisions_jsonl(
            inbox_dir=_config.routing_inbox_dir,
            rows=[
                _make_routing_row(
                    chat_request_id=chat_request_id,
                    model_selected=model_name,
                    routing_mode="cascade",
                    context_bucket="chat:cascade",
                    confidence=confidence,
                    score=score,
                    panel_score=panel_score,
                )
            ],
        )

        # Cascade-path web search (default off). The panel branch returned above,
        # so this only reaches the single-model cascade -- the MVP scope.
        web_search_on = _web_search_enabled()

        if request.stream and _streaming_enabled():
            stream_response = await _build_stream_response(
                model_name=model_name,
                confidence=confidence,
                score=score,
                client=client,
                system=system,
                non_system=non_system,
                max_tokens=request.max_tokens,
                enable_web_search=web_search_on,
            )
            stream_response.headers["X-PDP-Prediction-Id"] = chat_request_id
            return stream_response

        if request.stream and not _streaming_enabled():
            log.info(
                "Stream requested but pipeline.proxy_streaming_enabled is off; "
                "returning non-streaming response."
            )

        return _execute_single(
            request=request,
            client=client,
            model_name=model_name,
            confidence=confidence,
            score=score,
            system=system,
            non_system=non_system,
            start=start,
            enable_web_search=web_search_on,
        )
    except HTTPException as e:
        merged_headers = dict(e.headers) if e.headers else {}
        merged_headers["X-PDP-Prediction-Id"] = chat_request_id
        raise HTTPException(
            status_code=e.status_code, detail=e.detail, headers=merged_headers
        ) from e


# -- Streaming helpers --


def _sse_event(payload: dict) -> str:
    """Encode a single SSE data event, including the trailing blank line."""
    return f"data: {json.dumps(payload)}\n\n"


async def _iter_sse(
    *,
    model_name: str,
    confidence: float,
    score: int,
    client: object,
    system: str,
    non_system: list[ChatMessage],
    max_tokens: int,
    enable_web_search: bool = False,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted events: route_info -> chunks -> [DONE].

    With enable_web_search the system prompt is augmented and the provider
    web-search tool attached; the text-delta filter in the client streams only
    answer text (server_tool_use / web_search_tool_result blocks are skipped),
    so there is a brief pause while the search runs but the SSE shape is
    unchanged.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # First event: routing metadata so callers see which model the router picked.
    yield _sse_event(
        {
            "type": "route_info",
            "model": model_name,
            "confidence": confidence,
            "score": score,
            "object": "pdp.route_info",
        }
    )

    if enable_web_search:
        system = _augment_system_for_search(system)

    try:
        if len(non_system) == 1:
            stream = client.stream_complete(  # type: ignore[attr-defined]
                system=system,
                user_message=non_system[0].content,
                max_tokens=max_tokens,
                enable_web_search=enable_web_search,
            )
        else:
            messages = [{"role": m.role, "content": m.content} for m in non_system]
            stream = client.stream_complete_multi(  # type: ignore[attr-defined]
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                enable_web_search=enable_web_search,
            )

        async for token in stream:
            yield _sse_event(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
            )
    except CreditExhaustionError as e:
        log.warning("Credit exhaustion mid-stream on %s: %s", model_name, e)
        yield _sse_event({"type": "error", "message": str(e), "object": "pdp.error"})
    except Exception as e:
        log.error("Stream failed on %s: %s", model_name, e)
        yield _sse_event({"type": "error", "message": f"Stream failed: {e}", "object": "pdp.error"})

    # Terminator. OpenAI clients look for this exact sentinel.
    yield "data: [DONE]\n\n"


async def _build_stream_response(
    *,
    model_name: str,
    confidence: float,
    score: int,
    client: object,
    system: str,
    non_system: list[ChatMessage],
    max_tokens: int,
    enable_web_search: bool = False,
) -> StreamingResponse:
    """Wrap _iter_sse in a FastAPI StreamingResponse with text/event-stream."""
    return StreamingResponse(
        _iter_sse(
            model_name=model_name,
            confidence=confidence,
            score=score,
            client=client,
            system=system,
            non_system=non_system,
            max_tokens=max_tokens,
            enable_web_search=enable_web_search,
        ),
        media_type="text/event-stream",
    )
