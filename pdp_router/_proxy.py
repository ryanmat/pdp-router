# Description: PDP Router Proxy -- OpenAI-compatible HTTP endpoint for PDP routing.
# Description: Classifies request complexity, routes to best model via confidence cascade.

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pdp_router._clients import CompletionResult, get_client
from pdp_router._effort import level_for_score, supports_effort
from pdp_router._lineage import classify_lineage
from pdp_router._models import (
    GEMINI_FLASH,
    GEMINI_PRO,
    OPUS,
    SONNET,
    CreditExhaustionError,
    expand_canonical_to_live,
)
from pdp_router._panel import (
    PanelMemberResponse,
    append_panel_transcript_jsonl,
    append_routing_decisions_jsonl,
    build_chair_prompt,
    compose_panel,
    synthesize_chair,
)
from pdp_router._proxy_config import ProxyConfig
from pdp_router._router import DEFAULT_REGISTRY, confidence_cascade
from pdp_router._tracing import init_tracing, shutdown_tracing
from pdp_router._utils import strip_markdown_fences

log = logging.getLogger(__name__)

# Soft import for the clawflag flag system. When it is absent (a standalone
# build), _flag_enabled() falls back to PROXY_*_ENABLED environment variables so
# the flag-gated features stay reachable without it.
_clawflag: object | None = None
try:
    import clawflag as _clawflag_mod  # type: ignore[import-not-found]

    _clawflag_mod.init()
    _clawflag = _clawflag_mod
except ImportError:
    _clawflag = None


def _flag_enabled(flag_key: str, env_var: str, default: bool) -> bool:
    """Resolve a boolean feature flag.

    With the clawflag flag system present the value is read from clawflag,
    preserving its behavior. Without it (a standalone build) the same flag is
    togglable via an environment variable, so the flag-gated features stay
    reachable outside the private flag system. The env value is truthy for
    1/true/yes/on (case-insensitive); unset falls back to the default.
    """
    if _clawflag is not None:
        return _clawflag.get_bool(flag_key, default=default)  # type: ignore[attr-defined]
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _streaming_enabled() -> bool:
    """Return True if pipeline.proxy_streaming_enabled is on (default False)."""
    return _flag_enabled(
        "pipeline.proxy_streaming_enabled", "PROXY_STREAMING_ENABLED", default=False
    )


def _autopanel_enabled() -> bool:
    """Return True if pipeline.proxy_autopanel_enabled is on (default False).

    The kill-switch flag for Sprint X.K auto-panel decompose+synth on
    /v1/chat/completions. When off, every request follows the single-model
    cascade path (current behavior). When on, complexity-classified
    panel-worthy requests fan out to N=3 lineage-diverse models and a
    chair-synth pass returns one coherent answer.
    """
    return _flag_enabled(
        "pipeline.proxy_autopanel_enabled", "PROXY_AUTOPANEL_ENABLED", default=False
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
    return _flag_enabled(
        "pipeline.proxy_web_search_enabled", "PROXY_WEB_SEARCH_ENABLED", default=False
    )


def _effort_routing_enabled() -> bool:
    """Return True if pipeline.proxy_effort_routing_enabled is on (default False).

    Gates deterministic effort-aware routing: when on, the proxy maps the
    classifier complexity score to a reasoning-effort level (low/medium/high) and
    threads it to the arms that support a dial (Anthropic non-fast, OpenRouter).
    Default OFF -- ship dark, smoke on real traffic, then enable.
    """
    return _flag_enabled(
        "pipeline.proxy_effort_routing_enabled", "PROXY_EFFORT_ROUTING_ENABLED", default=False
    )


def _route_footer_enabled() -> bool:
    """Return True if pipeline.proxy_route_footer_enabled is on (default False).

    When on, the openai-faithful streaming surface appends a one-line footer naming
    the routed model (+ complexity score / effort) to each response, so a client
    that renders only its configured model label (e.g. Crush, which always shows
    "pdp-auto") can still see which concrete model the cascade picked. Faithful
    surface only -- the default /v1 path is untouched (its consumers already get
    the model via the route_info event / the JSON `model` field).
    """
    return _flag_enabled(
        "pipeline.proxy_route_footer_enabled", "PROXY_ROUTE_FOOTER_ENABLED", default=False
    )


def _panel_transcript_enabled() -> bool:
    """Return True if pipeline.proxy_panel_transcript_enabled is on (default False).

    Gates the panel-transcript sidecar: when on, every panel turn persists its
    prompt + member texts + chair synthesis to {panel_transcript_dir}/panel-*.jsonl
    so a downstream Ryan-grade / judge-ensemble eval can compare synthesis vs the
    single best member. Default OFF -- ship dark, enable to start accruing. Capture
    only (no LLM calls, no effect on the response); independent of the panel flags so
    it can be toggled without touching routing behavior.
    """
    return _flag_enabled(
        "pipeline.proxy_panel_transcript_enabled",
        "PROXY_PANEL_TRANSCRIPT_ENABLED",
        default=False,
    )


def _panel_streaming_enabled() -> bool:
    """Return True if pipeline.proxy_panel_streaming_enabled is on (default True).

    Dedicated kill-switch for streaming the auto-panel chair synthesis on the
    OpenAI-faithful /openai/v1 surface (the Crush front door). Independent of
    pipeline.proxy_autopanel_enabled so panel-on-Crush can be disabled without
    killing panel-on-the-non-streaming-bot, and vice versa. Default ON: the
    feature is live wherever autopanel + streaming are already on. When clawflag
    is absent (test envs) this returns True so the streaming-panel path is
    exercisable; the live kill-switch is the flag read here.
    """
    return _flag_enabled(
        "pipeline.proxy_panel_streaming_enabled",
        "PROXY_PANEL_STREAMING_ENABLED",
        default=True,
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


# -- Web-search intent gating --

# Models that reliably honor an attached web-search / grounding tool: Anthropic
# Sonnet/Opus and Gemini Pro/Flash. Haiku has the tool attached but underuses it
# in practice; meta/* (Vertex Llama) reject the tool; the OpenRouter arms
# (openai/*, qwen*) accept-and-ignore it; DeepSeek has no key on the proxy.
# Gemini Flash-Lite is excluded on purpose -- it is the budget classifier arm and
# its grounding quality is not validated, so a search-intent pick of it is floored
# up to a confirmed searcher rather than trusted to search. Ordered for the floor:
# Sonnet is the default target; the rest are availability fallbacks.
_RELIABLE_SEARCHERS: tuple[str, ...] = (SONNET, OPUS, GEMINI_PRO, GEMINI_FLASH)

# Lexical pre-gate for explicit web-search intent on the latest user message.
# Deterministic (no classifier LLM round-trip) so it needs no test mock and CI
# stays live-call-free. Drives the auto-panel skip + the capability floor so an
# explicit "search the web for X" stays on the cascade path where the web_search
# tool attaches and a model that actually searches answers it.
#
# Precision-first for a heavy coding user: every alternative anchors the search
# verb to a web target ("search the web", "look it up", "google it") or pairs a
# recency word with a news noun, so routine coding phrasing does NOT trip it --
# "binary search for X", "hashtable lookup", "google cloud run", "latest release
# of pytest", "url = https://...", "search for the user in the db" all stay False.
# The cost of that precision is some missed paraphrases; that is acceptable
# because the web_search tool is still attached to every cascade call regardless,
# so a missed query can still search if the model judges it needs to -- it simply
# is not floored or de-paneled.
_SEARCH_INTENT_RE = re.compile(
    r"\bsearch\s+(?:the\s+web|online|the\s+internet)\b"
    r"|\bweb\s+search\b"
    r"|\bbrowse\s+the\s+(?:web|internet)\b"
    r"|\blook\s+(?:it|that|this|them|these|those)\s+up\b"
    r"|\bgoogle\s+(?:it|that|for|the|how|why|what|whether|when|where|who)\b"
    r"|\b(?:latest|current|recent|breaking|today'?s)\s+(?:\w+\s+){0,2}?(?:news|headlines?)\b"
    r"|\bas\s+of\s+(?:today|now)\b",
    re.IGNORECASE,
)


def _is_reliable_searcher(model_name: str) -> bool:
    """True if model_name reliably invokes an attached web-search / grounding tool."""
    return model_name in _RELIABLE_SEARCHERS


def _has_search_intent(messages: list[ChatMessage]) -> bool:
    """Detect explicit web-search intent in the latest user message.

    Mirrors the classifier's latest-user-only convention. Deterministic regex,
    not an LLM call, so it adds no latency, cost, or test-mock surface.
    """
    user_msgs = [m.content for m in messages if m.role == "user"]
    if not user_msgs:
        return False
    return bool(_SEARCH_INTENT_RE.search(user_msgs[-1][:2000]))


def _search_floor_model() -> str | None:
    """Return the preferred AVAILABLE reliable searcher for the search-intent
    floor, or None if none are available.

    Honors the registry's availability contract (route_with_fallback skips
    unavailable models) so the floor never forces a retired or down model: it
    walks _RELIABLE_SEARCHERS in preference order and returns the first one
    marked available. None means "leave the cascade pick" rather than force a
    dead model -- defeating availability fallback for the request class that most
    needs a working model would be worse than not flooring.
    """
    for name in _RELIABLE_SEARCHERS:
        cap = DEFAULT_REGISTRY.get(name)
        if cap is not None and cap.available:
            return name
    return None


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


def _classify_retryable(exc: Exception) -> bool:
    """True if a classifier exception looks transient (worth a bounded retry).

    Provider-SDK-decoupled: prefers a numeric status-code attribute when present
    (429 rate-limit + any 5xx), and falls back to message substrings for the
    common transient signals (the Gemini 503 UNAVAILABLE capacity spike, rate
    limits, timeouts). A bad-credential / bad-request error carries none of these
    and is not retried, so a persistent misconfig falls back immediately rather
    than adding latency to every request.
    """
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code <= 599):
        return True
    text = str(exc).lower()
    return any(
        sig in text
        for sig in (
            "unavailable",
            "resource_exhausted",
            "overloaded",
            "try again",
            "timeout",
            "deadline",
            "503",
            "429",
        )
    )


def _classify_request(messages: list[ChatMessage], config: ProxyConfig) -> tuple[float, int, int]:
    """Classify request complexity, return (confidence, score, panel_score).

    The classifier call is retried a bounded number of times on a TRANSIENT error
    (a provider 503/429, a timeout) before falling back to (0.55, 3, 0) -- the
    Sonnet-tier, no-panel default. The transient retry matters because a single
    capacity blip on the classifier model otherwise zeroes panel_score and
    silently disables the whole auto-panel (streaming + bot). Non-transient errors
    (bad key, etc.) fall back immediately. Retry budget + backoff are env-tunable
    (PROXY_CLASSIFY_RETRIES default 2, PROXY_CLASSIFY_RETRY_BACKOFF_S default 0.2).
    The backoff is a blocking sleep -- acceptable for this single-user proxy, where
    the classifier call is already synchronous on the request path.
    """
    # Classify against the LATEST user message only, not the whole history.
    # The classifier is asking "how complex is THIS request?" -- conversation
    # context isn't relevant and joining all user turns then truncating to
    # 2000 chars drops the newest message (which is the one being asked
    # about) when history exceeds the cap. Caught empirically post-X.K
    # ship 2026-05-23: bot multi-turn DMs to a complex query were getting
    # panel_score=0 because the new query had been truncated out.
    user_msgs = [m.content for m in messages if m.role == "user"]
    user_text = (user_msgs[-1] if user_msgs else "")[:2000]
    retries = int(os.getenv("PROXY_CLASSIFY_RETRIES", "2"))
    backoff = float(os.getenv("PROXY_CLASSIFY_RETRY_BACKOFF_S", "0.2"))

    score, panel_score = 3, 0
    try:
        client = get_client(
            config.classify_model,
            api_key=config.gemini_api_key or config.anthropic_api_key,
            project=config.gcp_project,
            location=config.gcp_location,
        )
        for attempt in range(retries + 1):
            try:
                result = client.complete(
                    system=CLASSIFY_SYSTEM,
                    user_message=user_text,
                    max_tokens=config.classify_max_tokens,
                )
                score, panel_score = _parse_classifier(result.text)
                break
            except Exception as e:
                if attempt < retries and _classify_retryable(e):
                    log.info(
                        "Classifier transient failure (attempt %d/%d): %s; retrying",
                        attempt + 1,
                        retries + 1,
                        e,
                    )
                    if backoff > 0:
                        time.sleep(backoff * (attempt + 1))
                    continue
                raise
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
@app.get("/openai/v1/models")
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
) -> tuple[str, float, int, int, bool, str, list[ChatMessage]]:
    """Apply classification and routing.

    Returns (model_name, confidence, score, panel_score, search_intent, system,
    non_system). panel_score is 0 for explicit-model requests (caller did
    selection; auto-panel never triggers). search_intent is True only when web
    search is enabled AND the latest user message shows explicit search intent;
    it drives the auto-panel skip (handler) and the Sonnet capability floor.

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
        search_intent = False
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

        # Web-search capability floor. Web search attaches only on the cascade
        # path, and only the reliable searchers actually invoke it. When search
        # is on and the user explicitly asked to search, but the cascade (or its
        # ~10% explore branch) picked a model that will not search -- haiku
        # underuses it, meta/* reject it, the OpenRouter arms ignore it -- floor
        # to the preferred AVAILABLE reliable searcher. A model that already
        # searches (e.g. an Opus low-confidence pick) is left untouched, and if
        # no reliable searcher is available the cascade pick stands rather than
        # forcing a down model (that would defeat route_with_fallback's
        # availability contract).
        search_intent = _web_search_enabled() and _has_search_intent(non_system)
        if search_intent and not _is_reliable_searcher(model_name):
            floor_model = _search_floor_model()
            if floor_model is not None:
                log.info(
                    "Search-intent floor: %s -> %s (cascade pick will not web-search)",
                    model_name,
                    floor_model,
                )
                model_name = floor_model
            else:
                log.warning(
                    "Search intent detected but no reliable searcher is available; "
                    "leaving %s -- the search may not fire",
                    model_name,
                )

    return model_name, confidence, score, panel_score, search_intent, system, non_system


def _client_kwargs(model_name: str) -> dict[str, str]:
    """Provider credentials + base_url for get_client, chosen by model-name prefix.

    OpenRouter-fronted arms (openai/*, qwen*) take the OpenRouter key + base_url;
    gemini-* takes the Gemini key; everything else (claude-*, meta/*) takes the
    Anthropic key with the Vertex project/location riding along for meta/* MaaS.
    Shared by _build_client (cascade + chair) and the panel member builder so the
    routing for the new arms lives in exactly one place.
    """
    assert _config is not None
    if model_name.startswith("openai/") or model_name.startswith("qwen"):
        return {
            "api_key": _config.openrouter_api_key,
            "base_url": _config.openrouter_base_url,
        }
    return {
        "api_key": (
            _config.gemini_api_key if model_name.startswith("gemini") else _config.anthropic_api_key
        ),
        "auth_token": "",
        "project": _config.gcp_project,
        "location": _config.gcp_location,
    }


def _build_client(model_name: str) -> object:
    """Instantiate the LLM client for a model, raising HTTPException on failure."""
    assert _config is not None
    try:
        return get_client(model_name, **_client_kwargs(model_name))
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
    search_intent: bool = False,
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
        "search_intent": search_intent,
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
    effort: str | None = None,
) -> ChatCompletionResponse:
    """Single-model cascade execution. Behavior matches the pre-X.K path.

    Extracted from chat_completions so the panel-empty fallback inside
    _execute_panel_with_synth can re-use the same code path without
    duplicating the try/except + log + response-construction logic.

    enable_web_search attaches the provider web-search tool and appends the
    capability hint to the system prompt. The panel-empty fallback leaves it
    False (the panel branch is search-free for the MVP). effort (low/medium/high
    or None) is the deterministic reasoning-effort level for arms that support a
    dial; the panel-empty fallback leaves it None.
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
                effort=effort,
            )
        else:
            messages = [{"role": m.role, "content": m.content} for m in non_system]
            result = client.complete_multi(  # type: ignore[attr-defined]
                system=system,
                messages=messages,
                max_tokens=request.max_tokens,
                enable_web_search=enable_web_search,
                effort=effort,
            )
    except CreditExhaustionError as e:
        raise HTTPException(
            status_code=402,
            detail=str(
                ErrorResponse(error=ErrorDetail(message=str(e), type="billing_error")).model_dump()
            ),
        ) from e
    except Exception as e:
        log.exception("Completion failed on %s", model_name)
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
        "Routed: score=%d conf=%.2f effort=%s model=%s tokens=%d/%d cost=$%.6f latency=%.2fs",
        score,
        confidence,
        effort or "-",
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


async def _run_panel_members(
    *,
    request: ChatCompletionRequest,
    system: str,
    non_system: list[ChatMessage],
    trust_weights: dict[str, float],
    chair_model: str,
    panel_n: int,
) -> tuple[list[PanelMemberResponse], list[str]]:
    """Compose an N-member lineage-diverse panel and run all members concurrently.

    Returns (results, members). `members` is the composed model-id list (empty
    when compose_panel drains the pool, e.g. only the chair is available);
    `results` is the per-member response (an errored member carries error != None
    and empty text). Survivor filtering and the empty-members fallback are the
    caller's job -- the streaming and non-streaming chair paths degrade
    differently. Shared by _execute_panel_with_synth (non-streaming) and
    _iter_panel_sse (faithful streaming) so the fan-out lives in one place.
    """
    members = compose_panel(
        n=panel_n,
        exclude_models=[chair_model],
        trust_weights=trust_weights,
    )
    if not members:
        return [], []

    user_text = non_system[-1].content if non_system else ""
    panel_msgs = [{"role": m.role, "content": m.content} for m in non_system]

    async def _one(model_id: str) -> PanelMemberResponse:
        t0 = time.monotonic()
        try:
            cli = get_client(model_id, **_client_kwargs(model_id))
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
    return list(results), members


def _panel_routing_rows(
    *,
    chat_request_id: str,
    survivors: list[PanelMemberResponse],
    chair_model: str,
    confidence: float,
    score: int,
    panel_score: int,
) -> list[dict[str, Any]]:
    """Build the routing-decision rows for one panel turn.

    One row per surviving member (role=panel_member, routing_mode=panel) plus one
    chair row (role=chair, routing_mode=panel_chair), all sharing chat_request_id.
    Only survivors are passed in, so an errored arm never enters the learning
    substrate. Shared by the streaming and non-streaming panel paths so both write
    identical rows to the inbox.
    """
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
        for r in survivors
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
    return rows


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

    results, members = await _run_panel_members(
        request=request,
        system=system,
        non_system=non_system,
        trust_weights=_trust_cache.get_weights(),
        chair_model=chair_model,
        panel_n=panel_n,
    )
    if not members:
        log.warning(
            "compose_panel returned empty (chair=%s excluded); falling back to cascade",
            chair_model,
        )
        bandit_states = _bandit_cache.get_states() if _config.routing_mode == "bandit" else None
        model_name = confidence_cascade(
            confidence=confidence,
            trust_weights=_trust_cache.get_weights(),
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
        # Panel paths are effort-free in v1 (the dial is cascade/single-model only),
        # so this panel-empty fallback runs at provider-default effort by design.
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

    user_text = non_system[-1].content if non_system else ""
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

    append_routing_decisions_jsonl(
        inbox_dir=_config.routing_inbox_dir,
        rows=_panel_routing_rows(
            chat_request_id=chat_request_id,
            survivors=survivors,
            chair_model=chair_model,
            confidence=confidence,
            score=score,
            panel_score=panel_score,
        ),
    )

    if _panel_transcript_enabled():
        # chair.text (not the fallback-substituted content): an empty synthesis_text
        # records that the chair failed, so the grader never scores single-arm
        # fallback text as a synthesis. synthesis_status mirrors the streaming path's
        # vocabulary so both surfaces produce filter-compatible records.
        synthesis_status = (
            "error" if chair.error else ("complete" if chair.text.strip() else "chair_empty")
        )
        append_panel_transcript_jsonl(
            transcript_dir=_config.panel_transcript_dir,
            chat_request_id=chat_request_id,
            surface="nonstream",
            prompt=user_text,
            messages=[{"role": m.role, "content": m.content} for m in non_system],
            system=system,
            members=survivors,
            synthesis_text=chair.text,
            synthesis_status=synthesis_status,
            chair_model=chair_model,
            panel_score=panel_score,
            score=score,
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


async def _handle_chat(
    request: ChatCompletionRequest,
    response: Response,
    *,
    openai_faithful: bool,
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
        model_name, confidence, score, panel_score, search_intent, system, non_system = (
            _route_request(request)
        )

        effective_stream = request.stream and _streaming_enabled()
        panel_base = (
            request.model in ("pdp-auto", "")
            and _autopanel_enabled()
            and panel_score >= int(os.getenv("PROXY_AUTOPANEL_THRESHOLD", "7"))
        )
        if panel_base and search_intent:
            log.info(
                "Auto-panel skipped for search intent (panel_score=%d); routing to "
                "the cascade so the web_search tool can attach",
                panel_score,
            )
        if panel_base and not search_intent:
            # Streaming panel: only on the OpenAI-faithful surface (the Crush front
            # door) and only when the dedicated kill-switch is on. The default /v1
            # stream (Rust CLI) keeps its single-model route_info shape and never
            # panels, so its contract is untouched.
            if effective_stream and openai_faithful and _panel_streaming_enabled():
                stream_response = await _build_panel_stream_response(
                    request=request,
                    chat_request_id=chat_request_id,
                    confidence=confidence,
                    score=score,
                    panel_score=panel_score,
                    system=system,
                    non_system=non_system,
                )
                stream_response.headers["X-PDP-Prediction-Id"] = chat_request_id
                return stream_response
            # Non-streaming panel (Telegram bot). Gate on the genuine request shape,
            # NOT effective_stream: when the master streaming kill-switch is OFF, a
            # stream:true request must still fall through to the single-model path
            # (as it did pre-change), not fire the ~12x panel.
            if not request.stream:
                return await _execute_panel_with_synth(
                    request=request,
                    chat_request_id=chat_request_id,
                    confidence=confidence,
                    score=score,
                    panel_score=panel_score,
                    system=system,
                    non_system=non_system,
                )
            # Panel-eligible but streaming on the default /v1 surface, or the
            # kill-switch is off: fall through to the single-model cascade below.

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
                    search_intent=search_intent,
                )
            ],
        )

        # Cascade-path web search (default off). The panel branch returned above,
        # so this only reaches the single-model cascade -- the MVP scope.
        web_search_on = _web_search_enabled()

        # Deterministic effort dial (default off). Map the classifier complexity
        # score to a reasoning-effort level for the arms that support one. Only for
        # pdp-auto routed picks -- explicit-model requests carry no real score
        # (score=0) -- and only when supports_effort gates the model in. None means
        # "leave the knob unset" (the no-dial arms: Gemini, DeepSeek, Llama, Haiku).
        effort_level: str | None = None
        if (
            _effort_routing_enabled()
            and request.model in ("pdp-auto", "")
            and supports_effort(model_name)
        ):
            effort_level = level_for_score(
                score,
                low_max=_config.effort_score_low_max,
                high_min=_config.effort_score_high_min,
            )

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
                openai_faithful=openai_faithful,
                effort=effort_level,
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
            effort=effort_level,
        )
    except HTTPException as e:
        merged_headers = dict(e.headers) if e.headers else {}
        merged_headers["X-PDP-Prediction-Id"] = chat_request_id
        raise HTTPException(
            status_code=e.status_code, detail=e.detail, headers=merged_headers
        ) from e


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    response: Response,
) -> ChatCompletionResponse | StreamingResponse:
    """Default chat surface. Streaming emits the pdp `route_info` first SSE event,
    which the Rust CLI requires (it errors without it) and the Telegram bot ignores
    on its non-streaming path. Byte-identical to the pre-faithful behavior."""
    return await _handle_chat(request, response, openai_faithful=False)


@app.post("/openai/v1/chat/completions", response_model=None)
async def chat_completions_openai(
    request: ChatCompletionRequest,
    response: Response,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-faithful chat surface for strict agent clients (e.g. Crush). Streaming
    omits the non-standard `route_info` event and frames the stream with a leading
    assistant-role delta and a terminal `finish_reason: "stop"` chunk. Crush forces
    streaming and rejects the route_info first event as "unexpected EOF"; this surface
    is what it accepts. See claude-code-concerns.md #44."""
    return await _handle_chat(request, response, openai_faithful=True)


# -- Streaming helpers --


def _sse_event(payload: dict) -> str:
    """Encode a single SSE data event, including the trailing blank line."""
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_event(
    completion_id: str,
    created: int,
    model_name: str,
    delta: dict,
    finish_reason: str | None,
) -> str:
    """Encode one OpenAI `chat.completion.chunk` SSE event.

    Single source for the chunk shape so the default and openai-faithful paths
    emit byte-identical content chunks (the default-path bytes must not change --
    the Rust CLI streams against them).
    """
    return _sse_event(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


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
    openai_faithful: bool = False,
    effort: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted events for a streaming completion.

    Two surface shapes, selected by `openai_faithful`:
      - default (False): pdp `route_info` first event -> content chunks -> [DONE],
        and a non-standard `pdp.error` event on mid-stream failure. This is the
        legacy contract the Rust CLI (which REQUIRES a route_info event and errors
        without one) and the Telegram bot rely on; its bytes must not change.
      - faithful (True): a leading assistant-role delta -> content chunks -> a
        terminal `finish_reason: "stop"` chunk -> [DONE], with NO route_info and NO
        pdp.error event. Strict OpenAI agent clients (e.g. Crush) reject the
        route_info first event as "unexpected EOF"; this is the surface they accept.

    With enable_web_search the system prompt is augmented and the provider
    web-search tool attached; the text-delta filter in the client streams only
    answer text (server_tool_use / web_search_tool_result blocks are skipped),
    so there is a brief pause while the search runs but the SSE shape is
    unchanged.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if openai_faithful:
        # OpenAI convention: lead with an assistant-role delta so a strict reader
        # sees a well-formed first chat.completion.chunk rather than the pdp
        # route_info object it does not recognize.
        yield _chunk_event(completion_id, created, model_name, {"role": "assistant"}, None)
    else:
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

    error_message: str | None = None
    try:
        if len(non_system) == 1:
            stream = client.stream_complete(  # type: ignore[attr-defined]
                system=system,
                user_message=non_system[0].content,
                max_tokens=max_tokens,
                enable_web_search=enable_web_search,
                effort=effort,
            )
        else:
            messages = [{"role": m.role, "content": m.content} for m in non_system]
            stream = client.stream_complete_multi(  # type: ignore[attr-defined]
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                enable_web_search=enable_web_search,
                effort=effort,
            )

        async for token in stream:
            yield _chunk_event(completion_id, created, model_name, {"content": token}, None)
    except CreditExhaustionError as e:
        log.warning(
            "Credit exhaustion mid-stream on %s (completion_id=%s): %s",
            model_name,
            completion_id,
            e,
        )
        error_message = str(e)
    except Exception as e:
        # log.exception so the stack trace is captured before we swallow -- on the
        # faithful surface the error frame below is the only client-visible signal,
        # so the server-side record must be complete.
        log.exception("Stream failed on %s (completion_id=%s)", model_name, completion_id)
        error_message = f"Stream failed: {e}"

    if openai_faithful:
        if error_message is not None:
            # OpenAI-compatible mid-stream error frame (the vLLM / LiteLLM
            # convention the OpenAI SDKs surface). Emit it instead of a
            # finish_reason:"stop" so a strict agent client sees a FAILURE rather
            # than reading a truncated or empty turn as a clean completion.
            yield _sse_event({"error": {"message": error_message, "type": "upstream_error"}})
        else:
            # Optional routing footer (default off): a strict client like Crush
            # renders only its configured model label, so append the concrete
            # routed model as a final content chunk before the stop. ASCII-only.
            if _route_footer_enabled():
                tag = f"[routed: {model_name}"
                if score:
                    tag += f" | score {score}"
                if effort:
                    tag += f" | effort {effort}"
                tag += "]"
                yield _chunk_event(
                    completion_id, created, model_name, {"content": f"\n\n`{tag}`"}, None
                )
            # Clean completion: terminal finish_reason chunk so a strict client
            # sees a well-formed end of turn.
            yield _chunk_event(completion_id, created, model_name, {}, "stop")
    elif error_message is not None:
        yield _sse_event({"type": "error", "message": error_message, "object": "pdp.error"})

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
    openai_faithful: bool = False,
    effort: str | None = None,
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
            openai_faithful=openai_faithful,
            effort=effort,
        ),
        media_type="text/event-stream",
    )


# -- Faithful panel streaming (Crush front door) --


def _open_token_stream(
    *,
    client: object,
    system: str,
    non_system: list[ChatMessage],
    max_tokens: int,
    effort: str | None,
) -> AsyncIterator[str]:
    """Open a client token stream (single- or multi-turn) for the faithful panel
    paths. Returns the async iterator; the caller iterates and handles backend
    errors. Web search is off (panel paths are search-free by design)."""
    if len(non_system) == 1:
        return client.stream_complete(  # type: ignore[attr-defined,no-any-return]
            system=system,
            user_message=non_system[0].content,
            max_tokens=max_tokens,
            enable_web_search=False,
            effort=effort,
        )
    messages = [{"role": m.role, "content": m.content} for m in non_system]
    return client.stream_complete_multi(  # type: ignore[attr-defined,no-any-return]
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        enable_web_search=False,
        effort=effort,
    )


async def _faithful_stream_tail(
    *,
    completion_id: str,
    created: int,
    model_label: str,
    token_source: AsyncIterator[str],
    footer: str | None,
    empty_fallback_text: str | None,
    fallback_model_label: str | None = None,
    outcome: list[str] | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens as faithful content chunks, then close the turn.

    Emits each token as a chat.completion.chunk content delta. On a clean finish:
    if nothing was emitted and empty_fallback_text is given, emit that fallback
    (the chair-empty -> first-survivor case), then the optional footer, then a
    terminal finish_reason:"stop" chunk. On a backend failure mid-stream: emit an
    OpenAI {"error":...} frame and NO stop chunk, so a strict client reads a
    failure rather than a clean (empty/truncated) turn -- the same convention as
    _iter_sse's faithful path. Does NOT emit the [DONE] terminator; the caller
    owns it so it can run after writing rows / logging.

    When `outcome` is given, the terminal status is appended before each return so
    the caller can record it: "error" (backend failure mid-stream), "chair_empty"
    (nothing emitted, empty-fallback streamed), or "complete" (clean non-empty
    finish). A caller that never sees an append knows the generator was closed mid
    flight (client disconnect) and the turn was truncated.
    """
    emitted = False
    error_message: str | None = None
    try:
        async for token in token_source:
            # Whitespace-only output does not count as real content -- parity with
            # the non-streaming chair-empty check (not content.strip()).
            if token.strip():
                emitted = True
            yield _chunk_event(completion_id, created, model_label, {"content": token}, None)
    except CreditExhaustionError as e:
        log.warning("Faithful stream credit exhaustion (%s): %s", model_label, e)
        error_message = str(e)
    except Exception as e:
        log.exception("Faithful stream failed (%s)", model_label)
        error_message = f"Stream failed: {e}"

    if error_message is not None:
        if outcome is not None:
            outcome.append("error")
        yield _sse_event({"error": {"message": error_message, "type": "upstream_error"}})
        return

    # When the empty-fallback fires, the closing frames carry fallback_model_label
    # (e.g. ...+chair_fallback) so a downstream reader of the chunk model field does
    # not attribute single-arm output to the synthesis pipeline -- parity with the
    # non-streaming path's response.model relabel.
    used_fallback = not emitted and empty_fallback_text is not None
    close_label = fallback_model_label if (used_fallback and fallback_model_label) else model_label
    if used_fallback:
        yield _chunk_event(
            completion_id, created, close_label, {"content": empty_fallback_text}, None
        )
    if footer:
        yield _chunk_event(
            completion_id, created, close_label, {"content": f"\n\n`{footer}`"}, None
        )
    yield _chunk_event(completion_id, created, close_label, {}, "stop")
    if outcome is not None:
        outcome.append("chair_empty" if used_fallback else "complete")


async def _iter_panel_sse(
    *,
    request: ChatCompletionRequest,
    chat_request_id: str,
    confidence: float,
    score: int,
    panel_score: int,
    system: str,
    non_system: list[ChatMessage],
) -> AsyncGenerator[str, None]:
    """Faithful-surface SSE for a STREAMING auto-panel: fan out N lineage-diverse
    members, then stream the chair synthesis.

    Shape (openai-faithful, no route_info): a leading assistant-role delta, then
    SSE keep-alive comments while the panel fans out (the unavoidable dead-air
    while members run and the chair starts), then chair content chunks, an
    optional footer naming the members, and a terminal finish_reason:"stop" ->
    [DONE]. Failures (zero survivors, chair stream error) surface as an OpenAI
    {"error":...} frame instead of a clean stop, mirroring _iter_sse. Routing rows
    (survivors + chair) are written exactly as the non-streaming path writes them,
    so the learning substrate sees panel exposures identically.
    """
    assert _config is not None
    assert _trust_cache is not None
    assert _bandit_cache is not None

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    start = time.monotonic()

    panel_n = int(os.getenv("PROXY_AUTOPANEL_N", "3"))
    chair_model = os.getenv("PROXY_CHAIR_MODEL", "claude-sonnet-4-6")
    chair_max_tokens = int(os.getenv("PROXY_CHAIR_MAX_TOKENS", "2048"))
    keepalive_s = float(os.getenv("PROXY_PANEL_KEEPALIVE_S", "10"))

    # Lead with the assistant-role delta so a strict reader (Crush) sees a
    # well-formed first chunk immediately, before the long fan-out. The model field
    # is a generic "pdp-panel" placeholder: survivor count (and thus the concrete
    # pdp-panel-N+chair label on the body chunks) is not known until the fan-out
    # completes, and the chunk id -- which OpenAI accumulators key on -- stays constant.
    yield _chunk_event(completion_id, created, "pdp-panel", {"role": "assistant"}, None)

    # Fan the panel out as a task; keep the connection warm with SSE comment
    # keep-alives during the wait. Comment lines (": ...") are ignored by OpenAI
    # stream parsers, so they cost nothing on the rendered content.
    fanout = asyncio.ensure_future(
        _run_panel_members(
            request=request,
            system=system,
            non_system=non_system,
            trust_weights=_trust_cache.get_weights(),
            chair_model=chair_model,
            panel_n=panel_n,
        )
    )
    try:
        while not fanout.done():
            done, _pending = await asyncio.wait(
                {fanout}, timeout=keepalive_s if keepalive_s > 0 else None
            )
            if not done and keepalive_s > 0:
                yield ": panel-keepalive\n\n"
    finally:
        # On client disconnect / GeneratorExit the wait is cancelled; cancel the
        # fan-out task so it is not orphaned. The to_thread member work itself is
        # non-cancellable, so this reclaims the Task object, not the in-flight calls.
        if not fanout.done():
            fanout.cancel()
    try:
        results, members = fanout.result()
    except Exception as e:  # defensive: compose/gather machinery, not member errors
        log.exception("Panel fan-out failed (completion_id=%s)", completion_id)
        yield _sse_event({"error": {"message": f"Panel failed: {e}", "type": "server_error"}})
        yield "data: [DONE]\n\n"
        return

    # Empty members (pool drained, e.g. only the chair available): degrade to a
    # faithful single-model cascade stream, mirroring the non-streaming fallback.
    if not members:
        log.warning(
            "compose_panel empty (chair=%s); streaming cascade fallback (completion_id=%s)",
            chair_model,
            completion_id,
        )
        bandit_states = _bandit_cache.get_states() if _config.routing_mode == "bandit" else None
        model_name = confidence_cascade(
            confidence=confidence,
            trust_weights=_trust_cache.get_weights(),
            explore_rate=_config.explore_rate,
            cost_adjusted=True,
            routing_mode=_config.routing_mode,
            bandit_states=bandit_states,
        )
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
        footer = None
        if _route_footer_enabled():
            tag = f"[routed: {model_name}"
            if score:
                tag += f" | score {score}"
            footer = tag + "]"
        # Build + open AFTER the role delta is already flushed, so a build failure
        # must surface as a faithful error frame, not a torn stream.
        try:
            cascade_cli = _build_client(model_name)
            cascade_source = _open_token_stream(
                client=cascade_cli,
                system=system,
                non_system=non_system,
                max_tokens=request.max_tokens,
                effort=None,
            )
        except Exception as e:
            log.exception("Cascade-fallback client open failed (completion_id=%s)", completion_id)
            yield _sse_event(
                {"error": {"message": f"Cascade fallback failed: {e}", "type": "server_error"}}
            )
            yield "data: [DONE]\n\n"
            return
        async for ev in _faithful_stream_tail(
            completion_id=completion_id,
            created=created,
            model_label=model_name,
            token_source=cascade_source,
            footer=footer,
            empty_fallback_text=None,
        ):
            yield ev
        yield "data: [DONE]\n\n"
        return

    survivors = [r for r in results if r.error is None and r.text.strip()]
    if not survivors:
        reasons = "; ".join(f"{r.model_id}: {r.error or 'empty'}" for r in results)
        log.warning("All panel members failed (completion_id=%s): %s", completion_id, reasons)
        yield _sse_event({"error": {"message": "all panel members failed", "type": "server_error"}})
        yield "data: [DONE]\n\n"
        return

    # Persist the routing rows up front (survivors + chair) so they survive even if
    # the chair stream drops mid-flight. The chair row is attributed to the chair
    # model regardless of synthesis success, matching the non-streaming path.
    append_routing_decisions_jsonl(
        inbox_dir=_config.routing_inbox_dir,
        rows=_panel_routing_rows(
            chat_request_id=chat_request_id,
            survivors=survivors,
            chair_model=chair_model,
            confidence=confidence,
            score=score,
            panel_score=panel_score,
        ),
    )
    log.info(
        "Panel stream: panel_score=%d members=%d survivors=%d chair=%s fanout_latency=%.2fs",
        panel_score,
        len(members),
        len(survivors),
        chair_model,
        time.monotonic() - start,
    )

    response_model_id = f"pdp-panel-{len(survivors)}+{chair_model}"
    footer = None
    if _route_footer_enabled():
        member_names = ", ".join(r.model_id for r in survivors)
        footer = f"[panel: {member_names} | chair: {chair_model}]"

    user_prompt = non_system[-1].content if non_system else ""
    chair_system, chair_user = build_chair_prompt(
        user_prompt=user_prompt,
        system=system,
        survivors=survivors,
    )
    # Build + open AFTER the role delta is flushed, so a build failure surfaces as a
    # faithful error frame rather than a torn stream.
    try:
        chair_cli = _build_client(chair_model)
        chair_source = _open_token_stream(
            client=chair_cli,
            system=chair_system,
            non_system=[ChatMessage(role="user", content=chair_user)],
            max_tokens=chair_max_tokens,
            effort=None,
        )
    except Exception as e:
        log.exception("Chair client open failed (completion_id=%s)", completion_id)
        yield _sse_event({"error": {"message": f"Chair setup failed: {e}", "type": "server_error"}})
        yield "data: [DONE]\n\n"
        return

    # Tee the chair tokens into a buffer so the transcript captures the synthesis the
    # client received -- _faithful_stream_tail consumes the source and never
    # materializes it. Written in the finally so a mid-stream client disconnect
    # (GeneratorExit) still records the complete members plus the partial synthesis.
    chair_buf: list[str] = []
    tail_outcome: list[str] = []

    async def _tee(src: AsyncIterator[str]) -> AsyncGenerator[str, None]:
        async for tok in src:
            chair_buf.append(tok)
            yield tok

    completed_normally = False
    try:
        async for ev in _faithful_stream_tail(
            completion_id=completion_id,
            created=created,
            model_label=response_model_id,
            token_source=_tee(chair_source),
            footer=footer,
            # Chair produced nothing -> stream the first survivor's text (parity with the
            # non-streaming first-survivor fallback), and relabel the closing frames to
            # +chair_fallback so the chunk model field does not report single-arm output
            # as a successful synthesis.
            empty_fallback_text=survivors[0].text,
            fallback_model_label=f"pdp-panel-{len(survivors)}+chair_fallback",
            outcome=tail_outcome,
        ):
            yield ev
        completed_normally = True
    finally:
        # chair_buf is the chair's OWN output (the empty->first-survivor fallback emits
        # from _faithful_stream_tail, not through the tee). synthesis_status lets the
        # grader exclude truncated turns: "complete"/"chair_empty"/"error" come from the
        # tail; if the loop never finished (client disconnect / GeneratorExit) the tail
        # appended nothing and the turn is "disconnect" with a partial synthesis_text.
        if _panel_transcript_enabled():
            if not completed_normally:
                synthesis_status = "disconnect"
            elif tail_outcome:
                synthesis_status = tail_outcome[0]
            else:
                synthesis_status = "complete"
            append_panel_transcript_jsonl(
                transcript_dir=_config.panel_transcript_dir,
                chat_request_id=chat_request_id,
                surface="stream",
                prompt=user_prompt,
                messages=[{"role": m.role, "content": m.content} for m in non_system],
                system=system,
                members=survivors,
                synthesis_text="".join(chair_buf),
                synthesis_status=synthesis_status,
                chair_model=chair_model,
                panel_score=panel_score,
                score=score,
            )
    yield "data: [DONE]\n\n"


async def _build_panel_stream_response(
    *,
    request: ChatCompletionRequest,
    chat_request_id: str,
    confidence: float,
    score: int,
    panel_score: int,
    system: str,
    non_system: list[ChatMessage],
) -> StreamingResponse:
    """Wrap _iter_panel_sse in a FastAPI StreamingResponse (text/event-stream)."""
    return StreamingResponse(
        _iter_panel_sse(
            request=request,
            chat_request_id=chat_request_id,
            confidence=confidence,
            score=score,
            panel_score=panel_score,
            system=system,
            non_system=non_system,
        ),
        media_type="text/event-stream",
    )
