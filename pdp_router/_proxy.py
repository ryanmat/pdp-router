# Description: PDP Router Proxy -- OpenAI-compatible HTTP endpoint for PDP routing.
# Description: Classifies request complexity, routes to best model via confidence cascade.

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError, model_validator

from pdp_router._clients import CompletionResult, get_client
from pdp_router._effort import level_for_score, supports_effort
from pdp_router._lineage import classify_lineage
from pdp_router._models import (
    GEMINI_FLASH,
    GEMINI_PRO,
    GPT_5_5,
    OPUS,
    QWEN_3_7_PLUS,
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
from pdp_router._router import DEFAULT_REGISTRY, bandit_branch_active, confidence_cascade
from pdp_router._tools import StreamFinish, ToolCallDelta, ToolTranslationError
from pdp_router._tracing import SERVICE_VERSION, init_tracing, shutdown_tracing
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


def _tool_passthrough_enabled() -> bool:
    """Return True if pipeline.proxy_tool_passthrough_enabled is on (default False).

    Gates OpenAI-compatible tool-call passthrough on the /openai/v1 faithful
    surface: `tools`/`tool_choice` in, assistant `tool_calls` out, so a strict
    agent client can run its own tools while the proxy still routes the model.
    When off, /openai/v1 behaves exactly as it does today -- tool fields are
    dropped and every request takes the plain path. Default OFF because turning
    it on makes the surface tool-capable for any client, not just the one it was
    built for. /v1 is never affected either way.
    """
    return _flag_enabled(
        "pipeline.proxy_tool_passthrough_enabled",
        "PROXY_TOOL_PASSTHROUGH_ENABLED",
        default=False,
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


# -- Tool-call models, used by the /openai/v1 faithful surface only --
#
# Subclasses rather than new fields on the models above, so /v1 keeps its exact
# request contract and its responses stay incapable of carrying a tool field.
# No ConfigDict anywhere in the chain, which keeps pydantic's extra="ignore" in
# force: a strict agent client sends temperature/top_p/stream_options and must
# keep getting them dropped rather than rejected.


class FunctionCallSpec(BaseModel):
    name: str
    # Raw JSON text. Never parsed at the schema layer -- a payload truncated by
    # max_tokens has to reach the translation layer, which owns that error,
    # rather than failing here as a malformed request.
    arguments: str


class ToolCallSpec(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCallSpec


class LenientToolChatMessage(ChatMessage):
    """The message shape /openai/v1 binds: tool fields carried, not checked.

    The endpoint cannot bind ToolChatMessage below, because FastAPI validates in
    its dependency layer -- which runs before the passthrough flag is read. A
    malformed tool call would then be refused even with passthrough OFF, where
    those fields are dropped and the request has always been served. Carrying
    them as Any preserves that answer; _validate_tool_request applies the strict
    shape once the flag is on.
    """

    # Defaulted, not merely nullable: OpenAI clients omit the key entirely on a
    # tool_calls turn rather than sending null.
    content: str | None = None
    tool_calls: Any = None
    # Set on a role:"tool" turn to bind the result back to the call it answers.
    tool_call_id: Any = None
    name: Any = None

    @model_validator(mode="after")
    def _require_content_or_tool_calls(self) -> LenientToolChatMessage:
        """Reject a message that carries neither text nor tool calls.

        Widening content to nullable would otherwise let an empty turn through to
        a provider, which rejects it less helpfully and a good deal later. The
        message text is user-visible: both _openai_validation_handler and
        _validate_tool_request prefix it with the field path, so it has to read
        well as "messages.0: Value error, <this>" and has to name the field.
        """
        if self.content is None and not self.tool_calls:
            raise ValueError("content is required unless the message carries tool_calls")
        return self


class ToolChatMessage(LenientToolChatMessage):
    """The faithful message shape, applied when tool passthrough is on."""

    tool_calls: list[ToolCallSpec] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class LenientToolChatCompletionRequest(ChatCompletionRequest):
    """The request /openai/v1 binds. Same reason as LenientToolChatMessage."""

    messages: list[LenientToolChatMessage]
    tools: Any = None
    tool_choice: Any = None
    parallel_tool_calls: Any = None


class ToolChatCompletionRequest(LenientToolChatCompletionRequest):
    messages: list[ToolChatMessage]
    # Passed through verbatim. Modelling the JSON-Schema internals would mean
    # re-implementing a spec the provider already validates, and any gap between
    # the two would reject tools that actually work.
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None


class ToolResponseMessage(ChatMessage):
    content: str | None = None
    tool_calls: list[ToolCallSpec] | None = None


class ToolChatCompletionChoice(ChatCompletionChoice):
    # Narrowing this annotation is load-bearing, not tidiness. Pydantic serializes
    # a field through its ANNOTATED type, so a ToolResponseMessage left in the
    # inherited `message: ChatMessage` slot dumps without tool_calls -- silently,
    # no error and no warning.
    message: ToolResponseMessage


class ToolChatCompletionResponse(ChatCompletionResponse):
    # Same reason as the choice above, one level up.
    choices: list[ToolChatCompletionChoice]


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
    # `or ""` because a ToolChatMessage carrying tool_calls may omit content
    # entirely, and this runs before the with-tools branch is reached. A no-op
    # for /v1, whose model still requires a string.
    user_msgs = [m.content or "" for m in messages if m.role == "user"]
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


# Tool-call driver floor: the arms tool requests are allowed to route to.
# Membership is policy, not raw client capability: Gemini, Vertex-Llama and
# Ollama raise NotImplementedError on any tool request; Haiku's client can
# serve tools but the spec keeps it out (invariant 5: never receives tool
# requests); DeepSeek's client can too, but _client_kwargs carries no DeepSeek
# credential path. Live registry ids in preference order, mirroring
# _RELIABLE_SEARCHERS.
_TOOL_DRIVERS: tuple[str, ...] = (SONNET, OPUS, GPT_5_5, QWEN_3_7_PLUS)


def _tool_shaped_history(messages: list[ChatMessage]) -> bool:
    """True when the transcript carries tool results or assistant tool_calls.

    Shared by the pre-route 422 guard and the tool branch so the two cannot
    disagree about what counts as tool-shaped.
    """
    return any(m.role == "tool" or getattr(m, "tool_calls", None) for m in messages)


def _first_user_text(messages: list[ChatMessage]) -> str:
    """Content of the FIRST user message, stripped; "" when none exists.

    The driver pin keys on the first user turn because it is stable for the
    life of a conversation; the search gate's latest-user convention would
    re-pin on every turn.
    """
    for message in messages:
        if message.role == "user":
            return (message.content or "").strip()
    return ""


def _is_usable_tool_driver(model_name: str, config: ProxyConfig) -> bool:
    """A driver-set member that is neither retired in the registry nor
    credential-less.

    Composes the registry's availability kill switch with _has_credentials_for
    -- the same two notions _search_floor_model and _build_client already route
    by -- rather than inventing a third availability predicate.
    """
    if model_name not in _TOOL_DRIVERS:
        return False
    cap = DEFAULT_REGISTRY.get(model_name)
    return cap is not None and cap.available and _has_credentials_for(model_name, config)


def _tool_pin_digest(first_user_text: str) -> str:
    """sha256 hexdigest of the conversation pin key.

    The single source for both consumers: the driver walk indexes
    int(digest, 16) % len(_TOOL_DRIVERS) and the routing row records
    digest[:8] as tool_pin_key, so the two cannot drift apart.
    """
    return hashlib.sha256(first_user_text.encode("utf-8")).hexdigest()


def _tool_driver_model(first_user_text: str, config: ProxyConfig) -> str | None:
    """Return the pinned usable tool driver for this conversation, or None.

    sha256 of the first user text picks a starting index into the FULL driver
    tuple, then the walk moves forward with wrap-around past unusable drivers.
    Indexing before filtering keeps the start index independent of driver
    state: a conversation pinned to a usable driver never moves when OTHER
    drivers gain or lose keys. (A conversation already walked past its pin is
    degraded mode; its landing spot follows the rest of the roster.) None
    means no driver is usable; the caller keeps the cascade pick and the
    client's NotImplementedError surfaces on the existing faithful error path
    rather than this floor inventing a dead route.
    """
    start = int(_tool_pin_digest(first_user_text), 16) % len(_TOOL_DRIVERS)
    for offset in range(len(_TOOL_DRIVERS)):
        model_name = _TOOL_DRIVERS[(start + offset) % len(_TOOL_DRIVERS)]
        if _is_usable_tool_driver(model_name, config):
            return model_name
    return None


def _tool_row_context(
    request: ChatCompletionRequest,
    non_system: list[ChatMessage],
    *,
    model_selected: str,
    cascade_pick: str | None,
) -> dict[str, Any]:
    """Request-side observability fields for a tools routing row.

    Everything rides inside context_json -- the drain takes explicit kwargs, so
    top-level row keys are closed -- and every field is omit-when-absent (the
    cascade_explored precedent): model_cascade_pick and tool_pin_key exist only
    for routed picks (an explicit pin consulted neither), tool_choice only when
    the caller sent one, and provider_path only for the two driver families (a
    degraded no-driver pick has no translation path to name).
    """
    tools = getattr(request, "tools", None) or []
    context: dict[str, Any] = {
        "tools_present": bool(tools),
        "tool_count": len(tools),
        "loop_depth": sum(
            1 for m in non_system if m.role == "assistant" and getattr(m, "tool_calls", None)
        ),
    }
    tool_choice = getattr(request, "tool_choice", None)
    if tool_choice is not None:
        context["tool_choice"] = str(tool_choice)
    if cascade_pick is not None:
        context["model_cascade_pick"] = cascade_pick
        context["tool_pin_key"] = _tool_pin_digest(_first_user_text(non_system))[:8]
    if model_selected.startswith("claude-"):
        context["provider_path"] = "anthropic-translated"
    elif model_selected.startswith("openai/") or model_selected.startswith("qwen"):
        context["provider_path"] = "openai-native"
    return context


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


def _parse_classifier(text: str) -> tuple[int, int] | None:
    """Parse the classifier reply into (complexity, panel_score).

    Handles two shapes:
      - '4 8'  -> (4, 8)            Sprint X.K two-int format.
      - '4'    -> (4, 0)            Pre-X.K single-int back-compat.

    Complexity clamped to [1, 5]; panel_score clamped to [0, 10]. Returns
    None on an unparseable reply so _classify_request can treat it as a
    classifier failure (retry/fallback) instead of a silent (3, 0) collapse
    that would disable the auto-panel with no trace.
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
    return None


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
    (a provider 503/429, a timeout). The transient retry matters because a single
    capacity blip on the classifier model otherwise zeroes panel_score and
    silently disables the whole auto-panel (streaming + bot). Retry budget +
    backoff are env-tunable (PROXY_CLASSIFY_RETRIES default 2,
    PROXY_CLASSIFY_RETRY_BACKOFF_S default 0.2). The backoff is a blocking
    sleep -- acceptable for this single-user proxy, where the classifier call is
    already synchronous on the request path.

    When the primary fails outright (retries exhausted, a non-transient error,
    or client construction), one cross-lineage fallback attempt runs on
    config.classify_fallback_model (default Haiku; requires its provider key)
    before the final (0.55, 3, 0) collapse -- the Sonnet-tier, no-panel default.
    """
    # Classify against the LATEST user message only, not the whole history.
    # The classifier is asking "how complex is THIS request?" -- conversation
    # context isn't relevant and joining all user turns then truncating to
    # 2000 chars drops the newest message (which is the one being asked
    # about) when history exceeds the cap. Caught empirically post-X.K
    # ship 2026-05-23: bot multi-turn DMs to a complex query were getting
    # panel_score=0 because the new query had been truncated out.
    user_msgs = [m.content or "" for m in messages if m.role == "user"]
    user_text = (user_msgs[-1] if user_msgs else "")[:2000]
    retries = int(os.getenv("PROXY_CLASSIFY_RETRIES", "2"))
    backoff = float(os.getenv("PROXY_CLASSIFY_RETRY_BACKOFF_S", "0.2"))

    score, panel_score = 3, 0
    try:
        # Preflight the credentials rather than learning it from an exception on
        # every request. The default classifier is gemini-2.5-flash-lite, so a
        # single-provider (Anthropic-only) user would otherwise construct a
        # doomed Gemini client per request and log a ValueError traceback before
        # recovering via the cross-lineage fallback: correct routing that reads
        # as a crash. A genuine failure on a credentialed model still raises and
        # is still logged with its traceback below.
        if not _has_credentials_for(config.classify_model, config):
            raise _MissingClassifierCredentialsError(config.classify_model)
        client = get_client(
            config.classify_model, **_client_kwargs(config.classify_model, config)
        )
        for attempt in range(retries + 1):
            try:
                result = client.complete(
                    system=CLASSIFY_SYSTEM,
                    user_message=user_text,
                    max_tokens=config.classify_max_tokens,
                )
                parsed = _parse_classifier(result.text)
                if parsed is None:
                    raise ValueError(
                        f"unparseable classifier reply: {result.text[:80]!r}"
                    )
                score, panel_score = parsed
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
    except Exception as primary_error:
        fallback = config.classify_fallback_model
        fb_kwargs = _client_kwargs(fallback, config) if fallback else {}
        if fallback and fallback != config.classify_model and fb_kwargs.get("api_key"):
            # Absent credentials are a configuration fact, not a fault. The
            # substitution is announced once at startup (see _lifespan), so per
            # request it is INFO and carries no traceback. A real failure on a
            # credentialed model stays a WARNING with its traceback.
            uncredentialed = isinstance(primary_error, _MissingClassifierCredentialsError)
            log.log(
                logging.INFO if uncredentialed else logging.WARNING,
                "Classifier %s unavailable (%s); using cross-lineage fallback %s",
                config.classify_model,
                "no credentials configured" if uncredentialed else "failed",
                fallback,
                exc_info=not uncredentialed,
            )
            try:
                fb_client = get_client(fallback, **fb_kwargs)
                result = fb_client.complete(
                    system=CLASSIFY_SYSTEM,
                    user_message=user_text,
                    max_tokens=config.classify_max_tokens,
                )
                parsed = _parse_classifier(result.text)
                if parsed is None:
                    raise ValueError(
                        f"unparseable fallback classifier reply: {result.text[:80]!r}"
                    )
                score, panel_score = parsed
            except Exception:
                log.warning(
                    "Fallback classifier %s also failed, falling back to (3, 0)",
                    fallback,
                    exc_info=True,
                )
                score, panel_score = 3, 0
        else:
            log.warning("Classifier failed, falling back to (3, 0)", exc_info=True)
            score, panel_score = 3, 0

    return _SCORE_TO_CONFIDENCE.get(score, 0.55), score, panel_score


# -- Trust cache --


_CACHE_POLL_INTERVAL_S = 5.0


class _MtimeCache:
    """Poll-throttled, mtime-invalidated reader for the read-only trust DB.

    Shared by the trust-weight and bandit-posterior caches: same freshness and
    failure policy, different query.

    Freshness: at most one filesystem poll per _CACHE_POLL_INTERVAL_S; between
    polls the cached value is returned untouched. A poll re-reads when the file
    mtime changed, or when `ttl` has elapsed since the last successful read.

    Failure policy: an absent file is a supported configuration (the confidence
    cascade routes on defaults), so it stays quiet. A file that exists but
    cannot be opened or queried is a user error that silently disables the
    learned layer, so it warns -- once per failure episode rather than once per
    request, and again only after a successful read resets the state.
    """

    # Named in operator-facing warnings; overridden per subclass.
    label = "trust DB"

    def __init__(self, db_path: str, ttl: int = 300) -> None:
        self._db_path = db_path
        self._ttl = ttl
        self._last_mtime: float = 0.0
        # -inf, not time.monotonic(). Priming these to "now" made the poll
        # throttle swallow the very first call and hand back the empty initial
        # value, so a ROUTING_MODE=bandit deployment ran the plain cascade for
        # its first 5 seconds with no indication that it was doing so.
        self._last_poll: float = float("-inf")
        self._last_read: float = float("-inf")
        self._warned_unreadable = False

    def _query(self, conn: sqlite3.Connection) -> None:
        """Populate the subclass's cached value from an open read-only connection."""
        raise NotImplementedError

    def _refresh(self) -> None:
        now = time.monotonic()
        if now - self._last_poll < _CACHE_POLL_INTERVAL_S:
            return
        self._last_poll = now

        try:
            mtime = os.path.getmtime(self._db_path)
            if mtime == self._last_mtime and (now - self._last_read) < self._ttl:
                return

            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            try:
                self._query(conn)
            finally:
                conn.close()

            self._last_mtime = mtime
            self._last_read = now
            self._warned_unreadable = False
        except FileNotFoundError:
            log.debug(
                "%s absent at %s; routing on cascade defaults",
                self.label,
                self._db_path,
            )
        except Exception:
            if not self._warned_unreadable:
                log.warning(
                    "%s at %s exists but could not be read. The learned layer is "
                    "inactive and routing has fallen back to the static cascade. "
                    "Check the schema against the README.",
                    self.label,
                    self._db_path,
                    exc_info=True,
                )
                self._warned_unreadable = True


class TrustCache(_MtimeCache):
    """Mtime-cached trust weights from pdp-tracker SQLite DB."""

    label = "trust DB (model_trust)"

    def __init__(self, db_path: str, ttl: int = 300) -> None:
        super().__init__(db_path, ttl)
        self._weights: dict[str, float] = {}

    def _query(self, conn: sqlite3.Connection) -> None:
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

    def get_weights(self) -> dict[str, float]:
        self._refresh()
        return self._weights


# -- Bandit cache --


class BanditCache(_MtimeCache):
    """Mtime-cached bandit posteriors from pdp-tracker SQLite DB."""

    label = "bandit posterior store (bandit_state)"

    def __init__(self, db_path: str, ttl: int = 300) -> None:
        super().__init__(db_path, ttl)
        self._states: dict | None = None

    def _query(self, conn: sqlite3.Connection) -> None:
        from pdp_router._bandit import BanditState

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

    def get_states(self) -> dict | None:
        """Read bandit_state table, return dict[str, BanditState] or None."""
        self._refresh()
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

    # Configuring bandit mode is not the same as the bandit running: with no
    # readable posteriors, confidence_cascade falls through to the thresholds.
    # Say so at startup rather than letting the operator believe a learned
    # policy is live while the static cascade serves every request.
    if _config.routing_mode == "bandit" and not _bandit_cache.get_states():
        log.warning(
            "ROUTING_MODE=bandit but no bandit posteriors were readable from %s. "
            "Every request will route on the static confidence cascade until the "
            "bandit_state table is populated.",
            _config.trust_db_path,
        )

    # Announce a classifier substitution once here rather than on every request.
    # One provider key is a supported setup, so the per-request path stays quiet.
    if not _has_credentials_for(_config.classify_model, _config):
        fallback = _config.classify_fallback_model
        if fallback and _has_credentials_for(fallback, _config):
            log.warning(
                "Classifier %s has no credentials; every request will use the "
                "cross-lineage fallback %s. Set PROXY_CLASSIFY_MODEL to a model you "
                "have credentials for to remove the extra hop.",
                _config.classify_model,
                fallback,
            )
        else:
            log.warning(
                "Classifier %s has no credentials and no usable fallback; requests "
                "will route at the default complexity (3, 0) with the auto-panel off.",
                _config.classify_model,
            )

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


# Real package metadata, not a literal: the deploy gate reads /health's version
# to prove which build is serving, so a hardcoded string here would lie.
app = FastAPI(title="PDP Router Proxy", version=SERVICE_VERSION, lifespan=_lifespan)


@app.exception_handler(HTTPException)
async def _openai_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Render HTTPException as an OpenAI-compatible top-level {"error": {...}} body.

    FastAPI's default handler nests the detail under "detail", which no
    OpenAI-compatible client reads. Raise sites pass an ErrorResponse.model_dump()
    dict; a plain-string detail is wrapped so ad-hoc raises stay compatible.
    Mirrors the shape the streaming surface already emits in its SSE error frames.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        content = detail
    else:
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        content = ErrorResponse(
            error=ErrorDetail(message=str(detail), type=error_type)
        ).model_dump()
    return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _openai_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render request-validation failures in the same OpenAI error envelope.

    FastAPI's default emits {"detail": [ ... ]}, so without this the surface is
    inconsistent: a client could parse a routing error but not a malformed-request
    error. Real OpenAI returns the envelope for both. The field path is folded
    into the message because that is the part worth reading, and the raw error
    list has no place in an OpenAI-shaped body.
    """
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=ErrorDetail(
                message=_fold_validation_problems(exc.errors()),
                type="invalid_request_error",
            )
        ).model_dump(),
    )


def _fold_validation_problems(errors: list[Any]) -> str:
    """Fold pydantic error records into one OpenAI-shaped message string.

    Shared so a validation failure reads the same whether FastAPI's dependency
    layer raised it or _validate_tool_request did, which is what lets tool
    validation move behind the flag without changing the body a client sees.
    """
    problems = []
    for err in errors:
        # loc starts with "body"/"query" from the dependency layer and with the
        # field itself from a direct model_validate; dropping "body" makes both
        # produce the same path.
        path = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = err.get("msg", "invalid value")
        problems.append(f"{path}: {msg}" if path else msg)
    return "; ".join(problems) or "invalid request"


def _configured_providers(config: ProxyConfig) -> dict[str, bool]:
    """Which provider credentials are actually present in this process.

    Registry size says nothing about reachability: the roster is a static list,
    so /health reports 11 models with zero keys configured. This is what a
    first-run user needs instead, without spending a billable request to find out.
    """
    return {
        "anthropic": bool(config.anthropic_api_key),
        "gemini": bool(config.gemini_api_key),
        "vertex": bool(config.gcp_project),
        "openrouter": bool(config.openrouter_api_key),
    }


def _trust_db_status(config: ProxyConfig) -> dict[str, Any]:
    """Presence and readability of the trust DB.

    Absent is a supported configuration (the cascade routes on defaults), so it
    is reported rather than treated as an error. Present-but-unreadable is a user
    error worth surfacing, because the learned layer silently does nothing.
    """
    path = config.trust_db_path
    present = path.exists()
    readable = False
    if present:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
                readable = True
            finally:
                conn.close()
        except Exception:
            log.warning("Trust DB at %s exists but is not readable", path, exc_info=True)
    return {"path": str(path), "present": present, "readable": readable}


@app.get("/health")
async def health() -> dict:
    config = _config if _config is not None else ProxyConfig()
    return {
        "status": "ok",
        "version": SERVICE_VERSION,
        "models": len(DEFAULT_REGISTRY.available_models()),
        "providers": _configured_providers(config),
        "trust_db": _trust_db_status(config),
        "routing_mode": config.routing_mode,
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
    *,
    for_tools: bool = False,
) -> tuple[str, float, int, int, bool, str, list[ChatMessage], _RouteProvenance]:
    """Apply classification and routing.

    Returns (model_name, confidence, score, panel_score, search_intent, system,
    non_system, provenance). panel_score is 0 for explicit-model requests (caller
    did selection; auto-panel never triggers). search_intent is True only when web
    search is enabled AND the latest user message shows explicit search intent;
    it drives the auto-panel skip (handler) and the Sonnet capability floor.
    provenance carries which policy actually produced the pick, so the routing
    row records the executed mode rather than a hardcoded guess.

    for_tools suppresses the search-intent capability floor: a tools request
    never attaches proxy web search (the tool branch returns above that), so
    flooring its pick to a searcher would only override the driver pin for no
    benefit. search_intent is still computed and recorded; only the model
    rewrite is skipped.

    Raises HTTPException(400) on unknown explicit model.
    """
    assert _config is not None
    assert _trust_cache is not None
    assert _bandit_cache is not None

    system_parts = [m.content or "" for m in request.messages if m.role == "system"]
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
        # No routing ran: the caller chose. Recording this as "cascade" would
        # put caller-pinned picks into the cascade's own outcome statistics.
        provenance = _RouteProvenance(mode="explicit")
    else:
        confidence, score, panel_score = _classify_request(non_system, _config)
        model_name, provenance = _cascade_with_provenance(
            confidence=confidence,
            trust_weights=_trust_cache.get_weights(),
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
        if search_intent and not for_tools and not _is_reliable_searcher(model_name):
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

    return (
        model_name,
        confidence,
        score,
        panel_score,
        search_intent,
        system,
        non_system,
        provenance,
    )


@dataclass(frozen=True)
class _RouteProvenance:
    """How the executed pick was actually made, for the routing row.

    Derived where the decision happens rather than restated at the log site, so
    the recorded policy cannot drift from the policy that ran.
    """

    # "bandit" (Thompson Sampling ran), "cascade" (confidence thresholds), or
    # "explicit" (caller pinned a model, no routing at all).
    mode: str
    # True when the cascade took its epsilon-greedy branch, so the pick is
    # uniform-random rather than threshold-driven. None where no cascade ran.
    explored: bool | None = None


def _cascade_with_provenance(
    *, confidence: float, trust_weights: dict[str, float]
) -> tuple[str, _RouteProvenance]:
    """Run the configured routing policy and report which one actually ran.

    The single place that pairs a routing call with its provenance, so the three
    call sites (main path plus the two panel-empty fallbacks) cannot disagree
    about how to label a decision.
    """
    assert _config is not None
    assert _bandit_cache is not None

    bandit_states = _bandit_cache.get_states() if _config.routing_mode == "bandit" else None
    # return_debug reports whether the epsilon-greedy branch fired, which the
    # drain's cascade_explored column exists to record and nothing populated.
    model_name, explored = confidence_cascade(
        confidence=confidence,
        trust_weights=trust_weights,
        explore_rate=_config.explore_rate,
        cost_adjusted=True,
        routing_mode=_config.routing_mode,
        bandit_states=bandit_states,
        return_debug=True,
    )
    # Configured mode is not executed mode: bandit with an absent or unreadable
    # posterior store falls through to the cascade. Ask the same predicate the
    # router branches on instead of restating the condition.
    if bandit_branch_active(_config.routing_mode, bandit_states):
        return model_name, _RouteProvenance(mode="bandit", explored=False)
    return model_name, _RouteProvenance(mode="cascade", explored=explored)


class _MissingClassifierCredentialsError(RuntimeError):
    """The configured classifier has no credentials in this process.

    Distinguishes "you did not configure this provider" from "the provider
    broke", so the first can be reported in one line and the second keeps its
    traceback. Not part of the public API.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__(f"no credentials configured for {model_name}")
        self.model_name = model_name


def _has_credentials_for(model_name: str, config: ProxyConfig) -> bool:
    """Whether this process holds credentials that can reach model_name.

    Reads the same per-provider mapping the client builder uses
    (_client_kwargs), so the two cannot disagree about what "configured" means.
    Local models need no credentials. Prefix matching here mirrors
    _client_kwargs and moves to the registry when provider dispatch does.
    """
    if model_name.startswith(("ollama/", "local/")):
        return True
    kwargs = _client_kwargs(model_name, config)
    if model_name.startswith("meta/"):
        return bool(kwargs.get("project"))
    return bool(kwargs.get("api_key"))


def _client_kwargs(model_name: str, config: ProxyConfig) -> dict[str, str]:
    """Provider credentials + base_url for get_client, chosen by model-name prefix.

    OpenRouter-fronted arms (openai/*, qwen*) take the OpenRouter key + base_url;
    gemini-* takes the Gemini key; everything else (claude-*, meta/*) takes the
    Anthropic key with the Vertex project/location riding along for meta/* MaaS.
    Shared by _build_client (cascade + chair), the panel member builder, and the
    classifier (primary + fallback) so the routing for the arms lives in exactly
    one place.
    """
    if model_name.startswith("openai/") or model_name.startswith("qwen"):
        return {
            "api_key": config.openrouter_api_key,
            "base_url": config.openrouter_base_url,
        }
    return {
        "api_key": (
            config.gemini_api_key if model_name.startswith("gemini") else config.anthropic_api_key
        ),
        "auth_token": "",
        "project": config.gcp_project,
        "location": config.gcp_location,
    }


def _build_client(model_name: str) -> object:
    """Instantiate the LLM client for a model, raising HTTPException on failure."""
    assert _config is not None
    try:
        return get_client(model_name, **_client_kwargs(model_name, _config))
    except Exception as e:
        log.error("Failed to create client for %s: %s", model_name, e)
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error=ErrorDetail(message=f"Client creation failed: {e}", type="server_error")
            ).model_dump(),
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
    explored: bool | None = None,
    tool_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one routing-decisions row matching pdp-tracker's record_routing_decision schema.

    Every key here must be a parameter of that tool: it takes explicit keyword
    arguments, so a drain doing `record_routing_decision(**row)` raises TypeError
    on anything extra. That constraint is why the per-request UUID rides inside
    `context_json` as `chat_request_id` rather than becoming a top-level field.
    That value is the join key for outcomes, and it equals the
    X-PDP-Prediction-Id response header; `alert_id` carries it too as
    `chat-<uuid>`.

    `prediction_id` is a literal 0, not an id. The tool coerces both 0 and None
    to NULL in the DB row, so the proxy writes 0 as a stable "no upstream
    prediction id" sentinel the drain can read unambiguously. Do not join on it.

    `routing_mode` is the policy that actually produced the pick, and `explored`
    populates `cascade_explored` so uniform-random epsilon-greedy picks can be
    excluded from agreement-rate analytics. Bandit-mode rows record False, not
    None: posterior sampling is the bandit's own exploration mechanism, so it is
    not epsilon-greedy exploration (see confidence_cascade's return_debug).
    `explored=None` omits the key entirely, for rows where the concept does not
    apply at all -- panel members, the chair, and caller-pinned models.

    `tool_context` merges the tools-request observability fields (built by
    _tool_row_context, plus the post-completion fields on the non-stream leg)
    into context_json; None keeps non-tool rows exactly as they are.
    """
    context: dict[str, Any] = {
        "chat_request_id": chat_request_id,
        "complexity": score,
        "panel_score": panel_score,
        "search_intent": search_intent,
    }
    if role is not None:
        context["role"] = role
    if tool_context is not None:
        context.update(tool_context)
    row: dict[str, Any] = {
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
    if explored is not None:
        row["cascade_explored"] = explored
    return row


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
            detail=ErrorResponse(
                error=ErrorDetail(message=str(e), type="billing_error")
            ).model_dump(),
        ) from e
    except Exception as e:
        log.exception("Completion failed on %s", model_name)
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error=ErrorDetail(message=f"Completion failed: {e}", type="server_error")
            ).model_dump(),
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


def _reject_tool_shaped_messages(request: ChatCompletionRequest) -> None:
    """Re-reject a tool_calls turn that carries no content, when passthrough is off.

    Widening the /openai/v1 annotation is itself what changes flag-off behavior:
    LenientToolChatMessage defaults content to None, so a shape that 422s under
    the legacy models would start parsing, and the legacy path would drop its
    tool_calls and hand the model a gutted transcript -- plausible output from a
    request the caller never made. The ruling (2026-07-27) is to keep the legacy
    answer exactly: 422, same envelope, before any routing happens.

    Raised ahead of _route_request so a request that has never reached the
    handler still costs no classifier call.
    """
    for index, message in enumerate(request.messages):
        if getattr(message, "tool_calls", None) and message.content is None:
            raise HTTPException(
                status_code=422,
                detail=ErrorResponse(
                    error=ErrorDetail(
                        message=(
                            f"messages.{index}: content is required because tool "
                            "passthrough is disabled on this proxy"
                        ),
                        type="invalid_request_error",
                    )
                ).model_dump(),
            )


def _validate_tool_request(request: ChatCompletionRequest) -> ToolChatCompletionRequest:
    """Apply the faithful tool schema to a leniently-bound request.

    /openai/v1 binds LenientToolChatCompletionRequest so that the flag-off answer
    for a malformed tool-shaped payload stays what it has always been: the fields
    are dropped and the request is served. Validating them at the annotation
    instead would run in FastAPI's dependency layer, before the flag is read, and
    turn those requests into 422s on a surface whose contract promises identity.

    No strictness is lost -- it moves here, behind the flag, where it guards the
    requests that actually carry tools to a provider. Every shape the annotation
    refused is still refused, with the same status and the same envelope, and the
    field path is folded by the same helper the dependency layer uses.

    What does change is the pydantic message clause, which the contract excludes:
    model_validate reports a mismatch as "or instance of ToolCallSpec" where the
    dependency layer says "or object to extract fields from", and a payload that
    already fails the lenient bind is refused there, so its tool-field problems
    are never reached and the message lists fewer of them.

    Raises:
        HTTPException: 422, in the OpenAI error envelope, on a payload that does
            not match the faithful shape.
    """
    try:
        return ToolChatCompletionRequest.model_validate(request.model_dump())
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error=ErrorDetail(
                    message=_fold_validation_problems(e.errors()),
                    type="invalid_request_error",
                )
            ).model_dump(),
        ) from e


def _execute_single_with_tools(
    *,
    request: ChatCompletionRequest,
    client: object,
    model_name: str,
    confidence: float,
    score: int,
    system: str,
    non_system: list[ChatMessage],
    start: float,
    effort: str | None = None,
) -> ChatCompletionResponse:
    """Single-model execution for a request carrying tools or tool results.

    A sibling of _execute_single rather than a mode on it: that path hardcodes
    finish_reason "stop" and always emits a string content, and both have to
    stay exactly as they are for /v1 and every non-tool request. Web search is
    never attached here -- it would replace the caller's own tools.
    """
    # exclude_none drops an absent content key rather than sending null, which
    # is the shape OpenAI clients themselves send on a tool_calls turn.
    messages = [m.model_dump(exclude_none=True) for m in non_system]
    try:
        result: CompletionResult = client.complete_with_tools(  # type: ignore[attr-defined]
            system=system,
            messages=messages,
            tools=getattr(request, "tools", None) or [],
            tool_choice=getattr(request, "tool_choice", None),
            max_tokens=request.max_tokens,
            effort=effort,
            parallel_tool_calls=getattr(request, "parallel_tool_calls", None),
        )
    except CreditExhaustionError as e:
        raise HTTPException(
            status_code=402,
            detail=ErrorResponse(
                error=ErrorDetail(message=str(e), type="billing_error")
            ).model_dump(),
        ) from e
    except ToolTranslationError as e:
        # The caller's payload cannot be expressed to this provider, which is a
        # bad request rather than a proxy fault.
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error=ErrorDetail(message=str(e), type="invalid_request_error")
            ).model_dump(),
        ) from e
    except Exception as e:
        log.exception("Tool completion failed on %s", model_name)
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error=ErrorDetail(message=f"Completion failed: {e}", type="server_error")
            ).model_dump(),
        ) from e

    elapsed = time.monotonic() - start
    log.info(
        "Routed (tools): score=%d conf=%.2f effort=%s model=%s finish=%s calls=%d "
        "tokens=%d/%d cost=$%.6f latency=%.2fs",
        score,
        confidence,
        effort or "-",
        model_name,
        result.finish_reason,
        len(result.tool_calls),
        result.input_tokens,
        result.output_tokens,
        result.estimated_cost_usd,
        elapsed,
    )

    # Only on a turn that claims to have finished answering. A tool-only turn
    # has no text by design, and warning on it would fire on every tool call.
    if result.finish_reason == "stop" and not (result.text or "").strip():
        log.warning(
            "Empty content from %s (input_tokens=%d, output_tokens=%d) -- "
            "likely safety filter or no-output condition. concerns.md item 27.",
            model_name,
            result.input_tokens,
            result.output_tokens,
        )

    tool_calls = [
        ToolCallSpec(
            id=call.id,
            type="function",
            function=FunctionCallSpec(name=call.name, arguments=call.arguments),
        )
        for call in result.tool_calls
    ]
    return ToolChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            ToolChatCompletionChoice(
                index=0,
                message=ToolResponseMessage(
                    role="assistant",
                    # null rather than "" on a tool-only turn: the OpenAI shape,
                    # and what a strict client expects alongside tool_calls.
                    content=result.text or None,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=result.finish_reason,
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
            cli = get_client(model_id, **_client_kwargs(model_id, _config))
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
        model_name, provenance = _cascade_with_provenance(
            confidence=confidence,
            trust_weights=_trust_cache.get_weights(),
        )
        client = _build_client(model_name)
        append_routing_decisions_jsonl(
            inbox_dir=_config.routing_inbox_dir,
            rows=[
                _make_routing_row(
                    chat_request_id=chat_request_id,
                    model_selected=model_name,
                    # Carries the executed policy: this fallback runs the bandit
                    # when one is configured, so a flat literal would mislabel it.
                    routing_mode=f"{provenance.mode}_panel_empty_fallback",
                    context_bucket="chat:cascade",
                    confidence=confidence,
                    score=score,
                    panel_score=panel_score,
                    explored=provenance.explored,
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
            detail=ErrorResponse(
                error=ErrorDetail(message="all panel members failed", type="server_error")
            ).model_dump(),
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
        tool_passthrough = openai_faithful and _tool_passthrough_enabled()
        if tool_passthrough:
            # Both branches run ahead of _route_request so a refused request
            # still costs no classifier call.
            request = _validate_tool_request(request)
            if _tool_shaped_history(request.messages) and not getattr(request, "tools", None):
                # Anthropic rejects tool blocks without a tools param, and the
                # legacy path may never receive tool-shaped messages: fail loud
                # here instead of leaking a confusing provider 400 downstream.
                raise HTTPException(
                    status_code=422,
                    detail=ErrorResponse(
                        error=ErrorDetail(
                            message="tool history requires tools",
                            type="invalid_request_error",
                        )
                    ).model_dump(),
                )
        elif openai_faithful:
            _reject_tool_shaped_messages(request)

        # A request that will take the tool branch is routed with the search
        # floor suppressed: web search never attaches to a tools request, so
        # flooring its pick to a searcher would only override the driver pin.
        tools_active = tool_passthrough and (
            bool(getattr(request, "tools", None)) or _tool_shaped_history(request.messages)
        )

        (
            model_name,
            confidence,
            score,
            panel_score,
            search_intent,
            system,
            non_system,
            provenance,
        ) = _route_request(request, for_tools=tools_active)

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

        effective_stream = request.stream and _streaming_enabled()
        panel_base = (
            request.model in ("pdp-auto", "")
            and _autopanel_enabled()
            and panel_score >= int(os.getenv("PROXY_AUTOPANEL_THRESHOLD", "7"))
        )
        # Tool passthrough pre-empts everything below: no panel (members are
        # text-only forever) and no proxy web search (attaching it would replace
        # the caller's tools). Keyed on tool-shaped HISTORY as well as a tools
        # param, so a transcript carrying tool_calls or tool results can never
        # fall down the legacy path and have those fields silently dropped.
        if tool_passthrough:
            tools_present = bool(getattr(request, "tools", None))
            if tools_present or _tool_shaped_history(request.messages):
                # The policy's own pick before any override or floor; None for
                # an explicit request, where the caller picked and no cascade
                # ran. Feeds model_cascade_pick/tool_pin_key on the routing row.
                routed_pick: str | None = None
                if request.model and request.model != "pdp-auto":
                    # The caller's explicit pick is honored inside the driver
                    # set and refused outside it -- routing must not silently
                    # replace a model the caller named.
                    if request.model not in _TOOL_DRIVERS:
                        raise HTTPException(
                            status_code=400,
                            detail=ErrorResponse(
                                error=ErrorDetail(
                                    message=(
                                        f"{request.model} does not support tool calls on this proxy"
                                    ),
                                    type="invalid_request_error",
                                )
                            ).model_dump(),
                        )
                else:
                    # Driver floor on the routed path: the pick came from
                    # policy, so a pick that is not a usable driver is replaced,
                    # never refused. PROXY_TOOL_MODEL is the operator override
                    # and beats the pin and any driver pick; an unusable value
                    # degrades to the pin walk instead of failing the request.
                    routed_pick = model_name
                    override = os.getenv("PROXY_TOOL_MODEL", "").strip()
                    if override and _is_usable_tool_driver(override, _config):
                        model_name = override
                    else:
                        if override:
                            log.warning(
                                "PROXY_TOOL_MODEL=%s is not a usable tool driver; ignoring",
                                override,
                            )
                        # Usability, not bare membership: a driver that is
                        # retired (available=False) or credential-less must also
                        # be walked past, or it would be dispatched and fail.
                        if not _is_usable_tool_driver(model_name, _config):
                            driver = _tool_driver_model(_first_user_text(non_system), _config)
                            if driver is not None:
                                log.info(
                                    "Tool-driver floor: %s -> %s (pick is not a usable "
                                    "tool driver)",
                                    model_name,
                                    driver,
                                )
                                model_name = driver
                            else:
                                # No usable driver at all. Invariant 5 is the
                                # router's to keep, not the client's: leaving a
                                # non-driver pick here would dispatch a tool
                                # request to whatever the cascade chose, and a
                                # claude-* non-driver (Haiku) would SERVE it
                                # rather than refuse. Refuse explicitly, before
                                # any client build or routing row, so the
                                # guarantee holds for every provider.
                                log.warning(
                                    "Tool request but no tool driver is usable; refusing "
                                    "(cascade pick was %s)",
                                    model_name,
                                )
                                raise HTTPException(
                                    status_code=503,
                                    detail=ErrorResponse(
                                        error=ErrorDetail(
                                            message=(
                                                "No tool-capable model is currently available"
                                            ),
                                            type="server_error",
                                        )
                                    ).model_dump(),
                                )
                    if (
                        model_name != routed_pick
                        and _effort_routing_enabled()
                        and supports_effort(model_name)
                    ):
                        # The dial ran against the pre-floor pick: a haiku or
                        # gemini pick computed None while every driver supports
                        # a level. The search floor gets its level naturally by
                        # running before the dial; this floor runs after, so it
                        # recomputes.
                        effort_level = level_for_score(
                            score,
                            low_max=_config.effort_score_low_max,
                            high_min=_config.effort_score_high_min,
                        )
                client = _build_client(model_name)
                tool_context = _tool_row_context(
                    request,
                    non_system,
                    model_selected=model_name,
                    cascade_pick=routed_pick,
                )
                row_kwargs: dict[str, Any] = {
                    "chat_request_id": chat_request_id,
                    "model_selected": model_name,
                    "routing_mode": provenance.mode,
                    "context_bucket": "chat:cascade",
                    "confidence": confidence,
                    "score": score,
                    "panel_score": panel_score,
                    "search_intent": search_intent,
                    "explored": provenance.explored,
                }
                if effective_stream:
                    # The stream row is written before the body runs, so it
                    # carries the request-side fields only; the outcome is not
                    # knowable here.
                    append_routing_decisions_jsonl(
                        inbox_dir=_config.routing_inbox_dir,
                        rows=[_make_routing_row(**row_kwargs, tool_context=tool_context)],
                    )
                    stream_response = await _build_tool_stream_response(
                        model_name=model_name,
                        client=client,
                        request=request,
                        system=system,
                        non_system=non_system,
                        score=score,
                        effort=effort_level,
                    )
                    stream_response.headers["X-PDP-Prediction-Id"] = chat_request_id
                    return stream_response
                # The non-stream row is written after completion so it can
                # carry finish_reason/tool_call_count/tool_names. A failed
                # completion is still a real exposure: its row is written from
                # the except path, without outcome fields it never produced.
                try:
                    tool_response = _execute_single_with_tools(
                        request=request,
                        client=client,
                        model_name=model_name,
                        confidence=confidence,
                        score=score,
                        system=system,
                        non_system=non_system,
                        start=start,
                        effort=effort_level,
                    )
                except HTTPException:
                    # Catches every executor failure because
                    # _execute_single_with_tools wraps each raise in an
                    # HTTPException (bare Exception -> 503); the regression
                    # wall pins that a bare client raise still lands here
                    # rather than silently dropping the exposure row.
                    append_routing_decisions_jsonl(
                        inbox_dir=_config.routing_inbox_dir,
                        rows=[_make_routing_row(**row_kwargs, tool_context=tool_context)],
                    )
                    raise
                choice = tool_response.choices[0]
                calls = choice.message.tool_calls or []
                tool_context.update(
                    finish_reason=choice.finish_reason,
                    tool_call_count=len(calls),
                    tool_names=[call.function.name for call in calls[:8]],
                )
                append_routing_decisions_jsonl(
                    inbox_dir=_config.routing_inbox_dir,
                    rows=[_make_routing_row(**row_kwargs, tool_context=tool_context)],
                )
                return tool_response

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
                    routing_mode=provenance.mode,
                    context_bucket="chat:cascade",
                    confidence=confidence,
                    score=score,
                    panel_score=panel_score,
                    search_intent=search_intent,
                    explored=provenance.explored,
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
    request: LenientToolChatCompletionRequest,
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


def _route_footer_content(model_name: str, score: int, effort: str | None) -> str:
    """Render the routing footer as a content delta.

    Shared by the plain and with-tools faithful generators so a tool turn that
    ends in text cannot drift into a different footer than a plain one.
    """
    tag = f"[routed: {model_name}"
    if score:
        tag += f" | score {score}"
    if effort:
        tag += f" | effort {effort}"
    tag += "]"
    return f"\n\n`{tag}`"


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
                yield _chunk_event(
                    completion_id,
                    created,
                    model_name,
                    {"content": _route_footer_content(model_name, score, effort)},
                    None,
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


def _tool_call_fragment(delta: ToolCallDelta) -> dict:
    """Render one ToolCallDelta as an OpenAI streaming tool_calls fragment.

    The announcing fragment of a call carries id, type and function.name; every
    later fragment carries only argument text against the same index, which is
    what ties the pieces of a call together when several stream in parallel.
    """
    if delta.id is not None or delta.name is not None:
        return {
            "index": delta.index,
            "id": delta.id,
            "type": "function",
            "function": {"name": delta.name, "arguments": delta.arguments},
        }
    return {"index": delta.index, "function": {"arguments": delta.arguments}}


async def _iter_sse_with_tools(
    *,
    model_name: str,
    client: object,
    request: ChatCompletionRequest,
    system: str,
    non_system: list[ChatMessage],
    score: int,
    effort: str | None = None,
) -> AsyncGenerator[str, None]:
    """Yield SSE events for a streaming turn that may call tools.

    A sibling of _iter_sse rather than a mode on it, for the reason the clients
    give: _iter_sse backs live plain streaming on BOTH surfaces and its emitted
    bytes are pinned by a golden. This generator serves /openai/v1 only, because
    tool passthrough is gated on the faithful surface, so it has no route_info
    branch at all.

    Shape: a leading assistant-role delta, then content deltas and tool-call
    fragments in arrival order, then a terminal finish_reason chunk and [DONE].
    On a mid-stream failure it emits an OpenAI {"error":...} frame and NO finish
    chunk, the same convention as _iter_sse's faithful path -- a truncated tool
    call read as a completed one is worse than an explicit failure, because the
    caller would execute it.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    yield _chunk_event(completion_id, created, model_name, {"role": "assistant"}, None)

    # exclude_none drops an absent content key rather than sending null, the
    # shape OpenAI clients themselves send on a tool_calls turn.
    messages = [m.model_dump(exclude_none=True) for m in non_system]
    # None until a StreamFinish arrives, so an absent reason stays tellable from
    # a reported one. What the provider sends is passed through verbatim.
    finish_reason: str | None = None
    # What this generator actually put on the wire. The provider's label cannot
    # answer "was this a tool turn": an Anthropic stop_reason outside
    # _STOP_REASONS (pause_turn, refusal) maps to "stop" on a genuine tool turn,
    # and a provider can stream calls and then end without any reason at all.
    saw_tool_call = False
    error_message: str | None = None
    # Frame-type parity with the non-stream leg: once the stream is open the HTTP
    # status is already 200, so a client-payload error can only be signalled by
    # the error frame's type, not a 4xx. A translation failure is the caller's
    # bad request there (400 invalid_request_error), so it says so here too.
    error_type = "upstream_error"
    try:
        stream = client.stream_with_tools(  # type: ignore[attr-defined]
            system=system,
            messages=messages,
            tools=getattr(request, "tools", None) or [],
            tool_choice=getattr(request, "tool_choice", None),
            max_tokens=request.max_tokens,
            effort=effort,
            parallel_tool_calls=getattr(request, "parallel_tool_calls", None),
        )
        async for event in stream:
            if isinstance(event, ToolCallDelta):
                saw_tool_call = True
                yield _chunk_event(
                    completion_id,
                    created,
                    model_name,
                    {"tool_calls": [_tool_call_fragment(event)]},
                    None,
                )
            elif isinstance(event, StreamFinish):
                finish_reason = event.finish_reason
            else:
                yield _chunk_event(completion_id, created, model_name, {"content": event}, None)
    except CreditExhaustionError as e:
        log.warning(
            "Credit exhaustion mid-tool-stream on %s (completion_id=%s): %s",
            model_name,
            completion_id,
            e,
        )
        error_message = str(e)
    except ToolTranslationError as e:
        # The caller's payload cannot be expressed to this provider -- a bad
        # request, classified the same as the non-stream leg's 400.
        log.warning(
            "Tool translation failed mid-stream on %s (completion_id=%s): %s",
            model_name,
            completion_id,
            e,
        )
        error_message = str(e)
        error_type = "invalid_request_error"
    except Exception as e:
        # log.exception before the swallow: the error frame below is the only
        # client-visible signal, so the server-side record has to be complete.
        log.exception("Tool stream failed on %s (completion_id=%s)", model_name, completion_id)
        error_message = f"Stream failed: {e}"

    if error_message is not None:
        yield _sse_event({"error": {"message": error_message, "type": error_type}})
    else:
        # Only the INVENTED reason is inferred; a reported one rides through
        # verbatim (d926aa8 pinned that). A provider that streamed calls and then
        # ended without a reason produced a tool turn, not a text one.
        if finish_reason is None:
            finish_reason = "tool_calls" if saw_tool_call else "stop"
        # Footer only on a turn that emitted no tool call, keyed on what this
        # generator sent rather than on the provider's label. It is display text
        # the model never wrote, and appending it to a turn whose payload is a
        # call corrupts the caller's agent loop -- the client is not rendering
        # prose, it is about to execute the call.
        if not saw_tool_call and finish_reason == "stop" and _route_footer_enabled():
            yield _chunk_event(
                completion_id,
                created,
                model_name,
                {"content": _route_footer_content(model_name, score, effort)},
                None,
            )
        yield _chunk_event(completion_id, created, model_name, {}, finish_reason)

    yield "data: [DONE]\n\n"


async def _build_tool_stream_response(
    *,
    model_name: str,
    client: object,
    request: ChatCompletionRequest,
    system: str,
    non_system: list[ChatMessage],
    score: int,
    effort: str | None = None,
) -> StreamingResponse:
    """Wrap _iter_sse_with_tools in a FastAPI StreamingResponse."""
    return StreamingResponse(
        _iter_sse_with_tools(
            model_name=model_name,
            client=client,
            request=request,
            system=system,
            non_system=non_system,
            score=score,
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
        model_name, provenance = _cascade_with_provenance(
            confidence=confidence,
            trust_weights=_trust_cache.get_weights(),
        )
        append_routing_decisions_jsonl(
            inbox_dir=_config.routing_inbox_dir,
            rows=[
                _make_routing_row(
                    chat_request_id=chat_request_id,
                    model_selected=model_name,
                    # Carries the executed policy: this fallback runs the bandit
                    # when one is configured, so a flat literal would mislabel it.
                    routing_mode=f"{provenance.mode}_panel_empty_fallback",
                    context_bucket="chat:cascade",
                    confidence=confidence,
                    score=score,
                    panel_score=panel_score,
                    explored=provenance.explored,
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
