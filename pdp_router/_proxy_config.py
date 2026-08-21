# Description: Configuration for the PDP Router Proxy service.
# Description: Reads settings from environment variables with sensible defaults.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from pdp_router._domain_overrides import parse_overrides
from pdp_router._models import HAIKU


@dataclass(frozen=True)
class ProxyConfig:
    """Configuration for the PDP Router Proxy."""

    host: str = field(default_factory=lambda: os.getenv("PROXY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PROXY_PORT", "7741")))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gcp_project: str = field(default_factory=lambda: os.getenv("GCP_PROJECT", ""))
    gcp_location: str = field(default_factory=lambda: os.getenv("GCP_LOCATION", "us-east5"))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    classify_model: str = field(
        default_factory=lambda: os.getenv("PROXY_CLASSIFY_MODEL", "gemini-2.5-flash-lite")
    )
    classify_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("PROXY_CLASSIFY_MAX_TOKENS", "16"))
    )
    # Cross-lineage fallback when the primary classifier fails outright: one
    # attempt before the (3, 0) collapse that disables the auto-panel. Empty
    # string disables the fallback.
    classify_fallback_model: str = field(
        default_factory=lambda: os.getenv("PROXY_CLASSIFY_FALLBACK_MODEL", HAIKU)
    )
    trust_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "PROXY_TRUST_DB",
                os.path.expanduser("~/.pdp-router/pdp_tracker.db"),
            )
        )
    )
    trust_cache_ttl: int = field(
        default_factory=lambda: int(os.getenv("PROXY_TRUST_CACHE_TTL", "300"))
    )
    explore_rate: float = field(
        default_factory=lambda: float(os.getenv("PROXY_EXPLORE_RATE", "0.10"))
    )
    routing_mode: str = field(default_factory=lambda: os.getenv("ROUTING_MODE", "cascade"))
    routing_mode_overrides: dict = field(
        default_factory=lambda: parse_overrides(os.getenv("ROUTING_MODE_OVERRIDES"))
    )
    routing_inbox_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "PROXY_ROUTING_INBOX_DIR",
                os.path.expanduser("~/.pdp-router/inbox"),
            )
        )
    )
    # Sidecar for full panel-turn transcripts (prompt + member texts + synthesis),
    # the only on-disk source the chat-quality eval can grade. Written only when the
    # proxy_panel_transcript_enabled flag is on. Defaults beside the inbox: when
    # PROXY_PANEL_TRANSCRIPT_DIR is unset, __post_init__ derives this from
    # routing_inbox_dir so the two cannot drift (see __post_init__).
    panel_transcript_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "PROXY_PANEL_TRANSCRIPT_DIR",
                os.path.expanduser("~/.pdp-router/panel-transcripts"),
            )
        )
    )
    # Deterministic effort-routing thresholds on the 1-5 classifier complexity
    # score: score <= low_max -> low effort, score >= high_min -> high, else medium.
    # Only consulted when the proxy_effort_routing_enabled flag is on.
    effort_score_low_max: int = field(
        default_factory=lambda: int(os.getenv("PROXY_EFFORT_SCORE_LOW_MAX", "2"))
    )
    effort_score_high_min: int = field(
        default_factory=lambda: int(os.getenv("PROXY_EFFORT_SCORE_HIGH_MIN", "4"))
    )
    # Per-conversation state cache (sticky tool driver, spend accounting,
    # implicit-feedback lineage). Entries are keyed by the first-user-turn
    # digest; bounds keep a long-running proxy's memory flat.
    conversation_cache_max: int = field(
        default_factory=lambda: int(os.getenv("PROXY_CONVERSATION_CACHE_MAX", "512"))
    )
    conversation_cache_ttl_s: float = field(
        default_factory=lambda: float(os.getenv("PROXY_CONVERSATION_CACHE_TTL_S", "7200"))
    )
    # Per-conversation spend ceiling (only consulted when the
    # proxy_spend_cap_enabled flag is on): one footer warning at warn, an
    # insufficient_quota refusal for requests arriving past max. Spend is the
    # soft in-memory total -- it resets on restart and undercounts turns whose
    # provider reported no usage, so max is a runaway-loop backstop, not a
    # billing limit.
    budget_warn_usd: float = field(
        default_factory=lambda: float(os.getenv("PROXY_CONVERSATION_BUDGET_WARN_USD", "1.00"))
    )
    budget_max_usd: float = field(
        default_factory=lambda: float(os.getenv("PROXY_CONVERSATION_BUDGET_MAX_USD", "5.00"))
    )
    # Uplift gate for bandit-mode routing: when a baseline model id is set,
    # other arms are sampleable only after their posterior beats the
    # baseline's with probability >= uplift_min_prob on >= uplift_min_obs
    # observations. Env-only and default OFF (empty baseline) -- flipping it
    # grants the bandit bounded promotion authority, which is an operator
    # decision, not a flag flip; it also only matters under ROUTING_MODE=bandit.
    uplift_baseline: str = field(default_factory=lambda: os.getenv("PROXY_UPLIFT_BASELINE", ""))
    uplift_min_prob: float = field(
        default_factory=lambda: float(os.getenv("PROXY_UPLIFT_MIN_PROB", "0.75"))
    )
    uplift_min_obs: int = field(
        default_factory=lambda: int(os.getenv("PROXY_UPLIFT_MIN_OBS", "10"))
    )

    def __post_init__(self) -> None:
        # Keep the transcript dir beside the inbox WITHOUT a second env var to set:
        # when PROXY_PANEL_TRANSCRIPT_DIR is unset, derive it from routing_inbox_dir so
        # an inbox pointed at ~/.pdp-router/inbox (via PROXY_ROUTING_INBOX_DIR) lands
        # transcripts at the sibling ~/.pdp-router/panel-transcripts. The two
        # cannot drift, so flipping the flag can never silently write to a base the eval
        # does not read. An explicit PROXY_PANEL_TRANSCRIPT_DIR still wins (frozen
        # dataclass -> object.__setattr__).
        if "PROXY_PANEL_TRANSCRIPT_DIR" not in os.environ:
            object.__setattr__(
                self,
                "panel_transcript_dir",
                self.routing_inbox_dir.parent / "panel-transcripts",
            )
