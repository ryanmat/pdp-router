# Description: Configuration for the PDP Router Proxy service.
# Description: Reads settings from environment variables with sensible defaults.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from pdp_router._domain_overrides import parse_overrides


@dataclass(frozen=True)
class ProxyConfig:
    """Configuration for the PDP Router Proxy."""

    host: str = field(default_factory=lambda: os.getenv("PROXY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("PROXY_PORT", "7741")))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gcp_project: str = field(default_factory=lambda: os.getenv("GCP_PROJECT", ""))
    gcp_location: str = field(default_factory=lambda: os.getenv("GCP_LOCATION", "us-east5"))
    classify_model: str = field(
        default_factory=lambda: os.getenv("PROXY_CLASSIFY_MODEL", "gemini-2.5-flash-lite")
    )
    classify_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("PROXY_CLASSIFY_MAX_TOKENS", "16"))
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
