# Description: Shared pytest fixtures for the pdp-router test suite.
# Description: Redirects on-disk writes into tmp_path so no test touches real user state.

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_on_disk_state(tmp_path_factory):
    """Point every path-valued setting at a temp directory for the whole suite.

    Without this, any test that builds its own TestClient gets a ProxyConfig
    with the shipped defaults, so serving one request appends a routing-decision
    row to the developer's real `~/.pdp-router/inbox`. That silently accumulates
    synthetic rows in the exact place a user is told to drain for outcomes, and a
    contributor running `pytest` would litter their own inbox with test traffic.

    Autouse and function-scoped so it applies before the client fixtures build a
    config. A test that sets these variables itself still wins, because its own
    patch.dict is applied inside this one.
    """
    base = tmp_path_factory.mktemp("pdp-isolated")
    with patch.dict(
        os.environ,
        {
            "PROXY_ROUTING_INBOX_DIR": str(base / "inbox"),
            "PROXY_PANEL_TRANSCRIPT_DIR": str(base / "panel-transcripts"),
            "PROXY_TRUST_DB": str(base / "absent-trust.db"),
        },
    ):
        yield base


@pytest.fixture(autouse=True)
def _pin_flags_to_defaults(monkeypatch):
    """Detach every flag read from the host's live flag store for the suite.

    _flag_enabled prefers the flag library whenever it is importable, and on a
    deployed host the live store has features switched ON. Without this pin, a
    test that exercises one feature can be silently affected by an unrelated
    live flag (a feedback row shifting a row-index assertion was the incident
    that added this). Setting the module handle to None routes every read to
    the env fallback, and the suite's env carries no PROXY_*_ENABLED values,
    so all flags sit at their coded defaults. Tests that want a flag on keep
    patching the reader function; tests of the flag plumbing itself re-patch
    the handle and still win.
    """
    from pdp_router import _proxy

    monkeypatch.setattr(_proxy, "_clawflag", None)
    for var in list(os.environ):
        if var.startswith("PROXY_") and var.endswith("_ENABLED"):
            monkeypatch.delenv(var)
