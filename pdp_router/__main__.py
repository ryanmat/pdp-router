# Description: Console entry point for the PDP Router Proxy (pdp-router-proxy).
# Description: Loads .env, resolves host/port from ProxyConfig, and serves the FastAPI app.

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Import string rather than the app object so uvicorn owns the import and the
# FastAPI/proxy dependency stays inside the `proxy` extra.
APP = "pdp_router._proxy:app"


def _load_env_file() -> None:
    """Load a .env from the working directory into the environment, if present.

    Existing environment variables win (python-dotenv defaults to
    override=False), so a systemd EnvironmentFile or a shell export is never
    clobbered by a stale .env sitting in the working directory.

    python-dotenv ships with the `proxy` extra. A missing install is not fatal:
    the process may already have its configuration in the environment, which is
    how the systemd unit runs.

    usecwd=True is required, not cosmetic: bare load_dotenv() resolves through
    find_dotenv(), which walks up from the *calling module's* file rather than
    the working directory. For an installed package that searches site-packages,
    and for a source checkout it can pick up a .env belonging to a parent repo
    instead of the one the user just created. The quickstart says to copy
    .env.example in the directory you run from, so that is what we search.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        log.debug("python-dotenv not installed; reading configuration from the environment only")
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


def main() -> None:
    """Serve the proxy on the configured host and port.

    Configuration comes from the environment, which `.env` populates. This is
    the entry point the README documents, and the only launch path that reads
    `.env` -- bare `uvicorn pdp_router._proxy:app` still expects the
    environment to be populated already (use `uv run --env-file .env` there).
    """
    _load_env_file()

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "pdp-router-proxy requires the 'proxy' extra. Install it with:\n"
            "  uv sync --all-extras\n"
            "or:\n"
            "  uv sync --extra proxy"
        ) from None

    # Imported after _load_env_file so a .env-supplied host/port is honored.
    from pdp_router._proxy_config import ProxyConfig

    config = ProxyConfig()
    uvicorn.run(APP, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
