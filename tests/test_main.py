# Description: Tests for the pdp-router-proxy console entry point.
# Description: Covers .env loading, host/port resolution, and missing-extra handling.

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("dotenv")

from pdp_router.__main__ import main


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot os.environ around every test.

    main() loads a .env into the process environment, and monkeypatch only
    reverts keys it set itself, so a loaded value would otherwise leak into
    later tests.
    """
    with patch.dict(os.environ, {}, clear=False):
        yield


class TestEnvFileLoading:
    """The documented quickstart puts credentials in .env; they must reach the process.

    Nothing else in the package reads .env, and `uv run` does not load it, so
    this entry point is the only thing that makes `cp .env.example .env` work.
    """

    def test_env_file_supplies_config(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("PROXY_PORT", raising=False)
        monkeypatch.delenv("PROXY_HOST", raising=False)
        (tmp_path / ".env").write_text("PROXY_HOST=0.0.0.0\nPROXY_PORT=9931\n")
        monkeypatch.chdir(tmp_path)

        with patch("uvicorn.run") as run:
            main()

        assert run.call_args.kwargs["host"] == "0.0.0.0"
        assert run.call_args.kwargs["port"] == 9931

    def test_env_file_values_reach_the_environment(self, tmp_path, monkeypatch) -> None:
        """A sentinel stands in for a credential so no real key can be echoed on failure."""
        import os

        monkeypatch.delenv("PDP_DOTENV_SENTINEL", raising=False)
        (tmp_path / ".env").write_text("PDP_DOTENV_SENTINEL=reached\n")
        monkeypatch.chdir(tmp_path)

        with patch("uvicorn.run"):
            main()

        assert os.environ.get("PDP_DOTENV_SENTINEL") == "reached"

    def test_ignores_env_file_outside_the_working_directory(self, tmp_path, monkeypatch) -> None:
        """Bare load_dotenv() walks up from the package file; it must use the CWD instead.

        Without usecwd=True an installed package searches site-packages, and a
        source checkout can pick up a parent repo's .env.
        """
        monkeypatch.delenv("PDP_DOTENV_SENTINEL", raising=False)
        workdir = tmp_path / "clone"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        with patch("uvicorn.run") as run:
            main()

        import os

        assert "PDP_DOTENV_SENTINEL" not in os.environ
        assert run.call_args.kwargs["port"] == 7741

    def test_real_env_wins_over_env_file(self, tmp_path, monkeypatch) -> None:
        """systemd EnvironmentFile and shell exports must not be clobbered."""
        monkeypatch.setenv("PROXY_PORT", "7741")
        (tmp_path / ".env").write_text("PROXY_PORT=9931\n")
        monkeypatch.chdir(tmp_path)

        with patch("uvicorn.run") as run:
            main()

        assert run.call_args.kwargs["port"] == 7741


class TestHostPortResolution:
    def test_defaults(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("PROXY_HOST", raising=False)
        monkeypatch.delenv("PROXY_PORT", raising=False)
        monkeypatch.chdir(tmp_path)

        with patch("uvicorn.run") as run:
            main()

        assert run.call_args.kwargs["host"] == "127.0.0.1"
        assert run.call_args.kwargs["port"] == 7741

    def test_targets_the_proxy_app(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("uvicorn.run") as run:
            main()
        assert run.call_args.args[0] == "pdp_router._proxy:app"


class TestMissingProxyExtra:
    def test_missing_uvicorn_explains_the_fix(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with patch.dict(sys.modules, {"uvicorn": None}), pytest.raises(SystemExit) as exc:
            main()
        assert "proxy" in str(exc.value)
