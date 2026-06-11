# Description: Tests for OTel resource construction in pdp_router._tracing.
# Description: Covers service-name env override and generic defaults.

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

from pdp_router._tracing import DEFAULT_SERVICE_NAME, _build_resource, _service_name


class TestServiceName:
    def test_default_is_generic(self, monkeypatch) -> None:
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        assert _service_name() == DEFAULT_SERVICE_NAME == "pdp-router-proxy"

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("OTEL_SERVICE_NAME", "my-deployment-proxy")
        assert _service_name() == "my-deployment-proxy"


class TestBuildResource:
    def test_resource_uses_env_service_name(self, monkeypatch) -> None:
        monkeypatch.setenv("OTEL_SERVICE_NAME", "my-deployment-proxy")
        attrs = _build_resource().attributes
        assert attrs["service.name"] == "my-deployment-proxy"

    def test_resource_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        monkeypatch.delenv("OTEL_SERVICE_NAMESPACE", raising=False)
        attrs = _build_resource().attributes
        assert attrs["service.name"] == "pdp-router-proxy"
        assert attrs["service.namespace"] == "pdp-router"
