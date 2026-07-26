# Description: Tests for OTel resource construction in pdp_router._tracing.
# Description: Covers service-name env override, defaults, and GenAI instrumentation gating.

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

pytest.importorskip("opentelemetry")

from pdp_router._tracing import (
    DEFAULT_SERVICE_NAME,
    _build_resource,
    _capture_content,
    _service_name,
    init_tracing,
)


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


@contextmanager
def _isolated_init():
    """Run init_tracing() without touching global providers or opening sockets.

    Patches every side-effecting collaborator so the test observes only the
    gating decision. Keeps output pristine: no OTLP exporter retries against a
    dead endpoint, no "Overriding of current TracerProvider" warnings.
    """
    with (
        patch("pdp_router._tracing.OTLPSpanExporter"),
        patch("pdp_router._tracing.OTLPMetricExporter"),
        patch("pdp_router._tracing._init_log_provider"),
        patch("pdp_router._tracing.trace.set_tracer_provider"),
        patch("pdp_router._tracing.metrics.set_meter_provider"),
        patch("pdp_router._tracing._init_openlit") as openlit,
        patch("pdp_router._tracing._init_traceloop") as traceloop,
    ):
        yield openlit, traceloop


class TestGenAIInstrumentationGating:
    """GenAI auto-instrumentation must not initialize without an OTLP endpoint.

    With no endpoint there is no configured LoggerProvider, so openlit's GenAI
    log records fall back to the console and print the full prompt and
    completion text of every request to stdout.
    """

    def test_no_endpoint_skips_genai_instrumentation(self, monkeypatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_TRACING_ENABLED", raising=False)
        with _isolated_init() as (openlit, traceloop):
            init_tracing()
        openlit.assert_not_called()
        traceloop.assert_not_called()

    def test_endpoint_enables_genai_instrumentation(self, monkeypatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.delenv("OTEL_TRACING_ENABLED", raising=False)
        with _isolated_init() as (openlit, traceloop):
            init_tracing()
        openlit.assert_called_once()
        traceloop.assert_called_once()

    def test_kill_switch_skips_genai_instrumentation(self, monkeypatch) -> None:
        """OTEL_TRACING_ENABLED=false is documented as off; it must actually be off."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "false")
        with _isolated_init() as (openlit, traceloop):
            init_tracing()
        openlit.assert_not_called()
        traceloop.assert_not_called()


class TestCaptureContent:
    """Prompt and completion text is only captured on explicit opt-in.

    Enabling it ships user prompts to whatever OTLP backend is configured, so
    the default must be off regardless of whether tracing itself is on.
    """

    def test_off_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("OTEL_CAPTURE_CONTENT", raising=False)
        assert _capture_content() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_opt_in_values(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv("OTEL_CAPTURE_CONTENT", value)
        assert _capture_content() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_off_values(self, monkeypatch, value: str) -> None:
        monkeypatch.setenv("OTEL_CAPTURE_CONTENT", value)
        assert _capture_content() is False

    def test_openlit_receives_the_flag(self, monkeypatch) -> None:
        monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "1")
        openlit = pytest.importorskip("openlit")
        with patch.object(openlit, "init") as init:
            from pdp_router._tracing import _init_openlit

            _init_openlit()
        assert init.call_args.kwargs["capture_message_content"] is True

    def test_openlit_defaults_to_no_content(self, monkeypatch) -> None:
        monkeypatch.delenv("OTEL_CAPTURE_CONTENT", raising=False)
        openlit = pytest.importorskip("openlit")
        with patch.object(openlit, "init") as init:
            from pdp_router._tracing import _init_openlit

            _init_openlit()
        assert init.call_args.kwargs["capture_message_content"] is False
