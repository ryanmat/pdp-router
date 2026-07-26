# Description: OpenTelemetry tracing, metrics, and log export for the pdp-router proxy.
# Description: Configures TracerProvider, MeterProvider, LoggerProvider, OpenLIT, Traceloop.

from __future__ import annotations

import importlib.metadata
import logging
import os

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

log = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "pdp-router-proxy"

# Version comes from the installed package metadata so OTel resources can never
# drift from pyproject.toml. Uninstalled source trees (e.g. ad-hoc imports
# outside a venv) fall back to a sentinel rather than crashing telemetry setup.
try:
    SERVICE_VERSION = importlib.metadata.version("pdp-router")
except importlib.metadata.PackageNotFoundError:
    SERVICE_VERSION = "0.0.0+uninstalled"


def _service_name() -> str:
    """Service name for all OTel providers; per-deployment via OTEL_SERVICE_NAME."""
    return os.environ.get("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME)


def _capture_content() -> bool:
    """True when GenAI instrumentation may record prompt and completion text.

    Off unless OTEL_CAPTURE_CONTENT opts in. Capturing content ships the text of
    every request and response to whatever OTLP backend is configured, so it is
    a deliberate choice rather than a side effect of enabling tracing.
    """
    return os.environ.get("OTEL_CAPTURE_CONTENT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

_tracer: trace.Tracer | None = None
_logger_provider: LoggerProvider | None = None


def _build_resource() -> Resource:
    """Build the shared OTel resource for all providers."""
    return Resource.create(
        {
            "service.name": _service_name(),
            "service.version": SERVICE_VERSION,
            "service.namespace": os.environ.get("OTEL_SERVICE_NAMESPACE", "pdp-router"),
            "host.name": os.environ.get("OTEL_HOST_NAME", "127.0.0.1"),
            "deployment.environment": os.environ.get("OTEL_ENVIRONMENT", "lab"),
        }
    )


def init_tracing() -> trace.Tracer:
    """Initialize OTel tracing, metrics, logs, OpenLIT, and Traceloop.

    Reads configuration from environment variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP HTTP endpoint (required for export).
        OTEL_EXPORTER_OTLP_HEADERS: Comma-separated key=value headers for auth.
        OTEL_TRACING_ENABLED: Set to "false" to disable tracing (default: true).
        OTEL_EXPORT_BATCH: Set to "false" to use SimpleSpanProcessor (tests).
    """
    global _tracer

    enabled = os.environ.get("OTEL_TRACING_ENABLED", "true").lower() != "false"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    resource = _build_resource()
    provider = TracerProvider(resource=resource)

    if enabled and endpoint:
        headers = _parse_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""))
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", headers=headers)

        use_batch = os.environ.get("OTEL_EXPORT_BATCH", "true").lower() != "false"
        if use_batch:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            provider.add_span_processor(SimpleSpanProcessor(exporter))

        metric_exporter = OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics", headers=headers)
        reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

        _init_log_provider(endpoint, headers, resource, use_batch)

        log.info(
            "OTel tracing, metrics, and logs enabled: endpoint=%s, service=%s",
            endpoint,
            _service_name(),
        )
    else:
        log.info("OTel export disabled (no endpoint or OTEL_TRACING_ENABLED=false)")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_service_name(), SERVICE_VERSION)

    # GenAI auto-instrumentation only when there is somewhere to send it. With no
    # endpoint there is no configured LoggerProvider, so openlit's GenAI log
    # records fall back to the console and print the full prompt and completion
    # text of every request to stdout. Gated after set_tracer_provider because
    # openlit auto-detects the provider it attaches to.
    if enabled and endpoint:
        _init_openlit()
        _init_traceloop()

    return _tracer


def _init_log_provider(
    endpoint: str,
    headers: dict[str, str],
    resource: Resource,
    use_batch: bool,
) -> None:
    """Set up OTel LoggerProvider and bridge Python logging to OTLP export."""
    global _logger_provider

    log_exporter = OTLPLogExporter(endpoint=f"{endpoint}/v1/logs", headers=headers)

    if use_batch:
        processor = BatchLogRecordProcessor(log_exporter)
    else:
        processor = SimpleLogRecordProcessor(log_exporter)

    _logger_provider = LoggerProvider(resource=resource)
    _logger_provider.add_log_record_processor(processor)
    set_logger_provider(_logger_provider)

    handler = LoggingHandler(
        level=logging.INFO,
        logger_provider=_logger_provider,
    )
    logging.getLogger().addHandler(handler)
    log.info("OTel log export initialized")


def get_tracer() -> trace.Tracer:
    """Return the configured tracer. Falls back to no-op if not initialized."""
    if _tracer is not None:
        return _tracer
    return trace.get_tracer(_service_name(), SERVICE_VERSION)


def shutdown_tracing() -> None:
    """Flush pending spans, metrics, and logs, then shut down providers."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
        log.info("OTel tracer provider shut down")

    meter_provider = metrics.get_meter_provider()
    if isinstance(meter_provider, MeterProvider):
        meter_provider.shutdown()
        log.info("OTel meter provider shut down")

    if _logger_provider is not None:
        _logger_provider.shutdown()
        log.info("OTel logger provider shut down")


def _init_openlit() -> None:
    """Enable OpenLIT GenAI auto-instrumentation if available.

    Generates OTel traces and metrics for Anthropic + Gemini SDK calls (token
    usage, latency, cost). openlit auto-detects the existing TracerProvider
    and MeterProvider so all telemetry flows through the configured OTLP
    exporters. Prompt and completion text is recorded only when
    OTEL_CAPTURE_CONTENT opts in (see _capture_content).
    """
    try:
        import openlit

        openlit.init(
            environment=os.environ.get("OTEL_ENVIRONMENT", "lab"),
            application_name=_service_name(),
            collect_gpu_stats=False,
            capture_message_content=_capture_content(),
            disable_metrics=False,
        )
        log.info("OpenLIT GenAI instrumentation initialized")
    except ImportError:
        log.warning("openlit not installed, skipping GenAI instrumentation")
    except Exception:
        log.warning("Failed to initialize OpenLIT", exc_info=True)


def _init_traceloop() -> None:
    """Enable Traceloop GenAI tracing if available.

    Provides complementary metrics to OpenLIT: operation duration histograms,
    exception counts (llm.anthropic.completion.exceptions), and streaming
    time-to-first-token. Shares the same OTLP endpoint so all telemetry
    flows through LMOTEL.
    """
    try:
        from traceloop.sdk import Traceloop

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if not endpoint:
            log.info("Traceloop skipped (no OTLP endpoint)")
            return

        Traceloop.init(
            app_name=_service_name(),
            api_endpoint=endpoint,
            disable_batch=False,
        )
        log.info("Traceloop GenAI tracing initialized")
    except ImportError:
        log.warning("traceloop-sdk not installed, skipping Traceloop instrumentation")
    except Exception:
        log.warning("Failed to initialize Traceloop", exc_info=True)


def _parse_headers(header_str: str) -> dict[str, str]:
    """Parse OTEL_EXPORTER_OTLP_HEADERS format: key=value,key2=value2."""
    if not header_str:
        return {}
    headers = {}
    for pair in header_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers
