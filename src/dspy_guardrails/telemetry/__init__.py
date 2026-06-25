"""
Telemetry - 可观测性模块

提供 OpenTelemetry 集成、结构化日志和运行时指标。

Usage:
    from dspy_guardrails.telemetry import (
        setup_telemetry,
        get_metrics,
        get_logger,
    )

    # 初始化
    setup_telemetry(service_name="my-guardrail-service")

    # 获取结构化 logger
    log = get_logger("my_module")
    log.info("check_completed", check="no_injection", result=True, latency_ms=1.2)

    # 获取 metrics collector
    metrics = get_metrics()
    print(metrics.summary())
"""

from dspy_guardrails.telemetry.logging import get_logger, setup_logging
from dspy_guardrails.telemetry.metrics import GuardrailMetrics, get_metrics
from dspy_guardrails.telemetry.tracing import (
    instrument_guardrail,
    setup_tracing,
    traced,
)

_initialized = False


def setup_telemetry(
    service_name: str = "dspy-guardrails",
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    otlp_endpoint: str | None = None,
    log_format: str = "json",
) -> None:
    """Initialize all telemetry subsystems.

    Args:
        service_name: Service name for traces and metrics.
        enable_tracing: Enable OpenTelemetry tracing.
        enable_metrics: Enable metrics collection.
        enable_logging: Enable structured logging.
        otlp_endpoint: OTLP exporter endpoint (e.g. "http://localhost:4317").
        log_format: Log format - "json" or "console".
    """
    global _initialized
    if _initialized:
        return

    if enable_logging:
        setup_logging(format=log_format)

    if enable_tracing:
        setup_tracing(service_name=service_name, otlp_endpoint=otlp_endpoint)

    if enable_metrics:
        get_metrics()  # Initialize singleton

    _initialized = True


__all__ = [
    "setup_telemetry",
    "setup_tracing",
    "setup_logging",
    "traced",
    "instrument_guardrail",
    "get_logger",
    "get_metrics",
    "GuardrailMetrics",
]
