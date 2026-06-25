"""
OpenTelemetry Tracing - Span/Trace 封装

每次 guardrail 调用生成 span，支持 OTLP 导出。
"""

import functools
import time
from collections.abc import Callable
from typing import Any

_tracer = None
_otel_available = False

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _otel_available = True
except ImportError:
    pass


def setup_tracing(
    service_name: str = "dspy-guardrails",
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for resource identification.
        otlp_endpoint: OTLP gRPC endpoint. If None, uses console exporter.
    """
    global _tracer

    if not _otel_available:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("dspy_guardrails")


def _get_tracer():
    global _tracer
    if _tracer is None and _otel_available:
        _tracer = trace.get_tracer("dspy_guardrails")
    return _tracer


def traced(name: str | None = None) -> Callable:
    """Decorator to add tracing span to a function.

    Args:
        name: Span name. Defaults to function name.
    """

    def decorator(fn: Callable) -> Callable:
        span_name = name or f"guardrail.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _get_tracer()
            if tracer is None:
                return fn(*args, **kwargs)

            with tracer.start_as_current_span(span_name) as span:
                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    span.set_attribute("guardrail.result", str(result))
                    span.set_attribute("guardrail.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("guardrail.success", False)
                    span.set_attribute("guardrail.error", str(e))
                    span.record_exception(e)
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("guardrail.latency_ms", elapsed_ms)

        return wrapper

    return decorator


def instrument_guardrail(guardrail_cls: type) -> type:
    """Class decorator to instrument all check methods with tracing.

    Usage:
        @instrument_guardrail
        class MyGuardrail:
            def no_injection(self, text): ...
    """
    check_methods = [
        "no_injection",
        "no_pii",
        "no_toxicity",
        "safe",
        "safe_input",
        "safe_output",
        "injection_score",
        "toxicity",
        "pii_score",
        "no_mcp_attack",
        "mcp_security_score",
        "safe_mcp",
        "check",
    ]

    for method_name in check_methods:
        method = getattr(guardrail_cls, method_name, None)
        if method is not None and callable(method):
            setattr(guardrail_cls, method_name, traced(f"guardrail.{method_name}")(method))

    return guardrail_cls
