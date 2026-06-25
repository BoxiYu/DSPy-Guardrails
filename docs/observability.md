# Observability Guide

## Setup

```bash
pip install -e ".[telemetry]"
```

```python
from dspy_guardrails.telemetry import setup_telemetry

setup_telemetry(
    service_name="my-guardrail-service",
    enable_tracing=True,
    enable_metrics=True,
    enable_logging=True,
    otlp_endpoint="http://localhost:4317",  # optional
    log_format="json",
)
```

## Metrics

```python
from dspy_guardrails.telemetry import get_metrics

metrics = get_metrics()

# Track with context manager
with metrics.track("no_injection") as t:
    result = guardrail.no_injection(text)
    t.set_result(result)

# Or record directly
metrics.record("no_injection", passed=True, latency_ms=0.5)

# Get summary
print(metrics.summary())
# {
#   "uptime_seconds": 3600.0,
#   "total_checks": 1000,
#   "total_blocks": 50,
#   "overall_block_rate": 0.05,
#   "checks": {
#     "no_injection": {"total": 500, "blocked": 30, ...},
#     "no_toxicity": {"total": 500, "blocked": 20, ...}
#   }
# }
```

## Structured Logging

```python
from dspy_guardrails.telemetry import get_logger
from dspy_guardrails.telemetry.logging import log_check

log = get_logger("my_module")
log.info("processing_request", user_id="123", action="check")

# Log guardrail check results
log_check(log, "no_injection", text, result=True, latency_ms=0.5)
# Output (JSON): {"event": "guardrail_check", "check": "no_injection", "input_hash": "a1b2c3...", "result": true, "latency_ms": 0.5}
```

## Tracing

```python
from dspy_guardrails.telemetry import traced, instrument_guardrail

# Decorator for individual functions
@traced("my_pipeline.check")
def check_input(text):
    return guardrail.safe(text)

# Instrument a class
@instrument_guardrail
class MyGuardrail:
    def no_injection(self, text): ...
    def check(self, text): ...
```

Traces are exported via OTLP to Jaeger, Zipkin, or any OpenTelemetry-compatible backend.

## Components

| Module | Purpose |
|--------|---------|
| `telemetry.metrics` | Call counts, block rates, latency distribution |
| `telemetry.logging` | Structured JSON logging with structlog |
| `telemetry.tracing` | OpenTelemetry spans for each guardrail call |
