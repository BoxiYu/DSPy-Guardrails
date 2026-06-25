# API Reference

## Core Functions (`guardrail`)

```python
from dspy_guardrails import guardrail
```

### Boolean Checks

| Function | Returns | Description |
|----------|---------|-------------|
| `guardrail.no_injection(text)` | `bool` | True if no prompt injection detected |
| `guardrail.no_pii(text)` | `bool` | True if no PII detected |
| `guardrail.no_toxicity(text)` | `bool` | True if no toxic content |
| `guardrail.safe(text)` | `bool` | Combined: no_injection + no_toxicity |
| `guardrail.safe_input(text)` | `bool` | Combined: no_injection + no_pii |
| `guardrail.safe_output(text)` | `bool` | Combined: no_toxicity + no_pii |
| `guardrail.no_mcp_attack(text)` | `bool` | True if no MCP attack detected |
| `guardrail.safe_mcp(text)` | `bool` | Combined MCP safety |

### Score Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `guardrail.injection_score(text)` | `float` | 0.0-1.0, higher = more risk |
| `guardrail.toxicity(text)` | `float` | 0.0-1.0, higher = more toxic |
| `guardrail.pii_score(text)` | `float` | 0.0-1.0, higher = more PII |
| `guardrail.mcp_security_score(text)` | `float` | 0.0-1.0, MCP risk score |
| `guardrail.factuality(text)` | `float` | 0.0-1.0, factuality score |
| `guardrail.quality(text)` | `float` | 0.0-1.0, quality score |

## Async API

```python
from dspy_guardrails.async_guardrail import (
    no_injection_async, no_pii_async, no_toxicity_async,
    safe_async, injection_score_async,
    check_all_async, batch_check_async,
    AsyncLLMGuardrail, AsyncHybridGuardrail,
)
```

All functions mirror their sync counterparts with `await` support.

### Utility Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `check_all_async` | `(text, checks=None) -> dict` | Run multiple checks concurrently |
| `batch_check_async` | `(texts, check, max_concurrency=50) -> list` | Check many texts |

## Streaming

```python
from dspy_guardrails.streaming import StreamGuardrail, StreamViolation
```

### StreamGuardrail

| Method | Returns | Description |
|--------|---------|-------------|
| `filter(stream)` | `AsyncIterator[str]` | Filter token stream |
| `check_stream(stream)` | `(str, list[StreamViolation])` | Collect and check |
| `reset()` | `None` | Reset state |

Properties: `violations`, `is_clean`

## LLM Guardrail

```python
from dspy_guardrails import LLMGuardrail, HybridGuardrail
```

### LLMGuardrail

| Method | Returns | Description |
|--------|---------|-------------|
| `check(text, category)` | `Prediction` | LLM-based check |
| `no_injection(text)` | `bool` | Injection check |
| `no_toxicity(text)` | `bool` | Toxicity check |
| `safe(text)` | `bool` | Combined check |

### HybridGuardrail

| Method | Returns | Description |
|--------|---------|-------------|
| `check(text, category)` | `(bool, float)` | (is_unsafe, confidence) |

## Decorators

```python
from dspy_guardrails import Guarded, guarded
```

### @Guarded (class decorator)

```python
@Guarded(
    input_checks=["no_injection"],
    output_checks=["no_toxicity"],
    on_violation="assert",  # "assert", "suggest", "log", "ignore"
)
class MyModule(dspy.Module): ...
```

Supports both sync and async `forward()` methods.

### @guarded (function decorator)

```python
@guarded(input_checks=["no_injection"])
def my_function(text): ...

@guarded(input_checks=["no_injection"])
async def my_async_function(text): ...
```

## Telemetry

```python
from dspy_guardrails.telemetry import (
    setup_telemetry, get_metrics, get_logger,
    traced, instrument_guardrail,
)
```

### GuardrailMetrics

| Method | Description |
|--------|-------------|
| `track(name)` | Context manager for tracking |
| `record(name, passed, latency_ms)` | Direct recording |
| `get_stats(name)` | Get stats for a check |
| `summary()` | Full metrics summary |
| `reset()` | Reset all metrics |

## Server

```python
from dspy_guardrails.server import create_app, ServerConfig
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/check` | POST | Single text check |
| `/v1/check/batch` | POST | Batch check |
| `/v1/score` | POST | Risk scores |
| `/v1/health` | GET | Health check |
| `/v1/config` | GET | Configuration |
