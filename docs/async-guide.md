# Async & Streaming Guide

## Async API

All core guardrail functions have async variants that use `asyncio.to_thread()` to avoid blocking the event loop.

### Basic Usage

```python
from dspy_guardrails.async_guardrail import (
    no_injection_async,
    no_pii_async,
    no_toxicity_async,
    safe_async,
    injection_score_async,
)

result = await no_injection_async(text)
score = await injection_score_async(text)
```

### Concurrent Checks

```python
from dspy_guardrails.async_guardrail import check_all_async

# Run multiple checks on the same text concurrently
results = await check_all_async(text, checks=["no_injection", "no_toxicity", "no_pii"])
# Returns: {"no_injection": True, "no_toxicity": True, "no_pii": True}
```

### Batch Processing

```python
from dspy_guardrails.async_guardrail import batch_check_async

texts = ["text1", "text2", "text3", ...]
results = await batch_check_async(texts, check="safe", max_concurrency=50)
```

### Async LLM Guardrail

```python
from dspy_guardrails.async_guardrail import AsyncLLMGuardrail, AsyncHybridGuardrail

guard = AsyncLLMGuardrail()
result = await guard.check(text, "injection")

# Hybrid
hybrid = AsyncHybridGuardrail()
is_unsafe, confidence = await hybrid.check(text, "injection")
```

### Async Decorators

The `@Guarded` and `@guarded` decorators auto-detect async functions:

```python
from dspy_guardrails import Guarded, guarded

@Guarded(input_checks=["no_injection"])
class AsyncModule(dspy.Module):
    async def forward(self, question):
        return await self.generate(question=question)

@guarded(input_checks=["no_injection"])
async def my_async_handler(text: str) -> str:
    return await process(text)
```

## Streaming

### StreamGuardrail

Incremental checking of token streams with sentence-level buffering.

```python
from dspy_guardrails.streaming import StreamGuardrail

guard = StreamGuardrail(
    checks=["no_injection", "no_toxicity"],
    on_violation="block",  # "block", "warn", "pass"
    buffer_size=5,
)

# Filter mode: yields safe tokens, stops on violation
async for token in guard.filter(token_stream):
    print(token, end="", flush=True)

if not guard.is_clean:
    print(f"\nBlocked: {guard.violations}")
```

### Check Mode

Collect all text and return violations without filtering:

```python
full_text, violations = await guard.check_stream(token_stream)
if violations:
    for v in violations:
        print(f"[{v.check}] at position {v.position}: {v.text!r}")
```

### Violation Modes

- `"block"` — Stop yielding tokens on first violation
- `"warn"` — Continue yielding but record violations
- `"pass"` — Record violations only (no filtering)

## Integration with FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dspy_guardrails.async_guardrail import no_injection_async
from dspy_guardrails.streaming import StreamGuardrail

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    if not await no_injection_async(message):
        return {"error": "Injection detected"}

    guard = StreamGuardrail(checks=["no_toxicity"])

    async def generate():
        async for token in guard.filter(llm_stream(message)):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")
```

## Install

```bash
pip install -e ".[async]"  # aiohttp, anyio
```
