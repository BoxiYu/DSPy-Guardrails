# Server Mode Guide

Run dspy-guardrails as an HTTP service accessible from any language or system.

## Quick Start

```bash
pip install -e ".[server]"
dspy-guardrails serve --port 8000
```

API docs available at `http://localhost:8000/docs`.

## Endpoints

### POST /v1/check

Check a single text against guardrails.

```bash
curl -X POST http://localhost:8000/v1/check \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "checks": ["no_injection", "no_toxicity"]}'
```

Response:
```json
{
  "is_safe": true,
  "results": [
    {"check": "no_injection", "passed": true, "score": null},
    {"check": "no_toxicity", "passed": true, "score": null}
  ]
}
```

### POST /v1/check/batch

Check multiple texts at once.

```bash
curl -X POST http://localhost:8000/v1/check/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["text1", "text2"], "checks": ["no_injection"]}'
```

### POST /v1/score

Get risk scores across dimensions.

```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{"text": "some text", "dimensions": ["injection", "toxicity", "pii"]}'
```

Response:
```json
{
  "scores": {"injection": 0.0, "toxicity": 0.0, "pii": 0.0},
  "overall_risk": 0.0
}
```

### GET /v1/health

```bash
curl http://localhost:8000/v1/health
```

### GET /v1/config

```bash
curl http://localhost:8000/v1/config
```

## Available Checks

| Check | Description |
|-------|-------------|
| `no_injection` | Prompt injection detection |
| `no_pii` | PII detection |
| `no_toxicity` | Toxicity detection |
| `safe` | Combined safety |
| `safe_input` | Input safety |
| `safe_output` | Output safety |
| `no_mcp_attack` | MCP attack detection |
| `safe_mcp` | MCP combined safety |

## Score Dimensions

| Dimension | Description |
|-----------|-------------|
| `injection` | Injection risk (0-1) |
| `toxicity` | Toxicity score (0-1) |
| `pii` | PII exposure (0-1) |
| `mcp_security` | MCP risk (0-1) |
| `factuality` | Factuality score (0-1) |
| `quality` | Quality score (0-1) |

## CLI Options

```
dspy-guardrails serve [OPTIONS]

Options:
  -h, --host TEXT     Host to bind [default: 0.0.0.0]
  -p, --port INTEGER  Port to bind [default: 8000]
  -w, --workers INT   Number of workers [default: 1]
  --reload            Enable auto-reload
```

## Programmatic Usage

```python
from dspy_guardrails.server import create_app, ServerConfig
import uvicorn

config = ServerConfig(port=9000, workers=4)
app = create_app(config)
uvicorn.run(app, host=config.host, port=config.port, workers=config.workers)
```

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[server]"
CMD ["dspy-guardrails", "serve", "--port", "8000", "--workers", "4"]
EXPOSE 8000
```
