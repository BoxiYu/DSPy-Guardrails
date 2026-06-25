"""
Server Mode - FastAPI 服务

提供 HTTP API 使其他语言/系统可以调用 guardrail 检测。

Usage:
    # CLI 启动
    dspy-guardrails serve --port 8000

    # 代码启动
    from dspy_guardrails.server import create_app
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from dspy_guardrails.server.app import create_app
from dspy_guardrails.server.config import ServerConfig
from dspy_guardrails.server.schemas import (
    BatchCheckRequest,
    BatchCheckResponse,
    CheckRequest,
    CheckResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)

__all__ = [
    "create_app",
    "ServerConfig",
    "CheckRequest",
    "CheckResponse",
    "BatchCheckRequest",
    "BatchCheckResponse",
    "ScoreRequest",
    "ScoreResponse",
    "HealthResponse",
]
