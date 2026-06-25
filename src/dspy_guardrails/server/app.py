"""FastAPI application factory."""

from __future__ import annotations

from dspy_guardrails.server.config import ServerConfig


def create_app(config: ServerConfig | None = None) -> FastAPI:  # noqa: F821
    """Create and configure the FastAPI application.

    Args:
        config: Server configuration. Uses defaults if None.

    Returns:
        Configured FastAPI application instance.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    if config is None:
        config = ServerConfig()

    app = FastAPI(
        title="DSPy Guardrails API",
        description="Security guardrail checks for LLM applications",
        version="0.5.0",
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from dspy_guardrails.server.routes import router

    app.include_router(router)

    return app
