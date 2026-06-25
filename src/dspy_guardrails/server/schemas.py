"""Pydantic models for server request/response."""

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    """Single text check request."""

    text: str = Field(..., description="Text to check")
    checks: list[str] = Field(
        default=["no_injection", "no_toxicity"],
        description="Guardrail checks to run",
    )


class CheckResult(BaseModel):
    """Result of a single check."""

    check: str
    passed: bool
    score: float | None = None


class CheckResponse(BaseModel):
    """Response for a single check request."""

    is_safe: bool
    results: list[CheckResult]


class BatchCheckRequest(BaseModel):
    """Batch check request."""

    texts: list[str] = Field(..., description="Texts to check")
    checks: list[str] = Field(
        default=["no_injection", "no_toxicity"],
        description="Guardrail checks to run",
    )


class BatchCheckResponse(BaseModel):
    """Response for batch check request."""

    results: list[CheckResponse]


class ScoreRequest(BaseModel):
    """Score request returning numeric scores."""

    text: str = Field(..., description="Text to score")
    dimensions: list[str] = Field(
        default=["injection", "toxicity", "pii"],
        description="Dimensions to score",
    )


class ScoreResponse(BaseModel):
    """Response with dimension scores."""

    scores: dict[str, float]
    overall_risk: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    checks_available: list[str]


class ConfigResponse(BaseModel):
    """Current configuration response."""

    checks_available: list[str]
    score_dimensions: list[str]
    version: str
