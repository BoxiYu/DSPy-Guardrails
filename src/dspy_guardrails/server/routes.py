"""API route definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from fastapi import APIRouter

from dspy_guardrails import __version__
from dspy_guardrails.guardrail import guardrail

from .schemas import (
    BatchCheckRequest,
    BatchCheckResponse,
    CheckRequest,
    CheckResponse,
    CheckResult,
    ConfigResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)

router = APIRouter()

AVAILABLE_CHECKS = [
    "no_injection",
    "no_pii",
    "no_toxicity",
    "safe",
    "safe_input",
    "safe_output",
    "no_mcp_attack",
    "safe_mcp",
]

SCORE_DIMENSIONS = [
    "injection",
    "toxicity",
    "pii",
    "mcp_security",
    "factuality",
    "quality",
]

_CHECK_FNS = {
    "no_injection": guardrail.no_injection,
    "no_pii": guardrail.no_pii,
    "no_toxicity": guardrail.no_toxicity,
    "safe": guardrail.safe,
    "safe_input": guardrail.safe_input,
    "safe_output": guardrail.safe_output,
    "no_mcp_attack": guardrail.no_mcp_attack,
    "safe_mcp": guardrail.safe_mcp,
}

_SCORE_FNS = {
    "injection": guardrail.injection_score,
    "toxicity": guardrail.toxicity,
    "pii": guardrail.pii_score,
    "mcp_security": guardrail.mcp_security_score,
    "factuality": guardrail.factuality,
    "quality": guardrail.quality,
}


def _run_checks(text: str, checks: list[str]) -> CheckResponse:
    results = []
    all_passed = True
    for check_name in checks:
        fn = _CHECK_FNS.get(check_name)
        if fn is None:
            continue
        passed = fn(text)
        if not passed:
            all_passed = False
        results.append(CheckResult(check=check_name, passed=passed))
    return CheckResponse(is_safe=all_passed, results=results)


@router.post("/v1/check", response_model=CheckResponse)
async def check(request: CheckRequest) -> CheckResponse:
    """Run guardrail checks on a single text."""
    return _run_checks(request.text, request.checks)


@router.post("/v1/check/batch", response_model=BatchCheckResponse)
async def check_batch(request: BatchCheckRequest) -> BatchCheckResponse:
    """Run guardrail checks on multiple texts."""
    results = [_run_checks(text, request.checks) for text in request.texts]
    return BatchCheckResponse(results=results)


@router.post("/v1/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    """Get risk scores for multiple dimensions."""
    scores = {}
    for dim in request.dimensions:
        fn = _SCORE_FNS.get(dim)
        if fn is not None:
            scores[dim] = round(fn(request.text), 4)

    overall = max(scores.values()) if scores else 0.0
    return ScoreResponse(scores=scores, overall_risk=round(overall, 4))


@router.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=__version__,
        checks_available=AVAILABLE_CHECKS,
    )


@router.get("/v1/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """Return current configuration."""
    return ConfigResponse(
        checks_available=AVAILABLE_CHECKS,
        score_dimensions=SCORE_DIMENSIONS,
        version=__version__,
    )
