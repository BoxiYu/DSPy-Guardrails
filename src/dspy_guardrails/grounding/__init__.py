"""
Grounding module for intrinsic hallucination detection.

Detects contradictions between agent responses and source data
(knowledge base retrieval_context and tool_results).

Quick Start:
    from dspy_guardrails.grounding import HybridGroundingChecker, check_grounding

    # Using the checker class
    checker = HybridGroundingChecker(use_llm=True)
    result = checker.check(
        output="Your flight CA1234 costs $299.",
        retrieval_context=["Flight CA1234 info"],
        tool_results=[{"tool": "get_price", "result": {"price": 450}}],
    )
    print(result.has_contradiction)  # True
    print(result.contradictions[0].field)  # "price"

    # Using the convenience function
    result = check_grounding(
        output="Flight is on time.",
        tool_results=[{"tool": "status", "result": {"status": "delayed"}}],
    )

    # Simple boolean check
    from dspy_guardrails.grounding import is_grounded
    if not is_grounded(output, tool_results=results):
        print("Warning: Response may contain incorrect information")

Components:
    - HybridGroundingChecker: Main API combining rule + LLM detection
    - ContradictionDetector: Fast rule-based detection (~5ms)
    - LLMContradictionChecker: LLM-based detection (~500ms)
    - ValueExtractor: Extract values from text and JSON
    - check_grounding: Convenience function
    - is_grounded: Simple boolean check
"""

# Core data structures
# Main API
from .checker import (
    HybridGroundingChecker,
    check_grounding,
    is_grounded,
)

# Detectors
from .contradiction import ContradictionDetector
from .core import (
    Contradiction,
    ContradictionResult,
    ExtractedValue,
    FieldType,
    Severity,
    SourceType,
)

# Extractors
from .extractors import ValueExtractor
from .llm_grounding import LLMContradictionChecker

__all__ = [
    # Core types
    "Contradiction",
    "ContradictionResult",
    "ExtractedValue",
    "FieldType",
    "Severity",
    "SourceType",
    # Extractors
    "ValueExtractor",
    # Detectors
    "ContradictionDetector",
    "LLMContradictionChecker",
    # Main API
    "HybridGroundingChecker",
    "check_grounding",
    "is_grounded",
]
