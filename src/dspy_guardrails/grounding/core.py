"""
Core data structures for grounding/contradiction detection.

Detects Intrinsic Hallucination - when agent responses contradict
the source data (retrieval_context and tool_results).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """Source of the ground truth data."""

    RETRIEVAL = "retrieval"  # From knowledge base / RAG
    TOOL = "tool"  # From tool/function call results


class Severity(str, Enum):
    """Severity of a contradiction."""

    LOW = "low"  # Minor discrepancy (formatting, rounding)
    MEDIUM = "medium"  # Noticeable error (partial mismatch)
    HIGH = "high"  # Significant error (completely wrong value)


class FieldType(str, Enum):
    """Type of field being compared."""

    PRICE = "price"
    DATE = "date"
    TIME = "time"
    FLIGHT = "flight"
    STATUS = "status"
    PHONE = "phone"
    EMAIL = "email"
    PERCENTAGE = "percentage"
    COUNT = "count"
    NAME = "name"
    LOCATION = "location"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class Contradiction:
    """A single detected contradiction between agent output and source data."""

    claim: str  # The agent's claim (text snippet)
    source_value: str  # Value from the source data
    claim_value: str  # Value claimed by the agent
    field: str  # Field name (e.g., "price", "status", "departure_time")
    field_type: FieldType = FieldType.UNKNOWN
    source_type: SourceType = SourceType.RETRIEVAL
    severity: Severity = Severity.MEDIUM
    source_snippet: str = ""  # Context from source where the value was found
    confidence: float = 1.0  # Detection confidence (0-1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "claim": self.claim,
            "source_value": self.source_value,
            "claim_value": self.claim_value,
            "field": self.field,
            "field_type": self.field_type.value,
            "source_type": self.source_type.value,
            "severity": self.severity.value,
            "source_snippet": self.source_snippet,
            "confidence": self.confidence,
        }


@dataclass
class ContradictionResult:
    """Result from contradiction detection."""

    has_contradiction: bool
    contradiction_score: float  # 0.0 = no contradiction, 1.0 = severe contradiction
    contradictions: list[Contradiction] = field(default_factory=list)
    method: str = "rule"  # "rule" | "llm" | "hybrid"
    reasoning: str = ""  # Explanation of the detection process

    @classmethod
    def no_contradiction(cls, method: str = "rule") -> "ContradictionResult":
        """Create a result indicating no contradiction."""
        return cls(
            has_contradiction=False,
            contradiction_score=0.0,
            contradictions=[],
            method=method,
        )

    @classmethod
    def no_source_data(cls, method: str = "rule") -> "ContradictionResult":
        """Create a result when there's no source data to check against."""
        return cls(
            has_contradiction=False,
            contradiction_score=0.0,
            contradictions=[],
            method=method,
            reasoning="No source data available for comparison",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "has_contradiction": self.has_contradiction,
            "contradiction_score": self.contradiction_score,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "method": self.method,
            "reasoning": self.reasoning,
        }


@dataclass
class ExtractedValue:
    """A value extracted from text or structured data."""

    value: str  # The raw value
    normalized: str  # Normalized form for comparison
    field_type: FieldType
    source_text: str  # Original text context
    position: tuple[int, int] = (0, 0)  # Start and end position in source
    metadata: dict[str, Any] = field(default_factory=dict)
