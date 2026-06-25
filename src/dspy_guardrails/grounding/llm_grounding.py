"""
LLM-based contradiction detection using DSPy.

Uses LLM to detect subtle contradictions between agent output and source data
that rule-based detection might miss.
"""

import json
from typing import Any

import dspy

from .core import (
    Contradiction,
    ContradictionResult,
    FieldType,
    Severity,
    SourceType,
)


class ContradictionClassifier(dspy.Signature):
    """Detect contradictions between an agent response and source data.

    TASK: Check if the agent's response CONTRADICTS the source data.

    A contradiction is when the agent says something that CONFLICTS with facts in the source.

    Examples of CONTRADICTIONS:
    - Source: "flight status: delayed" → Response: "Your flight is on time" ❌
    - Source: "price: $450" → Response: "The ticket costs $299" ❌
    - Source: "departure: 10:30 AM" → Response: "leaving at 2:00 PM" ❌
    - Source: "seats available: 0" → Response: "Yes, seats are available" ❌

    Examples of NO CONTRADICTION:
    - Source: "departure: 10:30" → Response: "Departing at 10:30 AM" ✅ (same meaning)
    - Source: "price: $450" → Response: "The flight information is..." ✅ (price not mentioned)
    - Source: "status: confirmed" → Response: "Your booking is confirmed" ✅ (consistent)
    - Response mentions something not in source → NOT a contradiction (this is extrinsic hallucination, not intrinsic)

    IMPORTANT:
    - Only detect CONTRADICTIONS (intrinsic hallucination)
    - Do NOT flag responses that add information not in the source
    - Do NOT flag paraphrasing or reformatting of the same information
    - Focus on FACTUAL conflicts, not style differences
    """

    response: str = dspy.InputField(desc="The agent's response to analyze")
    source_data: str = dspy.InputField(
        desc="Combined data from knowledge base and tool results"
    )

    has_contradiction: bool = dspy.OutputField(
        desc="True if response contradicts source data"
    )
    contradiction_score: float = dspy.OutputField(
        desc="Severity of contradiction: 0.0 (none) to 1.0 (severe)"
    )
    contradictions_json: str = dspy.OutputField(
        desc='JSON list of contradictions: [{"field": "...", "source_value": "...", "claim_value": "...", "severity": "low|medium|high"}]'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the analysis"
    )


class LLMContradictionChecker(dspy.Module):
    """LLM-based contradiction detection using DSPy.

    Uses an LLM to detect contradictions between agent responses and source data.
    More accurate than rule-based detection but slower (~500ms vs ~5ms).

    Example:
        import dspy
        dspy.configure(lm=dspy.LM("openai/gpt-4"))

        checker = LLMContradictionChecker()
        result = checker(
            response="Your flight CA1234 is on time.",
            retrieval_context=["Flight info: CA1234"],
            tool_results=[{"tool": "get_status", "result": {"status": "delayed"}}],
        )
        print(result.has_contradiction)  # True
    """

    def __init__(self, use_cot: bool = True):
        super().__init__()
        Predictor = dspy.ChainOfThought if use_cot else dspy.Predict
        self.classifier = Predictor(ContradictionClassifier)

    def forward(
        self,
        response: str,
        retrieval_context: list[str] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ContradictionResult:
        """Check for contradictions between response and source data.

        Args:
            response: The agent's response text.
            retrieval_context: Text snippets from knowledge base/RAG.
            tool_results: Results from tool/function calls.

        Returns:
            ContradictionResult with detected contradictions.
        """
        retrieval_context = retrieval_context or []
        tool_results = tool_results or []

        # No source data - nothing to check
        if not retrieval_context and not tool_results:
            return ContradictionResult.no_source_data(method="llm")

        # Format source data for LLM
        source_data = self._format_source_data(retrieval_context, tool_results)

        # Call LLM
        try:
            prediction = self.classifier(
                response=response,
                source_data=source_data,
            )

            # Parse the result
            return self._parse_prediction(prediction)

        except Exception as e:
            # Fallback on error
            return ContradictionResult(
                has_contradiction=False,
                contradiction_score=0.0,
                contradictions=[],
                method="llm",
                reasoning=f"LLM check failed: {str(e)}",
            )

    def _format_source_data(
        self,
        retrieval_context: list[str],
        tool_results: list[dict[str, Any]],
    ) -> str:
        """Format source data for LLM consumption."""
        parts = []

        # Add retrieval context
        if retrieval_context:
            parts.append("=== Knowledge Base ===")
            for i, ctx in enumerate(retrieval_context, 1):
                parts.append(f"[{i}] {ctx}")

        # Add tool results
        if tool_results:
            parts.append("\n=== Database/Tool Results ===")
            for result in tool_results:
                tool_name = result.get("tool", result.get("tool_name", "unknown"))
                tool_output = result.get("result", result.get("output", {}))
                if isinstance(tool_output, dict):
                    formatted = json.dumps(tool_output, indent=2, ensure_ascii=False)
                else:
                    formatted = str(tool_output)
                parts.append(f"[{tool_name}] {formatted}")

        return "\n".join(parts)

    def _parse_prediction(self, prediction: dspy.Prediction) -> ContradictionResult:
        """Parse LLM prediction into ContradictionResult."""
        # Parse contradiction list
        contradictions = []
        try:
            if hasattr(prediction, "contradictions_json") and prediction.contradictions_json:
                raw_json = prediction.contradictions_json
                # Handle potential JSON wrapped in code blocks
                if "```" in raw_json:
                    raw_json = raw_json.split("```")[1]
                    if raw_json.startswith("json"):
                        raw_json = raw_json[4:]
                contradiction_list = json.loads(raw_json)
                for c in contradiction_list:
                    contradictions.append(
                        Contradiction(
                            claim=c.get("claim", ""),
                            source_value=c.get("source_value", ""),
                            claim_value=c.get("claim_value", ""),
                            field=c.get("field", "unknown"),
                            field_type=self._infer_field_type(c.get("field", "")),
                            source_type=SourceType.TOOL,  # Default
                            severity=self._parse_severity(c.get("severity", "medium")),
                            confidence=0.8,
                        )
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Keep empty contradiction list

        # Extract score, ensuring it's in valid range
        score = 0.0
        if hasattr(prediction, "contradiction_score"):
            try:
                score = float(prediction.contradiction_score)
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                pass

        # Extract has_contradiction
        has_contradiction = False
        if hasattr(prediction, "has_contradiction"):
            has_contradiction = bool(prediction.has_contradiction)

        # Extract reasoning
        reasoning = ""
        if hasattr(prediction, "reasoning"):
            reasoning = str(prediction.reasoning)

        return ContradictionResult(
            has_contradiction=has_contradiction or len(contradictions) > 0,
            contradiction_score=score if score > 0 else (0.5 if contradictions else 0.0),
            contradictions=contradictions,
            method="llm",
            reasoning=reasoning,
        )

    def _infer_field_type(self, field: str) -> FieldType:
        """Infer field type from field name."""
        field_lower = field.lower()
        if any(p in field_lower for p in ["price", "cost", "fee", "amount"]):
            return FieldType.PRICE
        if any(p in field_lower for p in ["status", "state"]):
            return FieldType.STATUS
        if any(p in field_lower for p in ["date", "day"]):
            return FieldType.DATE
        if any(p in field_lower for p in ["time", "hour"]):
            return FieldType.TIME
        if any(p in field_lower for p in ["flight"]):
            return FieldType.FLIGHT
        return FieldType.UNKNOWN

    def _parse_severity(self, severity: str) -> Severity:
        """Parse severity string to enum."""
        severity_lower = severity.lower().strip()
        if severity_lower == "high":
            return Severity.HIGH
        elif severity_lower == "low":
            return Severity.LOW
        return Severity.MEDIUM

    def check(
        self,
        response: str,
        retrieval_context: list[str] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ContradictionResult:
        """Convenience method - alias for forward()."""
        # Prefer module(...) over forward(...) to avoid DSPy warnings
        return self(response, retrieval_context, tool_results)
