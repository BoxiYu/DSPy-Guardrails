"""
Rule-based contradiction detection.

Fast (~5ms) detection of contradictions between agent output and source data
using pattern matching and value comparison.
"""

from typing import Any

from .core import (
    Contradiction,
    ContradictionResult,
    ExtractedValue,
    FieldType,
    Severity,
    SourceType,
)
from .extractors import ValueExtractor


class ContradictionDetector:
    """Rule-based contradiction detector.

    Detects contradictions between agent output and source data
    (retrieval_context and tool_results).

    Example:
        detector = ContradictionDetector()
        result = detector.detect(
            output="Your flight CA1234 departs at 10:30 and costs $299.",
            retrieval_context=["Flight CA1234: departure 10:30"],
            tool_results=[{"tool": "get_flight", "result": {"price": 450}}],
        )
        # result.has_contradiction == True
        # result.contradictions[0].field == "price"
    """

    # Tolerance thresholds for numeric comparisons
    PRICE_TOLERANCE_PERCENT = 0.01  # 1% tolerance for rounding
    TIME_TOLERANCE_MINUTES = 1  # 1 minute tolerance

    def __init__(self, extractor: ValueExtractor | None = None):
        self.extractor = extractor or ValueExtractor()

    def detect(
        self,
        output: str,
        retrieval_context: list[str] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ContradictionResult:
        """Detect contradictions between output and source data.

        Args:
            output: The agent's response text.
            retrieval_context: List of text snippets from knowledge base/RAG.
            tool_results: List of tool call results (dicts with tool name and result).

        Returns:
            ContradictionResult with detected contradictions.
        """
        retrieval_context = retrieval_context or []
        tool_results = tool_results or []

        # No source data - nothing to check against
        if not retrieval_context and not tool_results:
            return ContradictionResult.no_source_data()

        # Extract values from output
        output_values = self.extractor.extract_from_text(output)
        if not output_values:
            return ContradictionResult.no_contradiction()

        # Extract values from sources
        source_values: list[tuple[ExtractedValue, SourceType, str]] = []

        # From retrieval context
        for ctx in retrieval_context:
            for ev in self.extractor.extract_from_text(ctx):
                source_values.append((ev, SourceType.RETRIEVAL, ctx))

        # From tool results
        for ev in self.extractor.extract_from_tool_results(tool_results):
            # Get the tool name for source snippet
            _tool_name = ev.metadata.get("tool_name", ev.metadata.get("key", "").split(".")[0])
            source_values.append((ev, SourceType.TOOL, ev.source_text))

        if not source_values:
            return ContradictionResult.no_contradiction()

        # Find contradictions
        contradictions = self._find_contradictions(output_values, source_values, output)

        # Calculate score
        if contradictions:
            # Score based on severity
            severity_weights = {
                Severity.HIGH: 1.0,
                Severity.MEDIUM: 0.6,
                Severity.LOW: 0.3,
            }
            max_severity_score = max(
                severity_weights[c.severity] for c in contradictions
            )
            # Factor in number of contradictions
            count_factor = min(len(contradictions) / 3, 1.0)  # Cap at 3 contradictions
            score = max_severity_score * 0.7 + count_factor * 0.3
        else:
            score = 0.0

        return ContradictionResult(
            has_contradiction=len(contradictions) > 0,
            contradiction_score=score,
            contradictions=contradictions,
            method="rule",
            reasoning=self._generate_reasoning(contradictions),
        )

    def _find_contradictions(
        self,
        output_values: list[ExtractedValue],
        source_values: list[tuple[ExtractedValue, SourceType, str]],
        output: str,
    ) -> list[Contradiction]:
        """Find contradictions between output and source values."""
        contradictions: list[Contradiction] = []
        matched_output_indices: set[int] = set()

        for i, output_ev in enumerate(output_values):
            for source_ev, source_type, source_snippet in source_values:
                # Only compare same field types
                if not self._types_comparable(output_ev.field_type, source_ev.field_type):
                    continue

                # Check for contradiction
                is_contradiction, severity = self._values_contradict(
                    output_ev, source_ev
                )

                if is_contradiction:
                    matched_output_indices.add(i)

                    # Extract context around the value in output
                    start, end = output_ev.position
                    context_start = max(0, start - 30)
                    context_end = min(len(output), end + 30)
                    claim_context = output[context_start:context_end]

                    # Determine field name
                    field = self._determine_field_name(output_ev, source_ev)

                    contradictions.append(
                        Contradiction(
                            claim=claim_context,
                            source_value=source_ev.value,
                            claim_value=output_ev.value,
                            field=field,
                            field_type=output_ev.field_type,
                            source_type=source_type,
                            severity=severity,
                            source_snippet=source_snippet[:200],  # Truncate for readability
                            confidence=0.9 if severity == Severity.HIGH else 0.7,
                        )
                    )

        # Deduplicate contradictions for the same field
        return self._deduplicate_contradictions(contradictions)

    def _types_comparable(self, type1: FieldType, type2: FieldType) -> bool:
        """Check if two field types are comparable."""
        if type1 == type2:
            return True

        # Unknown type can match anything
        if type1 == FieldType.UNKNOWN or type2 == FieldType.UNKNOWN:
            return True

        # Price and count might be confused
        if {type1, type2} == {FieldType.PRICE, FieldType.COUNT}:
            return False  # Don't compare price with count

        return False

    def _values_contradict(
        self, output_ev: ExtractedValue, source_ev: ExtractedValue
    ) -> tuple[bool, Severity]:
        """Determine if two values contradict each other.

        Returns:
            Tuple of (is_contradiction, severity)
        """
        field_type = output_ev.field_type
        out_norm = output_ev.normalized
        src_norm = source_ev.normalized

        # Exact match - no contradiction
        if out_norm == src_norm:
            return False, Severity.LOW

        if field_type == FieldType.PRICE:
            return self._price_contradicts(out_norm, src_norm)
        elif field_type == FieldType.TIME:
            return self._time_contradicts(out_norm, src_norm)
        elif field_type == FieldType.STATUS:
            return self._status_contradicts(out_norm, src_norm)
        elif field_type == FieldType.DATE:
            return self._date_contradicts(out_norm, src_norm)
        elif field_type in (FieldType.PERCENTAGE, FieldType.COUNT):
            return self._numeric_contradicts(out_norm, src_norm)
        elif field_type == FieldType.BOOLEAN:
            return self._boolean_contradicts(output_ev.value, source_ev.value)
        else:
            # For other types, simple string comparison
            if out_norm != src_norm:
                return True, Severity.MEDIUM
            return False, Severity.LOW

    def _price_contradicts(self, out_norm: str, src_norm: str) -> tuple[bool, Severity]:
        """Check if prices contradict."""
        try:
            out_price = float(out_norm)
            src_price = float(src_norm)

            if out_price == src_price:
                return False, Severity.LOW

            # Calculate difference percentage
            diff_pct = abs(out_price - src_price) / max(src_price, 0.01)

            if diff_pct <= self.PRICE_TOLERANCE_PERCENT:
                return False, Severity.LOW  # Within rounding tolerance
            elif diff_pct <= 0.05:
                return True, Severity.LOW  # Small difference
            elif diff_pct <= 0.2:
                return True, Severity.MEDIUM  # Moderate difference
            else:
                return True, Severity.HIGH  # Large difference
        except ValueError:
            return True, Severity.MEDIUM

    def _time_contradicts(self, out_norm: str, src_norm: str) -> tuple[bool, Severity]:
        """Check if times contradict."""
        try:
            out_parts = out_norm.split(":")
            src_parts = src_norm.split(":")

            out_minutes = int(out_parts[0]) * 60 + int(out_parts[1])
            src_minutes = int(src_parts[0]) * 60 + int(src_parts[1])

            diff = abs(out_minutes - src_minutes)

            if diff <= self.TIME_TOLERANCE_MINUTES:
                return False, Severity.LOW
            elif diff <= 5:
                return True, Severity.LOW  # 5 minutes off
            elif diff <= 30:
                return True, Severity.MEDIUM  # 30 minutes off
            else:
                return True, Severity.HIGH  # More than 30 minutes off
        except (ValueError, IndexError):
            return out_norm != src_norm, Severity.MEDIUM

    def _status_contradicts(self, out_norm: str, src_norm: str) -> tuple[bool, Severity]:
        """Check if statuses contradict."""
        if out_norm == src_norm:
            return False, Severity.LOW

        # Define conflicting status pairs (HIGH severity)
        high_conflicts = [
            ("on_time", "delayed"),
            ("on_time", "cancelled"),
            ("departed", "cancelled"),
            ("confirmed", "cancelled"),
            ("confirmed", "failed"),
            ("available", "unavailable"),
            ("available", "sold_out"),
            ("active", "inactive"),
            ("active", "expired"),
        ]

        for s1, s2 in high_conflicts:
            if {out_norm, src_norm} == {s1, s2}:
                return True, Severity.HIGH

        # Any other status mismatch is medium severity
        return True, Severity.MEDIUM

    def _date_contradicts(self, out_norm: str, src_norm: str) -> tuple[bool, Severity]:
        """Check if dates contradict."""
        if out_norm == src_norm:
            return False, Severity.LOW

        # Try to compare as dates
        try:
            from datetime import datetime

            out_date = datetime.strptime(out_norm, "%Y-%m-%d")
            src_date = datetime.strptime(src_norm, "%Y-%m-%d")

            diff_days = abs((out_date - src_date).days)

            if diff_days == 0:
                return False, Severity.LOW
            elif diff_days == 1:
                return True, Severity.MEDIUM  # One day off
            else:
                return True, Severity.HIGH  # More than one day off
        except ValueError:
            return True, Severity.MEDIUM

    def _numeric_contradicts(
        self, out_norm: str, src_norm: str
    ) -> tuple[bool, Severity]:
        """Check if numeric values contradict."""
        try:
            out_val = float(out_norm)
            src_val = float(src_norm)

            if out_val == src_val:
                return False, Severity.LOW

            if src_val == 0:
                return True, Severity.HIGH

            diff_pct = abs(out_val - src_val) / abs(src_val)

            if diff_pct <= 0.01:
                return False, Severity.LOW  # Within 1%
            elif diff_pct <= 0.1:
                return True, Severity.LOW  # 10%
            elif diff_pct <= 0.5:
                return True, Severity.MEDIUM  # 50%
            else:
                return True, Severity.HIGH  # >50%
        except ValueError:
            return out_norm != src_norm, Severity.MEDIUM

    def _boolean_contradicts(self, out_val: str, src_val: str) -> tuple[bool, Severity]:
        """Check if boolean values contradict."""
        true_values = {"true", "yes", "1", "是", "对"}
        false_values = {"false", "no", "0", "否", "错"}

        out_lower = out_val.lower().strip()
        src_lower = src_val.lower().strip()

        out_is_true = out_lower in true_values
        out_is_false = out_lower in false_values
        src_is_true = src_lower in true_values
        src_is_false = src_lower in false_values

        # Can't determine - treat as medium severity mismatch
        if not (out_is_true or out_is_false) or not (src_is_true or src_is_false):
            return out_lower != src_lower, Severity.MEDIUM

        # Boolean contradiction
        if out_is_true != src_is_true:
            return True, Severity.HIGH

        return False, Severity.LOW

    def _determine_field_name(
        self, output_ev: ExtractedValue, source_ev: ExtractedValue
    ) -> str:
        """Determine a meaningful field name for the contradiction."""
        # Check if source has a key from JSON extraction
        if "key" in source_ev.metadata:
            key = source_ev.metadata["key"]
            # Get the last part of a dotted key
            parts = key.split(".")
            return parts[-1] if parts else key

        # Use field type as fallback
        return output_ev.field_type.value

    def _deduplicate_contradictions(
        self, contradictions: list[Contradiction]
    ) -> list[Contradiction]:
        """Remove duplicate contradictions for the same field."""
        seen: dict[tuple[str, str], Contradiction] = {}

        for c in contradictions:
            key = (c.field, c.claim_value)
            if key not in seen:
                seen[key] = c
            else:
                # Keep the higher severity one
                if c.severity.value > seen[key].severity.value:
                    seen[key] = c

        return list(seen.values())

    def _generate_reasoning(self, contradictions: list[Contradiction]) -> str:
        """Generate reasoning text for the detection result."""
        if not contradictions:
            return "No contradictions detected between output and source data."

        lines = [f"Found {len(contradictions)} contradiction(s):"]
        for c in contradictions:
            lines.append(
                f"  - {c.field}: agent says '{c.claim_value}' but source has '{c.source_value}' "
                f"({c.severity.value} severity)"
            )

        return "\n".join(lines)
