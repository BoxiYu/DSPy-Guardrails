"""
Hybrid grounding checker - combines rule-based and LLM detection.

Provides the main API for contradiction detection:
1. Fast rule-based detection (~5ms) for obvious contradictions
2. LLM-based detection (~500ms) for edge cases when rules are uncertain
"""

from typing import Any

from .contradiction import ContradictionDetector
from .core import ContradictionResult


class HybridGroundingChecker:
    """Hybrid contradiction detection combining rules and LLM.

    Strategy:
    1. Rule-based detection runs first (~5ms)
    2. If result is clear (high or low score), return immediately
    3. If uncertain (mid-range score), use LLM for deeper analysis

    Example:
        checker = HybridGroundingChecker(use_llm=True)

        result = checker.check(
            output="Your flight CA1234 departs at 10:30 and costs $299.",
            retrieval_context=["Flight CA1234: departure 10:30"],
            tool_results=[{"tool": "get_flight", "result": {"price": 450}}],
        )

        print(f"Contradiction found: {result.has_contradiction}")
        print(f"Score: {result.contradiction_score}")
        for c in result.contradictions:
            print(f"  - {c.field}: '{c.claim_value}' vs '{c.source_value}'")
    """

    def __init__(
        self,
        use_llm: bool = True,
        rule_confidence_threshold: float = 0.8,
        rule_no_contradiction_threshold: float = 0.1,
    ):
        """Initialize the hybrid checker.

        Args:
            use_llm: Whether to use LLM for uncertain cases.
            rule_confidence_threshold: Score above which rule detection is trusted
                for positive (contradiction found) cases.
            rule_no_contradiction_threshold: Score below which rule detection is
                trusted for negative (no contradiction) cases.
        """
        self.rule_detector = ContradictionDetector()
        self._llm_detector = None
        self.use_llm = use_llm
        self.high_threshold = rule_confidence_threshold
        self.low_threshold = rule_no_contradiction_threshold

    @property
    def llm_detector(self):
        """Lazy initialization of LLM detector."""
        if self._llm_detector is None and self.use_llm:
            from .llm_grounding import LLMContradictionChecker
            self._llm_detector = LLMContradictionChecker()
        return self._llm_detector

    def check(
        self,
        output: str,
        retrieval_context: list[str] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ContradictionResult:
        """Check for contradictions between output and source data.

        Args:
            output: The agent's response text.
            retrieval_context: Text snippets from knowledge base/RAG.
            tool_results: Results from tool/function calls.

        Returns:
            ContradictionResult with detected contradictions.
        """
        retrieval_context = retrieval_context or []
        tool_results = tool_results or []

        # No source data - nothing to check
        if not retrieval_context and not tool_results:
            return ContradictionResult.no_source_data(method="hybrid")

        # Step 1: Rule-based detection (fast)
        rule_result = self.rule_detector.detect(
            output=output,
            retrieval_context=retrieval_context,
            tool_results=tool_results,
        )

        # Step 2: Determine if we need LLM
        if not self.use_llm or self.llm_detector is None:
            return rule_result

        # Clear contradiction found - trust rules
        if rule_result.contradiction_score >= self.high_threshold:
            rule_result.method = "hybrid"
            return rule_result

        # Clearly no contradiction - trust rules
        if rule_result.contradiction_score <= self.low_threshold:
            rule_result.method = "hybrid"
            return rule_result

        # Step 3: Uncertain - use LLM
        try:
            llm_result = self.llm_detector(
                response=output,
                retrieval_context=retrieval_context,
                tool_results=tool_results,
            )
            llm_result.method = "hybrid"
            return llm_result
        except Exception:
            # LLM failed - fall back to rule result
            rule_result.method = "hybrid"
            rule_result.reasoning += " (LLM check failed, using rule result)"
            return rule_result

    def check_rule_only(
        self,
        output: str,
        retrieval_context: list[str] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ContradictionResult:
        """Check using only rule-based detection.

        Useful when you need fast results and can tolerate lower accuracy.

        Args:
            output: The agent's response text.
            retrieval_context: Text snippets from knowledge base/RAG.
            tool_results: Results from tool/function calls.

        Returns:
            ContradictionResult from rule-based detection only.
        """
        return self.rule_detector.detect(
            output=output,
            retrieval_context=retrieval_context or [],
            tool_results=tool_results or [],
        )

    def check_llm_only(
        self,
        output: str,
        retrieval_context: list[str] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> ContradictionResult:
        """Check using only LLM-based detection.

        Requires DSPy to be configured with a language model.

        Args:
            output: The agent's response text.
            retrieval_context: Text snippets from knowledge base/RAG.
            tool_results: Results from tool/function calls.

        Returns:
            ContradictionResult from LLM-based detection only.

        Raises:
            RuntimeError: If LLM is not enabled.
        """
        if not self.use_llm or self.llm_detector is None:
            raise RuntimeError("LLM detection is not enabled")

        return self.llm_detector(
            response=output,
            retrieval_context=retrieval_context or [],
            tool_results=tool_results or [],
        )


# Convenience function for simple usage
def check_grounding(
    output: str,
    retrieval_context: list[str] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    use_llm: bool = False,
) -> ContradictionResult:
    """Check for contradictions between agent output and source data.

    Convenience function that creates a checker and runs detection.

    Args:
        output: The agent's response text.
        retrieval_context: Text snippets from knowledge base/RAG.
        tool_results: Results from tool/function calls.
        use_llm: Whether to use LLM for uncertain cases.

    Returns:
        ContradictionResult with detected contradictions.

    Example:
        result = check_grounding(
            output="Your flight is on time.",
            tool_results=[{"tool": "get_status", "result": {"status": "delayed"}}],
        )
        if result.has_contradiction:
            print("Warning: Response may contain incorrect information")
    """
    checker = HybridGroundingChecker(use_llm=use_llm)
    return checker.check(output, retrieval_context, tool_results)


def is_grounded(
    output: str,
    retrieval_context: list[str] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    threshold: float = 0.3,
) -> bool:
    """Check if agent output is grounded in source data.

    Simple boolean check for use in assertions or validations.

    Args:
        output: The agent's response text.
        retrieval_context: Text snippets from knowledge base/RAG.
        tool_results: Results from tool/function calls.
        threshold: Maximum contradiction score to be considered grounded.

    Returns:
        True if output is grounded (no significant contradictions).

    Example:
        assert is_grounded(
            output=response,
            tool_results=tool_call_results,
        ), "Agent response contains contradictions"
    """
    result = check_grounding(
        output=output,
        retrieval_context=retrieval_context,
        tool_results=tool_results,
        use_llm=False,  # Fast check
    )
    return result.contradiction_score <= threshold
