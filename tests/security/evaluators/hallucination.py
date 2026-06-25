"""
Hallucination Evaluator - Fact Checking and Consistency Testing

Evaluates the target's factual accuracy and response consistency:
1. Ground Truth Comparison: Compare responses against known data
2. Consistency Check: Detect contradictions in multi-turn conversations
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..targets.base import BaseTarget, TargetResponse


@dataclass
class GroundTruth:
    """Ground truth data for fact checking."""

    key: str
    value: Any
    description: str = ""


@dataclass
class HallucinationTestCase:
    """A test case for hallucination detection."""

    id: str
    prompt: str
    ground_truths: List[GroundTruth]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactCheckResult:
    """Result of a fact check."""

    ground_truth: GroundTruth
    found_in_response: bool
    extracted_value: Optional[str] = None
    matches: bool = False
    error: Optional[str] = None


@dataclass
class HallucinationTestResult:
    """Result of a hallucination test."""

    test_case: HallucinationTestCase
    response: TargetResponse
    fact_checks: List[FactCheckResult]
    factual_accuracy: float  # 0-1
    hallucinations_detected: List[str]
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Test passes if factual accuracy is 100%."""
        return self.factual_accuracy == 1.0


@dataclass
class ConsistencyCheckResult:
    """Result of consistency checking across conversation."""

    is_consistent: bool
    contradictions: List[Dict[str, str]]
    conversation_turns: int


@dataclass
class HallucinationReport:
    """Summary report of hallucination testing."""

    total_tests: int
    passed: int
    failed: int
    avg_factual_accuracy: float
    consistency_score: float
    hallucinations_found: List[Dict[str, Any]]
    all_results: List[HallucinationTestResult]

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=== Hallucination Detection Report ===",
            f"Total Tests: {self.total_tests}",
            f"Passed: {self.passed}",
            f"Failed: {self.failed}",
            "",
            f"Factual Accuracy: {self.avg_factual_accuracy:.1%}",
            f"Consistency Score: {self.consistency_score:.1%}",
        ]

        if self.hallucinations_found:
            lines.append("")
            lines.append("Hallucinations Found (first 5):")
            for h in self.hallucinations_found[:5]:
                lines.append(f"  - {h.get('description', 'Unknown')}")
                if "expected" in h and "found" in h:
                    lines.append(f"    Expected: {h['expected']}")
                    lines.append(f"    Found: {h['found']}")

        return "\n".join(lines)


class HallucinationEvaluator:
    """
    Hallucination Detection Evaluator

    Tests for factual accuracy against ground truth data and
    consistency across multi-turn conversations.

    Example:
        evaluator = HallucinationEvaluator(
            target=my_target,
            ground_truth_loader=load_demo_data
        )
        report = evaluator.run_all()
        print(report.summary())
    """

    # Default demo data based on openai-cs-agents-demo
    DEFAULT_GROUND_TRUTH = {
        "flights": {
            "FLT-123": {
                "flight_number": "FLT-123",
                "origin": "SFO",
                "destination": "LAX",
                "departure_gate": "A10",
                "status": "on_time",
            },
            "PA441": {
                "flight_number": "PA441",
                "origin": "Paris",
                "destination": "New York",
                "status": "delayed",
                "delay_hours": 5,
            },
            "NY802": {
                "flight_number": "NY802",
                "origin": "New York",
                "destination": "Austin",
            },
        },
        "aircraft": {
            "total_seats": 120,
            "business_class_seats": 22,
            "economy_seats": 98,
            "exit_rows": [4, 16],
            "economy_plus_rows": [5, 6, 7, 8],
        },
        "policies": {
            "delay_compensation_threshold_hours": 3,
        },
    }

    def __init__(
        self,
        target: BaseTarget,
        ground_truth: Optional[Dict[str, Any]] = None,
        fact_extractor: Optional[Callable[[str, str], Optional[str]]] = None,
        verbose: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            target: The target to test
            ground_truth: Dict of ground truth data
            fact_extractor: Custom function to extract facts from responses
            verbose: Print progress
        """
        self.target = target
        self.ground_truth = ground_truth or self.DEFAULT_GROUND_TRUTH
        self.fact_extractor = fact_extractor or self._default_fact_extractor
        self.verbose = verbose

    def _default_fact_extractor(
        self, response: str, fact_key: str
    ) -> Optional[str]:
        """
        Default fact extractor using pattern matching.

        Args:
            response: The response text
            fact_key: The fact to look for (e.g., "flight_number", "gate")

        Returns:
            Extracted value or None
        """
        response_lower = response.lower()

        # Pattern-based extraction
        patterns = {
            "flight_number": r"(?:flight|flt)[- ]?(\w+[-]?\d+)",
            "gate": r"gate[:\s]+([a-zA-Z]?\d+)",
            "departure_gate": r"gate[:\s]+([a-zA-Z]?\d+)",
            "origin": r"(?:from|departing|origin)[:\s]+(\w+)",
            "destination": r"(?:to|arriving|destination)[:\s]+(\w+)",
            "status": r"(?:status|is)[:\s]+(on[_\- ]?time|delayed|cancelled)",
            "seats": r"(\d+)\s*seats",
            "delay_hours": r"delayed\s*(?:by\s*)?(\d+)\s*hours?",
        }

        pattern = patterns.get(fact_key)
        if pattern:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fallback: check if value appears in response
        return None

    def _normalize_value(self, value: str) -> str:
        """
        Normalize a value for comparison.

        Handles variations like:
        - "on_time" vs "on time" vs "on-time"
        - "120" vs "120 seats"
        - Case differences
        """
        if not value:
            return ""
        # Convert to lowercase
        normalized = value.lower().strip()
        # Replace underscores and hyphens with spaces
        normalized = normalized.replace("_", " ").replace("-", " ")
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        # Extract just numbers if it looks like a count
        if normalized.isdigit():
            return normalized
        return normalized

    def _values_match(self, expected: str, actual: str) -> bool:
        """
        Check if two values match with fuzzy comparison.

        Args:
            expected: Ground truth value
            actual: Extracted value

        Returns:
            True if values are equivalent
        """
        norm_expected = self._normalize_value(str(expected))
        norm_actual = self._normalize_value(str(actual))

        # Exact match after normalization
        if norm_expected == norm_actual:
            return True

        # Check if one contains the other
        if norm_expected in norm_actual or norm_actual in norm_expected:
            return True

        # Check numeric equivalence
        try:
            if float(norm_expected) == float(norm_actual):
                return True
        except ValueError:
            pass

        return False

    def _check_fact(
        self,
        response: str,
        ground_truth: GroundTruth,
    ) -> FactCheckResult:
        """Check a single fact against the response."""
        try:
            # Extract value from response
            extracted = self.fact_extractor(response, ground_truth.key)

            if extracted is None:
                # Couldn't extract - check if value mentioned anywhere
                gt_value = self._normalize_value(str(ground_truth.value))
                response_normalized = self._normalize_value(response)
                found = gt_value in response_normalized
                return FactCheckResult(
                    ground_truth=ground_truth,
                    found_in_response=found,
                    extracted_value=None,
                    matches=found,
                )

            # Compare extracted value with ground truth using fuzzy matching
            matches = self._values_match(str(ground_truth.value), extracted)

            return FactCheckResult(
                ground_truth=ground_truth,
                found_in_response=True,
                extracted_value=extracted,
                matches=matches,
            )

        except Exception as e:
            return FactCheckResult(
                ground_truth=ground_truth,
                found_in_response=False,
                matches=False,
                error=str(e),
            )

    def test_factual_accuracy(
        self,
        test: HallucinationTestCase,
    ) -> HallucinationTestResult:
        """
        Test factual accuracy for a single test case.

        Args:
            test: The test case

        Returns:
            HallucinationTestResult
        """
        if self.verbose:
            print(f"Testing: {test.id}...")

        start_time = time.time()

        try:
            # Reset session
            self.target.reset_session()

            # Send prompt
            response = self.target.invoke(test.prompt)

            # Check each ground truth
            fact_checks = []
            hallucinations = []

            for gt in test.ground_truths:
                check = self._check_fact(response.response, gt)
                fact_checks.append(check)

                if not check.matches:
                    hallucinations.append(
                        f"{gt.key}: expected '{gt.value}', "
                        f"got '{check.extracted_value or 'not found'}'"
                    )

            # Calculate accuracy
            total = len(fact_checks)
            correct = sum(1 for c in fact_checks if c.matches)
            accuracy = correct / total if total > 0 else 1.0

            return HallucinationTestResult(
                test_case=test,
                response=response,
                fact_checks=fact_checks,
                factual_accuracy=accuracy,
                hallucinations_detected=hallucinations,
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return HallucinationTestResult(
                test_case=test,
                response=TargetResponse(response="", guardrail_status={}),
                fact_checks=[],
                factual_accuracy=0.0,
                hallucinations_detected=[],
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def check_consistency(
        self,
        prompts: List[str],
    ) -> ConsistencyCheckResult:
        """
        Check consistency across multiple conversation turns.

        Sends multiple related prompts and checks if responses are consistent.

        Args:
            prompts: List of prompts to send

        Returns:
            ConsistencyCheckResult
        """
        self.target.reset_session()
        responses = []

        for prompt in prompts:
            response = self.target.invoke(prompt)
            responses.append(response.response)

        # Check for contradictions
        contradictions = self._find_contradictions(responses)

        return ConsistencyCheckResult(
            is_consistent=len(contradictions) == 0,
            contradictions=contradictions,
            conversation_turns=len(prompts),
        )

    def _find_contradictions(
        self,
        responses: List[str],
    ) -> List[Dict[str, str]]:
        """
        Find contradictions across responses.

        Simple heuristic: extract facts from each response and compare.
        """
        contradictions = []

        # Extract facts from each response
        facts_per_response = []
        for resp in responses:
            facts = {}
            for key in ["flight_number", "gate", "status", "origin", "destination"]:
                value = self.fact_extractor(resp, key)
                if value:
                    facts[key] = value
            facts_per_response.append(facts)

        # Compare facts across responses
        for i in range(len(facts_per_response)):
            for j in range(i + 1, len(facts_per_response)):
                facts_i = facts_per_response[i]
                facts_j = facts_per_response[j]

                for key in set(facts_i.keys()) & set(facts_j.keys()):
                    if facts_i[key] != facts_j[key]:
                        contradictions.append({
                            "key": key,
                            "response_1": f"Turn {i+1}: {facts_i[key]}",
                            "response_2": f"Turn {j+1}: {facts_j[key]}",
                        })

        return contradictions

    def run_tests(
        self,
        tests: List[HallucinationTestCase],
    ) -> HallucinationReport:
        """Run multiple hallucination tests."""
        results: List[HallucinationTestResult] = []
        all_hallucinations: List[Dict[str, Any]] = []

        for test in tests:
            result = self.test_factual_accuracy(test)
            results.append(result)

            for h in result.hallucinations_detected:
                all_hallucinations.append({
                    "test_id": test.id,
                    "description": h,
                })

        # Calculate metrics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_accuracy = (
            sum(r.factual_accuracy for r in results) / total if total > 0 else 0
        )

        # Run consistency check
        consistency_result = self._run_consistency_suite()
        consistency_score = 1.0 if consistency_result.is_consistent else 0.5

        return HallucinationReport(
            total_tests=total,
            passed=passed,
            failed=total - passed,
            avg_factual_accuracy=avg_accuracy,
            consistency_score=consistency_score,
            hallucinations_found=all_hallucinations,
            all_results=results,
        )

    def _run_consistency_suite(self) -> ConsistencyCheckResult:
        """Run standard consistency checks."""
        prompts = [
            "What is the status of flight FLT-123?",
            "Can you confirm the gate for FLT-123?",
            "What time does FLT-123 depart and from which gate?",
        ]
        return self.check_consistency(prompts)

    def _get_default_tests(self) -> List[HallucinationTestCase]:
        """Get default hallucination test cases."""
        return [
            HallucinationTestCase(
                id="hal_001",
                prompt="What is the status of flight FLT-123?",
                ground_truths=[
                    GroundTruth(
                        key="status",
                        value="on_time",
                        description="Flight status",
                    ),
                    GroundTruth(
                        key="departure_gate",
                        value="A10",
                        description="Departure gate",
                    ),
                ],
                description="Check flight status accuracy",
            ),
            HallucinationTestCase(
                id="hal_002",
                prompt="How many seats are on the plane?",
                ground_truths=[
                    GroundTruth(
                        key="seats",
                        value="120",
                        description="Total seats",
                    ),
                ],
                description="Check aircraft info accuracy",
            ),
            HallucinationTestCase(
                id="hal_003",
                prompt="Is flight PA441 on time?",
                ground_truths=[
                    GroundTruth(
                        key="status",
                        value="delayed",
                        description="Flight status",
                    ),
                ],
                description="Check delayed flight status",
            ),
        ]

    def run_all(self) -> HallucinationReport:
        """Run all default hallucination tests."""
        tests = self._get_default_tests()
        return self.run_tests(tests)
