"""
Efficiency Evaluator - Performance Testing

Evaluates the operational efficiency of AI agent systems including:
- Response latency
- Token usage and cost
- Accuracy and relevance
- Resolution metrics
- Throughput under load
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..targets.base import BaseTarget, TargetResponse


class MetricCategory(Enum):
    """Categories of efficiency metrics."""

    LATENCY = "latency"
    TOKENS = "tokens"
    ACCURACY = "accuracy"
    RESOLUTION = "resolution"
    THROUGHPUT = "throughput"
    COST = "cost"


@dataclass
class EfficiencyTestCase:
    """A test case for efficiency evaluation."""

    id: str
    prompt: str
    expected_response_pattern: Optional[str] = None
    expected_keywords: List[str] = field(default_factory=list)
    category: str = "general"
    complexity: str = "medium"  # simple, medium, complex
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EfficiencyTestResult:
    """Result of an efficiency test."""

    test_case: EfficiencyTestCase
    response: TargetResponse
    latency_ms: float
    token_count_estimate: int  # Estimated tokens in response
    is_accurate: bool
    is_relevant: bool
    keywords_found: List[str]
    error: Optional[str] = None

    @property
    def response_quality_score(self) -> float:
        """Calculate response quality (0-1)."""
        if self.error:
            return 0.0

        score = 0.0
        if self.is_accurate:
            score += 0.5
        if self.is_relevant:
            score += 0.3

        # Bonus for keywords found
        if self.test_case.expected_keywords:
            keyword_ratio = len(self.keywords_found) / len(self.test_case.expected_keywords)
            score += 0.2 * keyword_ratio
        else:
            score += 0.2  # No keywords expected, full bonus

        return min(score, 1.0)


@dataclass
class LatencyMetrics:
    """Latency statistics."""

    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float


@dataclass
class ThroughputMetrics:
    """Throughput under load."""

    requests_per_second: float
    concurrent_users: int
    success_rate: float
    error_rate: float
    avg_latency_under_load_ms: float


@dataclass
class CostMetrics:
    """Cost estimation metrics."""

    total_tokens: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    cost_per_request_usd: float
    tokens_per_request: float


@dataclass
class EfficiencyReport:
    """Comprehensive efficiency report."""

    total_tests: int
    successful_tests: int
    failed_tests: int

    # Latency
    latency: LatencyMetrics

    # Quality
    avg_accuracy: float
    avg_relevance: float
    avg_quality_score: float

    # Cost
    cost: Optional[CostMetrics] = None

    # Throughput (if load testing was performed)
    throughput: Optional[ThroughputMetrics] = None

    # By category breakdown
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_complexity: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # All results
    all_results: List[EfficiencyTestResult] = field(default_factory=list)

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=== Efficiency Test Report ===",
            f"Total Tests: {self.total_tests}",
            f"Successful: {self.successful_tests} ({self.successful_tests/max(self.total_tests,1):.1%})",
            f"Failed: {self.failed_tests}",
            "",
            "Latency Metrics:",
            f"  Mean: {self.latency.mean_ms:.1f}ms",
            f"  Median: {self.latency.median_ms:.1f}ms",
            f"  P95: {self.latency.p95_ms:.1f}ms",
            f"  P99: {self.latency.p99_ms:.1f}ms",
            "",
            "Quality Metrics:",
            f"  Accuracy: {self.avg_accuracy:.1%}",
            f"  Relevance: {self.avg_relevance:.1%}",
            f"  Overall Quality: {self.avg_quality_score:.1%}",
        ]

        if self.cost:
            lines.extend([
                "",
                "Cost Metrics:",
                f"  Total Tokens: {self.cost.total_tokens:,}",
                f"  Estimated Cost: ${self.cost.estimated_cost_usd:.4f}",
                f"  Cost/Request: ${self.cost.cost_per_request_usd:.6f}",
            ])

        if self.throughput:
            lines.extend([
                "",
                "Throughput Metrics:",
                f"  Requests/Second: {self.throughput.requests_per_second:.1f}",
                f"  Success Rate: {self.throughput.success_rate:.1%}",
                f"  Avg Latency Under Load: {self.throughput.avg_latency_under_load_ms:.1f}ms",
            ])

        return "\n".join(lines)


class EfficiencyEvaluator:
    """
    Efficiency Evaluator

    Evaluates operational efficiency of AI agent systems.

    Example:
        evaluator = EfficiencyEvaluator(target)
        report = evaluator.run_all()
        print(report.summary())
    """

    # Token cost estimates (per 1K tokens)
    TOKEN_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "moonshot-v1-8k": {"input": 0.012, "output": 0.012},
        "default": {"input": 0.01, "output": 0.03},
    }

    def __init__(
        self,
        target: BaseTarget,
        model_name: str = "default",
        verbose: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            target: The target to evaluate
            model_name: Name of the model for cost estimation
            verbose: Enable verbose output
        """
        self.target = target
        self.model_name = model_name
        self.verbose = verbose
        self.token_cost = self.TOKEN_COSTS.get(model_name, self.TOKEN_COSTS["default"])

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (rough: ~4 chars per token)."""
        return max(1, len(text) // 4)

    def check_accuracy(self, response: str, test_case: EfficiencyTestCase) -> bool:
        """
        Check if response is accurate.

        Uses pattern matching and keyword checking.
        """
        if not response:
            return False

        response_lower = response.lower()

        # Check for error patterns
        error_patterns = ["error", "i cannot", "i'm sorry", "unable to"]
        for pattern in error_patterns:
            if pattern in response_lower:
                return False

        # If expected pattern specified, check it
        if test_case.expected_response_pattern:
            import re
            if not re.search(test_case.expected_response_pattern, response, re.IGNORECASE):
                return False

        return True

    def check_relevance(self, response: str, test_case: EfficiencyTestCase) -> Tuple[bool, List[str]]:
        """
        Check if response is relevant to the query.

        Returns (is_relevant, keywords_found)
        """
        if not response:
            return False, []

        response_lower = response.lower()
        keywords_found = []

        # Check expected keywords
        for keyword in test_case.expected_keywords:
            if keyword.lower() in response_lower:
                keywords_found.append(keyword)

        # Consider relevant if at least 50% of keywords found (or no keywords expected)
        if test_case.expected_keywords:
            is_relevant = len(keywords_found) >= len(test_case.expected_keywords) * 0.5
        else:
            # Simple relevance check based on response length
            is_relevant = len(response) > 20

        return is_relevant, keywords_found

    def run_test(self, test_case: EfficiencyTestCase) -> EfficiencyTestResult:
        """Run a single efficiency test."""
        start_time = time.time()
        error = None

        try:
            response = self.target.invoke(test_case.prompt)
            latency_ms = (time.time() - start_time) * 1000

            response_text = response.response or ""
            token_count = self.estimate_tokens(test_case.prompt + response_text)

            is_accurate = self.check_accuracy(response_text, test_case)
            is_relevant, keywords_found = self.check_relevance(response_text, test_case)

            return EfficiencyTestResult(
                test_case=test_case,
                response=response,
                latency_ms=latency_ms,
                token_count_estimate=token_count,
                is_accurate=is_accurate,
                is_relevant=is_relevant,
                keywords_found=keywords_found,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return EfficiencyTestResult(
                test_case=test_case,
                response=TargetResponse(response="", guardrail_status={}, latency_ms=latency_ms),
                latency_ms=latency_ms,
                token_count_estimate=0,
                is_accurate=False,
                is_relevant=False,
                keywords_found=[],
                error=str(e),
            )

    def run_load_test(
        self,
        test_cases: List[EfficiencyTestCase],
        concurrent_users: int = 5,
        duration_seconds: int = 10,
    ) -> ThroughputMetrics:
        """
        Run load test with concurrent users.

        Args:
            test_cases: Test cases to run
            concurrent_users: Number of concurrent simulated users
            duration_seconds: Duration of load test
        """
        if self.verbose:
            print(f"Running load test: {concurrent_users} concurrent users for {duration_seconds}s")

        results: List[EfficiencyTestResult] = []
        errors = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            test_idx = 0

            while time.time() - start_time < duration_seconds:
                test_case = test_cases[test_idx % len(test_cases)]
                futures.append(executor.submit(self.run_test, test_case))
                test_idx += 1

                # Small delay to avoid overwhelming
                time.sleep(0.1)

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                    if result.error:
                        errors += 1
                except Exception:
                    errors += 1

        elapsed = time.time() - start_time
        total_requests = len(results) + errors

        latencies = [r.latency_ms for r in results if not r.error]
        avg_latency = statistics.mean(latencies) if latencies else 0

        return ThroughputMetrics(
            requests_per_second=total_requests / elapsed if elapsed > 0 else 0,
            concurrent_users=concurrent_users,
            success_rate=len([r for r in results if not r.error]) / max(total_requests, 1),
            error_rate=errors / max(total_requests, 1),
            avg_latency_under_load_ms=avg_latency,
        )

    def calculate_latency_metrics(self, latencies: List[float]) -> LatencyMetrics:
        """Calculate latency statistics."""
        if not latencies:
            return LatencyMetrics(0, 0, 0, 0, 0, 0, 0)

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return LatencyMetrics(
            min_ms=min(latencies),
            max_ms=max(latencies),
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            p95_ms=sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1],
            p99_ms=sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1],
            std_dev_ms=statistics.stdev(latencies) if n > 1 else 0,
        )

    def calculate_cost_metrics(self, results: List[EfficiencyTestResult]) -> CostMetrics:
        """Calculate cost metrics."""
        total_tokens = sum(r.token_count_estimate for r in results)
        # Rough split: 40% input, 60% output
        input_tokens = int(total_tokens * 0.4)
        output_tokens = total_tokens - input_tokens

        input_cost = (input_tokens / 1000) * self.token_cost["input"]
        output_cost = (output_tokens / 1000) * self.token_cost["output"]
        total_cost = input_cost + output_cost

        num_requests = len(results)

        return CostMetrics(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=total_cost,
            cost_per_request_usd=total_cost / max(num_requests, 1),
            tokens_per_request=total_tokens / max(num_requests, 1),
        )

    def get_default_test_cases(self) -> List[EfficiencyTestCase]:
        """Get default efficiency test cases."""
        return [
            # Simple queries
            EfficiencyTestCase(
                id="EFF-001",
                prompt="What are your business hours?",
                expected_keywords=["hours", "open", "available"],
                category="general",
                complexity="simple",
            ),
            EfficiencyTestCase(
                id="EFF-002",
                prompt="How can I contact customer support?",
                expected_keywords=["contact", "support", "help", "phone", "email"],
                category="support",
                complexity="simple",
            ),
            # Medium complexity
            EfficiencyTestCase(
                id="EFF-003",
                prompt="I need to change my flight from New York to Los Angeles from March 15 to March 17. What are my options?",
                expected_keywords=["change", "flight", "options", "fee", "available"],
                category="booking",
                complexity="medium",
            ),
            EfficiencyTestCase(
                id="EFF-004",
                prompt="Can you explain the refund policy for cancelled flights?",
                expected_keywords=["refund", "policy", "cancel", "hours", "days"],
                category="policy",
                complexity="medium",
            ),
            # Complex queries
            EfficiencyTestCase(
                id="EFF-005",
                prompt="I booked a flight last week but I need to change the date, add extra baggage, and also upgrade to business class. Can you help with all of these?",
                expected_keywords=["change", "baggage", "upgrade", "business"],
                category="booking",
                complexity="complex",
            ),
            EfficiencyTestCase(
                id="EFF-006",
                prompt="I'm a Gold member and I've had two flights cancelled this month. I want to know my compensation options and whether I can use my points for an upgrade on my next flight.",
                expected_keywords=["gold", "compensation", "points", "upgrade", "cancelled"],
                category="loyalty",
                complexity="complex",
            ),
            # Chinese queries
            EfficiencyTestCase(
                id="EFF-CN-001",
                prompt="请问如何查询我的航班状态？",
                expected_keywords=["航班", "状态", "查询"],
                category="general",
                complexity="simple",
                metadata={"language": "zh"},
            ),
            EfficiencyTestCase(
                id="EFF-CN-002",
                prompt="我想申请退款，请问需要什么条件？",
                expected_keywords=["退款", "条件", "政策"],
                category="refund",
                complexity="medium",
                metadata={"language": "zh"},
            ),
        ]

    def run_all(
        self,
        test_cases: Optional[List[EfficiencyTestCase]] = None,
        run_load_test: bool = False,
        load_test_users: int = 5,
        load_test_duration: int = 10,
    ) -> EfficiencyReport:
        """
        Run all efficiency tests.

        Args:
            test_cases: Custom test cases (uses defaults if None)
            run_load_test: Whether to run load testing
            load_test_users: Concurrent users for load test
            load_test_duration: Duration of load test in seconds

        Returns:
            EfficiencyReport with all metrics
        """
        test_cases = test_cases or self.get_default_test_cases()

        if self.verbose:
            print(f"Running {len(test_cases)} efficiency tests...")

        # Run all tests
        results: List[EfficiencyTestResult] = []
        for i, test_case in enumerate(test_cases):
            if self.verbose:
                print(f"  [{i+1}/{len(test_cases)}] {test_case.id}: {test_case.prompt[:40]}...")

            result = self.run_test(test_case)
            results.append(result)

        # Calculate metrics
        successful = [r for r in results if not r.error]
        failed = [r for r in results if r.error]

        latencies = [r.latency_ms for r in successful]
        latency_metrics = self.calculate_latency_metrics(latencies)

        accuracy_scores = [1.0 if r.is_accurate else 0.0 for r in successful]
        relevance_scores = [1.0 if r.is_relevant else 0.0 for r in successful]
        quality_scores = [r.response_quality_score for r in successful]

        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0
        avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0

        cost_metrics = self.calculate_cost_metrics(results)

        # By category breakdown
        by_category: Dict[str, Dict[str, float]] = {}
        by_complexity: Dict[str, Dict[str, float]] = {}

        for result in successful:
            cat = result.test_case.category
            comp = result.test_case.complexity

            if cat not in by_category:
                by_category[cat] = {"count": 0, "avg_latency": 0, "avg_quality": 0}
            by_category[cat]["count"] += 1
            by_category[cat]["avg_latency"] += result.latency_ms
            by_category[cat]["avg_quality"] += result.response_quality_score

            if comp not in by_complexity:
                by_complexity[comp] = {"count": 0, "avg_latency": 0, "avg_quality": 0}
            by_complexity[comp]["count"] += 1
            by_complexity[comp]["avg_latency"] += result.latency_ms
            by_complexity[comp]["avg_quality"] += result.response_quality_score

        # Calculate averages
        for cat, stats in by_category.items():
            stats["avg_latency"] /= stats["count"]
            stats["avg_quality"] /= stats["count"]

        for comp, stats in by_complexity.items():
            stats["avg_latency"] /= stats["count"]
            stats["avg_quality"] /= stats["count"]

        # Load testing (optional)
        throughput_metrics = None
        if run_load_test:
            throughput_metrics = self.run_load_test(
                test_cases,
                concurrent_users=load_test_users,
                duration_seconds=load_test_duration,
            )

        return EfficiencyReport(
            total_tests=len(results),
            successful_tests=len(successful),
            failed_tests=len(failed),
            latency=latency_metrics,
            avg_accuracy=avg_accuracy,
            avg_relevance=avg_relevance,
            avg_quality_score=avg_quality,
            cost=cost_metrics,
            throughput=throughput_metrics,
            by_category=by_category,
            by_complexity=by_complexity,
            all_results=results,
        )
