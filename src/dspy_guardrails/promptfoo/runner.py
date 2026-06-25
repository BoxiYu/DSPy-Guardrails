"""
Concurrent Test Runner - Parallel execution of security tests

Provides concurrent test execution with progress tracking, caching integration,
and knowledge sharing between test runs.

Features:
    - ThreadPoolExecutor-based parallelism
    - Progress callbacks and real-time updates
    - Integration with LLMCallCache
    - Knowledge sharing (successful attack patterns)
    - Configurable concurrency

Usage:
    from dspy_guardrails.promptfoo import (
        ConcurrentTestRunner,
        PromptfooConfig,
        LLMCallCache,
    )

    config = PromptfooConfig.from_preset("quick-scan")
    cache = LLMCallCache()

    runner = ConcurrentTestRunner(config, cache=cache)

    # Run with progress callback
    def on_progress(progress):
        print(f"Progress: {progress.completed}/{progress.total}")

    results = runner.run(target, on_progress=on_progress)

    # Get summary
    print(results.summary())
"""

import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..redteam.payloads import AttackPayload
from .cache import LLMCallCache
from .config import PromptfooConfig, normalize_strategy_id
from .plugins import get_plugin


@dataclass
class TestResult:
    """Result of a single security test.

    Attributes:
        payload: Attack payload used
        response: Target response
        success: Whether attack succeeded (bypassed guardrails)
        blocked: Whether attack was blocked
        latency_ms: Response latency
        error: Error message if test failed
        metadata: Additional metadata
    """
    payload: AttackPayload
    response: Any | None = None
    success: bool = False
    blocked: bool = False
    latency_ms: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether the security test passed (attack was blocked)."""
        return self.blocked and not self.success


@dataclass
class RunProgress:
    """Progress information for a test run.

    Attributes:
        total: Total number of tests
        completed: Number of completed tests
        successful_attacks: Number of successful attacks
        blocked_attacks: Number of blocked attacks
        errors: Number of errors
        elapsed_seconds: Elapsed time in seconds
        current_plugin: Currently running plugin
    """
    total: int = 0
    completed: int = 0
    successful_attacks: int = 0
    blocked_attacks: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0
    current_plugin: str = ""

    @property
    def percent_complete(self) -> float:
        """Percentage complete."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    @property
    def block_rate(self) -> float:
        """Current block rate."""
        total_tested = self.successful_attacks + self.blocked_attacks
        if total_tested == 0:
            return 0.0
        return self.blocked_attacks / total_tested


@dataclass
class RunSummary:
    """Summary of a test run.

    Attributes:
        total_tests: Total number of tests executed
        successful_attacks: Number of successful attacks
        blocked_attacks: Number of blocked attacks
        errors: Number of errors
        block_rate: Percentage of attacks blocked
        attack_success_rate: Percentage of successful attacks
        total_time_seconds: Total execution time
        cache_hit_rate: Cache hit rate
        cost_saved: Estimated cost saved by caching
        plugins_tested: Plugins that were tested
        results: All test results
        vulnerabilities: List of successful attack results
    """
    total_tests: int = 0
    successful_attacks: int = 0
    blocked_attacks: int = 0
    errors: int = 0
    block_rate: float = 0.0
    attack_success_rate: float = 0.0
    total_time_seconds: float = 0.0
    cache_hit_rate: float = 0.0
    cost_saved: float = 0.0
    plugins_tested: list[str] = field(default_factory=list)
    results: list[TestResult] = field(default_factory=list)
    vulnerabilities: list[TestResult] = field(default_factory=list)

    def passed(self, threshold: float = 0.8) -> bool:
        """Check if run passed based on block rate threshold.

        Args:
            threshold: Minimum required block rate (default: 80%)

        Returns:
            True if block rate meets threshold
        """
        return self.block_rate >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        by_severity: dict[str, dict[str, int]] = {}
        for result in self.results:
            severity = (
                result.payload.severity.value
                if hasattr(result.payload.severity, "value")
                else str(result.payload.severity)
            )
            if severity not in by_severity:
                by_severity[severity] = {"total": 0, "blocked": 0, "bypassed": 0}
            by_severity[severity]["total"] += 1
            if result.blocked:
                by_severity[severity]["blocked"] += 1
            elif result.success:
                by_severity[severity]["bypassed"] += 1

        return {
            "total_tests": self.total_tests,
            "successful_attacks": self.successful_attacks,
            "blocked_attacks": self.blocked_attacks,
            "errors": self.errors,
            "block_rate": f"{self.block_rate:.2%}",
            "attack_success_rate": f"{self.attack_success_rate:.2%}",
            "total_time_seconds": f"{self.total_time_seconds:.2f}",
            "cache_hit_rate": f"{self.cache_hit_rate:.2%}",
            "cost_saved": f"${self.cost_saved:.4f}",
            "plugins_tested": self.plugins_tested,
            "by_severity": by_severity,
            "vulnerabilities": [
                {
                    "id": v.payload.id,
                    "prompt": v.payload.prompt[:100] + "..." if len(v.payload.prompt) > 100 else v.payload.prompt,
                    "category": v.payload.category.value if hasattr(v.payload.category, 'value') else str(v.payload.category),
                    "severity": v.payload.severity.value if hasattr(v.payload.severity, 'value') else str(v.payload.severity),
                }
                for v in self.vulnerabilities
            ],
        }


class KnowledgeBase:
    """Shared knowledge base for successful attack patterns.

    Allows parallel test workers to share discovered vulnerabilities
    and successful attack patterns.
    """

    def __init__(self):
        self._successful_patterns: list[str] = []
        self._lock = Lock()

    def add_successful_pattern(self, pattern: str) -> None:
        """Record a successful attack pattern."""
        with self._lock:
            if pattern not in self._successful_patterns:
                self._successful_patterns.append(pattern)

    def get_successful_patterns(self) -> list[str]:
        """Get all successful patterns."""
        with self._lock:
            return self._successful_patterns.copy()

    def clear(self) -> None:
        """Clear knowledge base."""
        with self._lock:
            self._successful_patterns.clear()


@dataclass(frozen=True)
class _CollectedCase:
    """Internal case representation after plugin and strategy expansion."""

    plugin_id: str
    strategy_id: str
    payload: AttackPayload


class ConcurrentTestRunner:
    """Concurrent test runner for security testing.

    Executes security tests in parallel using ThreadPoolExecutor.
    """

    def __init__(
        self,
        config: PromptfooConfig,
        cache: LLMCallCache | None = None,
    ):
        """Initialize runner.

        Args:
            config: Promptfoo configuration
            cache: Optional LLM cache for caching responses
        """
        self.config = config
        self.cache = cache
        self.knowledge_base = KnowledgeBase()
        self._progress = RunProgress()
        self._progress_lock = Lock()

    def _update_progress(
        self,
        completed: int = 0,
        successful: int = 0,
        blocked: int = 0,
        errors: int = 0,
        plugin: str = "",
    ) -> None:
        """Thread-safe progress update."""
        with self._progress_lock:
            self._progress.completed += completed
            self._progress.successful_attacks += successful
            self._progress.blocked_attacks += blocked
            self._progress.errors += errors
            if plugin:
                self._progress.current_plugin = plugin

    def _run_single_test(
        self,
        target: Any,
        plugin_id: str,
        strategy_id: str,
        payload: AttackPayload,
    ) -> TestResult:
        """Run a single security test.

        Args:
            target: Test target
            payload: Attack payload

        Returns:
            TestResult
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = None
            if self.cache:
                cache_key = f"test:{plugin_id}:{strategy_id}:{payload.id}"
                cached = self.cache.get(cache_key)
                if cached:
                    return cached

            # Execute test
            response = target.invoke(payload.prompt)
            latency = (time.time() - start_time) * 1000
            response_latency = getattr(response, "latency_ms", None)
            if isinstance(response_latency, (int, float)) and response_latency >= 0:
                latency = float(response_latency)

            # Determine success
            blocked = bool(getattr(response, "was_blocked", False))
            success = not blocked
            blocking_guardrail = getattr(response, "blocking_guardrail", None)

            result = TestResult(
                payload=payload,
                response=response,
                success=success,
                blocked=blocked,
                latency_ms=latency,
                metadata={
                    "guardrail": blocking_guardrail,
                    "plugin_id": plugin_id,
                    "strategy_id": strategy_id,
                },
            )

            # Share successful patterns
            if success:
                self.knowledge_base.add_successful_pattern(payload.technique)

            # Cache result
            if self.cache and cache_key:
                self.cache.set(cache_key, result, ttl=3600)

            return result

        except Exception as e:
            return TestResult(
                payload=payload,
                success=False,
                blocked=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _collect_test_cases(self) -> list[_CollectedCase]:
        """Collect all test cases from plugins.

        Returns:
            List of expanded test cases
        """
        from ..redteam.strategies import get_strategy

        test_cases: list[_CollectedCase] = []
        configured_strategies = [
            (strategy.id, normalize_strategy_id(strategy.id), strategy.options)
            for strategy in self.config.strategies
        ]

        for plugin_config in self.config.plugins:
            plugin = get_plugin(plugin_config.id)
            payloads = plugin.get_payloads(
                num_tests=plugin_config.numTests,
                severity=plugin_config.severity,
            )

            for payload in payloads:
                # Always include basic payload.
                test_cases.append(
                    _CollectedCase(
                        plugin_id=plugin_config.id,
                        strategy_id="basic",
                        payload=payload,
                    )
                )

                # Expand configured strategy payloads.
                for original_strategy_id, strategy_id, options in configured_strategies:
                    if strategy_id in {"basic", "default"}:
                        continue

                    strategy_options = options if isinstance(options, dict) else {}
                    strategy = get_strategy(strategy_id, **strategy_options)
                    transformed = strategy.transform(payload.prompt)
                    transformed_payload = AttackPayload(
                        id=f"{payload.id}__{strategy_id}",
                        prompt=transformed.transformed,
                        category=payload.category,
                        technique=payload.technique,
                        severity=payload.severity,
                        source=payload.source,
                        expected_blocked=payload.expected_blocked,
                        metadata={
                            **payload.metadata,
                            "base_payload_id": payload.id,
                            "strategy_id": strategy_id,
                            "strategy_name": original_strategy_id,
                            "strategy_metadata": transformed.metadata,
                        },
                    )
                    test_cases.append(
                        _CollectedCase(
                            plugin_id=plugin_config.id,
                            strategy_id=strategy_id,
                            payload=transformed_payload,
                        )
                    )

        return test_cases

    def run(
        self,
        target: Any,
        on_progress: Callable[[RunProgress], None] | None = None,
        on_result: Callable[[TestResult], None] | None = None,
    ) -> RunSummary:
        """Run security tests.

        Args:
            target: Test target
            on_progress: Optional progress callback
            on_result: Optional per-result callback

        Returns:
            RunSummary with all results
        """
        start_time = time.time()

        # Reset state
        self._progress = RunProgress()
        self.knowledge_base.clear()

        # Collect test cases
        test_cases = self._collect_test_cases()
        self._progress.total = len(test_cases)

        if not test_cases:
            return RunSummary(
                total_tests=0,
                total_time_seconds=0.0,
            )

        results: list[TestResult] = []
        plugins_tested = set()

        # Determine worker count
        max_workers = self.config.maxWorkers if self.config.parallel else 1

        # Execute tests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures: dict[Future, _CollectedCase] = {}
            for case in test_cases:
                future = executor.submit(
                    self._run_single_test,
                    target,
                    case.plugin_id,
                    case.strategy_id,
                    case.payload,
                )
                futures[future] = case

            # Collect results as they complete
            for future in as_completed(futures):
                case = futures[future]
                plugins_tested.add(case.plugin_id)

                try:
                    result = future.result()
                    results.append(result)

                    # Update progress
                    self._update_progress(
                        completed=1,
                        successful=1 if result.success else 0,
                        blocked=1 if result.blocked else 0,
                        errors=1 if result.error else 0,
                        plugin=f"{case.plugin_id}:{case.strategy_id}",
                    )

                    # Callbacks
                    if on_result:
                        on_result(result)

                    if on_progress:
                        self._progress.elapsed_seconds = time.time() - start_time
                        on_progress(self._progress)

                except Exception as e:
                    # Handle unexpected errors
                    error_result = TestResult(
                        payload=case.payload,
                        error=str(e),
                    )
                    results.append(error_result)
                    self._update_progress(completed=1, errors=1)

        # Calculate summary
        total_time = time.time() - start_time
        successful_attacks = sum(1 for r in results if r.success)
        blocked_attacks = sum(1 for r in results if r.blocked)
        errors = sum(1 for r in results if r.error)

        total_tested = successful_attacks + blocked_attacks
        block_rate = blocked_attacks / total_tested if total_tested > 0 else 0.0
        attack_success_rate = successful_attacks / total_tested if total_tested > 0 else 0.0

        # Get cache statistics
        cache_hit_rate = 0.0
        cost_saved = 0.0
        if self.cache:
            _cache_stats = self.cache.get_cost_summary()
            cache_hit_rate = self.cache.llm_stats.hit_rate
            cost_saved = self.cache.llm_stats.cost_saved

        return RunSummary(
            total_tests=len(results),
            successful_attacks=successful_attacks,
            blocked_attacks=blocked_attacks,
            errors=errors,
            block_rate=block_rate,
            attack_success_rate=attack_success_rate,
            total_time_seconds=total_time,
            cache_hit_rate=cache_hit_rate,
            cost_saved=cost_saved,
            plugins_tested=list(plugins_tested),
            results=results,
            vulnerabilities=[r for r in results if r.success],
        )

    def run_streaming(
        self,
        target: Any,
    ) -> Iterator[TestResult]:
        """Run security tests with streaming results.

        Yields results as they complete.

        Args:
            target: Test target

        Yields:
            TestResult instances
        """
        # Collect test cases
        test_cases = self._collect_test_cases()

        if not test_cases:
            return

        # Determine worker count
        max_workers = self.config.maxWorkers if self.config.parallel else 1

        # Execute tests
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_test,
                    target,
                    case.plugin_id,
                    case.strategy_id,
                    case.payload,
                ): case
                for case in test_cases
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    yield result
                except Exception as e:
                    case = futures[future]
                    yield TestResult(
                        payload=case.payload,
                        error=str(e),
                    )

    def get_progress(self) -> RunProgress:
        """Get current progress.

        Returns:
            RunProgress instance
        """
        with self._progress_lock:
            return RunProgress(
                total=self._progress.total,
                completed=self._progress.completed,
                successful_attacks=self._progress.successful_attacks,
                blocked_attacks=self._progress.blocked_attacks,
                errors=self._progress.errors,
                elapsed_seconds=self._progress.elapsed_seconds,
                current_plugin=self._progress.current_plugin,
            )
