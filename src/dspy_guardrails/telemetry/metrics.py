"""
Runtime Metrics - 运行时指标收集

收集 guardrail 调用量、拦截量、延迟分布等指标。
"""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class CheckStats:
    """Statistics for a single check type."""

    total: int = 0
    blocked: int = 0
    passed: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total > 0 else 0.0

    @property
    def block_rate(self) -> float:
        return self.blocked / self.total if self.total > 0 else 0.0


class GuardrailMetrics:
    """Global metrics collector for guardrail operations.

    Thread-safe singleton that tracks call counts, block rates, and latency
    for each guardrail check type.

    Usage:
        metrics = get_metrics()

        with metrics.track("no_injection") as tracker:
            result = guardrail.no_injection(text)
            tracker.set_result(result)

        print(metrics.summary())
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, CheckStats] = defaultdict(CheckStats)
        self._start_time = time.monotonic()

    @contextmanager
    def track(self, check_name: str):
        """Context manager to track a guardrail check.

        Usage:
            with metrics.track("no_injection") as t:
                result = check(text)
                t.set_result(result)  # True=passed, False=blocked
        """
        tracker = _Tracker(check_name)
        start = time.perf_counter()
        error_occurred = False
        try:
            yield tracker
        except Exception:
            error_occurred = True
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                stats = self._stats[check_name]
                stats.total += 1
                stats.total_latency_ms += elapsed_ms
                stats.min_latency_ms = min(stats.min_latency_ms, elapsed_ms)
                stats.max_latency_ms = max(stats.max_latency_ms, elapsed_ms)
                if error_occurred:
                    stats.errors += 1
                elif tracker.result is True:
                    stats.passed += 1
                elif tracker.result is False:
                    stats.blocked += 1

    def record(self, check_name: str, passed: bool, latency_ms: float) -> None:
        """Record a check result directly."""
        with self._lock:
            stats = self._stats[check_name]
            stats.total += 1
            stats.total_latency_ms += latency_ms
            stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
            stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)
            if passed:
                stats.passed += 1
            else:
                stats.blocked += 1

    def get_stats(self, check_name: str) -> CheckStats:
        with self._lock:
            return self._stats[check_name]

    def summary(self) -> dict:
        """Return summary of all metrics."""
        uptime = time.monotonic() - self._start_time
        with self._lock:
            checks = {}
            total_calls = 0
            total_blocks = 0
            for name, stats in self._stats.items():
                total_calls += stats.total
                total_blocks += stats.blocked
                checks[name] = {
                    "total": stats.total,
                    "blocked": stats.blocked,
                    "passed": stats.passed,
                    "errors": stats.errors,
                    "block_rate": round(stats.block_rate, 4),
                    "avg_latency_ms": round(stats.avg_latency_ms, 3),
                    "min_latency_ms": round(stats.min_latency_ms, 3) if stats.total > 0 else 0,
                    "max_latency_ms": round(stats.max_latency_ms, 3),
                }

        return {
            "uptime_seconds": round(uptime, 1),
            "total_checks": total_calls,
            "total_blocks": total_blocks,
            "overall_block_rate": round(total_blocks / total_calls, 4) if total_calls > 0 else 0,
            "checks": checks,
        }

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()
            self._start_time = time.monotonic()


class _Tracker:
    def __init__(self, check_name: str):
        self.check_name = check_name
        self.result: bool | None = None

    def set_result(self, result: bool) -> None:
        self.result = result


_metrics_instance: GuardrailMetrics | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> GuardrailMetrics:
    """Get the global metrics singleton."""
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = GuardrailMetrics()
    return _metrics_instance
