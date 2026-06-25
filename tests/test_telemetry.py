"""Tests for telemetry module."""

import pytest

from dspy_guardrails.telemetry.metrics import GuardrailMetrics


class TestGuardrailMetrics:

    def test_record(self):
        metrics = GuardrailMetrics()
        metrics.record("no_injection", passed=True, latency_ms=0.5)
        metrics.record("no_injection", passed=False, latency_ms=1.2)

        stats = metrics.get_stats("no_injection")
        assert stats.total == 2
        assert stats.passed == 1
        assert stats.blocked == 1
        assert stats.avg_latency_ms == pytest.approx(0.85, abs=0.01)

    def test_track_context_manager(self):
        metrics = GuardrailMetrics()

        with metrics.track("no_toxicity") as t:
            t.set_result(True)

        stats = metrics.get_stats("no_toxicity")
        assert stats.total == 1
        assert stats.passed == 1
        assert stats.blocked == 0

    def test_track_blocked(self):
        metrics = GuardrailMetrics()

        with metrics.track("no_injection") as t:
            t.set_result(False)

        stats = metrics.get_stats("no_injection")
        assert stats.total == 1
        assert stats.blocked == 1
        assert stats.block_rate == 1.0

    def test_track_error(self):
        metrics = GuardrailMetrics()

        with pytest.raises(ValueError):
            with metrics.track("no_injection"):
                raise ValueError("test error")

        stats = metrics.get_stats("no_injection")
        assert stats.total == 1
        assert stats.errors == 1

    def test_summary(self):
        metrics = GuardrailMetrics()
        metrics.record("no_injection", passed=True, latency_ms=0.5)
        metrics.record("no_injection", passed=False, latency_ms=1.0)
        metrics.record("no_toxicity", passed=True, latency_ms=0.3)

        summary = metrics.summary()
        assert summary["total_checks"] == 3
        assert summary["total_blocks"] == 1
        assert "no_injection" in summary["checks"]
        assert "no_toxicity" in summary["checks"]
        assert summary["checks"]["no_injection"]["total"] == 2

    def test_reset(self):
        metrics = GuardrailMetrics()
        metrics.record("no_injection", passed=True, latency_ms=0.5)
        metrics.reset()

        summary = metrics.summary()
        assert summary["total_checks"] == 0

    def test_block_rate(self):
        metrics = GuardrailMetrics()
        for _ in range(7):
            metrics.record("safe", passed=True, latency_ms=0.5)
        for _ in range(3):
            metrics.record("safe", passed=False, latency_ms=0.5)

        stats = metrics.get_stats("safe")
        assert stats.block_rate == pytest.approx(0.3)

    def test_latency_min_max(self):
        metrics = GuardrailMetrics()
        metrics.record("check", passed=True, latency_ms=1.0)
        metrics.record("check", passed=True, latency_ms=5.0)
        metrics.record("check", passed=True, latency_ms=2.0)

        stats = metrics.get_stats("check")
        assert stats.min_latency_ms == 1.0
        assert stats.max_latency_ms == 5.0


class TestStructuredLogging:

    def test_get_logger(self):
        from dspy_guardrails.telemetry.logging import get_logger

        logger = get_logger("test")
        assert logger is not None

    def test_log_check(self, capsys):
        import logging

        from dspy_guardrails.telemetry.logging import get_logger, log_check

        logger = get_logger("test_log")
        # Just verify it doesn't raise
        log_check(logger, "no_injection", "test text", True, 0.5)


class TestTracing:

    def test_traced_decorator(self):
        from dspy_guardrails.telemetry.tracing import traced

        @traced("test.check")
        def my_check(text: str) -> bool:
            return "bad" not in text

        assert my_check("hello") is True
        assert my_check("bad stuff") is False

    def test_instrument_guardrail(self):
        from dspy_guardrails.telemetry.tracing import instrument_guardrail

        @instrument_guardrail
        class MyGuard:
            def no_injection(self, text: str) -> bool:
                return True

            def check(self, text: str) -> bool:
                return True

        guard = MyGuard()
        assert guard.no_injection("test") is True
        assert guard.check("test") is True
