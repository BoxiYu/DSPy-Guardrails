"""D5: Performance benchmark tests for dspyGuardrails (mock mode, no API keys)."""

import os
import sys
import time
import tracemalloc

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dspy_guardrails.guardrail import guardrail


class TestDetectionLatency:
    """Throughput benchmarks for pattern-based detection functions."""

    @pytest.mark.benchmark
    def test_injection_detection_throughput(self):
        start = time.time()
        for _ in range(1000):
            guardrail.no_injection("What is the weather?")
        elapsed = time.time() - start
        assert elapsed < 5, f"1000 injection checks took {elapsed:.2f}s (limit 5s)"

    @pytest.mark.benchmark
    def test_pii_detection_throughput(self):
        start = time.time()
        for _ in range(1000):
            guardrail.no_pii("Hello world")
        elapsed = time.time() - start
        assert elapsed < 5, f"1000 PII checks took {elapsed:.2f}s (limit 5s)"

    @pytest.mark.benchmark
    def test_mcp_detection_throughput(self):
        start = time.time()
        for _ in range(1000):
            guardrail.no_mcp_attack("Hello world")
        elapsed = time.time() - start
        assert elapsed < 5, f"1000 MCP checks took {elapsed:.2f}s (limit 5s)"

    @pytest.mark.benchmark
    def test_combined_safe_throughput(self):
        start = time.time()
        for _ in range(1000):
            guardrail.safe("Hello world")
        elapsed = time.time() - start
        assert elapsed < 10, f"1000 safe() checks took {elapsed:.2f}s (limit 10s)"


class TestMemoryUsage:
    """Memory consumption tests."""

    def test_memory_baseline(self):
        tracemalloc.start()
        snapshot_before = tracemalloc.get_traced_memory()

        for i in range(10000):
            guardrail.no_injection(f"Short test string number {i}")

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB exceeds 100MB limit"


class TestScalingBehavior:
    """Verify detection scales reasonably with input length."""

    def test_score_scales_with_input_length(self):
        for length in (100, 1000, 10000):
            text = "a" * length
            start = time.time()
            score = guardrail.injection_score(text)
            elapsed = time.time() - start

            assert elapsed < 1, (
                f"injection_score on {length}-char input took {elapsed:.2f}s (limit 1s)"
            )
            assert isinstance(score, float), f"Expected float, got {type(score)}"
            assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"
