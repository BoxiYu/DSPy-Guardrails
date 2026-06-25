#!/usr/bin/env python3
"""
Test script for the security testing framework.

Uses MockTarget to verify the framework works correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.security.targets.mock_target import MockOpenAICSAgentTarget
from tests.security.evaluators.redteam import RedTeamEvaluator
from tests.security.evaluators.blueteam import BlueTeamEvaluator
from tests.security.evaluators.hallucination import HallucinationEvaluator
from tests.security.reports.generator import ReportGenerator


def test_redteam():
    """Test RedTeam evaluator."""
    print("\n" + "=" * 60)
    print("Testing RedTeam Evaluator")
    print("=" * 60)

    # Use mock target with known accuracy
    target = MockOpenAICSAgentTarget(
        relevance_accuracy=0.9,
        jailbreak_accuracy=0.85,
    )

    evaluator = RedTeamEvaluator(target=target, verbose=True)

    # Run airline-specific attacks only (no external benchmarks)
    report = evaluator.run_attacks(evaluator._get_builtin_airline_attacks())

    print("\n" + report.summary())
    return report


def test_blueteam():
    """Test BlueTeam evaluator."""
    print("\n" + "=" * 60)
    print("Testing BlueTeam Evaluator")
    print("=" * 60)

    target = MockOpenAICSAgentTarget(
        relevance_accuracy=0.9,
        jailbreak_accuracy=0.85,
    )

    evaluator = BlueTeamEvaluator(target=target, verbose=True)
    report = evaluator.run_all()

    print("\n" + report.summary())
    return report


def test_hallucination():
    """Test Hallucination evaluator."""
    print("\n" + "=" * 60)
    print("Testing Hallucination Evaluator")
    print("=" * 60)

    target = MockOpenAICSAgentTarget(
        hallucination_rate=0.1,
    )

    evaluator = HallucinationEvaluator(target=target, verbose=True)
    report = evaluator.run_all()

    print("\n" + report.summary())
    return report


def test_report_generator():
    """Test report generation."""
    print("\n" + "=" * 60)
    print("Testing Report Generator")
    print("=" * 60)

    # Import here to avoid circular imports
    from tests.security.runner import SecurityTestResults

    # Create mock results
    target = MockOpenAICSAgentTarget()

    rt_eval = RedTeamEvaluator(target=target)
    bt_eval = BlueTeamEvaluator(target=target)
    hal_eval = HallucinationEvaluator(target=target)

    results = SecurityTestResults(
        redteam=rt_eval.run_attacks(rt_eval._get_builtin_airline_attacks()),
        blueteam=bt_eval.run_all(),
        hallucination=hal_eval.run_all(),
        execution_time_seconds=5.0,
        timestamp="2025-12-31T12:00:00",
    )

    print(f"\nOverall Score: {results.overall_score:.1f}/100")
    print(f"Critical Vulnerabilities: {len(results.critical_vulnerabilities)}")

    # Generate reports
    generator = ReportGenerator(output_dir="./tests/security/reports/output")
    outputs = generator.generate(results, ["json", "html"])

    print("\nReports generated:")
    for fmt, path in outputs.items():
        print(f"  {fmt}: {path}")

    return results, outputs


def main():
    """Run all tests."""
    print("=" * 60)
    print("   Security Testing Framework - Validation")
    print("=" * 60)

    errors = []

    # Test RedTeam
    try:
        rt_report = test_redteam()
        print(f"\n[PASS] RedTeam: {rt_report.block_rate:.1%} block rate")
    except Exception as e:
        errors.append(f"RedTeam: {e}")
        print(f"\n[FAIL] RedTeam: {e}")

    # Test BlueTeam
    try:
        bt_report = test_blueteam()
        print(f"\n[PASS] BlueTeam: {bt_report.accuracy:.1%} accuracy")
    except Exception as e:
        errors.append(f"BlueTeam: {e}")
        print(f"\n[FAIL] BlueTeam: {e}")

    # Test Hallucination
    try:
        hal_report = test_hallucination()
        print(f"\n[PASS] Hallucination: {hal_report.avg_factual_accuracy:.1%} accuracy")
    except Exception as e:
        errors.append(f"Hallucination: {e}")
        print(f"\n[FAIL] Hallucination: {e}")

    # Test Report Generator
    try:
        results, outputs = test_report_generator()
        print(f"\n[PASS] Report Generator: {len(outputs)} formats generated")
    except Exception as e:
        errors.append(f"Report Generator: {e}")
        print(f"\n[FAIL] Report Generator: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if errors:
        print(f"\nFailed: {len(errors)} tests")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
