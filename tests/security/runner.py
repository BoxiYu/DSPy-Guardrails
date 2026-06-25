"""
Security Test Runner

Main orchestrator for running security tests against targets.
"""

import os
import time
import yaml
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .targets import OpenAICSAgentTarget, AgentCopilotTarget
from .evaluators import (
    RedTeamEvaluator,
    BlueTeamEvaluator,
    HallucinationEvaluator,
    EfficiencyEvaluator,
)
from .evaluators.redteam import RedTeamReport
from .evaluators.blueteam import BlueTeamReport
from .evaluators.hallucination import HallucinationReport
from .evaluators.efficiency import EfficiencyReport
from .reports.generator import ReportGenerator


@dataclass
class SecurityTestConfig:
    """Configuration for security tests."""

    # Target configuration
    target_type: str = "openai-cs-agent"
    target_base_url: str = "http://localhost:8000"
    target_mode: str = "api"

    # Test configuration
    run_redteam: bool = True
    run_blueteam: bool = True
    run_hallucination: bool = True
    run_efficiency: bool = False

    # Efficiency settings
    efficiency_load_test: bool = False
    efficiency_load_users: int = 5
    efficiency_load_duration: int = 10

    # Red team settings
    redteam_benchmarks: List[str] = field(
        default_factory=lambda: ["jailbreakbench", "advbench"]
    )
    redteam_airline_specific: bool = True

    # Output settings
    output_dir: str = "./reports/output"
    report_formats: List[str] = field(
        default_factory=lambda: ["console", "json", "html"]
    )

    # Execution settings
    verbose: bool = False
    num_threads: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "SecurityTestConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            target_type=data.get("target", {}).get("type", "openai-cs-agent"),
            target_base_url=data.get("target", {}).get(
                "base_url", "http://localhost:8000"
            ),
            target_mode=data.get("target", {}).get("mode", "api"),
            run_redteam=data.get("tests", {}).get("redteam", {}).get("enabled", True),
            run_blueteam=data.get("tests", {}).get("blueteam", {}).get("enabled", True),
            run_hallucination=data.get("tests", {}).get("hallucination", {}).get(
                "enabled", True
            ),
            redteam_benchmarks=data.get("tests", {})
            .get("redteam", {})
            .get("benchmarks", ["jailbreakbench", "advbench"]),
            redteam_airline_specific=data.get("tests", {})
            .get("redteam", {})
            .get("airline_specific", True),
            output_dir=data.get("output", {}).get("dir", "./reports/output"),
            report_formats=data.get("output", {}).get(
                "formats", ["console", "json", "html"]
            ),
            verbose=data.get("verbose", False),
            num_threads=data.get("num_threads", 1),
        )


@dataclass
class SecurityTestResults:
    """Combined results from all security tests."""

    redteam: Optional[RedTeamReport] = None
    blueteam: Optional[BlueTeamReport] = None
    hallucination: Optional[HallucinationReport] = None
    efficiency: Optional[EfficiencyReport] = None
    execution_time_seconds: float = 0.0
    timestamp: str = ""
    config: Optional[SecurityTestConfig] = None

    @property
    def overall_score(self) -> float:
        """Calculate overall security score (0-100)."""
        scores = []
        weights = []

        if self.redteam:
            # Higher block rate = better
            scores.append(self.redteam.block_rate * 100)
            weights.append(0.4)  # 40% weight

        if self.blueteam:
            # Higher accuracy = better
            scores.append(self.blueteam.accuracy * 100)
            weights.append(0.35)  # 35% weight

        if self.hallucination:
            # Higher factual accuracy = better
            scores.append(self.hallucination.avg_factual_accuracy * 100)
            weights.append(0.25)  # 25% weight

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    @property
    def critical_vulnerabilities(self) -> List[str]:
        """Get list of critical vulnerabilities found."""
        vulns = []

        if self.redteam:
            for result in self.redteam.bypassed_attacks[:5]:
                vulns.append(
                    f"[RedTeam] {result.payload.category.value}: "
                    f"{result.payload.prompt[:50]}..."
                )

        if self.blueteam:
            for result in self.blueteam.failed_tests[:5]:
                if result.is_false_negative:
                    vulns.append(
                        f"[BlueTeam FN] {result.test_case.id}: "
                        f"Malicious input not blocked"
                    )

        return vulns

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "SECURITY TEST SUMMARY",
            "=" * 60,
            f"Overall Security Score: {self.overall_score:.1f}/100",
            f"Execution Time: {self.execution_time_seconds:.1f}s",
            "",
        ]

        if self.redteam:
            lines.append(f"Red Team: {self.redteam.block_rate:.1%} attacks blocked")

        if self.blueteam:
            lines.append(f"Blue Team: {self.blueteam.accuracy:.1%} accuracy")
            lines.append(f"  Precision: {self.blueteam.precision:.1%}")
            lines.append(f"  Recall: {self.blueteam.recall:.1%}")

        if self.hallucination:
            lines.append(
                f"Hallucination: {self.hallucination.avg_factual_accuracy:.1%} "
                f"factual accuracy"
            )

        if self.efficiency:
            lines.append(f"Efficiency: {self.efficiency.avg_quality_score:.1%} quality")
            lines.append(f"  Latency (P95): {self.efficiency.latency.p95_ms:.1f}ms")
            if self.efficiency.cost:
                lines.append(f"  Cost/Request: ${self.efficiency.cost.cost_per_request_usd:.6f}")

        vulns = self.critical_vulnerabilities
        if vulns:
            lines.append("")
            lines.append(f"Critical Vulnerabilities ({len(vulns)}):")
            for v in vulns[:5]:
                lines.append(f"  - {v}")

        return "\n".join(lines)


class SecurityTestRunner:
    """
    Main runner for security tests.

    Orchestrates red team, blue team, and hallucination tests against
    a target and generates comprehensive reports.

    Example:
        runner = SecurityTestRunner.from_yaml("config.yaml")
        results = runner.run_all()
        print(results.summary())
        runner.generate_report(results)
    """

    def __init__(self, config: SecurityTestConfig):
        """Initialize runner with config."""
        self.config = config
        self.target = self._create_target()
        self.report_generator = ReportGenerator(output_dir=config.output_dir)

    def _create_target(self):
        """Create target based on config."""
        if self.config.target_type == "openai-cs-agent":
            return OpenAICSAgentTarget(
                base_url=self.config.target_base_url,
                mode=self.config.target_mode,
            )
        elif self.config.target_type == "agent-copilot":
            return AgentCopilotTarget(
                base_url=self.config.target_base_url,
            )
        else:
            raise ValueError(f"Unknown target type: {self.config.target_type}")

    @classmethod
    def from_yaml(cls, path: str) -> "SecurityTestRunner":
        """Create runner from YAML config file."""
        config = SecurityTestConfig.from_yaml(path)
        return cls(config)

    @classmethod
    def with_defaults(
        cls,
        base_url: str = "http://localhost:8000",
        verbose: bool = False,
    ) -> "SecurityTestRunner":
        """Create runner with default config."""
        config = SecurityTestConfig(
            target_base_url=base_url,
            verbose=verbose,
        )
        return cls(config)

    def run_redteam(self) -> RedTeamReport:
        """Run red team attack tests."""
        if self.config.verbose:
            print("\n🔴 Running Red Team Tests...")

        evaluator = RedTeamEvaluator(
            target=self.target,
            verbose=self.config.verbose,
        )

        # Collect all payloads
        all_payloads = []

        # Add benchmark attacks
        for benchmark in self.config.redteam_benchmarks:
            try:
                report = evaluator.run_benchmark(benchmark)
                # We need the payloads, not the report
                # Re-run to get actual payloads
            except ImportError:
                if self.config.verbose:
                    print(f"  Benchmark {benchmark} not available, skipping...")

        # Add airline-specific attacks
        if self.config.redteam_airline_specific:
            pass  # Will be included in run_all

        return evaluator.run_all()

    def run_blueteam(self) -> BlueTeamReport:
        """Run blue team defense tests."""
        if self.config.verbose:
            print("\n🔵 Running Blue Team Tests...")

        evaluator = BlueTeamEvaluator(
            target=self.target,
            verbose=self.config.verbose,
        )

        return evaluator.run_all()

    def run_hallucination(self) -> HallucinationReport:
        """Run hallucination detection tests."""
        if self.config.verbose:
            print("\n🟡 Running Hallucination Tests...")

        evaluator = HallucinationEvaluator(
            target=self.target,
            verbose=self.config.verbose,
        )

        return evaluator.run_all()

    def run_efficiency(self) -> EfficiencyReport:
        """Run efficiency and performance tests."""
        if self.config.verbose:
            print("\n🟢 Running Efficiency Tests...")

        evaluator = EfficiencyEvaluator(
            target=self.target,
            verbose=self.config.verbose,
        )

        return evaluator.run_all(
            run_load_test=self.config.efficiency_load_test,
            load_test_users=self.config.efficiency_load_users,
            load_test_duration=self.config.efficiency_load_duration,
        )

    def run_all(self) -> SecurityTestResults:
        """
        Run all configured security tests.

        Returns:
            SecurityTestResults with all test results
        """
        import datetime

        start_time = time.time()
        results = SecurityTestResults(
            timestamp=datetime.datetime.now().isoformat(),
            config=self.config,
        )

        if self.config.run_redteam:
            try:
                results.redteam = self.run_redteam()
            except Exception as e:
                if self.config.verbose:
                    print(f"  Red team tests failed: {e}")

        if self.config.run_blueteam:
            try:
                results.blueteam = self.run_blueteam()
            except Exception as e:
                if self.config.verbose:
                    print(f"  Blue team tests failed: {e}")

        if self.config.run_hallucination:
            try:
                results.hallucination = self.run_hallucination()
            except Exception as e:
                if self.config.verbose:
                    print(f"  Hallucination tests failed: {e}")

        if self.config.run_efficiency:
            try:
                results.efficiency = self.run_efficiency()
            except Exception as e:
                if self.config.verbose:
                    print(f"  Efficiency tests failed: {e}")

        results.execution_time_seconds = time.time() - start_time

        return results

    def generate_report(
        self,
        results: SecurityTestResults,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Generate reports in specified formats.

        Args:
            results: Test results to report
            formats: List of formats ("console", "json", "html")

        Returns:
            Dict mapping format to output path/content
        """
        formats = formats or self.config.report_formats
        return self.report_generator.generate(results, formats)
