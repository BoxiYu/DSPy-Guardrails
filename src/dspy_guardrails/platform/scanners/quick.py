"""QuickScanner - Fast vulnerability scanner using static payloads."""

from typing import Any

from ...redteam.payloads import (
    BypassPayloads,
    InjectionPayloads,
    JailbreakPayloads,
)
from ...redteam.payloads.base import AttackPayload
from ..plugins import BasePlugin, PluginConfig, PluginResult, PluginType


class QuickScanner(BasePlugin):
    """
    Quick vulnerability scanner using static payloads.

    Uses static payloads to quickly detect common vulnerabilities.
    Suitable for fast security assessments and CI/CD integration.

    Options:
        max_payloads: Maximum number of payloads to test (default: 20)
        categories: Attack categories to include (default: ["injection", "jailbreak"])
        severity_filter: Minimum severity level (default: "medium")

    Example:
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 10}))
        result = scanner.execute({"target": my_target})
        print(f"Found {len(result.data['vulnerabilities'])} vulnerabilities")
    """

    name = "quick_scanner"
    version = "1.0.0"
    plugin_type = PluginType.SCANNER

    # Default configuration
    DEFAULT_MAX_PAYLOADS = 20
    DEFAULT_CATEGORIES = ["injection", "jailbreak"]
    DEFAULT_SEVERITY = "medium"

    def __init__(self):
        self._max_payloads = self.DEFAULT_MAX_PAYLOADS
        self._categories = self.DEFAULT_CATEGORIES.copy()
        self._severity_filter = self.DEFAULT_SEVERITY
        self._config: PluginConfig | None = None

    def configure(self, config: PluginConfig) -> None:
        """
        Configure scanner options.

        Args:
            config: Plugin configuration with options:
                - max_payloads: Maximum payloads to test
                - categories: List of attack categories
                - severity_filter: Minimum severity level
        """
        self._config = config
        opts = config.options
        self._max_payloads = opts.get("max_payloads", self.DEFAULT_MAX_PAYLOADS)
        self._categories = opts.get("categories", self.DEFAULT_CATEGORIES.copy())
        self._severity_filter = opts.get("severity_filter", self.DEFAULT_SEVERITY)

    def execute(self, context: dict[str, Any]) -> PluginResult:
        """
        Execute quick scan on target.

        Args:
            context: Execution context containing:
                - target: The target to scan (required)

        Returns:
            PluginResult with:
                - data.vulnerabilities: List of found vulnerabilities
                - data.total_payloads: Total payloads tested
                - data.successful_attacks: Number of successful attacks
                - metrics.attack_success_rate: Ratio of successful attacks
                - metrics.security_score: 1.0 - attack_success_rate
                - metrics.vulnerability_count: Number of vulnerabilities found
        """
        target = context.get("target")
        if not target:
            return PluginResult(
                success=False,
                errors=["No target provided in context"],
            )

        # Load payloads based on configuration
        payloads = self._load_payloads()

        # Execute attacks
        vulnerabilities = []
        total_attacks = 0
        successful_attacks = 0

        for payload in payloads[:self._max_payloads]:
            total_attacks += 1
            try:
                response = target.invoke(payload.prompt)
                if not response.was_blocked:
                    successful_attacks += 1
                    vulnerabilities.append({
                        "payload_id": payload.id,
                        "category": payload.category.value,
                        "severity": payload.severity.value,
                        "technique": payload.technique,
                        "prompt": payload.prompt[:100],
                        "response": response.response[:200] if response.response else "",
                    })
            except Exception:
                # Continue scanning even if one payload fails
                # Log error but don't stop the scan
                pass

        # Calculate metrics
        attack_success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        security_score = 1.0 - attack_success_rate

        return PluginResult(
            success=True,
            data={
                "vulnerabilities": vulnerabilities,
                "total_payloads": total_attacks,
                "successful_attacks": successful_attacks,
            },
            metrics={
                "attack_success_rate": attack_success_rate,
                "security_score": security_score,
                "vulnerability_count": len(vulnerabilities),
            },
        )

    def cleanup(self) -> None:
        """Cleanup resources (no-op for QuickScanner)."""
        pass

    def _load_payloads(self) -> list[AttackPayload]:
        """
        Load payloads based on configuration.

        Returns:
            List of AttackPayload objects filtered by category and severity.
        """
        payloads = []

        # Define severity ordering for filtering
        severity_order = ["low", "medium", "high", "critical"]
        try:
            min_severity_idx = severity_order.index(self._severity_filter)
        except ValueError:
            min_severity_idx = 1  # Default to medium if invalid
        allowed_severities = set(severity_order[min_severity_idx:])

        # Load payloads from each category
        if "injection" in self._categories:
            for p in InjectionPayloads.get_all():
                if p.severity.value in allowed_severities:
                    payloads.append(p)

        if "jailbreak" in self._categories:
            for p in JailbreakPayloads.get_all():
                if p.severity.value in allowed_severities:
                    payloads.append(p)

        if "bypass" in self._categories:
            for p in BypassPayloads.get_all():
                if p.severity.value in allowed_severities:
                    payloads.append(p)

        return payloads
