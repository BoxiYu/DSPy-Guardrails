"""StaticAttacker - Attacker plugin using pre-defined static payloads."""

import time
from typing import Any

from ...redteam.payloads import (
    BypassPayloads,
    InjectionPayloads,
    JailbreakPayloads,
    MCPPayloads,
)
from ...redteam.payloads.base import AttackPayload
from ..plugins import BasePlugin, PluginConfig, PluginResult, PluginType


class StaticAttacker(BasePlugin):
    """
    Static attacker plugin using pre-defined attack payloads.

    Executes pre-defined static attack payloads against a target and reports
    detailed results including success/failure status, latency, and attack metadata.

    Options:
        attack_budget: Maximum number of attacks to execute (default: 100)
        categories: Attack categories to include (default: all)
        severity_filter: Minimum severity level (default: "low")
        stop_on_success: Stop after first successful attack (default: False)

    Example:
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={"attack_budget": 50}))
        result = attacker.execute({"target": my_target})
        print(f"Attack success rate: {result.metrics['attack_success_rate']:.2%}")
    """

    name = "static_attacker"
    version = "1.0.0"
    plugin_type = PluginType.ATTACKER

    # Payload category mapping
    CATEGORY_LOADERS = {
        "injection": InjectionPayloads.get_all,
        "jailbreak": JailbreakPayloads.get_all,
        "bypass": BypassPayloads.get_all,
        "mcp": MCPPayloads.get_all,
    }

    # Default configuration values
    DEFAULT_ATTACK_BUDGET = 100
    DEFAULT_CATEGORIES = list(CATEGORY_LOADERS.keys())
    DEFAULT_SEVERITY = "low"

    def __init__(self):
        """Initialize StaticAttacker with default configuration."""
        self._attack_budget = self.DEFAULT_ATTACK_BUDGET
        self._categories = self.DEFAULT_CATEGORIES.copy()
        self._severity_filter = self.DEFAULT_SEVERITY
        self._stop_on_success = False
        self._config: PluginConfig | None = None

    def configure(self, config: PluginConfig) -> None:
        """
        Configure attacker options.

        Args:
            config: Plugin configuration with options:
                - attack_budget: Maximum number of attacks to execute
                - categories: List of attack categories to include
                - severity_filter: Minimum severity level ("low", "medium", "high", "critical")
                - stop_on_success: Stop after first successful attack
        """
        self._config = config
        opts = config.options
        self._attack_budget = opts.get("attack_budget", self.DEFAULT_ATTACK_BUDGET)
        self._categories = opts.get("categories", self.DEFAULT_CATEGORIES.copy())
        self._severity_filter = opts.get("severity_filter", self.DEFAULT_SEVERITY)
        self._stop_on_success = opts.get("stop_on_success", False)

    def execute(self, context: dict[str, Any]) -> PluginResult:
        """
        Execute static attacks on target.

        Args:
            context: Execution context containing:
                - target: The target to attack (required)

        Returns:
            PluginResult with:
                - data.attacks: List of attack results with details
                - data.successful_attacks: Number of attacks that bypassed guardrail
                - data.failed_attacks: Number of attacks that were blocked
                - data.total_attacks: Total number of attacks executed
                - metrics.attack_success_rate: Ratio of successful attacks
                - metrics.total_attacks: Total attacks executed
                - metrics.successful_attacks: Count of successful attacks
                - metrics.avg_latency_ms: Average attack latency
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
        attacks: list[dict[str, Any]] = []
        successful_attacks = 0
        failed_attacks = 0

        for payload in payloads[:self._attack_budget]:
            start_time = time.time()
            try:
                response = target.invoke(payload.prompt)
                latency_ms = (time.time() - start_time) * 1000

                # Attack succeeded if it bypassed the guardrail (was not blocked)
                attack_succeeded = not response.was_blocked
                if attack_succeeded:
                    successful_attacks += 1
                else:
                    failed_attacks += 1

                attacks.append({
                    "payload_id": payload.id,
                    "category": payload.category.value,
                    "severity": payload.severity.value,
                    "technique": payload.technique,
                    "prompt": payload.prompt[:100],
                    "response": response.response[:200] if response.response else "",
                    "success": attack_succeeded,
                    "was_blocked": response.was_blocked,
                    "latency_ms": latency_ms,
                })

                if self._stop_on_success and attack_succeeded:
                    break

            except Exception as e:
                failed_attacks += 1
                attacks.append({
                    "payload_id": payload.id,
                    "category": payload.category.value,
                    "severity": payload.severity.value,
                    "technique": payload.technique,
                    "prompt": payload.prompt[:100],
                    "response": "",
                    "success": False,
                    "was_blocked": True,
                    "latency_ms": 0.0,
                    "error": str(e),
                })

        total_attacks = len(attacks)
        attack_success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        total_latency = sum(a.get("latency_ms", 0) for a in attacks)
        avg_latency_ms = total_latency / total_attacks if total_attacks > 0 else 0.0

        return PluginResult(
            success=True,
            data={
                "attacks": attacks,
                "successful_attacks": successful_attacks,
                "failed_attacks": failed_attacks,
                "total_attacks": total_attacks,
            },
            metrics={
                "attack_success_rate": attack_success_rate,
                "total_attacks": float(total_attacks),
                "successful_attacks": float(successful_attacks),
                "avg_latency_ms": avg_latency_ms,
            },
        )

    def cleanup(self) -> None:
        """Cleanup resources (no-op for StaticAttacker)."""
        pass

    def _load_payloads(self) -> list[AttackPayload]:
        """
        Load payloads based on configuration.

        Returns:
            List of AttackPayload objects filtered by category and severity.
        """
        payloads: list[AttackPayload] = []

        # Define severity ordering for filtering
        severity_order = ["low", "medium", "high", "critical"]
        try:
            min_severity_idx = severity_order.index(self._severity_filter)
        except ValueError:
            min_severity_idx = 0  # Default to low if invalid
        allowed_severities = set(severity_order[min_severity_idx:])

        # Load payloads from each configured category
        for category in self._categories:
            loader = self.CATEGORY_LOADERS.get(category)
            if loader:
                for p in loader():
                    if p.severity.value in allowed_severities:
                        payloads.append(p)

        return payloads
