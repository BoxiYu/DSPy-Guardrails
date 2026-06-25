"""LLMAttacker - Attack plugin using LLM-based attack generation."""

import time
from typing import Any

from ..plugins import BasePlugin, PluginConfig, PluginResult, PluginType


class LLMAttacker(BasePlugin):
    """
    LLM-based dynamic attacker plugin.

    Uses DSPy-based attackers (PromptInjectionAttacker, JailbreakAttacker) to
    generate dynamic attacks based on target behavior and defense description.

    Unlike StaticAttacker which uses pre-defined payloads, LLMAttacker generates
    new attacks dynamically using LLM reasoning, making it more effective against
    adaptive defenses.

    Options:
        attack_types: Attack types to use (default: ["injection", "jailbreak"])
        num_attacks: Number of attacks per type (default: 10)
        target_behavior: Target behavior description (e.g., "reveal system prompt")
        defense_description: Defense description (e.g., "regex-based filtering")
        fallback_to_static: Fall back to static payloads when LLM not available (default: True)

    Example:
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "target_behavior": "extract sensitive data",
            "num_attacks": 5,
        }))
        result = attacker.execute({"target": my_target})
        print(f"Success rate: {result.metrics['success_rate']:.2%}")
    """

    name = "llm_attacker"
    version = "1.0.0"
    plugin_type = PluginType.ATTACKER

    DEFAULT_ATTACK_TYPES = ["injection", "jailbreak"]
    DEFAULT_NUM_ATTACKS = 10

    def __init__(self) -> None:
        """Initialize LLMAttacker with default configuration."""
        self._attack_types: list[str] = self.DEFAULT_ATTACK_TYPES.copy()
        self._num_attacks: int = self.DEFAULT_NUM_ATTACKS
        self._target_behavior: str = "bypass security controls"
        self._defense_description: str = "unknown defense"
        self._fallback_to_static: bool = True
        self._config: PluginConfig | None = None

    def configure(self, config: PluginConfig) -> None:
        """
        Configure attacker options.

        Args:
            config: Plugin configuration with options:
                - attack_types: List of attack types to use
                - num_attacks: Number of attacks per type
                - target_behavior: Description of target behavior to test
                - defense_description: Description of known defenses
                - fallback_to_static: Whether to use static payloads as fallback
        """
        self._config = config
        opts = config.options
        self._attack_types = opts.get("attack_types", self.DEFAULT_ATTACK_TYPES.copy())
        self._num_attacks = opts.get("num_attacks", self.DEFAULT_NUM_ATTACKS)
        self._target_behavior = opts.get("target_behavior", "bypass security controls")
        self._defense_description = opts.get("defense_description", "unknown defense")
        self._fallback_to_static = opts.get("fallback_to_static", True)

    def execute(self, context: dict[str, Any]) -> PluginResult:
        """
        Execute LLM-based attacks on target.

        Args:
            context: Execution context containing:
                - target: The target to attack (required)

        Returns:
            PluginResult with:
                - data.attack_results: List of attack results
                - data.successful_attacks: Number of successful attacks
                - data.total_attacks: Total attacks executed
                - data.llm_used: Whether LLM was used
                - metrics.success_rate: Attack success rate
                - metrics.total_attacks: Total attacks count
                - metrics.successful_attacks: Successful attacks count
        """
        target = context.get("target")
        if not target:
            return PluginResult(
                success=False,
                errors=["No target provided in context"],
            )

        attack_results: list[dict[str, Any]] = []
        warnings: list[str] = []

        # Check if LLM is available
        llm_available = self._check_llm_available()

        if llm_available:
            attack_results = self._execute_llm_attacks(target)
        elif self._fallback_to_static:
            warnings.append("LLM not configured, falling back to static payloads")
            attack_results = self._execute_static_fallback(target)
        else:
            return PluginResult(
                success=False,
                errors=["LLM not configured and fallback disabled"],
            )

        # Calculate metrics
        successful = sum(1 for a in attack_results if a.get("success", False))
        total = len(attack_results)
        success_rate = successful / total if total > 0 else 0.0

        return PluginResult(
            success=True,
            data={
                "attack_results": attack_results,
                "successful_attacks": successful,
                "total_attacks": total,
                "llm_used": llm_available,
            },
            warnings=warnings,
            metrics={
                "success_rate": success_rate,
                "total_attacks": float(total),
                "successful_attacks": float(successful),
            },
        )

    def cleanup(self) -> None:
        """Cleanup resources (no-op for LLMAttacker)."""
        pass

    def _check_llm_available(self) -> bool:
        """
        Check if DSPy LLM is configured.

        Returns:
            True if DSPy LM is configured, False otherwise.
        """
        try:
            import dspy
            return dspy.settings.lm is not None
        except Exception:
            return False

    def _execute_llm_attacks(self, target) -> list[dict[str, Any]]:
        """
        Execute attacks using LLM-based generators.

        Args:
            target: The target to attack.

        Returns:
            List of attack result dictionaries.
        """
        results: list[dict[str, Any]] = []

        try:
            from ...redteam import JailbreakAttacker, PromptInjectionAttacker

            # Initialize attackers based on configured types
            attackers: dict[str, Any] = {}
            if "injection" in self._attack_types:
                attackers["injection"] = PromptInjectionAttacker(use_llm=True)
            if "jailbreak" in self._attack_types:
                attackers["jailbreak"] = JailbreakAttacker(use_llm=True)

            # Execute attacks for each type
            for attack_type, attacker in attackers.items():
                for i in range(self._num_attacks):
                    try:
                        start_time = time.time()

                        # Generate attack based on type
                        if attack_type == "injection":
                            attack = attacker(
                                target_behavior=self._target_behavior,
                                defense_description=self._defense_description,
                            )
                        else:  # jailbreak
                            attack = attacker(
                                target_capability=self._target_behavior,
                                model_description=self._defense_description,
                            )

                        # Invoke target with generated attack
                        response = target.invoke(attack.prompt)
                        latency_ms = (time.time() - start_time) * 1000

                        results.append({
                            "payload_id": f"llm_{attack_type}_{i}",
                            "category": attack_type,
                            "technique": "llm_generated",
                            "success": not response.was_blocked,
                            "prompt": attack.prompt[:100] if attack.prompt else "",
                            "response": response.response[:200] if response.response else "",
                            "latency_ms": latency_ms,
                            "strategy": attack.strategy if hasattr(attack, 'strategy') else "unknown",
                        })
                    except Exception as e:
                        results.append({
                            "payload_id": f"llm_{attack_type}_{i}",
                            "category": attack_type,
                            "technique": "llm_generated",
                            "success": False,
                            "prompt": "",
                            "response": "",
                            "latency_ms": 0.0,
                            "error": str(e),
                        })
        except ImportError:
            # If redteam module not available, return empty results
            pass

        return results

    def _execute_static_fallback(self, target) -> list[dict[str, Any]]:
        """
        Fallback to static payloads when LLM is not available.

        Args:
            target: The target to attack.

        Returns:
            List of attack result dictionaries.
        """
        results: list[dict[str, Any]] = []

        try:
            from ...redteam.payloads import InjectionPayloads, JailbreakPayloads

            # Collect payloads based on configured attack types
            payloads = []
            if "injection" in self._attack_types:
                injection_payloads = InjectionPayloads.get_all()[:self._num_attacks]
                payloads.extend(injection_payloads)
            if "jailbreak" in self._attack_types:
                jailbreak_payloads = JailbreakPayloads.get_all()[:self._num_attacks]
                payloads.extend(jailbreak_payloads)

            # Execute each payload
            for payload in payloads:
                start_time = time.time()
                try:
                    response = target.invoke(payload.prompt)
                    latency_ms = (time.time() - start_time) * 1000

                    results.append({
                        "payload_id": payload.id,
                        "category": payload.category.value,
                        "technique": "static_fallback",
                        "success": not response.was_blocked,
                        "prompt": payload.prompt[:100] if payload.prompt else "",
                        "response": response.response[:200] if response.response else "",
                        "latency_ms": latency_ms,
                    })
                except Exception as e:
                    results.append({
                        "payload_id": payload.id,
                        "category": payload.category.value,
                        "technique": "static_fallback",
                        "success": False,
                        "prompt": payload.prompt[:100] if payload.prompt else "",
                        "response": "",
                        "latency_ms": 0.0,
                        "error": str(e),
                    })
        except ImportError:
            # If payloads module not available, return empty results
            pass

        return results
