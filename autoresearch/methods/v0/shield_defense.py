from dspy_guardrails.autoresearch.registry import DefenseAlgorithm


class ShieldDefenseBaseline(DefenseAlgorithm):
    algorithm_name = "shield_v0"
    version = 0
    description = "Shield baseline — pattern-based detection with default config"
    parent_version = None

    def create_target(self, base_lm=None, **kwargs):
        from dspy_guardrails.adversarial.evolvable_target import EvolvableShieldTarget
        return EvolvableShieldTarget.from_config(
            checks=kwargs.get("checks", ["injection"]),
            mode=kwargs.get("mode", "fast"),
        )
