from dspy_guardrails.autoresearch.registry import AttackAlgorithm


class MAPElitesBaseline(AttackAlgorithm):
    algorithm_name = "mapelites_v0"
    version = 0
    description = "MAP-Elites baseline — quality-diversity population search"
    parent_version = None

    def create_attack(self, target, attacker_lm=None, judge_fn=None, **kwargs):
        from dspy_guardrails.adversarial.attacks.mapelites import MAPElitesAttack
        return MAPElitesAttack(
            target=target,
            attacker_lm=attacker_lm,
            judge_fn=judge_fn,
            **kwargs,
        )
