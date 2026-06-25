from dspy_guardrails.autoresearch.registry import AttackAlgorithm


class TAPBaseline(AttackAlgorithm):
    algorithm_name = "tap_v0"
    version = 0
    description = "TAP baseline — tree-of-attacks with pruning"
    parent_version = None

    def create_attack(self, target, attacker_lm=None, judge_fn=None, **kwargs):
        from dspy_guardrails.adversarial.attacks.tap import TAPAttack
        return TAPAttack(
            target=target,
            attacker_lm=attacker_lm,
            judge_fn=judge_fn,
            **kwargs,
        )
