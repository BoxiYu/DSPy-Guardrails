from dspy_guardrails.autoresearch.registry import AttackAlgorithm


class PAIRBaseline(AttackAlgorithm):
    algorithm_name = "pair_v0"
    version = 0
    description = "PAIR baseline — iterative prompt refinement via attacker LLM"
    parent_version = None

    def create_attack(self, target, attacker_lm=None, judge_fn=None, **kwargs):
        from dspy_guardrails.adversarial.attacks.pair import PAIRAttack
        max_iterations = kwargs.pop("max_iterations", 20)
        return PAIRAttack(
            target=target,
            max_iterations=max_iterations,
            attacker_lm=attacker_lm,
            judge_fn=judge_fn,
            **kwargs,
        )
