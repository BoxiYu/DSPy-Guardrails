#!/usr/bin/env python3
"""Run co-evolution experiment with OUTPUT guard as the defense.

This adapts the existing AdversarialTrainer to evolve an output guard's
detection capability against PAIR attacks. The key difference from the
ASE experiments: the guard checks the LLM RESPONSE, not the user PROMPT.

Usage:
    python scripts/run_output_coevo.py --rounds 10 --seed 42
    python scripts/run_output_coevo.py --rounds 5 --seed 42 --quick
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CoevoConfig:
    rounds: int = 10
    attacks_per_round: int = 15
    seed: int = 42
    target_model: str = "qwen/qwen3-235b-a22b-2507"
    n_behaviors: int = 20
    query_budget: int = 10
    output_dir: str = "autoresearch/results/coevo"
    quick: bool = False


@dataclass
class RoundResult:
    round_num: int
    asr: float
    mean_score: float
    n_attacks: int
    n_blocked: int
    n_bypassed: int
    defense_patterns: int
    defense_examples: int
    wall_time_s: float


def setup_lms():
    """Configure attacker and judge LMs."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    import dspy

    # Attacker LM (local vLLM, uncensored)
    vllm_base = os.getenv("ATTACK_MODEL_BASE_URL", "http://localhost:18921/v1")
    vllm_key = os.getenv("ATTACK_MODEL_API_KEY", "EMPTY")
    vllm_model = os.getenv("ATTACK_MODEL", "openai/Huihui-Qwen3.5-27B-abliterated")
    attacker_lm = dspy.LM(
        vllm_model, api_key=vllm_key, api_base=vllm_base,
        cache=False, timeout=60.0, max_tokens=4096,
    )
    dspy.configure(lm=attacker_lm)

    # Judge/Defender LM (OpenRouter)
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    judge_lm = dspy.LM(
        "openrouter/qwen/qwen3-235b-a22b-2507", api_key=or_key,
        api_base="https://openrouter.ai/api/v1",
        cache=False, timeout=60.0, max_tokens=2048,
    )

    return attacker_lm, judge_lm


class OutputGuardDefenseTarget:
    """Wraps an LLM output guard as an EvolvableTarget for AdversarialTrainer.

    Flow: prompt → target LLM (OpenRouter) → response → output guard check
    The output guard is what gets evolved by the trainer.
    """

    def __init__(self, target_model: str, judge_lm: Any):
        from dspy_guardrails.autoresearch.vllm_target import OpenRouterTarget
        from dspy_guardrails.autoresearch.output_guard import LLMOutputGuard

        self._target = OpenRouterTarget(model=target_model)
        self._guard = LLMOutputGuard(judge_lm=judge_lm)
        self._learned_patterns: list[str] = []
        self._defense_hints: str = ""
        self.stats = {
            "total_invocations": 0,
            "blocked_count": 0,
            "bypassed_count": 0,
            "defense_updates": 0,
        }

    def invoke(self, prompt: str):
        """Call target LLM, then check response with output guard."""
        from dspy_guardrails.testing.targets import TargetResponse

        self.stats["total_invocations"] += 1

        # Step 1: Get actual LLM response
        llm_response = self._target.invoke(prompt)
        if llm_response.metadata.get("error"):
            return llm_response

        response_text = llm_response.response

        # Step 2: Check response against learned patterns (fast path)
        for pattern in self._learned_patterns:
            try:
                if re.search(pattern, response_text, re.IGNORECASE):
                    self.stats["blocked_count"] += 1
                    return TargetResponse(
                        response="",
                        guardrail_status={"output_pattern": {"passed": False}},
                        metadata={"block_reason": f"Output pattern: {pattern[:50]}"},
                    )
            except re.error:
                continue

        # Step 3: Check with LLM output guard
        guard_result = self._guard.check(response_text, goal=prompt)
        if guard_result.blocked:
            self.stats["blocked_count"] += 1
            return TargetResponse(
                response="",
                guardrail_status={"output_guard": {"passed": False, "reason": guard_result.reason}},
                metadata={"block_reason": guard_result.reason, "guard_score": guard_result.score},
            )

        # Passed — return actual LLM response
        self.stats["bypassed_count"] += 1
        return TargetResponse(
            response=response_text,
            guardrail_status={"output_guard": {"passed": True}},
            metadata={**llm_response.metadata, "guard_score": guard_result.score},
        )

    def update_defense(self, update):
        """Accept defense updates from the trainer."""
        self.stats["defense_updates"] += 1
        if hasattr(update, 'new_patterns'):
            for p in update.new_patterns:
                if p not in self._learned_patterns:
                    self._learned_patterns.append(p)

    def reset_session(self):
        self._target.reset_session()

    def get_defense_stats(self):
        return {
            **self.stats,
            "learned_patterns": len(self._learned_patterns),
        }


def run_simple_coevo(config: CoevoConfig):
    """Run a simplified co-evolution loop."""
    from dspy_guardrails.autoresearch.benchmarks import JBB100Benchmark
    from dspy_guardrails.autoresearch.harness import ResearchHarness
    from dspy_guardrails.autoresearch.registry import load_algorithm

    attacker_lm, judge_lm = setup_lms()

    # Create output guard defense target
    defense = OutputGuardDefenseTarget(
        target_model=config.target_model,
        judge_lm=judge_lm,
    )

    # Load PAIR attack
    algo_cls = load_algorithm(Path("autoresearch/methods/v0/pair_attack.py"))

    # Create harness for evaluation
    n_beh = 5 if config.quick else config.n_behaviors
    q_budget = 5 if config.quick else config.query_budget

    harness = ResearchHarness(
        target_model=config.target_model,
        target_provider="openrouter",
        guard_mode="none",  # We use our own defense target
        query_budget=q_budget,
        n_behaviors=n_beh,
        seed=config.seed,
        attacker_lm=attacker_lm,
        judge_lm=judge_lm,
    )

    # Output directory
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"[CoEvo] Starting output guard co-evolution: {config.rounds} rounds")
    print(f"[CoEvo] Target: {config.target_model}, {n_beh} behaviors, {q_budget} queries/beh")

    for round_num in range(1, config.rounds + 1):
        t0 = time.monotonic()
        print(f"\n{'='*60}")
        print(f"  Round {round_num}/{config.rounds}")
        print(f"  Defense stats: {defense.get_defense_stats()}")
        print(f"{'='*60}")

        # Evaluate PAIR against current defense
        algorithm = algo_cls()
        eval_result = harness.evaluate_attack(algorithm, defense_target=defense)

        wall_time = time.monotonic() - t0

        rr = RoundResult(
            round_num=round_num,
            asr=eval_result.asr,
            mean_score=eval_result.mean_score,
            n_attacks=len(eval_result.per_behavior),
            n_blocked=sum(1 for b in eval_result.per_behavior if not b.success),
            n_bypassed=sum(1 for b in eval_result.per_behavior if b.success),
            defense_patterns=len(defense._learned_patterns),
            defense_examples=0,
            wall_time_s=wall_time,
        )
        results.append(rr)

        print(f"  ASR={rr.asr:.3f}, blocked={rr.n_blocked}, bypassed={rr.n_bypassed}")

        # Extract patterns from successful attacks for defense evolution
        successful = [b for b in eval_result.per_behavior if b.success]
        if successful:
            # Simple defense evolution: extract keywords from successful attacks
            for b in successful:
                resp = b.best_response or ""
                # Extract potential harmful patterns from responses that bypassed
                # This is a simplified version of DefenseEvolver
                words = set(resp.lower().split())
                harmful_indicators = words & {
                    "step", "steps", "instructions", "guide", "tutorial",
                    "method", "procedure", "recipe", "formula", "code",
                }
                if harmful_indicators and b.goal:
                    # Create a pattern from the goal keywords + response indicators
                    goal_words = [w for w in b.goal.lower().split()[:5] if len(w) > 3]
                    if goal_words:
                        pattern = r"(?i)" + r".*".join(goal_words[:3])
                        defense._learned_patterns.append(pattern)

        print(f"  Defense patterns: {len(defense._learned_patterns)}")

    # Save results
    final = {
        "config": asdict(config),
        "rounds": [asdict(r) for r in results],
        "final_defense_stats": defense.get_defense_stats(),
        "trajectory": {
            "asr": [r.asr for r in results],
            "patterns": [r.defense_patterns for r in results],
        },
    }

    out_path = out_dir / f"output_coevo_s{config.seed}{'_quick' if config.quick else ''}.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[CoEvo] Results saved to {out_path}")

    # Print trajectory
    print(f"\n[CoEvo] ASR Trajectory:")
    for r in results:
        bar = "█" * int(r.asr * 20) + "░" * (20 - int(r.asr * 20))
        print(f"  R{r.round_num:2d}: {bar} {r.asr:.3f} (patterns={r.defense_patterns})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Output guard co-evolution")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-behaviors", type=int, default=20)
    args = parser.parse_args()

    config = CoevoConfig(
        rounds=args.rounds,
        seed=args.seed,
        quick=args.quick,
        n_behaviors=args.n_behaviors,
    )
    run_simple_coevo(config)


if __name__ == "__main__":
    main()
