"""Quick script to evaluate a custom output guard against PAIR attack."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

def main():
    from dotenv import load_dotenv
    load_dotenv()

    import dspy
    from dspy_guardrails.autoresearch.vllm_target import OpenRouterTarget, OutputGuardTarget
    from dspy_guardrails.autoresearch.harness import ResearchHarness
    from dspy_guardrails.autoresearch.registry import load_algorithm

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", default="d2", help="d1 or d2")
    parser.add_argument("--n-behaviors", type=int, default=10)
    parser.add_argument("--query-budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Configure LMs
    vllm_base = os.getenv("ATTACK_MODEL_BASE_URL", "http://localhost:18921/v1")
    vllm_key = os.getenv("ATTACK_MODEL_API_KEY", "EMPTY")
    vllm_model = os.getenv("ATTACK_MODEL", "openai/Huihui-Qwen3.5-27B-abliterated")
    attacker_lm = dspy.LM(
        vllm_model, api_key=vllm_key, api_base=vllm_base,
        cache=False, timeout=60.0, max_tokens=4096,
    )
    dspy.configure(lm=attacker_lm)

    or_key = os.getenv("OPENROUTER_API_KEY", "")
    judge_lm = dspy.LM(
        "openrouter/qwen/qwen3-235b-a22b-2507", api_key=or_key,
        api_base="https://openrouter.ai/api/v1",
        cache=False, timeout=60.0, max_tokens=2048,
    )

    # Create defense
    if args.defense == "d2":
        from autoresearch.methods.d2.defense import EnhancedLLMOutputGuard
        output_guard = EnhancedLLMOutputGuard(guard_lm=judge_lm)
        defense_name = "d2_enhanced_output"
    else:
        from dspy_guardrails.autoresearch.output_guard import LLMOutputGuard
        output_guard = LLMOutputGuard(judge_lm=judge_lm)
        defense_name = "d1_basic_output"

    # Create guarded target
    target_model = os.getenv("AUTORESEARCH_TARGET_MODEL", "qwen/qwen3-235b-a22b-2507")
    base_target = OpenRouterTarget(model=target_model)
    guarded_target = OutputGuardTarget(inner_target=base_target, output_guard=output_guard)

    # Load PAIR attack
    algo_cls = load_algorithm(Path("autoresearch/methods/v0/pair_attack.py"))
    algorithm = algo_cls()

    # Create harness and evaluate
    harness = ResearchHarness(
        target_model=target_model,
        target_provider="openrouter",
        guard_mode="none",  # We handle guarding manually via OutputGuardTarget
        query_budget=args.query_budget,
        n_behaviors=args.n_behaviors,
        seed=args.seed,
        attacker_lm=attacker_lm,
        judge_lm=judge_lm,
    )

    print(f"[eval] Defense: {defense_name}")
    print(f"[eval] Attack: PAIR v0")
    print(f"[eval] N={args.n_behaviors}, budget={args.query_budget}, seed={args.seed}")

    result = harness.evaluate_attack(algorithm, defense_target=guarded_target)
    print(f"\n{result.summary()}")

    if args.output:
        result.save(Path(args.output))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
