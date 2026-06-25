#!/usr/bin/env python3
"""LLM Defense Experiment — Pure LLM guardrail vs evolving attacks.

Compares LLM-based defense (Kimi K2.5) against a beta pattern baseline
under the same attack evolution conditions.

Experiment tiers:
  D1) Baseline — static attacks vs LLM defense
  D2) Attack evolution — evolved attacks vs LLM defense (3 mutation rates)

Usage:
  python scripts/run_llm_defense_experiment.py --quick     # 5 rounds, 10 attacks
  python scripts/run_llm_defense_experiment.py              # 10 rounds, 15 attacks
  python scripts/run_llm_defense_experiment.py --full       # 15 rounds, 30 attacks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure local source is used
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

LOCAL_SRC = PROJECT_ROOT / "src"
MAIN_SRC = WORKSPACE_ROOT / "dspyGuardrails" / "src"
for src in [LOCAL_SRC, MAIN_SRC]:
    if str(src) not in sys.path and src.exists():
        sys.path.insert(0, str(src))


# ─────────────────────────────────────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────────────────────────────────────

def configure_llm(model: str = "kimi-k2.5", verbose: bool = False) -> None:
    """Configure DSPy with the specified LLM."""
    from dotenv import load_dotenv

    # Load .env files — try multiple locations (worktree vs main repo)
    vag_root = Path("/Users/miracy/Documents/VAG")
    for env_path in [
        vag_root / "dspyGuardrails" / ".env",
        vag_root / ".env",
        WORKSPACE_ROOT / "dspyGuardrails" / ".env",
        WORKSPACE_ROOT / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)

    import dspy

    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("ERROR: MOONSHOT_API_KEY not found in .env")
        sys.exit(1)

    lm = dspy.LM(
        f"openai/{model}",
        api_key=api_key,
        api_base="https://api.moonshot.cn/v1",
    )
    dspy.configure(lm=lm)

    if verbose:
        print(f"  LLM configured: {model} via Moonshot API")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpConfig:
    name: str
    experiment_type: str  # D1_baseline_llm, D2_attack_vs_llm
    max_rounds: int = 10
    attacks_per_round: int = 15
    attack_categories: list[str] = field(
        default_factory=lambda: ["injection", "jailbreak", "bypass"]
    )
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    verbose: bool = False

    def label(self) -> str:
        return f"{self.experiment_type}_{self.name}"


@dataclass
class RoundMetric:
    round_num: int
    asr: float
    bypassed: int
    blocked: int
    total: int
    new_attacks: int
    avg_response_ms: float
    round_duration_s: float
    # Per-attack detail for analysis
    bypass_details: list[dict] = field(default_factory=list)


@dataclass
class ExpResult:
    config: dict[str, Any]
    rounds: list[RoundMetric]
    initial_asr: float
    final_asr: float
    asr_reduction: float
    total_duration_s: float
    total_evolved_attacks: int
    avg_latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Payload Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_payloads(categories: list[str], per_category: int):
    """Load attack payloads from the payload library."""
    from dspy_guardrails.adversarial.attack_evolver import EvolvedAttack
    from dspy_guardrails.redteam.payloads import (
        BypassPayloads,
        InjectionPayloads,
        JailbreakPayloads,
    )

    providers = {
        "injection": InjectionPayloads,
        "jailbreak": JailbreakPayloads,
        "bypass": BypassPayloads,
    }

    attacks: list[EvolvedAttack] = []
    for cat in categories:
        provider = providers.get(cat)
        if not provider:
            continue
        items = provider.get_all()
        for item in items[:per_category]:
            text = getattr(item, "prompt", None) or getattr(item, "template", str(item))
            sev = getattr(item, "severity", "medium")
            sev_val = sev.value if hasattr(sev, "value") else str(sev)
            attacks.append(EvolvedAttack(
                id=str(uuid.uuid4())[:8],
                payload=text,
                category=cat,
                severity=sev_val,
                evolution_type="initial",
                generation=0,
            ))
    return attacks


# ─────────────────────────────────────────────────────────────────────────────
# LLM Defense Target
# ─────────────────────────────────────────────────────────────────────────────

def create_llm_target():
    """Create LLM-based defense target."""
    from dspy_guardrails.adversarial.evolvable_target import EvolvableLLMTarget
    from dspy_guardrails.llm_guardrail import LLMGuardrail

    guardrail = LLMGuardrail(comprehensive=True, use_dspy=True)
    return EvolvableLLMTarget(guardrail=guardrail)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_llm(cfg: ExpConfig, output_dir: Path) -> ExpResult:
    """D1: Static attacks vs LLM defense — single round (static = deterministic)."""
    target = create_llm_target()
    payloads = load_payloads(
        cfg.attack_categories,
        cfg.attacks_per_round // max(1, len(cfg.attack_categories)),
    )

    rounds: list[RoundMetric] = []
    all_latencies: list[float] = []
    t0 = time.perf_counter()

    # Only 1 round needed — static attacks give same result each time
    round_start = time.perf_counter()
    bypassed = blocked = 0
    latencies: list[float] = []
    details: list[dict] = []

    for attack in payloads[:cfg.attacks_per_round]:
        target.reset_session()
        ts = time.perf_counter()
        resp = target.invoke(attack.payload)
        lat = (time.perf_counter() - ts) * 1000
        latencies.append(lat)
        all_latencies.append(lat)

        if resp.was_blocked:
            blocked += 1
        else:
            bypassed += 1
            details.append({
                "payload": attack.payload[:100],
                "category": attack.category,
            })

    total = bypassed + blocked
    asr = bypassed / total if total else 0.0
    rd = time.perf_counter() - round_start

    metric = RoundMetric(
        round_num=1, asr=asr, bypassed=bypassed, blocked=blocked,
        total=total, new_attacks=0,
        avg_response_ms=sum(latencies) / len(latencies) if latencies else 0,
        round_duration_s=rd, bypass_details=details,
    )
    rounds.append(metric)

    if cfg.verbose:
        print(f"  [D1-{cfg.name}] ASR={asr:.1%} ({bypassed}/{total}) "
              f"avg_lat={metric.avg_response_ms:.0f}ms")

    total_time = time.perf_counter() - t0

    return ExpResult(
        config=_cfg_to_dict(cfg),
        rounds=rounds,
        initial_asr=asr,
        final_asr=asr,
        asr_reduction=0.0,
        total_duration_s=total_time,
        total_evolved_attacks=0,
        avg_latency_ms=sum(all_latencies) / len(all_latencies) if all_latencies else 0,
    )


def run_attack_evolution_vs_llm(cfg: ExpConfig, output_dir: Path) -> ExpResult:
    """D2: Evolved attacks vs static LLM defense.

    Uses blind mutation: mutates ALL attacks each round (not just successful ones),
    simulating an attacker who doesn't know which attacks succeeded. The original
    AttackEvolver only mutates successful attacks, which produces no evolution when
    the defense blocks everything.
    """
    from dspy_guardrails.adversarial.attack_evolver import (
        ContextWrapMutation,
        EncodingMutation,
        EvolvedAttack,
        StructureMutation,
        SynonymMutation,
    )
    from dspy_guardrails.adversarial.metrics import AttackResult as AdvAttackResult

    target = create_llm_target()

    # Set up mutation strategies directly (blind mutation, not success-dependent)
    mutators = [SynonymMutation(), EncodingMutation(), ContextWrapMutation(), StructureMutation()]

    current_attacks = load_payloads(
        cfg.attack_categories,
        cfg.attacks_per_round // max(1, len(cfg.attack_categories)),
    )

    rounds: list[RoundMetric] = []
    all_latencies: list[float] = []
    t0 = time.perf_counter()

    for rnd in range(1, cfg.max_rounds + 1):
        round_start = time.perf_counter()
        successful: list[AdvAttackResult] = []
        failed: list[AdvAttackResult] = []
        latencies: list[float] = []
        details: list[dict] = []

        for attack in current_attacks[:cfg.attacks_per_round]:
            target.reset_session()
            ts = time.perf_counter()
            resp = target.invoke(attack.payload)
            lat = (time.perf_counter() - ts) * 1000
            latencies.append(lat)
            all_latencies.append(lat)

            result = AdvAttackResult(
                attack_id=attack.id, payload=attack.payload,
                category=attack.category, severity=attack.severity,
                bypassed=not resp.was_blocked, blocked=resp.was_blocked,
                response=resp.response[:200] if resp.response else "",
                response_time_ms=lat,
                block_reason=resp.metadata.get("block_reason") if resp.metadata else None,
            )
            if result.bypassed:
                successful.append(result)
                details.append({
                    "payload": attack.payload[:100],
                    "category": attack.category,
                    "evolution": attack.evolution_type,
                    "generation": attack.generation,
                })
            else:
                failed.append(result)

        total = len(successful) + len(failed)
        asr = len(successful) / total if total else 0.0

        # Blind mutation: mutate ALL attacks (successful + failed), not just successful
        # This simulates an attacker who keeps trying different mutations regardless
        all_results = successful + failed
        new_attacks: list[EvolvedAttack] = []
        for ar in all_results:
            # Apply each mutation with probability = mutation_rate
            for mutator in mutators:
                if random.random() < cfg.mutation_rate:
                    try:
                        mutated = mutator.mutate(ar.payload)
                        if mutated != ar.payload:
                            new_attacks.append(EvolvedAttack(
                                id=str(uuid.uuid4())[:8],
                                payload=mutated,
                                category=ar.category,
                                severity=ar.severity,
                                parent_id=ar.attack_id,
                                evolution_type=f"blind_{type(mutator).__name__}",
                                generation=rnd,
                            ))
                    except Exception:
                        continue

        # Keep original attacks + add mutations (cap at 2x attacks_per_round)
        if new_attacks:
            # Mix: keep some originals + add new mutations
            combined = current_attacks + new_attacks
            random.shuffle(combined)
            current_attacks = combined[:cfg.attacks_per_round * 2]

        rd = time.perf_counter() - round_start
        metric = RoundMetric(
            round_num=rnd, asr=asr, bypassed=len(successful), blocked=len(failed),
            total=total, new_attacks=len(new_attacks),
            avg_response_ms=sum(latencies) / len(latencies) if latencies else 0,
            round_duration_s=rd, bypass_details=details,
        )
        rounds.append(metric)

        if cfg.verbose:
            print(f"  [D2-{cfg.name}] Round {rnd}: ASR={asr:.1%} ({len(successful)}/{total}) "
                  f"new={len(new_attacks)} pool={len(current_attacks)} "
                  f"avg_lat={metric.avg_response_ms:.0f}ms")

    total_time = time.perf_counter() - t0
    initial_asr = rounds[0].asr if rounds else 0
    final_asr = rounds[-1].asr if rounds else 0

    return ExpResult(
        config=_cfg_to_dict(cfg),
        rounds=rounds,
        initial_asr=initial_asr,
        final_asr=final_asr,
        asr_reduction=initial_asr - final_asr,
        total_duration_s=total_time,
        total_evolved_attacks=len(current_attacks),
        avg_latency_ms=sum(all_latencies) / len(all_latencies) if all_latencies else 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_to_dict(cfg: ExpConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "experiment_type": cfg.experiment_type,
        "max_rounds": cfg.max_rounds,
        "attacks_per_round": cfg.attacks_per_round,
        "attack_categories": cfg.attack_categories,
        "mutation_rate": cfg.mutation_rate,
        "defense_mode": "llm",
        "llm_model": "kimi-k2.5",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: list[ExpResult], started_at: str, total_time: float) -> str:
    lines: list[str] = []
    lines.append("# LLM Defense Experiment Report")
    lines.append("")
    lines.append(f"**Generated**: {started_at}")
    lines.append(f"**Defense**: LLM (Kimi K2.5, comprehensive mode)")
    lines.append(f"**Total Runtime**: {total_time:.1f}s")
    lines.append(f"**Experiments**: {len(results)}")
    lines.append("")

    # Summary table
    lines.append("## 1. Summary")
    lines.append("")
    lines.append("| Experiment | Type | Mutation Rate | Rounds | Initial ASR | Final ASR | Avg Latency | Duration |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for res in results:
        cfg = res.config
        lines.append(
            f"| {cfg['name']} | {cfg['experiment_type']} | {cfg.get('mutation_rate', '-')} | "
            f"{len(res.rounds)} | {res.initial_asr:.1%} | {res.final_asr:.1%} | "
            f"{res.avg_latency_ms:.0f}ms | {res.total_duration_s:.1f}s |"
        )

    # Comparison with beta pattern baseline (historical reference)
    lines.append("")
    lines.append("## 2. LLM vs Beta Pattern Baseline (Historical Reference)")
    lines.append("")
    lines.append("| Metric | Pattern-only beta (fast) | LLM (Kimi K2.5) |")
    lines.append("|--------|---:|---:|")

    baseline_results = [r for r in results if "baseline" in r.config["experiment_type"]]
    attack_results = [r for r in results if "attack" in r.config["experiment_type"]]

    if baseline_results:
        llm_baseline = baseline_results[0].initial_asr
        lines.append(f"| Baseline ASR (static attacks) | 46.7% | {llm_baseline:.1%} |")

    if attack_results:
        avg_llm_attack = sum(r.final_asr for r in attack_results) / len(attack_results)
        lines.append(f"| Final ASR (evolved attacks, avg) | 91.1% | {avg_llm_attack:.1%} |")

    if baseline_results:
        lines.append(f"| Avg Latency | ~1ms | {baseline_results[0].avg_latency_ms:.0f}ms |")

    # ASR trajectory
    lines.append("")
    lines.append("## 3. ASR Trajectory")
    lines.append("")
    lines.append("```")
    header = "Round | " + " | ".join(
        f"{r.config['experiment_type']}_{r.config['name']}"[:25]
        for r in results
    )
    lines.append(header)

    max_rounds = max(len(r.rounds) for r in results) if results else 0
    for rnd_idx in range(max_rounds):
        cols = []
        for res in results:
            if rnd_idx < len(res.rounds):
                cols.append(f"{res.rounds[rnd_idx].asr:.3f}")
            else:
                cols.append("  -  ")
        lines.append(f"  {rnd_idx + 1:3d} | " + " | ".join(f"{c:>25s}" for c in cols))
    lines.append("```")

    # Bypass analysis
    lines.append("")
    lines.append("## 4. Bypass Analysis (attacks that fooled LLM)")
    lines.append("")

    for res in results:
        bypasses = []
        for rm in res.rounds:
            bypasses.extend(rm.bypass_details)
        if not bypasses:
            lines.append(f"### {res.config['name']}: **No bypasses** — LLM blocked 100%")
            lines.append("")
            continue

        lines.append(f"### {res.config['name']}: {len(bypasses)} total bypasses")
        lines.append("")

        # Category breakdown
        by_cat: dict[str, int] = {}
        for b in bypasses:
            by_cat[b.get("category", "unknown")] = by_cat.get(b.get("category", "unknown"), 0) + 1
        lines.append("| Category | Bypassed Count |")
        lines.append("|----------|---:|")
        for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {count} |")
        lines.append("")

        # Sample bypasses
        lines.append("**Sample bypassed payloads:**")
        lines.append("")
        for b in bypasses[:5]:
            lines.append(f"- `{b['payload'][:80]}...` ({b['category']})")
        if len(bypasses) > 5:
            lines.append(f"- ... and {len(bypasses) - 5} more")
        lines.append("")

    # Mutation rate impact
    if len(attack_results) > 1:
        lines.append("## 5. Mutation Rate Impact on LLM Defense")
        lines.append("")
        lines.append("| Mutation Rate | Initial ASR | Final ASR | ASR Change |")
        lines.append("|---:|---:|---:|---:|")
        for res in sorted(attack_results, key=lambda r: r.config.get("mutation_rate", 0)):
            mr = res.config.get("mutation_rate", 0)
            delta = res.final_asr - res.initial_asr
            lines.append(f"| {mr} | {res.initial_asr:.1%} | {res.final_asr:.1%} | {delta:+.1%} |")
        lines.append("")

    # Conclusions
    lines.append("## 6. Conclusions")
    lines.append("")

    if baseline_results:
        llm_asr = baseline_results[0].initial_asr
        if llm_asr < 0.2:
            lines.append(f"- **LLM defense is highly effective**: Baseline ASR of {llm_asr:.1%} vs 46.7% for beta pattern baseline")
        elif llm_asr < 0.4:
            lines.append(f"- **LLM defense is moderately effective**: Baseline ASR of {llm_asr:.1%} vs 46.7% for beta pattern baseline")
        else:
            lines.append(f"- **LLM defense shows limited improvement**: Baseline ASR of {llm_asr:.1%} vs 46.7% for beta pattern baseline")

    if attack_results:
        avg_final = sum(r.final_asr for r in attack_results) / len(attack_results)
        if avg_final < 0.3:
            lines.append(f"- **LLM resists attack evolution**: Avg final ASR {avg_final:.1%} vs 91.1% for beta pattern baseline — semantic understanding defeats mutation")
        elif avg_final < 0.6:
            lines.append(f"- **LLM partially resists attack evolution**: Avg final ASR {avg_final:.1%} vs 91.1% for beta pattern baseline — some mutations bypass LLM")
        else:
            lines.append(f"- **Attack evolution still effective vs LLM**: Avg final ASR {avg_final:.1%} vs 91.1% for beta pattern baseline")

    if baseline_results and attack_results:
        lines.append(f"- **Latency trade-off**: LLM defense ~{baseline_results[0].avg_latency_ms:.0f}ms vs ~1ms for beta pattern baseline")

    lines.append("")
    lines.append("---")
    lines.append(f"*Defense: Kimi K2.5 via Moonshot API (DSPy ComprehensiveSafetyClassifier)*")
    lines.append(f"*Generated by run_llm_defense_experiment.py*")

    return "\n".join(lines)


def generate_csv(results: list[ExpResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "round", "asr", "bypassed", "blocked", "total",
            "new_attacks", "avg_response_ms",
        ])
        for res in results:
            label = f"{res.config['experiment_type']}_{res.config['name']}"
            for rm in res.rounds:
                writer.writerow([
                    label, rm.round_num, f"{rm.asr:.4f}",
                    rm.bypassed, rm.blocked, rm.total,
                    rm.new_attacks, f"{rm.avg_response_ms:.1f}",
                ])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

RUNNER_MAP = {
    "D1_baseline_llm": run_baseline_llm,
    "D2_attack_vs_llm": run_attack_evolution_vs_llm,
}


def build_configs(mode: str = "default", experiments: str = "D1D2") -> list[ExpConfig]:
    if mode == "quick":
        max_rounds, attacks_per_round = 5, 10
        mutation_rates = [0.3]
    elif mode == "full":
        max_rounds, attacks_per_round = 15, 30
        mutation_rates = [0.1, 0.3, 0.5]
    else:  # default
        max_rounds, attacks_per_round = 10, 15
        mutation_rates = [0.1, 0.3, 0.5]

    configs: list[ExpConfig] = []

    if "D1" in experiments:
        configs.append(ExpConfig(
            name="static_llm",
            experiment_type="D1_baseline_llm",
            max_rounds=max_rounds,
            attacks_per_round=attacks_per_round,
        ))

    if "D2" in experiments:
        for mr in mutation_rates:
            configs.append(ExpConfig(
                name=f"attack_llm_mr{mr}",
                experiment_type="D2_attack_vs_llm",
                max_rounds=max_rounds,
                attacks_per_round=attacks_per_round,
                mutation_rate=mr,
            ))

    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM Defense Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 rounds, 10 attacks)")
    parser.add_argument("--full", action="store_true", help="Full experiment (15 rounds, 30 attacks)")
    parser.add_argument("--experiment", default="D1D2", help="Experiments: D1=baseline, D2=attack_evolution")
    parser.add_argument("--model", default="kimi-k2.5", help="LLM model (default: kimi-k2.5)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "llm_defense"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    mode = "quick" if args.quick else ("full" if args.full else "default")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure LLM
    print(f"{'=' * 70}")
    print(f"  LLM Defense Experiment (Kimi K2.5)")
    print(f"{'=' * 70}")
    print(f"  Configuring LLM...")
    configure_llm(model=args.model, verbose=True)

    configs = build_configs(mode=mode, experiments=args.experiment)
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"  Mode: {mode} | Experiments: {args.experiment} | Total runs: {len(configs)}")
    print(f"{'=' * 70}")

    all_results: list[ExpResult] = []
    t0 = time.perf_counter()

    for idx, cfg in enumerate(configs, 1):
        cfg.verbose = args.verbose
        print(f"\n[{idx}/{len(configs)}] Running: {cfg.label()}")
        print(f"  Type={cfg.experiment_type} rounds={cfg.max_rounds} "
              f"attacks/round={cfg.attacks_per_round} mr={cfg.mutation_rate}")

        runner = RUNNER_MAP.get(cfg.experiment_type)
        if not runner:
            print(f"  ERROR: Unknown experiment type: {cfg.experiment_type}")
            continue

        try:
            result = runner(cfg, out_dir)
            all_results.append(result)
            print(f"  Done: ASR {result.initial_asr:.1%} -> {result.final_asr:.1%} "
                  f"avg_lat={result.avg_latency_ms:.0f}ms in {result.total_duration_s:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.perf_counter() - t0

    # Save results
    json_path = out_dir / f"llm_defense_{started_at}.json"
    csv_path = out_dir / f"llm_defense_asr_{started_at}.csv"
    md_path = out_dir / f"llm_defense_{started_at}.md"

    json_data = {
        "experiment": "llm_defense_vs_evolving_attacks",
        "defense": "kimi-k2.5 (ComprehensiveSafetyClassifier)",
        "started_at": started_at,
        "mode": mode,
        "total_duration_s": total_time,
        "results": [
            {
                "config": r.config,
                "initial_asr": r.initial_asr,
                "final_asr": r.final_asr,
                "asr_reduction": r.asr_reduction,
                "total_duration_s": r.total_duration_s,
                "avg_latency_ms": r.avg_latency_ms,
                "total_evolved_attacks": r.total_evolved_attacks,
                "rounds": [
                    {
                        "round": rm.round_num,
                        "asr": rm.asr,
                        "bypassed": rm.bypassed,
                        "blocked": rm.blocked,
                        "total": rm.total,
                        "new_attacks": rm.new_attacks,
                        "avg_response_ms": rm.avg_response_ms,
                        "bypass_details": rm.bypass_details,
                    }
                    for rm in r.rounds
                ],
            }
            for r in all_results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    generate_csv(all_results, csv_path)

    md_text = generate_report(all_results, started_at, total_time)
    md_path.write_text(md_text, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"  Experiment Complete ({total_time:.1f}s)")
    print(f"{'=' * 70}")
    print(f"  JSON:     {json_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Report:   {md_path}")
    print()

    for r in all_results:
        label = f"{r.config['experiment_type']}_{r.config['name']}"
        print(f"  {label:45s}  ASR: {r.initial_asr:.1%} -> {r.final_asr:.1%}  "
              f"lat={r.avg_latency_ms:.0f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
