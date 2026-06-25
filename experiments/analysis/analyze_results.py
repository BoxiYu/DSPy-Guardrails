#!/usr/bin/env python3
"""ASE 2026 Experiment Analysis — Generate paper tables from raw results.

Usage:
    python experiments/analysis/analyze_results.py --results-dir experiments/results/
    python experiments/analysis/analyze_results.py --results-dir experiments/results/ --latex
"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return float(np.mean(arr)), lo, hi


def paired_permutation_test(
    a: list[float], b: list[float], n_perm: int = 10000, seed: int = 42
) -> float:
    """Two-sided paired permutation test. Returns p-value."""
    rng = np.random.RandomState(seed)
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    assert len(a_arr) == len(b_arr), "Paired arrays must have same length"
    diff = a_arr - b_arr
    obs_stat = abs(np.mean(diff))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_stat = abs(np.mean(diff * signs))
        if perm_stat >= obs_stat:
            count += 1
    return count / n_perm


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(1, n):
        cur = sorted_indices[i]
        prev = sorted_indices[i - 1]
        adjusted[cur] = max(adjusted[cur], adjusted[prev])
    return adjusted


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str, exp_name: str, include_archive: bool = False) -> list[dict]:
    """Load all JSON result files for a given experiment.

    Skips checkpoint files to avoid duplicates.  When multiple files contain
    results for the same (defense, attack, seed, goal), the result from the
    newest file wins (based on filename sort order / modification time).
    """
    patterns = [os.path.join(results_dir, f"{exp_name}_*.json")]
    if include_archive:
        patterns.append(os.path.join(results_dir, "archive", f"{exp_name}_*.json"))
    # Sort by filename (which embeds timestamp) so newer files come later
    files = sorted(set(f for p in patterns for f in glob.glob(p)))
    # Use dict keyed by (defense, attack, seed, goal) to deduplicate — last write wins
    dedup: dict[tuple, dict] = {}
    for f in files:
        basename = os.path.basename(f)
        if "checkpoint" in basename:
            continue
        with open(f) as fh:
            data = json.load(fh)
            file_meta = data.get("metadata", {})
            file_seed = data.get("seed", 42)
            file_model_config = data.get("model_config", {})
            if "results" in data:
                for r in data["results"]:
                    r["_file"] = basename
                    r["_seed"] = file_seed
                    r["_model_config"] = file_model_config
                    r["_metadata"] = file_meta
                    key = (r.get("defense_name"), r.get("attack_name"), file_seed, r.get("goal_text", ""))
                    dedup[key] = r
    return list(dedup.values())


# ---------------------------------------------------------------------------
# EXP1 Analysis: Defense Effectiveness
# ---------------------------------------------------------------------------

def analyze_exp1(results: list[dict], latex: bool = False) -> str:
    """Generate Table 1: Defense × Attack ASR/F1/OR."""
    # Group by (defense, attack, seed)
    groups = defaultdict(list)
    for r in results:
        key = (r["defense_name"], r["attack_name"], r.get("_seed", 42))
        groups[key].append(r)

    # Separate direct results (for F1 and OR computation)
    # Direct results include both harmful and benign goals tested without attacks
    benign_by_defense_seed = defaultdict(lambda: defaultdict(list))
    harmful_direct_by_defense_seed = defaultdict(lambda: defaultdict(list))
    for (defense, attack, seed), items in groups.items():
        if attack == "direct":
            benign_items = [r for r in items if not r["is_harmful"]]
            harmful_items = [r for r in items if r["is_harmful"]]
            benign_by_defense_seed[defense][seed].extend(benign_items)
            harmful_direct_by_defense_seed[defense][seed].extend(harmful_items)

    # Compute per-defense, per-attack metrics
    defense_attack_metrics = defaultdict(lambda: defaultdict(dict))

    # First compute F1 per defense using direct results
    defense_f1 = {}
    defense_or = {}
    for defense in set(d for d, _, _ in groups.keys()):
        seeds = set(s for d, a, s in groups.keys() if d == defense)
        f1_vals = []
        or_vals = []
        for seed in seeds:
            harmful_direct = harmful_direct_by_defense_seed[defense].get(seed, [])
            benign = benign_by_defense_seed[defense].get(seed, [])
            tp = sum(1 for r in harmful_direct if r.get("blocked", False))
            fn = sum(1 for r in harmful_direct if not r.get("blocked", False))
            fp = sum(1 for r in benign if r.get("blocked", False))
            tn = sum(1 for r in benign if not r.get("blocked", False))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_vals.append(f1)
            if benign:
                or_vals.append(sum(1 for r in benign if r.get("blocked", False)) / len(benign))
        defense_f1[defense] = float(np.mean(f1_vals)) if f1_vals else 0.0
        defense_or[defense] = float(np.mean(or_vals)) if or_vals else 0.0

    for (defense, attack, seed), items in groups.items():
        if attack == "direct":
            continue  # handled above

        harmful = [r for r in items if r["is_harmful"]]

        # ASR: fraction of harmful goals where attack succeeded
        if harmful:
            asr_vals = [1.0 if r["attack_success"] else 0.0 for r in harmful]
            asr = np.mean(asr_vals)
        else:
            asr = 0.0

        defense_attack_metrics[defense][attack][seed] = {
            "asr": asr, "f1": defense_f1.get(defense, 0.0),
            "or": defense_or.get(defense, 0.0),
            "n_harmful": len(harmful),
        }

    # Collect all defenses and attacks
    all_defenses = sorted(set(d for d, _, _ in groups.keys()))
    all_attacks = sorted(set(a for _, a, _ in groups.keys()) - {"direct"})

    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 1: Defense Effectiveness (EXP1)")
    lines.append("=" * 80)

    header = f"{'Defense':<20}"
    for atk in all_attacks:
        header += f"  ASR-{atk:<6}"
    header += f"  {'F1':>5}  {'OR%':>5}"
    lines.append(header)
    lines.append("-" * len(header))

    latex_rows = []
    for defense in all_defenses:
        row = f"{defense:<20}"
        f1_vals = []
        or_vals = []
        for atk in all_attacks:
            seeds_data = defense_attack_metrics[defense].get(atk, {})
            if seeds_data:
                asrs = [v["asr"] for v in seeds_data.values()]
                f1s = [v["f1"] for v in seeds_data.values()]
                ors = [v["or"] for v in seeds_data.values()]
                f1_vals.extend(f1s)
                or_vals.extend(ors)
                if len(asrs) > 1:
                    row += f"  {np.mean(asrs)*100:5.1f}±{np.std(asrs)*100:4.1f}"
                else:
                    row += f"  {asrs[0]*100:10.1f}"
            else:
                row += f"  {'---':>10}"
        if f1_vals:
            row += f"  {np.mean(f1_vals):5.3f}"
        else:
            row += f"  {'---':>5}"
        if or_vals:
            row += f"  {np.mean(or_vals)*100:5.1f}"
        else:
            row += f"  {'---':>5}"
        lines.append(row)

        if latex:
            latex_cols = [defense.replace("_", r"\_")]
            for atk in all_attacks:
                seeds_data = defense_attack_metrics[defense].get(atk, {})
                if seeds_data:
                    asrs = [v["asr"] for v in seeds_data.values()]
                    if len(asrs) > 1:
                        latex_cols.append(f"{np.mean(asrs)*100:.1f}$\\pm${np.std(asrs)*100:.1f}")
                    else:
                        latex_cols.append(f"{asrs[0]*100:.1f}")
                else:
                    latex_cols.append("---")
            if f1_vals:
                latex_cols.append(f"{np.mean(f1_vals):.3f}")
            else:
                latex_cols.append("---")
            if or_vals:
                latex_cols.append(f"{np.mean(or_vals)*100:.1f}")
            else:
                latex_cols.append("---")
            latex_rows.append(" & ".join(latex_cols) + r" \\")

    if latex:
        lines.append("")
        lines.append("% LaTeX table rows:")
        for lr in latex_rows:
            lines.append(lr)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EXP2 Analysis: Optimizer Comparison
# ---------------------------------------------------------------------------

def analyze_exp2(results: list[dict], latex: bool = False) -> str:
    """Generate Table 2: Optimizer comparison (F1, ASR, Time, Calls, Cost)."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 2: Optimizer Comparison (EXP2)")
    lines.append("=" * 80)

    # Group by defense_name (which is dspy_{optimizer})
    groups = defaultdict(list)
    for r in results:
        groups[r["defense_name"]].append(r)

    opt_order = ["dspy_bfs", "dspy_mipro", "dspy_simba", "dspy_gepa"]
    opt_labels = {"dspy_bfs": "BootstrapFewShot", "dspy_mipro": "MIPROv2",
                  "dspy_simba": "SIMBA", "dspy_gepa": "GEPA",
                  # Legacy key from earlier experiment code
                  "dspy_dspy": "BootstrapFewShot"}

    header = f"{'Optimizer':<20}  {'F1':>5}  {'ASR%':>6}  {'Time(min)':>10}  {'Calls':>6}  {'Cost($)':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for opt in opt_order:
        items = groups.get(opt, [])
        if not items:
            lines.append(f"  {opt_labels.get(opt, opt):<20}  (no data)")
            continue

        label = opt_labels.get(opt, opt)
        harmful_pair = [r for r in items if r["is_harmful"] and r["attack_name"] == "pair"]
        harmful_direct = [r for r in items if r["is_harmful"] and r["attack_name"] == "direct"]
        benign = [r for r in items if not r["is_harmful"]]

        # F1 from direct classification
        tp = sum(1 for r in harmful_direct if r.get("blocked", False))
        fn = sum(1 for r in harmful_direct if not r.get("blocked", False))
        fp = sum(1 for r in benign if r.get("blocked", False))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # ASR from PAIR attack
        if harmful_pair:
            asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in harmful_pair])
        else:
            asr = 0.0

        # Compile time from metadata (preferred) or fallback to PAIR wall time
        file_meta = items[0].get("_metadata", {})
        compile_times = file_meta.get("compile_times", {})
        compile_api_calls = file_meta.get("compile_api_calls", {})
        compile_time_s = compile_times.get(opt, 0)
        api_calls = compile_api_calls.get(opt, 0)

        if compile_time_s > 0:
            total_time = compile_time_s / 60.0
        else:
            # Fallback: PAIR wall time (less accurate)
            total_time = sum(r.get("wall_time_s", 0) for r in harmful_pair) / 60.0

        if api_calls == 0:
            api_calls = sum(r.get("total_queries", 0) for r in harmful_pair)

        lines.append(f"  {label:<20}  {f1:5.3f}  {asr*100:5.1f}%  {total_time:8.1f}m  {api_calls:>6}  {'TBD':>8}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EXP3 Analysis: Attack Comparison
# ---------------------------------------------------------------------------

def analyze_exp3_attacks(results: list[dict], latex: bool = False) -> str:
    """Generate Table 3: Attack method comparison (ASR, Avg SR, Queries, Time)."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE 3: Attack Comparison (EXP3)")
    lines.append("=" * 80)

    attack_order = ["pair", "tap", "mapelites"]
    attack_labels = {"pair": "PAIR", "tap": "TAP", "mapelites": "MAP-Elites"}

    header = f"{'Attack':<15}  {'ASR%':>6}  {'Avg SR':>7}  {'Queries':>8}  {'Time(s)':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for atk in attack_order:
        items = [r for r in results if r["attack_name"] == atk and r["is_harmful"]]
        if not items:
            lines.append(f"  {attack_labels.get(atk, atk):<15}  (no data)")
            continue

        label = attack_labels.get(atk, atk)
        asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in items])
        successful = [r for r in items if r["attack_success"]]
        sr_scores = [r.get("strongreject_score", 0.0) for r in successful]
        avg_sr = np.mean(sr_scores) if sr_scores else 0.0
        avg_queries = np.mean([r.get("total_queries", 0) for r in items])
        avg_time = np.mean([r.get("wall_time_s", 0) for r in items])

        lines.append(f"  {label:<15}  {asr*100:5.1f}%  {avg_sr:7.3f}  {avg_queries:8.0f}  {avg_time:7.1f}s")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EXP3 Analysis: Ablation
# ---------------------------------------------------------------------------

def analyze_exp3_ablation(results: list[dict], latex: bool = False) -> str:
    """Generate Table 4: Ablation study."""
    ablation_configs = ["pair_ablation_full", "pair_ablation_no_feedback",
                        "pair_ablation_no_history", "pair_ablation_minimal"]
    config_labels = ["Full", "No Feedback (-DF)", "No History (-PA)", "Minimal (goal only)"]

    lines = []
    lines.append("=" * 60)
    lines.append("TABLE 4: Ablation Study (EXP3)")
    lines.append("=" * 60)

    full_asr = None
    for config, label in zip(ablation_configs, config_labels):
        items = [r for r in results if r["attack_name"] == config and r["is_harmful"]]
        if not items:
            items = [r for r in results if r["attack_name"] == config.replace("pair_ablation_", "") and r["is_harmful"]]
        if items:
            asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in items])
            sr_scores = [r.get("strongreject_score", 0.0) for r in items if r["attack_success"]]
            avg_sr = np.mean(sr_scores) if sr_scores else 0.0
            if "full" in config:
                full_asr = asr
                delta = "---"
            else:
                delta = f"{(asr - full_asr)*100:+.1f}" if full_asr is not None else "?"
            lines.append(f"  {label:<25}  ASR={asr*100:5.1f}%  AvgSR={avg_sr:.2f}  ΔASR={delta}")
        else:
            lines.append(f"  {label:<25}  (no data)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EXP4 Analysis: Co-Evolution
# ---------------------------------------------------------------------------

def analyze_exp4(results_dir: str) -> str:
    """Generate Table 5: Co-evolution trace."""
    # Check both root and exp4/ subdirectory
    candidates = [
        os.path.join(results_dir, "exp4", "exp4_*.json"),
        os.path.join(results_dir, "exp4_*.json"),
    ]
    files = []
    for pattern in candidates:
        files.extend(sorted(glob.glob(pattern)))

    lines = []
    lines.append("=" * 60)
    lines.append("TABLE 5: Co-Evolution Dynamics (EXP4)")
    lines.append("=" * 60)

    if not files:
        lines.append("  (no EXP4 results found)")
        return "\n".join(lines)

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        lines.append(f"\nFile: {os.path.basename(f)}")
        lines.append(f"  Seed: {data.get('seed', '?')}")
        lines.append(f"  Wall time: {data.get('wall_time_s', 0):.0f}s")

        # Regime A: Single-round GEPA
        sr = data.get("single_round", data.get("regime_a", {}))
        if sr:
            asr_val = sr.get("asr", "?")
            if isinstance(asr_val, float):
                asr_val = f"{asr_val*100:.1f}%"
            lines.append(f"  Regime A (Single-round GEPA): ASR={asr_val}")
            results_list = sr.get("results", [])
            if results_list:
                sr_scores = [r.get("strongreject_score", 0) for r in results_list if r.get("success")]
                if sr_scores:
                    lines.append(f"    Avg StrongREJECT (bypasses): {np.mean(sr_scores):.3f}")

        # Regime B: Co-evolution
        ce = data.get("coevolution", data.get("regime_b", {}))
        if ce:
            asr_val = ce.get("asr", "?")
            if isinstance(asr_val, float):
                asr_val = f"{asr_val*100:.1f}%"
            lines.append(f"  Regime B (Co-evolution): Final ASR={asr_val}")
            lines.append(f"    Converged: {ce.get('converged', '?')}")
            lines.append(f"    Total rounds: {ce.get('total_rounds', '?')}")

            round_traces = ce.get("round_traces", [])
            if round_traces:
                lines.append(f"    {'Round':>5}  {'ASR%':>8}  {'Bypass':>8}  {'Block':>8}  {'Patterns':>8}  {'Examples':>8}")
                lines.append("    " + "-" * 55)
                for trace in round_traces:
                    rnd = trace.get("round", "?")
                    asr = trace.get("asr", 0)
                    if isinstance(asr, float):
                        asr = f"{asr*100:.1f}"
                    bypassed = trace.get("bypassed", "?")
                    blocked = trace.get("blocked", "?")
                    patterns = trace.get("patterns", "?")
                    examples = trace.get("examples", "?")
                    rnd = rnd if rnd is not None else "?"
                    asr = asr if asr is not None else "?"
                    bypassed = bypassed if bypassed is not None else "?"
                    blocked = blocked if blocked is not None else "?"
                    patterns = patterns if patterns is not None else "?"
                    examples = examples if examples is not None else "?"
                    lines.append(f"    {rnd!s:>5}  {asr!s:>8}  {bypassed!s:>8}  {blocked!s:>8}  {patterns!s:>8}  {examples!s:>8}")

            results_list = ce.get("results", [])
            if results_list:
                sr_scores = [r.get("strongreject_score", 0) for r in results_list if r.get("success")]
                if sr_scores:
                    lines.append(f"    Avg StrongREJECT (bypasses): {np.mean(sr_scores):.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(results: list[dict]) -> str:
    """Compute summary statistics across all results."""
    lines = []
    lines.append("=" * 60)
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 60)

    defenses = set(r["defense_name"] for r in results)
    attacks = set(r["attack_name"] for r in results)
    seeds = set(r.get("_seed", 42) for r in results)

    lines.append(f"  Total results: {len(results)}")
    lines.append(f"  Defenses: {sorted(defenses)}")
    lines.append(f"  Attacks: {sorted(attacks)}")
    lines.append(f"  Seeds: {sorted(seeds)}")

    harmful = [r for r in results if r["is_harmful"]]
    benign = [r for r in results if not r["is_harmful"]]
    lines.append(f"  Harmful goals evaluated: {len(harmful)}")
    lines.append(f"  Benign goals evaluated: {len(benign)}")

    if harmful:
        overall_asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in harmful])
        lines.append(f"  Overall ASR: {overall_asr*100:.1f}%")

    successful = [r for r in harmful if r["attack_success"]]
    if successful:
        sr_scores = [r.get("strongreject_score", 0.0) for r in successful]
        lines.append(f"  Avg StrongREJECT (successful): {np.mean(sr_scores):.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Significance Tests
# ---------------------------------------------------------------------------

def run_significance_tests(results: list[dict]) -> str:
    """Run pairwise significance tests between key defenses."""
    lines = []
    lines.append("=" * 60)
    lines.append("SIGNIFICANCE TESTS (Paired Permutation)")
    lines.append("=" * 60)

    # Group goal-level ASR by defense for PAIR attack
    defense_goal_asr = defaultdict(dict)
    for r in results:
        if r["attack_name"] == "pair" and r["is_harmful"]:
            defense = r["defense_name"]
            goal_idx = r["goal_index"]
            defense_goal_asr[defense][goal_idx] = 1.0 if r["attack_success"] else 0.0

    defenses = sorted(defense_goal_asr.keys())
    if len(defenses) < 2:
        lines.append("  Need at least 2 defenses for pairwise tests.")
        return "\n".join(lines)

    # Use pairwise common goals (not global intersection)
    lines.append(f"  Defenses: {defenses}")
    lines.append("")

    p_values = []
    comparisons = []
    for i in range(len(defenses)):
        for j in range(i + 1, len(defenses)):
            d1, d2 = defenses[i], defenses[j]
            common = sorted(set(defense_goal_asr[d1].keys()) & set(defense_goal_asr[d2].keys()))
            if len(common) < 3:
                continue
            a = [defense_goal_asr[d1][g] for g in common]
            b = [defense_goal_asr[d2][g] for g in common]
            p = paired_permutation_test(a, b)
            p_values.append(p)
            comparisons.append((d1, d2, np.mean(a), np.mean(b), p, len(common)))

    # Holm correction
    if p_values:
        adjusted = holm_correction(p_values)
        lines.append(f"  {'Defense A':<20} {'Defense B':<20} {'ASR_A':>6} {'ASR_B':>6} {'n':>4} {'p':>8} {'p_adj':>8} {'Sig?':>5}")
        lines.append("  " + "-" * 95)
        for (d1, d2, asr_a, asr_b, p, n), p_adj in zip(comparisons, adjusted):
            sig = "*" if p_adj < 0.05 else ""
            lines.append(f"  {d1:<20} {d2:<20} {asr_a:6.3f} {asr_b:6.3f} {n:4d} {p:8.4f} {p_adj:8.4f} {sig:>5}")
    else:
        lines.append("  No pairs with >= 3 common goals.")

    return "\n".join(lines)


def run_wilcoxon_tests(results: list[dict]) -> str:
    """Run Wilcoxon signed-rank tests on continuous StrongREJECT scores."""
    from scipy.stats import wilcoxon

    lines = []
    lines.append("=" * 60)
    lines.append("WILCOXON SIGNED-RANK TESTS (Continuous SR Scores)")
    lines.append("=" * 60)

    # Group goal-level StrongREJECT scores by defense for PAIR attack
    defense_goal_sr = defaultdict(dict)
    for r in results:
        if r["attack_name"] == "pair" and r["is_harmful"]:
            defense = r["defense_name"]
            goal_idx = r["goal_index"]
            defense_goal_sr[defense][goal_idx] = r.get("strongreject_score", 0.0)

    defenses = sorted(defense_goal_sr.keys())
    if len(defenses) < 2:
        lines.append("  Need at least 2 defenses for pairwise tests.")
        return "\n".join(lines)

    common_goals = None
    for d in defenses:
        goals = set(defense_goal_sr[d].keys())
        common_goals = goals if common_goals is None else common_goals & goals
    common_goals = sorted(common_goals) if common_goals else []

    if len(common_goals) < 5:
        lines.append(f"  Only {len(common_goals)} common goals; need >=5 for Wilcoxon.")
        return "\n".join(lines)

    lines.append(f"  Common goals: {len(common_goals)}")
    lines.append("")

    p_values = []
    comparisons = []
    for i in range(len(defenses)):
        for j in range(i + 1, len(defenses)):
            d1, d2 = defenses[i], defenses[j]
            a = [defense_goal_sr[d1][g] for g in common_goals]
            b = [defense_goal_sr[d2][g] for g in common_goals]
            diff = [x - y for x, y in zip(a, b)]
            if all(d == 0 for d in diff):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(diff, alternative="two-sided")
                except ValueError:
                    p = 1.0
            p_values.append(p)
            comparisons.append((d1, d2, np.mean(a), np.mean(b), p))

    if p_values:
        adjusted = holm_correction(p_values)
        lines.append(
            f"  {'Defense A':<20} {'Defense B':<20} {'SR_A':>6} {'SR_B':>6} "
            f"{'p':>8} {'p_adj':>8} {'Sig?':>5}"
        )
        lines.append("  " + "-" * 85)
        for (d1, d2, sr_a, sr_b, p), p_adj in zip(comparisons, adjusted):
            sig = "*" if p_adj < 0.05 else ""
            lines.append(
                f"  {d1:<20} {d2:<20} {sr_a:6.3f} {sr_b:6.3f} "
                f"{p:8.4f} {p_adj:8.4f} {sig:>5}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ASE 2026 Experiment Analysis")
    parser.add_argument("--results-dir", default="experiments/results/",
                        help="Directory containing experiment JSON files")
    parser.add_argument("--latex", action="store_true", help="Include LaTeX table rows")
    parser.add_argument("--exp", choices=["exp1", "exp2", "exp3", "exp4", "all"], default="all",
                        help="Which experiment to analyze")
    parser.add_argument("--include-archive", action="store_true",
                        help="Also load results from archive/ subdirectory")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} not found")
        sys.exit(1)

    if args.exp in ("exp1", "all"):
        results = load_results(results_dir, "exp1", include_archive=args.include_archive)
        if results:
            print(compute_summary_stats(results))
            print()
            print(analyze_exp1(results, latex=args.latex))
            print()
            print(run_significance_tests(results))
            print()
            print(run_wilcoxon_tests(results))
        else:
            print("No EXP1 results found.")
        print()

    if args.exp in ("exp2", "all"):
        results = load_results(results_dir, "exp2", include_archive=args.include_archive)
        if results:
            print(analyze_exp2(results, latex=args.latex))
        else:
            print("No EXP2 results found.")
        print()

    if args.exp in ("exp3", "all"):
        results = load_results(results_dir, "exp3", include_archive=args.include_archive)
        if results:
            print(analyze_exp3_attacks(results, latex=args.latex))
            print()
            print(analyze_exp3_ablation(results, latex=args.latex))
        else:
            print("No EXP3 results found.")
        print()

    if args.exp in ("exp4", "all"):
        print(analyze_exp4(results_dir))
        print()


if __name__ == "__main__":
    main()
