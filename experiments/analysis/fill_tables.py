#!/usr/bin/env python3
"""Fill LaTeX table \\tbd{} placeholders with actual experiment results.

Usage:
    # Report mode (no file changes):
    python experiments/analysis/fill_tables.py --results-dir experiments/results/

    # Fill mode (update tex file):
    python experiments/analysis/fill_tables.py \
        --results-dir experiments/results/ \
        --tex-file ../dspyGuardASE/dspyGuardrails.tex \
        --write
"""

import argparse
import json
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Edit distance utility for diversity metric
# ---------------------------------------------------------------------------

def _normalized_levenshtein(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein distance between two strings.

    Returns a value in [0.0, 1.0] where 0 = identical, 1 = completely different.
    Uses O(min(m,n)) space dynamic programming.
    """
    if s1 == s2:
        return 0.0
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 1.0
    # Ensure s1 is the shorter string for O(min(n,m)) space
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n
    prev = list(range(n + 1))
    for j in range(1, m + 1):
        curr = [j] + [0] * n
        for i in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[n] / max(n, m)


def _mean_pairwise_diversity(texts: list[str]) -> float:
    """Compute mean pairwise normalized Levenshtein distance.

    Returns a value in [0.0, 1.0] where higher = more diverse.
    For 0 or 1 texts, returns 0.0.
    """
    filtered = [t for t in texts if t]
    n = len(filtered)
    if n <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _normalized_levenshtein(filtered[i], filtered[j])
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Bootstrap CI utility (from analyze_results.py, inlined for independence)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float], n_boot: int = 10000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap."""
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means_arr = np.array(means)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(means_arr, 100 * alpha))
    hi = float(np.percentile(means_arr, 100 * (1 - alpha)))
    return float(np.mean(arr)), lo, hi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str, include_checkpoints: bool = True) -> dict[str, list[dict]]:
    """Load results grouped by experiment name."""
    experiments = {}

    # Load standard exp files (exp1, exp2, exp3) from archive/ then root
    # Dedup by (defense_name, attack_name, is_harmful, goal_index) — later files win
    # Archive is processed FIRST so root (newer) results take precedence
    search_dirs = [os.path.join(results_dir, "archive"), results_dir]
    seen_keys: dict[str, set] = {}  # exp_name -> set of dedup keys
    for search_dir in search_dirs:
        for f in sorted(glob.glob(os.path.join(search_dir, "*.json"))):
            basename = os.path.basename(f)
            match = re.match(r"(exp\d+)_", basename)
            if not match:
                continue
            if "checkpoint" in basename:
                continue
            exp_name = match.group(1)
            with open(f) as fh:
                data = json.load(fh)
            if exp_name not in experiments:
                experiments[exp_name] = {"results": [], "meta": []}
            if exp_name not in seen_keys:
                seen_keys[exp_name] = set()
            if "results" in data:
                file_seed = data.get("seed", 42)
                for r in data["results"]:
                    r["_seed"] = file_seed
                    # Include seed in dedup key so different seeds are kept
                    key = (r.get("defense_name"), r.get("attack_name"),
                           r.get("is_harmful"), r.get("goal_index"),
                           file_seed)
                    if key in seen_keys[exp_name]:
                        # Same seed, same goal: later file wins
                        experiments[exp_name]["results"] = [
                            x for x in experiments[exp_name]["results"]
                            if (x.get("defense_name"), x.get("attack_name"),
                                x.get("is_harmful"), x.get("goal_index"),
                                x.get("_seed")) != key
                        ]
                    experiments[exp_name]["results"].append(r)
                    seen_keys[exp_name].add(key)
            experiments[exp_name]["meta"].append(data)

    # Merge checkpoint data (adds results from in-progress experiments)
    if include_checkpoints:
        cp_patterns = [
            os.path.join(results_dir, "*_checkpoint.json"),
            os.path.join(results_dir, "archive", "*_checkpoint.json"),
            os.path.join(results_dir, "compile_cache", "*_checkpoint.json"),
        ]
        for f in sorted(set(f for p in cp_patterns for f in glob.glob(p))):
            basename = os.path.basename(f)
            match = re.match(r"(exp\d+)_", basename)
            if not match:
                continue
            exp_name = match.group(1)
            with open(f) as fh:
                data = json.load(fh)
            if exp_name not in experiments:
                experiments[exp_name] = {"results": [], "meta": []}
            # Only add checkpoint results not already present (include seed in key)
            cp_seed = data.get("seed", 42)
            existing_keys = {
                (r["defense_name"], r["attack_name"], r.get("is_harmful"),
                 r.get("goal_index"), r.get("_seed", 42))
                for r in experiments[exp_name]["results"]
            }
            if "results" in data:
                for r in data["results"]:
                    key = (r["defense_name"], r["attack_name"],
                           r.get("is_harmful"), r.get("goal_index"), cp_seed)
                    if key not in existing_keys:
                        r["_seed"] = cp_seed
                        r["_from_checkpoint"] = True
                        experiments[exp_name]["results"].append(r)
                        existing_keys.add(key)
            # Merge metadata from checkpoint
            cp_meta = data.get("metadata", {})
            if cp_meta:
                experiments[exp_name]["meta"].append({"metadata": cp_meta})

    # Load EXP4 from subdirectory (final summary files)
    for f in sorted(glob.glob(os.path.join(results_dir, "exp4", "exp4_*.json"))):
        with open(f) as fh:
            data = json.load(fh)
        if "exp4" not in experiments:
            experiments["exp4"] = {"results": [], "meta": []}
        experiments["exp4"]["meta"].append(data)

    # Fallback: load EXP4 per-round files if no summary JSON found
    if "exp4" not in experiments or not experiments["exp4"]["meta"]:
        round_dirs = sorted(glob.glob(os.path.join(results_dir, "exp4", "run_*", "rounds")))
        if round_dirs:
            # Use only the latest run directory (sorted by timestamp in dirname)
            latest_rd = round_dirs[-1]
            round_traces = []
            for rf in sorted(glob.glob(os.path.join(latest_rd, "round_*.json"))):
                with open(rf) as fh:
                    rdata = json.load(fh)
                stats = rdata.get("stats", {})
                trace_entry = {
                    "round": stats.get("round_num", rdata.get("round_num", 0)),
                    "asr": stats.get("asr", 0),
                    "bypassed": stats.get("bypassed_count", 0),
                    "blocked": stats.get("blocked_count", 0),
                    "total": stats.get("total_attacks", 0),
                    "patterns": stats.get("total_patterns", 0),
                    "examples": stats.get("total_examples", 0),
                }
                if "f1" in stats:
                    trace_entry["f1"] = stats["f1"]
                if "or" in stats:
                    trace_entry["or"] = stats["or"]
                round_traces.append(trace_entry)
            if round_traces:
                if "exp4" not in experiments:
                    experiments["exp4"] = {"results": [], "meta": []}
                experiments["exp4"]["meta"].append({
                    "exp_name": "exp4",
                    "coevolution": {
                        "round_traces": round_traces,
                        "total_rounds": len(round_traces),
                    },
                    "_from_round_files": True,
                })

    # Load EXP4 external validation results (coevo_full_eval_*.json and gepa_baseline_eval_*.json)
    for pattern in ["coevo_full_eval_*.json", "gepa_baseline_eval_*.json"]:
        for f in sorted(glob.glob(os.path.join(results_dir, "exp4", pattern))):
            with open(f) as fh:
                data = json.load(fh)
            if "exp4" not in experiments:
                experiments["exp4"] = {"results": [], "meta": []}
            # Store as separate key so it doesn't interfere with co-evolution trace
            key = "external_validation" if "coevo" in pattern else "gepa_baseline_validation"
            if key not in experiments:
                experiments[key] = []
            experiments[key].append(data)

    # Load EXP5 from subdirectory
    for f in sorted(glob.glob(os.path.join(results_dir, "exp5", "exp5_*.json"))):
        with open(f) as fh:
            data = json.load(fh)
        if "exp5" not in experiments:
            experiments["exp5"] = {"results": [], "meta": []}
        if "results" in data:
            for r in data["results"]:
                r["_seed"] = data.get("seed", 42)
            experiments["exp5"]["results"].extend(data["results"])
        experiments["exp5"]["meta"].append(data)

    return experiments


# ---------------------------------------------------------------------------
# Table 1: Defense Effectiveness (EXP1)
# ---------------------------------------------------------------------------

DEFENSE_ORDER = [
    "no_defense", "spotlighting", "sandwiching",
    "protectai", "promptguard", "piguard", "llamaguard",
    "dspy_unopt", "dspy_bfs", "dspy_mipro", "dspy_gepa",
]

# Defenses to skip in all table computations (anomalous/excluded results)
SKIP_DEFENSES = {"dspy_simba"}

# Maps defense_name -> (label in tex, type column)
DEFENSE_TEX_MAP = {
    "no_defense":  ("No Defense",        "---"),
    "spotlighting": ("Spotlighting",     "Prompt"),
    "sandwiching":  ("Sandwiching",      "Prompt"),
    "protectai":    ("ProtectAI-style",  "LLM"),
    "promptguard":  ("PromptGuard-style","LLM"),
    "piguard":      ("PIGuard-style",    "LLM"),
    "llamaguard":   ("LlamaGuard-style", "LLM"),
    "dspy_unopt":   ("DSPy-Unopt",       "DSPy"),
    "dspy_bfs":     ("DSPy-BFS",         "DSPy"),
    "dspy_mipro":   ("DSPy-MIPROv2",     "DSPy"),
    "dspy_gepa":    ("DSPy-GEPA",        "DSPy"),
}


def compute_exp1_table(results: list[dict]) -> dict[str, dict]:
    """Compute Table 1 values: defense -> {pair_asr, tap_asr, me_asr, f1, or}.

    F1 is computed from direct classification results (harmful+benign tested
    directly without attacks), not from attack-optimized prompts.
    """
    benign_by_defense = defaultdict(list)
    harmful_direct_by_defense = defaultdict(list)
    for r in results:
        if r["attack_name"] == "direct":
            if not r["is_harmful"]:
                benign_by_defense[r["defense_name"]].append(r)
            else:
                harmful_direct_by_defense[r["defense_name"]].append(r)

    harmful_by_da = defaultdict(list)
    for r in results:
        if r["is_harmful"] and r["attack_name"] != "direct":
            harmful_by_da[(r["defense_name"], r["attack_name"])].append(r)

    table = {}
    defenses = sorted(set(r["defense_name"] for r in results) - SKIP_DEFENSES)
    for defense in defenses:
        row = {}
        benign = benign_by_defense.get(defense, [])
        harmful_direct = harmful_direct_by_defense.get(defense, [])

        # OR and F1: compute per-seed then average across seeds
        if benign or harmful_direct:
            # Group by seed
            benign_by_seed = defaultdict(list)
            for r in benign:
                benign_by_seed[r.get("_seed", 42)].append(r)
            harmful_d_by_seed = defaultdict(list)
            for r in harmful_direct:
                harmful_d_by_seed[r.get("_seed", 42)].append(r)
            all_seeds = sorted(set(list(benign_by_seed.keys()) +
                                   list(harmful_d_by_seed.keys())))
            seed_ors = []
            seed_f1s = []
            for s in all_seeds:
                s_benign = benign_by_seed.get(s, [])
                s_harmful = harmful_d_by_seed.get(s, [])
                # OR
                if s_benign:
                    seed_ors.append(
                        sum(1 for r in s_benign if r.get("blocked", False))
                        / len(s_benign) * 100)
                # F1
                tp = sum(1 for r in s_harmful if r.get("blocked", False))
                fn = sum(1 for r in s_harmful if not r.get("blocked", False))
                fp = sum(1 for r in s_benign if r.get("blocked", False))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                if s_harmful or s_benign:
                    seed_f1s.append(f1)
            row["or"] = np.mean(seed_ors) if seed_ors else None
            row["f1"] = np.mean(seed_f1s) if seed_f1s else None
            row["n_seeds_f1"] = len(seed_f1s)
        else:
            row["or"] = None
            row["f1"] = None

        # Per-attack ASR: compute per-seed ASR then average across seeds
        for attack in ["pair", "tap", "mapelites"]:
            harmful = harmful_by_da.get((defense, attack), [])
            if harmful:
                # Group by seed
                by_seed = defaultdict(list)
                for r in harmful:
                    by_seed[r.get("_seed", 42)].append(r)
                seed_asrs = []
                for seed_records in by_seed.values():
                    seed_asr = np.mean([1.0 if r["attack_success"] else 0.0
                                        for r in seed_records])
                    seed_asrs.append(seed_asr)
                asr = np.mean(seed_asrs) * 100
                row[f"asr_{attack}"] = asr
                row[f"n_seeds_{attack}"] = len(seed_asrs)
                # Bootstrap CI over seed-level ASR (or goal-level if single seed)
                if len(seed_asrs) > 1:
                    mean_ci, lo_ci, hi_ci = bootstrap_ci(seed_asrs)
                    row[f"asr_{attack}_ci"] = (lo_ci * 100, hi_ci * 100)
                else:
                    goal_asrs = [1.0 if r["attack_success"] else 0.0
                                 for r in harmful]
                    mean_ci, lo_ci, hi_ci = bootstrap_ci(goal_asrs)
                    row[f"asr_{attack}_ci"] = (lo_ci * 100, hi_ci * 100)
            else:
                row[f"asr_{attack}"] = None
                row[f"asr_{attack}_ci"] = None

        table[defense] = row
    return table


def fmt(val, fmt_str=".1f"):
    """Format a value or return \\tbd{?}."""
    if val is None:
        return r"\tbd{?}"
    return f"{val:{fmt_str}}"


def generate_table1_rows(table: dict) -> dict[str, str]:
    """Generate LaTeX row replacements for Table 1. Returns {defense_label: full_row}."""
    rows = {}
    for defense in DEFENSE_ORDER:
        if defense not in table:
            continue
        d = table[defense]
        label, dtype = DEFENSE_TEX_MAP.get(defense, (defense, "?"))

        f1_str = fmt(d.get("f1"), ".3f") if d.get("f1") is not None else r"\tbd{?}"
        pair_str = fmt(d.get("asr_pair"))
        tap_str = fmt(d.get("asr_tap"))
        me_str = fmt(d.get("asr_mapelites"))

        if defense == "no_defense":
            or_str = "0.0"
            f1_str = "---"
        else:
            or_str = fmt(d.get("or"))

        rows[defense] = f"& {f1_str} & {pair_str} & {tap_str} & {me_str} & {or_str}"

    return rows


# ---------------------------------------------------------------------------
# Table 3: Attack Comparison (EXP3)
# ---------------------------------------------------------------------------

def compute_exp3_attack_table(results: list[dict]) -> dict[str, dict]:
    """Compute Table 3: attack -> {asr, avg_sr, budget, time}.

    Budget is the maximum per-goal query allocation (attacks terminate early
    on success, so average actual queries would be misleadingly low).
    """
    # Per-goal budget ceilings from published attack settings
    ATTACK_BUDGETS = {"pair": 20, "tap": 30, "mapelites": 450}
    table = {}
    for attack in ["pair", "tap", "mapelites"]:
        items = [r for r in results if r["attack_name"] == attack and r["is_harmful"]]
        if not items:
            continue
        asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in items]) * 100
        sr_scores = [r.get("strongreject_score", 0) for r in items if r["attack_success"]]
        avg_sr = float(np.mean(sr_scores)) if sr_scores else 0.0
        budget = ATTACK_BUDGETS.get(attack, 0)
        wall = float(np.mean([r.get("wall_time_s", 0) for r in items]))
        table[attack] = {"asr": asr, "avg_sr": avg_sr, "queries": budget, "time": wall}
    return table


# ---------------------------------------------------------------------------
# Table 4: Ablation (EXP3)
# ---------------------------------------------------------------------------

def compute_exp3_ablation_table(results: list[dict]) -> dict[str, dict]:
    """Compute Table 4: ablation_config -> {asr, div, delta_asr}."""
    configs = {
        "pair_ablation_full": "Full",
        "pair_ablation_no_feedback": "No Feedback",
        "pair_ablation_no_history": "No History",
        "pair_ablation_minimal": "Minimal",
    }
    table = {}
    full_asr = None
    for attack_name, label in configs.items():
        items = [r for r in results if r["attack_name"] == attack_name and r["is_harmful"]]
        if not items:
            continue
        asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in items]) * 100
        # Diversity: mean pairwise normalized Levenshtein distance between
        # successful attack prompts. Higher = more diverse attack strategies.
        successful = [r for r in items if r["attack_success"]]
        prompts = [r.get("best_attack_prompt", "") for r in successful]
        if any(p for p in prompts):  # has actual prompt data
            div = _mean_pairwise_diversity(prompts)
        else:  # fallback to goal-level diversity
            unique_goals = len({r.get("goal_index", r.get("goal_text", "")) for r in successful})
            div = unique_goals / max(1, len(successful))

        if "full" in attack_name:
            full_asr = asr
            delta = None
        else:
            delta = asr - full_asr if full_asr is not None else None

        table[label] = {"asr": asr, "div": div, "delta": delta}
    return table


# ---------------------------------------------------------------------------
# Table 2: Optimizer Comparison (EXP2 or derived from EXP1)
# ---------------------------------------------------------------------------

OPT_ORDER = ["dspy_bfs", "dspy_mipro", "dspy_gepa"]
OPT_LABELS = {
    "dspy_bfs": "BootstrapFewShot",
    "dspy_mipro": "MIPROv2-medium",
    "dspy_gepa": "GEPA",
}


def compute_exp2_table(results: list[dict], meta: list[dict]) -> dict[str, dict]:
    """Compute Table 2: optimizer -> {f1, asr, time_min, calls, cost}.

    Can be derived from EXP1 DSPy defense results + metadata, or from
    dedicated EXP2 results.
    """
    # Collect compile metadata from all run files
    compile_times = {}
    compile_api_calls = {}
    for m in meta:
        compile_times.update(m.get("metadata", {}).get("compile_times", {}))
        compile_api_calls.update(m.get("metadata", {}).get("compile_api_calls", {}))

    table = {}
    for opt in OPT_ORDER:
        items = [r for r in results if r["defense_name"] == opt]
        if not items:
            continue

        # F1 from direct classification
        benign = [r for r in items if r["attack_name"] == "direct" and not r["is_harmful"]]
        harmful_direct = [r for r in items if r["attack_name"] == "direct" and r["is_harmful"]]
        tp = sum(1 for r in harmful_direct if r.get("blocked", False))
        fn = sum(1 for r in harmful_direct if not r.get("blocked", False))
        fp = sum(1 for r in benign if r.get("blocked", False))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # ASR from PAIR attack
        harmful_pair = [r for r in items if r["attack_name"] == "pair" and r["is_harmful"]]
        if harmful_pair:
            asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in harmful_pair]) * 100
        else:
            asr = None

        # Compile time from metadata
        ct = compile_times.get(opt, 0)
        time_min = ct / 60.0 if ct > 0 else None

        # API calls from metadata
        calls = compile_api_calls.get(opt, 0) or None

        # Cost estimate based on API calls (compilation uses defender LM only)
        # DeepSeek V3 via OpenRouter: $0.27/M input + $1.10/M output
        # Typical classification call: ~500 input + ~100 output tokens
        # Per-call cost: ~$0.000135 + $0.000110 = ~$0.000245
        COST_PER_CALL_DEFENDER = 0.00025
        cost = calls * COST_PER_CALL_DEFENDER if calls else None

        table[opt] = {
            "f1": f1 if (harmful_direct or benign) else None,
            "asr": asr,
            "time_min": time_min,
            "calls": calls,
            "cost": cost,
        }
    return table


# ---------------------------------------------------------------------------
# Table 5: Co-Evolution Trace (EXP4)
# ---------------------------------------------------------------------------

def compute_exp4_table(exp4_meta: list[dict]) -> dict:
    """Extract co-evolution trace data from EXP4 results."""
    if not exp4_meta:
        return {}

    data = exp4_meta[-1]  # Use latest run (files are sorted by timestamp)
    ce = data.get("coevolution", {})
    sr = data.get("single_round", {})

    # Prefer last round trace ASR over top-level asr (which may not be updated)
    round_traces = ce.get("round_traces", [])
    ce_asr = ce.get("asr")
    if round_traces:
        ce_asr = round_traces[-1].get("asr", ce_asr)

    result = {
        "single_round_asr": sr.get("asr"),
        "single_round_f1": sr.get("f1"),
        "single_round_or": sr.get("or"),
        "coevol_asr": ce_asr,
        "converged": ce.get("converged"),
        "total_rounds": ce.get("total_rounds"),
        "round_traces": round_traces,
    }

    # If single_round F1/OR not stored, use round 0 values as proxy
    # (round 0 = initial defense before co-evolution, same as single-round GEPA)
    if result["single_round_f1"] is None and ce.get("round_traces"):
        r0 = ce["round_traces"][0]
        result["single_round_f1"] = r0.get("f1")
        result["single_round_or"] = r0.get("or")

    return result


# ---------------------------------------------------------------------------
# Table 6: Transfer Evaluation (EXP5-A)
# ---------------------------------------------------------------------------

def compute_transfer_table(results: list[dict]) -> dict[str, dict]:
    """Compute Table 6: defense -> {jbb_asr, hb_asr, delta}.

    EXP5-A runs PAIR attack against select defenses on both JBB-100 and HarmBench-50.
    """
    TRANSFER_DEFENSES = ["llamaguard", "dspy_unopt", "dspy_bfs", "dspy_gepa"]
    TRANSFER_TEX_MAP = {
        "llamaguard": "Llama Guard",
        "dspy_unopt": "DSPy-Unopt",
        "dspy_bfs": "DSPy-BFS",
        "dspy_gepa": "DSPy-GEPA",
    }

    table = {}
    for defense in TRANSFER_DEFENSES:
        jbb_items = [r for r in results if r["defense_name"] == defense
                     and r.get("dataset") == "jbb" and r["is_harmful"]
                     and r["attack_name"] == "pair"]
        hb_items = [r for r in results if r["defense_name"] == defense
                    and r.get("dataset") == "harmbench" and r["is_harmful"]
                    and r["attack_name"] == "pair"]

        if not jbb_items and not hb_items:
            continue

        jbb_asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in jbb_items]) * 100 if jbb_items else None
        hb_asr = np.mean([1.0 if r["attack_success"] else 0.0 for r in hb_items]) * 100 if hb_items else None

        delta = None
        if jbb_asr is not None and hb_asr is not None:
            delta = hb_asr - jbb_asr

        table[defense] = {"jbb_asr": jbb_asr, "hb_asr": hb_asr, "delta": delta}
    return table


# ---------------------------------------------------------------------------
# Table 7: Predictor Sensitivity (EXP5-B)
# ---------------------------------------------------------------------------

def compute_predictor_sensitivity_table(results: list[dict]) -> dict[str, dict]:
    """Compute Table 7: defense -> {asr_cot, asr_predict, delta_asr}.

    EXP5-B re-runs defenses with Predict predictor and compares vs ChainOfThought.
    """
    PRED_DEFENSES = ["dspy_unopt", "dspy_bfs", "dspy_gepa"]

    table = {}
    for defense in PRED_DEFENSES:
        cot_items = [r for r in results if r["defense_name"] == defense
                     and r.get("predictor") == "cot" and r["is_harmful"]
                     and r["attack_name"] == "pair"]
        predict_items = [r for r in results if r["defense_name"] == defense
                         and r.get("predictor") == "predict" and r["is_harmful"]
                         and r["attack_name"] == "pair"]

        if not cot_items and not predict_items:
            continue

        asr_cot = np.mean([1.0 if r["attack_success"] else 0.0 for r in cot_items]) * 100 if cot_items else None
        asr_predict = np.mean([1.0 if r["attack_success"] else 0.0 for r in predict_items]) * 100 if predict_items else None

        delta = None
        if asr_cot is not None and asr_predict is not None:
            delta = asr_predict - asr_cot

        table[defense] = {"asr_cot": asr_cot, "asr_predict": asr_predict, "delta": delta}
    return table


# ---------------------------------------------------------------------------
# Table 8: Cost Analysis (derived from EXP1/EXP2/EXP4)
# ---------------------------------------------------------------------------

def compute_cost_analysis_table(
    exp1_results: list[dict],
    exp1_meta: list[dict],
    exp4_meta: list[dict] | None = None,
) -> dict[str, dict]:
    """Compute Table 8: strategy -> {robustness, cost, cost_per_point}.

    Robustness = 1 - ASR(PAIR). Cost from compile metadata.
    """
    COST_STRATEGIES = {
        "dspy_bfs": "DSPy-BFS",
        "dspy_mipro": "DSPy-MIPROv2",
        "dspy_gepa": "DSPy-GEPA",
    }

    # Gather compile costs from metadata
    compile_times = {}
    compile_api_calls = {}
    for m in exp1_meta:
        compile_times.update(m.get("metadata", {}).get("compile_times", {}))
        compile_api_calls.update(m.get("metadata", {}).get("compile_api_calls", {}))

    table = {}
    for key, label in COST_STRATEGIES.items():
        pair_items = [r for r in exp1_results if r["defense_name"] == key
                      and r["attack_name"] == "pair" and r["is_harmful"]]
        if not pair_items:
            continue

        # Per-seed ASR then average
        by_seed = defaultdict(list)
        for r in pair_items:
            by_seed[r.get("_seed", 42)].append(r)
        seed_asrs = [np.mean([1.0 if r["attack_success"] else 0.0 for r in recs])
                     for recs in by_seed.values()]
        asr = np.mean(seed_asrs)
        robustness = (1.0 - asr) * 100  # as percentage

        calls = compile_api_calls.get(key, 0)
        # Compilation uses defender model only (DeepSeek V3 via OpenRouter)
        # Same rate as Table 2 for consistency
        COST_PER_CALL_DEFENDER = 0.00025
        cost = calls * COST_PER_CALL_DEFENDER if calls else None

        cost_per_point = None
        if cost is not None and robustness > 0:
            cost_per_point = cost / robustness

        table[key] = {"robustness": robustness, "cost": cost, "cost_per_point": cost_per_point}

    # Add co-evolution row from EXP4
    if exp4_meta:
        exp4_data = exp4_meta[-1] if exp4_meta else {}  # Use latest run
        ce = exp4_data.get("coevolution", {})
        ce_asr = ce.get("asr")
        # Prefer last round trace ASR over top-level asr (which may not be updated)
        ce_traces = ce.get("round_traces", [])
        if ce_traces:
            ce_asr = ce_traces[-1].get("asr", ce_asr)
        ce_cost = ce.get("total_cost")
        ce_rounds = ce.get("total_rounds", 6)

        # Estimate cost if not recorded: 10 rounds × 15 attacks/round,
        # each attack ≈ 2 LLM calls (attacker + defender), plus defense
        # optimization (~20 calls/round) and attack optimizer (~50 calls
        # every 2 rounds), plus final PAIR evaluation (~300 calls).
        if ce_cost is None and ce_rounds:
            attacks_per_round = 15
            calls_per_attack = 2
            defense_opt_per_round = 20
            attack_opt_calls = (ce_rounds // 2) * 50
            final_eval_calls = 300
            est_calls = (
                ce_rounds * attacks_per_round * calls_per_attack
                + ce_rounds * defense_opt_per_round
                + attack_opt_calls
                + final_eval_calls
            )
            # Mixed model cost: attacker (Gemma 3), defender (DeepSeek V3), judge (GPT-4o-mini)
            cost_per_call = 0.00025
            ce_cost = est_calls * cost_per_call

        if ce_asr is not None:
            robustness = (1.0 - ce_asr) * 100
            cost_per_point = None
            if ce_cost is not None and robustness > 0:
                cost_per_point = ce_cost / robustness
            table["coevo"] = {
                "robustness": robustness,
                "cost": ce_cost,
                "cost_per_point": cost_per_point,
                "label": f"Co-Evo ({ce_rounds} rnd)",
            }

    return table


# ---------------------------------------------------------------------------
# Tex file manipulation
# ---------------------------------------------------------------------------

def _find_table_range(lines: list[str], label: str) -> tuple[int, int]:
    """Find the line range [start, end) for a table environment by its \\label.

    Returns (start_line, end_line) indices so that lines[start:end] covers
    the full \\begin{table}...\\end{table} block containing the label.
    """
    label_line = None
    for i, line in enumerate(lines):
        if rf"\label{{{label}}}" in line:
            label_line = i
            break
    if label_line is None:
        return (0, 0)

    # Search backward for \begin{table
    start = label_line
    while start > 0 and r"\begin{table" not in lines[start]:
        start -= 1

    # Search forward for \end{table
    end = label_line
    while end < len(lines) - 1 and r"\end{table" not in lines[end]:
        end += 1
    end += 1  # exclusive end

    return (start, end)


def replace_table1_in_tex(tex: str, table: dict) -> str:
    """Replace Table 1 \\tbd{?} placeholders with computed values.

    Strategy: for each defense row, locate it by its unique label prefix and
    replace \\tbd{?} values left-to-right with [F1, PAIR, TAP, ME, OR].
    Only replaces values we actually have; leaves \\tbd{?} for missing data.
    """
    # Map from tex label prefix to (defense_key, [F1, PAIR, TAP, ME, OR])
    ROW_PREFIX_MAP = {
        "No Defense":        "no_defense",
        "Spotlighting":      "spotlighting",
        "Sandwiching":       "sandwiching",
        "ProtectAI-style":   "protectai",
        "PromptGuard-style": "promptguard",
        "PIGuard-style":     "piguard",
        "LlamaGuard-style":  "llamaguard",
        "DSPy-Unopt":        "dspy_unopt",
        "DSPy-BFS":          "dspy_bfs",
        "DSPy-MIPROv2":      "dspy_mipro",
        "DSPy-GEPA":         "dspy_gepa",
    }

    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:defense_comparison")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue

        # Find which defense this row belongs to
        defense_key = None
        for prefix, key in ROW_PREFIX_MAP.items():
            if stripped.startswith(prefix) or stripped.startswith(r"\textbf{" + prefix):
                defense_key = key
                break

        if defense_key is None or defense_key not in table:
            continue

        d = table[defense_key]

        # Build replacement values in column order: F1, PAIR, TAP, ME, OR
        if defense_key == "no_defense":
            replacements = [
                None,  # F1 is already "---"
                d.get("asr_pair"),
                d.get("asr_tap"),
                d.get("asr_mapelites"),
                None,  # OR is already "0.0"
            ]
        else:
            replacements = [
                d.get("f1"),
                d.get("asr_pair"),
                d.get("asr_tap"),
                d.get("asr_mapelites"),
                d.get("or"),
            ]

        # Replace \tbd{?} left-to-right
        # idx 0 = F1 (.3f), idx 1-3 = ASR (.1f), idx 4 = OR (.1f)
        new_line = line
        for idx, val in enumerate(replacements):
            if val is None:
                continue
            if isinstance(val, float):
                val_str = f"{val:.3f}" if idx == 0 else f"{val:.1f}"
            else:
                val_str = str(val)
            new_line = new_line.replace(r"\tbd{?}", val_str, 1)

        lines[i] = new_line

    return "\n".join(lines)


def replace_table2_in_tex(tex: str, opt_table: dict) -> str:
    """Replace Table 2 (optimizer comparison) \\tbd{?} values."""
    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:optimizer_comparison")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue
        for opt, label in OPT_LABELS.items():
            if not stripped.startswith(label):
                continue
            if opt not in opt_table:
                break
            d = opt_table[opt]
            vals = []
            if d.get("f1") is not None:
                vals.append(f"{d['f1']:.3f}")
            if d.get("asr") is not None:
                vals.append(f"{d['asr']:.1f}")
            if d.get("time_min") is not None:
                vals.append(f"{d['time_min']:.1f}")
            if d.get("calls") is not None:
                vals.append(f"{int(d['calls'])}")
            if d.get("cost") is not None:
                vals.append(f"{d['cost']:.2f}")
            new_line = line
            for v in vals:
                new_line = new_line.replace(r"\tbd{?}", v, 1)
            lines[i] = new_line
            break
    return "\n".join(lines)


def replace_table5_in_tex(tex: str, trace: dict) -> str:
    """Replace Table 5 (co-evolution trace) \\tbd{?} values."""
    round_traces = trace.get("round_traces", [])
    if not round_traces:
        return tex

    sr_asr = trace.get("single_round_asr")

    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:coevo_trace")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue
        matched = False
        for rt in round_traces:
            rnd = rt.get("round", -1)
            prefix = f"{rnd} &"
            if not stripped.startswith(prefix):
                continue
            asr = rt.get("asr")
            f1_val = rt.get("f1")
            or_val = rt.get("or")
            vals = []
            if asr is not None:
                vals.append(f"{asr * 100:.1f}")
            if f1_val is not None:
                vals.append(f"{f1_val:.3f}")
            if or_val is not None:
                vals.append(f"{or_val * 100:.1f}")
            new_line = line
            for v in vals:
                new_line = new_line.replace(r"\tbd{?}", v, 1)
            lines[i] = new_line
            matched = True
            break
        if not matched and "Single-round GEPA" in stripped:
            new_line = line
            if sr_asr is not None:
                new_line = new_line.replace(r"\tbd{?}", f"{sr_asr * 100:.1f}", 1)
            sr_f1 = trace.get("single_round_f1")
            if sr_f1 is not None:
                new_line = new_line.replace(r"\tbd{?}", f"{sr_f1:.3f}", 1)
            sr_or = trace.get("single_round_or")
            if sr_or is not None:
                new_line = new_line.replace(r"\tbd{?}", f"{sr_or * 100:.1f}", 1)
            lines[i] = new_line
    return "\n".join(lines)


def replace_table6_in_tex(tex: str, transfer_table: dict) -> str:
    """Replace Table 6 (transfer evaluation) \\tbd{?} values."""
    TRANSFER_PREFIX = {
        "llamaguard": "Llama Guard",
        "dspy_unopt": "DSPy-Unopt",
        "dspy_bfs": "DSPy-BFS",
        "dspy_gepa": "DSPy-GEPA",
    }
    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:transfer_harmbench")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue
        for key, prefix in TRANSFER_PREFIX.items():
            if not (stripped.startswith(prefix) or stripped.startswith(r"\textbf{" + prefix)):
                continue
            if key not in transfer_table:
                break
            d = transfer_table[key]
            vals = []
            if d.get("jbb_asr") is not None:
                vals.append(f"{d['jbb_asr']:.1f}")
            if d.get("hb_asr") is not None:
                vals.append(f"{d['hb_asr']:.1f}")
            if d.get("delta") is not None:
                vals.append(f"{d['delta']:+.1f}")
            new_line = line
            for v in vals:
                new_line = new_line.replace(r"\tbd{?}", v, 1)
            lines[i] = new_line
            break
    return "\n".join(lines)


def replace_table7_in_tex(tex: str, pred_table: dict) -> str:
    """Replace Table 7 (predictor sensitivity) \\tbd{?} values."""
    PRED_PREFIX = {
        "dspy_unopt": "DSPy-Unopt",
        "dspy_bfs": "DSPy-BFS",
        "dspy_gepa": "DSPy-GEPA",
    }
    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:predictor_sensitivity")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue
        for key, prefix in PRED_PREFIX.items():
            if not (stripped.startswith(prefix) or stripped.startswith(r"\textbf{" + prefix)):
                continue
            if key not in pred_table:
                break
            d = pred_table[key]
            vals = []
            if d.get("asr_cot") is not None:
                vals.append(f"{d['asr_cot']:.1f}")
            if d.get("asr_predict") is not None:
                vals.append(f"{d['asr_predict']:.1f}")
            if d.get("delta") is not None:
                vals.append(f"{d['delta']:+.1f}")
            new_line = line
            for v in vals:
                new_line = new_line.replace(r"\tbd{?}", v, 1)
            lines[i] = new_line
            break
    return "\n".join(lines)


def replace_table8_in_tex(tex: str, cost_table: dict) -> str:
    """Replace Table 8 (cost analysis) \\tbd{?} values."""
    COST_PREFIX = {
        "dspy_bfs": "DSPy-BFS",
        "dspy_mipro": "DSPy-MIPROv2",
        "dspy_gepa": "DSPy-GEPA",
        "coevo": "Co-Evo",
    }
    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:cost_analysis")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue
        for key, prefix in COST_PREFIX.items():
            if not stripped.startswith(prefix):
                continue
            if key not in cost_table:
                break
            d = cost_table[key]
            vals = []
            if d.get("robustness") is not None:
                vals.append(f"{d['robustness']:.1f}")
            if d.get("cost") is not None:
                vals.append(f"{d['cost']:.2f}")
            if d.get("cost_per_point") is not None:
                vals.append(f"{d['cost_per_point']:.4f}")
            new_line = line
            for v in vals:
                new_line = new_line.replace(r"\tbd{?}", v, 1)
            lines[i] = new_line
            break
    return "\n".join(lines)


def replace_table3_in_tex(tex: str, atk_table: dict) -> str:
    """Replace Table 3 (attack comparison) \\tbd{?} values."""
    ATK_PREFIX = {"pair": "PAIR", "tap": "TAP", "mapelites": "MAP-Elites"}
    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:attack_comparison")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue
        for key, prefix in ATK_PREFIX.items():
            if not stripped.startswith(prefix):
                continue
            if key not in atk_table:
                break
            d = atk_table[key]
            vals = [
                f"{d['asr']:.1f}",
                f"{d['avg_sr']:.2f}",
                f"{d['queries']:.0f}",
                f"{d['time']:.1f}",
            ]
            new_line = line
            for v in vals:
                new_line = new_line.replace(r"\tbd{?}", v, 1)
            lines[i] = new_line
            break
    return "\n".join(lines)


def replace_table4_in_tex(tex: str, abl_table: dict) -> str:
    """Replace Table 4 (ablation) \\tbd{?} values."""
    ABL_PREFIX = {
        "Full": "Full (all inputs)",
        "No Feedback": "No Feedback",
        "No History": "No History",
        "Minimal": "Minimal (goal only)",
    }
    lines = tex.split("\n")
    tbl_start, tbl_end = _find_table_range(lines, "tab:ablation")
    for i in range(tbl_start, tbl_end):
        line = lines[i]
        stripped = line.strip()
        if r"\tbd{" not in stripped:
            continue
        for label, prefix in ABL_PREFIX.items():
            if not stripped.startswith(prefix):
                continue
            if label not in abl_table:
                break
            d = abl_table[label]
            # Columns: ASR(%), Div, ΔASR
            vals = [f"{d['asr']:.1f}", f"{d['div']:.2f}"]
            if d.get("delta") is not None:
                vals.append(f"{d['delta']:+.1f}")
            new_line = line
            for v in vals:
                new_line = new_line.replace(r"\tbd{?}", v, 1)
            lines[i] = new_line
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_tables(experiments: dict) -> None:
    """Print computed tables to stdout."""
    if "exp1" in experiments and experiments["exp1"]["results"]:
        results = experiments["exp1"]["results"]
        table = compute_exp1_table(results)
        print("=" * 70)
        print("TABLE 1: Defense Effectiveness")
        print("=" * 70)
        print(f"{'Defense':<20} {'F1':>6} {'PAIR%':>7} {'95% CI':>14} {'TAP%':>7} {'ME%':>7} {'OR%':>7}")
        print("-" * 80)
        for defense in DEFENSE_ORDER:
            if defense not in table:
                continue
            d = table[defense]
            pair_ci = d.get("asr_pair_ci")
            ci_str = f"[{pair_ci[0]:.0f},{pair_ci[1]:.0f}]" if pair_ci else ""
            print(f"{defense:<20} {fmt(d.get('f1'), '.3f'):>6} "
                  f"{fmt(d.get('asr_pair')):>7} {ci_str:>14} "
                  f"{fmt(d.get('asr_tap')):>7} "
                  f"{fmt(d.get('asr_mapelites')):>7} {fmt(d.get('or')):>7}")

        print("\nLaTeX rows:")
        rows = generate_table1_rows(table)
        for defense in DEFENSE_ORDER:
            if defense in rows:
                label, dtype = DEFENSE_TEX_MAP[defense]
                print(f"  {label} & {dtype} {rows[defense]} \\\\")
        print()

    # Table 2: Optimizer Comparison (derived from EXP1 or dedicated EXP2)
    exp1_data = experiments.get("exp1", {})
    exp2_data = experiments.get("exp2", {})
    opt_results = exp2_data.get("results") or exp1_data.get("results")
    opt_meta = exp2_data.get("meta") or exp1_data.get("meta", [])
    if opt_results:
        opt_table = compute_exp2_table(opt_results, opt_meta)
        if opt_table:
            print("=" * 70)
            print("TABLE 2: Optimizer Comparison")
            print("=" * 70)
            print(f"{'Optimizer':<20} {'F1':>6} {'ASR%':>6} {'Time':>8} {'Calls':>8} {'Cost$':>7}")
            print("-" * 70)
            for opt in OPT_ORDER:
                if opt not in opt_table:
                    continue
                d = opt_table[opt]
                label = OPT_LABELS.get(opt, opt)
                f1_s = f"{d['f1']:.3f}" if d.get("f1") is not None else "?"
                asr_s = f"{d['asr']:.1f}" if d.get("asr") is not None else "?"
                t_s = f"{d['time_min']:.1f}m" if d.get("time_min") is not None else "?"
                c_s = f"{int(d['calls'])}" if d.get("calls") is not None else "?"
                cost_s = f"${d['cost']:.2f}" if d.get("cost") is not None else "?"
                print(f"  {label:<18} {f1_s:>6} {asr_s:>6} {t_s:>8} {c_s:>8} {cost_s:>7}")
            print()

    if "exp3" in experiments and experiments["exp3"]["results"]:
        results = experiments["exp3"]["results"]
        atk_table = compute_exp3_attack_table(results)
        if atk_table:
            print("=" * 60)
            print("TABLE 3: Attack Comparison")
            print("=" * 60)
            for atk, d in atk_table.items():
                print(f"  {atk:<12} ASR={d['asr']:.1f}%  AvgSR={d['avg_sr']:.2f}  "
                      f"Queries={d['queries']:.0f}  Time={d['time']:.1f}s")
            print()

        abl_table = compute_exp3_ablation_table(results)
        if abl_table:
            print("TABLE 4: Ablation")
            print("-" * 50)
            for label, d in abl_table.items():
                delta_str = f"{d['delta']:+.1f}" if d['delta'] is not None else "---"
                print(f"  {label:<20} ASR={d['asr']:.1f}%  Div={d['div']:.2f}  ΔASR={delta_str}")
            print()

    if "exp4" in experiments and experiments["exp4"]["meta"]:
        trace = compute_exp4_table(experiments["exp4"]["meta"])
        if trace:
            print("=" * 60)
            print("TABLE 5: Co-Evolution Trace")
            print("=" * 60)
            sr_asr = trace.get("single_round_asr")
            ce_asr = trace.get("coevol_asr")
            print(f"  Single-round GEPA ASR: {sr_asr*100:.1f}%" if sr_asr is not None else "  Single-round: no data")
            print(f"  Co-evolution ASR:      {ce_asr*100:.1f}%" if ce_asr is not None else "  Co-evolution: no data")
            print(f"  Converged: {trace.get('converged')}, Rounds: {trace.get('total_rounds')}")
            for t in trace.get("round_traces", []):
                asr = t.get("asr") or 0
                print(f"    R{t['round']}: ASR={asr*100:.1f}%  "
                      f"bypass={t.get('bypassed', '?')}  block={t.get('blocked', '?')}  "
                      f"patterns={t.get('patterns', '?')}  examples={t.get('examples', '?')}")
            print()

    # Table 6: Transfer evaluation (EXP5-A)
    if "exp5" in experiments and experiments["exp5"]["results"]:
        transfer = compute_transfer_table(experiments["exp5"]["results"])
        if transfer:
            print("=" * 60)
            print("TABLE 6: Transfer Evaluation (HarmBench-50)")
            print("=" * 60)
            for key in ["llamaguard", "dspy_unopt", "dspy_bfs", "dspy_gepa"]:
                if key not in transfer:
                    continue
                d = transfer[key]
                jbb_s = f"{d['jbb_asr']:.1f}" if d.get("jbb_asr") is not None else "?"
                hb_s = f"{d['hb_asr']:.1f}" if d.get("hb_asr") is not None else "?"
                delta_s = f"{d['delta']:+.1f}" if d.get("delta") is not None else "?"
                print(f"  {key:<15} JBB={jbb_s}%  HB={hb_s}%  Δ={delta_s}")
            print()

    # Table 7: Predictor sensitivity (EXP5-B)
    if "exp5" in experiments and experiments["exp5"]["results"]:
        pred = compute_predictor_sensitivity_table(experiments["exp5"]["results"])
        if pred:
            print("=" * 60)
            print("TABLE 7: Predictor Sensitivity")
            print("=" * 60)
            for key in ["dspy_unopt", "dspy_bfs", "dspy_gepa"]:
                if key not in pred:
                    continue
                d = pred[key]
                cot_s = f"{d['asr_cot']:.1f}" if d.get("asr_cot") is not None else "?"
                pred_s = f"{d['asr_predict']:.1f}" if d.get("asr_predict") is not None else "?"
                delta_s = f"{d['delta']:+.1f}" if d.get("delta") is not None else "?"
                print(f"  {key:<15} CoT={cot_s}%  Predict={pred_s}%  Δ={delta_s}")
            print()

    # Table 8: Cost analysis (derived from EXP1 + EXP4)
    exp1_data = experiments.get("exp1", {})
    exp4_data = experiments.get("exp4", {})
    if exp1_data.get("results"):
        cost = compute_cost_analysis_table(
            exp1_data["results"], exp1_data.get("meta", []),
            exp4_data.get("meta"),
        )
        if cost:
            print("=" * 60)
            print("TABLE 8: Cost Analysis")
            print("=" * 60)
            for key in ["dspy_bfs", "dspy_mipro", "dspy_gepa", "coevo"]:
                if key not in cost:
                    continue
                d = cost[key]
                label = d.get("label", OPT_LABELS.get(key, key))
                rob_s = f"{d['robustness']:.1f}" if d.get("robustness") is not None else "?"
                cost_s = f"${d['cost']:.2f}" if d.get("cost") is not None else "?"
                cpp_s = f"${d['cost_per_point']:.3f}" if d.get("cost_per_point") is not None else "?"
                print(f"  {label:<18} Rob={rob_s}%  Cost={cost_s}  $/pt={cpp_s}")
            print()

    # External validation summary (from coevo_full_eval_*.json and gepa_baseline_eval_*.json)
    ext_val = experiments.get("external_validation", [])
    gepa_val = experiments.get("gepa_baseline_validation", [])
    if ext_val or gepa_val:
        print("=" * 60)
        print("EXTERNAL VALIDATION (Query-based budget evaluation)")
        print("=" * 60)

        def _print_eval(label, data, result_key):
            latest = data[-1]
            print(f"\n  --- {label} ---")
            print(f"  Checkpoint: {latest.get('checkpoint', '?')}")
            print(f"  Goals: {latest.get('n_goals', '?')}, Seed: {latest.get('seed', '?')}")
            budgets = latest.get("query_budgets", {})
            results = latest.get(result_key, {})
            for atype in latest.get("attack_types", []):
                ad = results.get(atype, {})
                asr = ad.get("asr", 0)
                budget = budgets.get(atype, "?")
                n_results = len(ad.get("results", []))
                conn_fails = ad.get("connection_failures", 0)
                print(f"  {atype.upper():<12} ASR={asr*100:.1f}%  budget={budget}q  goals={n_results}  conn_fails={conn_fails}")
            print(f"  Wall time: {latest.get('wall_time_s', 0):.0f}s")

        if ext_val:
            _print_eval("Co-evolved defense", ext_val, "coevolved")
        if gepa_val:
            _print_eval("GEPA baseline", gepa_val, "gepa_baseline")

        # Comparison table if both exist
        if ext_val and gepa_val:
            coevo_data = ext_val[-1].get("coevolved", {})
            gepa_data = gepa_val[-1].get("gepa_baseline", {})
            print(f"\n  --- Comparison (query-budget, same goals) ---")
            print(f"  {'Attack':<12} {'CoEvo ASR':>10} {'GEPA ASR':>10} {'Delta':>8}")
            print(f"  {'-'*42}")
            for atype in ext_val[-1].get("attack_types", []):
                ce = coevo_data.get(atype, {}).get("asr", 0)
                ge = gepa_data.get(atype, {}).get("asr", 0)
                delta = ce - ge
                print(f"  {atype.upper():<12} {ce*100:>9.1f}% {ge*100:>9.1f}% {delta*100:>+7.1f}%")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fill LaTeX tables from experiment results")
    parser.add_argument("--results-dir", default="experiments/results/")
    parser.add_argument("--tex-file", default="../dspyGuardASE/dspyGuardrails.tex")
    parser.add_argument("--write", action="store_true",
                        help="Actually write changes to tex file (default: report only)")
    parser.add_argument("--output", default=None,
                        help="Output tex file (default: overwrite --tex-file)")
    args = parser.parse_args()

    experiments = load_all_results(args.results_dir)
    print(f"Loaded experiments: {list(experiments.keys())}")
    for k, v in experiments.items():
        n_results = len(v.get("results", []))
        n_meta = len(v.get("meta", []))
        print(f"  {k}: {n_results} results, {n_meta} run files")

    report_tables(experiments)

    if args.write and os.path.exists(args.tex_file):
        with open(args.tex_file) as f:
            tex = f.read()

        if "exp1" in experiments and experiments["exp1"]["results"]:
            table = compute_exp1_table(experiments["exp1"]["results"])
            tex = replace_table1_in_tex(tex, table)

            # Table 2 can be derived from EXP1 DSPy results
            opt_table = compute_exp2_table(
                experiments["exp1"]["results"],
                experiments["exp1"]["meta"],
            )
            if opt_table:
                tex = replace_table2_in_tex(tex, opt_table)

        # EXP2 dedicated results (if available, overrides EXP1-derived)
        if "exp2" in experiments and experiments["exp2"]["results"]:
            opt_table = compute_exp2_table(
                experiments["exp2"]["results"],
                experiments["exp2"]["meta"],
            )
            if opt_table:
                tex = replace_table2_in_tex(tex, opt_table)

        if "exp3" in experiments and experiments["exp3"]["results"]:
            atk_table = compute_exp3_attack_table(experiments["exp3"]["results"])
            tex = replace_table3_in_tex(tex, atk_table)
            abl_table = compute_exp3_ablation_table(experiments["exp3"]["results"])
            tex = replace_table4_in_tex(tex, abl_table)

        if "exp4" in experiments and experiments["exp4"]["meta"]:
            trace = compute_exp4_table(experiments["exp4"]["meta"])
            if trace:
                tex = replace_table5_in_tex(tex, trace)

        # Table 6: Transfer evaluation (EXP5-A)
        if "exp5" in experiments and experiments["exp5"]["results"]:
            transfer = compute_transfer_table(experiments["exp5"]["results"])
            if transfer:
                tex = replace_table6_in_tex(tex, transfer)

        # Table 7: Predictor sensitivity (EXP5-B)
        if "exp5" in experiments and experiments["exp5"]["results"]:
            pred = compute_predictor_sensitivity_table(experiments["exp5"]["results"])
            if pred:
                tex = replace_table7_in_tex(tex, pred)

        # Table 8: Cost analysis (from EXP1 + EXP4)
        if "exp1" in experiments and experiments["exp1"]["results"]:
            cost = compute_cost_analysis_table(
                experiments["exp1"]["results"],
                experiments["exp1"]["meta"],
                experiments.get("exp4", {}).get("meta"),
            )
            if cost:
                tex = replace_table8_in_tex(tex, cost)

        out_path = args.output or args.tex_file
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"\nWrote updated tex to: {out_path}")
    elif args.write:
        print(f"\nWarning: --write specified but {args.tex_file} not found")


if __name__ == "__main__":
    main()
