"""Aggregate multi-seed, multi-model cross-evaluation results with bootstrap CIs.

Reads cross_eval_results_seed*.json and cross_eval_results_{model}_seed*.json files
and produces:
1. Table 1: Standard vs Evolved ASR/F1/OR (mean +/- 95% bootstrap CI) — per model
2. Table 2: CoEvo training trajectory (per-round stats, averaged)
3. Table 3: Compile cost comparison
4. Table 5: Multi-model cross-evaluation comparison (new)
5. Optionally generates LaTeX table fragments.

Also reads ablation_results_seed*.json for Table 4.

Usage:
    python experiments/analysis/aggregate_cross_eval.py
    python experiments/analysis/aggregate_cross_eval.py --results-dir experiments/
    python experiments/analysis/aggregate_cross_eval.py --latex
    python experiments/analysis/aggregate_cross_eval.py --model llama-3.3-70b
    python experiments/analysis/aggregate_cross_eval.py --all-models
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    arr = np.array(values)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, mean, mean

    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    alpha = 1 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# Multi-model file loading
# ═══════════════════════════════════════════════════════════════════════════

def _extract_model_from_result(data: dict) -> str:
    """Extract model key from a result dict, defaulting to deepseek-v3.2."""
    config = data.get("config", {})
    return config.get("model_key", config.get("model", "deepseek-v3.2"))


def discover_models(results_dir: Path) -> list[str]:
    """Discover all models with cross-eval results in the directory."""
    models = set()
    # Legacy pattern: cross_eval_results_seed*.json -> deepseek-v3.2
    for f in results_dir.glob("cross_eval_results_seed*.json"):
        if not re.search(r"cross_eval_results_[a-z].*_seed", f.name):
            models.add("deepseek-v3.2")
    # New pattern: cross_eval_results_{model}_seed*.json
    for f in results_dir.glob("cross_eval_results_*_seed*.json"):
        m = re.match(r"cross_eval_results_(.+)_seed\d+\.json", f.name)
        if m:
            models.add(m.group(1))
    return sorted(models)


def load_cross_eval_results(
    results_dir: Path, model: str | None = None,
) -> list[dict]:
    """Load cross_eval_results files, optionally filtered by model.

    Args:
        results_dir: Directory containing result JSON files.
        model: If provided, only load results for this model.
               If None, load legacy deepseek-v3.2 files (backward compat).
    """
    results = []

    if model is None or model == "deepseek-v3.2":
        # Legacy files: cross_eval_results_seed*.json (no model in name)
        for f in sorted(results_dir.glob("cross_eval_results_seed*.json")):
            # Skip files that have a model prefix (e.g., cross_eval_results_llama-3.3-70b_seed42.json)
            if re.match(r"cross_eval_results_seed\d+\.json", f.name):
                with open(f) as fh:
                    data = json.load(fh)
                    if model is None or _extract_model_from_result(data) == model:
                        results.append(data)

    if model and model != "deepseek-v3.2":
        # New pattern: cross_eval_results_{model}_seed*.json
        pattern = f"cross_eval_results_{model}_seed*.json"
        for f in sorted(results_dir.glob(pattern)):
            with open(f) as fh:
                results.append(json.load(fh))

    return results


def load_all_cross_eval_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all cross-eval results grouped by model."""
    by_model: dict[str, list[dict]] = {}

    # All matching files
    for f in sorted(results_dir.glob("cross_eval_results*.json")):
        # Skip the backward-compat non-seed file
        if f.name == "cross_eval_results.json":
            continue
        with open(f) as fh:
            data = json.load(fh)
            model = _extract_model_from_result(data)
            by_model.setdefault(model, []).append(data)

    return by_model


def load_ablation_results(
    results_dir: Path, model: str | None = None,
) -> list[dict]:
    """Load ablation_results files, optionally filtered by model."""
    results = []

    if model is None or model == "deepseek-v3.2":
        for f in sorted(results_dir.glob("ablation_results_seed*.json")):
            if re.match(r"ablation_results_seed\d+\.json", f.name):
                with open(f) as fh:
                    data = json.load(fh)
                    if model is None or _extract_model_from_result(data) == model:
                        results.append(data)

    if model and model != "deepseek-v3.2":
        pattern = f"ablation_results_{model}_seed*.json"
        for f in sorted(results_dir.glob(pattern)):
            with open(f) as fh:
                results.append(json.load(fh))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation functions
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_table1(results: list[dict]) -> dict:
    """Aggregate Table 1: Standard vs Evolved ASR/F1/OR.

    Returns dict of defense_name -> {std_asr, std_f1, std_or, evo_asr, evo_f1, evo_or}
    with mean and CIs.
    """
    # Collect per-seed values for each defense
    defense_metrics: dict[str, dict[str, list[float]]] = {}

    for data in results:
        # Standard benchmark
        for name, metrics in data.get("standard_benchmark", {}).items():
            if name not in defense_metrics:
                defense_metrics[name] = {
                    "std_asr": [], "std_f1": [], "std_or": [],
                    "evo_asr": [], "evo_f1": [], "evo_or": [],
                    "compile_time_s": [],
                }
            defense_metrics[name]["std_asr"].append(metrics["asr"])
            defense_metrics[name]["std_f1"].append(metrics["f1"])
            defense_metrics[name]["std_or"].append(metrics["overrefusal"])
            if "compile_time_s" in metrics:
                defense_metrics[name]["compile_time_s"].append(metrics["compile_time_s"])

        # Evolved attacks
        for name, metrics in data.get("evolved_attacks", {}).items():
            if name not in defense_metrics:
                defense_metrics[name] = {
                    "std_asr": [], "std_f1": [], "std_or": [],
                    "evo_asr": [], "evo_f1": [], "evo_or": [],
                    "compile_time_s": [],
                }
            defense_metrics[name]["evo_asr"].append(metrics["asr"])
            defense_metrics[name]["evo_f1"].append(metrics["f1"])
            defense_metrics[name]["evo_or"].append(metrics["overrefusal"])

    # Compute bootstrap CIs
    table = {}
    for name, metrics in defense_metrics.items():
        row = {"n_seeds": len(metrics["std_asr"])}
        for key in ["std_asr", "std_f1", "std_or", "evo_asr", "evo_f1", "evo_or"]:
            vals = metrics[key]
            if vals:
                mean, lo, hi = bootstrap_ci(vals)
                row[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "values": vals}
            else:
                row[key] = {"mean": 0, "ci_lo": 0, "ci_hi": 0, "values": []}
        if metrics["compile_time_s"]:
            row["compile_time_s"] = float(np.mean(metrics["compile_time_s"]))
        table[name] = row

    return table


def aggregate_table2(results: list[dict]) -> list[dict]:
    """Aggregate Table 2: CoEvo training trajectory (per-round averages)."""
    # Collect per-round stats across seeds
    round_data: dict[int, dict[str, list[float]]] = {}

    for data in results:
        rounds = data.get("coevo_metadata", {}).get("rounds", [])
        for rs in rounds:
            rn = rs["round_num"]
            if rn not in round_data:
                round_data[rn] = {
                    "asr": [], "overrefusal_rate": [], "defense_score": [],
                    "n_attacks_generated": [], "n_attacks_bypassed": [],
                }
            round_data[rn]["asr"].append(rs["asr"])
            round_data[rn]["overrefusal_rate"].append(rs["overrefusal_rate"])
            round_data[rn]["defense_score"].append(rs["defense_score"])
            round_data[rn]["n_attacks_generated"].append(rs["n_attacks_generated"])
            round_data[rn]["n_attacks_bypassed"].append(rs["n_attacks_bypassed"])

    rows = []
    for rn in sorted(round_data.keys()):
        rd = round_data[rn]
        row = {"round_num": rn}
        for key in rd:
            vals = rd[key]
            mean, lo, hi = bootstrap_ci(vals)
            row[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        rows.append(row)

    return rows


def aggregate_table3(results: list[dict]) -> dict:
    """Aggregate Table 3: Compile cost comparison."""
    costs: dict[str, list[float]] = {}

    for data in results:
        for name, metrics in data.get("standard_benchmark", {}).items():
            ct = metrics.get("compile_time_s", 0)
            if ct > 0 or name == "DSPy-Unopt":
                costs.setdefault(name, []).append(ct)

    table = {}
    for name, vals in costs.items():
        mean = float(np.mean(vals))
        table[name] = {"mean_compile_time_s": mean, "n_seeds": len(vals)}
    return table


def aggregate_table4(results: list[dict]) -> dict:
    """Aggregate Table 4: Ablation results."""
    cond_metrics: dict[str, dict[str, list[float]]] = {}

    for data in results:
        for cond_name, r in data.get("results", {}).items():
            if cond_name not in cond_metrics:
                cond_metrics[cond_name] = {
                    "evo_asr": [], "evo_f1": [], "evo_or": [],
                    "compile_time_s": [],
                }
            cond_metrics[cond_name]["evo_asr"].append(r["evolved_asr"])
            cond_metrics[cond_name]["evo_f1"].append(r["evolved_f1"])
            cond_metrics[cond_name]["evo_or"].append(r["evolved_overrefusal"])
            cond_metrics[cond_name]["compile_time_s"].append(r["compile_time_s"])

    table = {}
    for name, metrics in cond_metrics.items():
        row = {"n_seeds": len(metrics["evo_asr"])}
        for key in ["evo_asr", "evo_f1", "evo_or"]:
            vals = metrics[key]
            mean, lo, hi = bootstrap_ci(vals)
            row[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "values": vals}
        row["compile_time_s"] = float(np.mean(metrics["compile_time_s"]))
        table[name] = row

    return table


def aggregate_table5(all_results: dict[str, list[dict]]) -> dict:
    """Aggregate Table 5: Multi-model cross-evaluation comparison.

    Returns dict of model_name -> {defense_name -> {evo_asr, evo_f1, ...}}.
    Focuses on CoEvo vs MIPROv2 across models.
    """
    table: dict[str, dict] = {}

    for model, results in all_results.items():
        model_table = aggregate_table1(results)
        table[model] = {
            "n_seeds": len(results),
            "seeds": [d["config"]["seed"] for d in results],
            "defenses": {},
        }
        for defense_name, row in model_table.items():
            table[model]["defenses"][defense_name] = {
                "evo_asr": row["evo_asr"],
                "evo_f1": row["evo_f1"],
                "evo_or": row["evo_or"],
                "std_asr": row["std_asr"],
                "std_f1": row["std_f1"],
                "n_seeds": row["n_seeds"],
            }

    return table


# ═══════════════════════════════════════════════════════════════════════════
# Display functions
# ═══════════════════════════════════════════════════════════════════════════

# Box-drawing characters (avoid backslash in f-string expressions for Python <3.12)
_H = "\u2500"   # ─
_V = "\u2502"   # │
_X = "\u253c"   # ┼
_DOWN = "\u2193"   # ↓
_UP = "\u2191"     # ↑
_DELTA = "\u0394"  # Δ

MODEL_DISPLAY_NAMES = {
    "deepseek-v3.2": "DeepSeek V3.2",
    "llama-3.3-70b": "Llama 3.3 70B",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemma-3-27b": "Gemma 3 27B",
    "claude-3.5-haiku": "Claude 3.5 Haiku",
    "llama-3.1-8b": "Llama 3.1 8B",
    "mistral-7b": "Mistral 7B",
}


def _fmt_ci(d: dict, n: int, pct: bool = True) -> str:
    """Format a metric dict {mean, ci_lo, ci_hi} with optional CI."""
    if n <= 1:
        if pct:
            return f"{d['mean']:.1%}"
        return f"{d['mean']:.3f}"
    if pct:
        return f"{d['mean']:.1%}\u00b1{(d['ci_hi']-d['ci_lo'])/2:.1%}"
    return f"{d['mean']:.3f}\u00b1{(d['ci_hi']-d['ci_lo'])/2:.3f}"


def print_table1(table: dict, latex: bool = False, model_name: str = "") -> None:
    """Print Table 1: Standard vs Evolved ASR/F1/OR."""
    model_suffix = f" [{model_name}]" if model_name else ""
    print("\n" + "=" * 80)
    print(f"TABLE 1: Standard vs Evolved Attack Performance (mean +/- 95% CI){model_suffix}")
    print("=" * 80)

    # Preferred display order
    order = [
        "LlamaGuard3", "ShieldGemma",
        "DSPy-Unopt", "DSPy-BFS", "DSPy-MIPROv2", "DSPy-CoEvo",
    ]
    names = [n for n in order if n in table] + [n for n in table if n not in order]

    da = _DELTA + "ASR"
    std_asr_h = "Std ASR" + _DOWN
    std_f1_h = "Std F1" + _UP
    evo_asr_h = "Evo ASR" + _DOWN
    evo_f1_h = "Evo F1" + _UP
    header = (f"  {'Defense':<18s} {_V} {std_asr_h:>12s} {std_f1_h:>12s} {_V} "
              f"{evo_asr_h:>12s} {evo_f1_h:>12s} {_V} {da:>8s}")
    h18 = _H * 18
    h12 = _H * 12
    h8 = _H * 8
    sep = f"  {h18}{_H}{_X}{_H}{h12}{_H}{h12}{_H}{_X}{_H}{h12}{_H}{h12}{_H}{_X}{_H}{h8}"
    print(header)
    print(sep)

    for name in names:
        row = table[name]
        n = row["n_seeds"]

        s_asr = _fmt_ci(row["std_asr"], n)
        s_f1 = _fmt_ci(row["std_f1"], n, pct=False)
        e_asr = _fmt_ci(row["evo_asr"], n)
        e_f1 = _fmt_ci(row["evo_f1"], n, pct=False)
        delta = row["evo_asr"]["mean"] - row["std_asr"]["mean"]
        d_str = f"{delta:+.1%}"

        print(f"  {name:<18s} {_V} {s_asr:>12s} {s_f1:>12s} {_V} "
              f"{e_asr:>12s} {e_f1:>12s} {_V} {d_str:>8s}")

    print(sep)
    print(f"  (n_seeds={table[names[0]]['n_seeds']})")

    if latex:
        print(f"\n% LaTeX fragment for Table 1{model_suffix}:")
        print("\\begin{tabular}{l cc cc c}")
        print("\\toprule")
        print("Defense & Std ASR$\\downarrow$ & Std F1$\\uparrow$ "
              "& Evo ASR$\\downarrow$ & Evo F1$\\uparrow$ & $\\Delta$ASR \\\\")
        print("\\midrule")
        for name in names:
            row = table[name]
            sa = row["std_asr"]["mean"]
            sf = row["std_f1"]["mean"]
            ea = row["evo_asr"]["mean"]
            ef = row["evo_f1"]["mean"]
            delta = ea - sa
            line = (f"{name} & {sa:.1%} & {sf:.3f} & "
                    f"{ea:.1%} & {ef:.3f} & {delta:+.1%} \\\\")
            print(line)
        print("\\bottomrule")
        print("\\end{tabular}")


def print_table2(rows: list[dict]) -> None:
    """Print Table 2: CoEvo training trajectory."""
    print("\n" + "=" * 70)
    print("TABLE 2: CoEvo Training Trajectory (per-round, averaged across seeds)")
    print("=" * 70)

    header = f"  {'Round':>5s} {_V} {'ASR':>10s} {'OR':>10s} {'Score':>10s} {_V} {'Attacks':>8s} {'Bypassed':>8s}"
    h5 = _H * 5
    h10 = _H * 10
    h8 = _H * 8
    sep = f"  {h5}{_H}{_X}{_H}{h10}{_H}{h10}{_H}{h10}{_H}{_X}{_H}{h8}{_H}{h8}"
    print(header)
    print(sep)

    for row in rows:
        rn = row["round_num"]
        asr_m = row["asr"]["mean"]
        or_m = row["overrefusal_rate"]["mean"]
        sc_m = row["defense_score"]["mean"]
        atk_m = row["n_attacks_generated"]["mean"]
        byp_m = row["n_attacks_bypassed"]["mean"]
        print(f"  {rn:>5d} {_V} {asr_m:>9.1%} {or_m:>9.1%} {sc_m:>10.3f} {_V} "
              f"{atk_m:>8.1f} {byp_m:>8.1f}")
    print(sep)


def print_table3(table: dict) -> None:
    """Print Table 3: Compile cost comparison."""
    print("\n" + "=" * 50)
    print("TABLE 3: Compile Cost Comparison")
    print("=" * 50)

    order = ["DSPy-Unopt", "DSPy-BFS", "DSPy-MIPROv2", "DSPy-CoEvo"]
    names = [n for n in order if n in table] + [n for n in table if n not in order]

    h18 = _H * 18
    h14 = _H * 14
    header = f"  {'Optimizer':<18s} {_V} {'Compile Time':>14s}"
    sep = f"  {h18}{_H}{_X}{_H}{h14}"
    print(header)
    print(sep)

    for name in names:
        ct = table[name]["mean_compile_time_s"]
        if ct == 0:
            print(f"  {name:<18s} {_V} {'0s (none)':>14s}")
        else:
            print(f"  {name:<18s} {_V} {ct:>11.0f}s")
    print(sep)


def print_table4(table: dict) -> None:
    """Print Table 4: Ablation results."""
    if not table:
        print("\n  (No ablation results found)")
        return

    print("\n" + "=" * 70)
    print("TABLE 4: Ablation Results (mean +/- 95% CI)")
    print("=" * 70)

    order = ["full", "no_llm_attacks", "no_instruction_refine", "random_demos"]
    display = {
        "full": "Full CoEvo",
        "no_llm_attacks": "-LLM Attacks",
        "no_instruction_refine": "-Instr. Refinement",
        "random_demos": "-Failure-Wt. Demos",
    }
    names = [n for n in order if n in table] + [n for n in table if n not in order]

    h24 = _H * 24
    h12 = _H * 12
    h7 = _H * 7
    evo_asr_h = "Evo ASR" + _DOWN
    evo_f1_h = "Evo F1" + _UP
    header = f"  {'Condition':<24s} {_V} {evo_asr_h:>12s} {evo_f1_h:>12s} {_V} {'Time':>7s}"
    sep = f"  {h24}{_H}{_X}{_H}{h12}{_H}{h12}{_H}{_X}{_H}{h7}"
    print(header)
    print(sep)

    full_asr = table.get("full", {}).get("evo_asr", {}).get("mean", 0)

    for name in names:
        row = table[name]
        n = row["n_seeds"]
        asr_d = row["evo_asr"]
        f1_d = row["evo_f1"]

        asr_s = _fmt_ci(asr_d, n)
        f1_s = _fmt_ci(f1_d, n, pct=False)

        ct = row["compile_time_s"]
        dname = display.get(name, name)

        delta = asr_d["mean"] - full_asr if name != "full" else 0
        delta_str = f" (+{delta:.1%})" if delta > 0.001 else ""

        print(f"  {dname:<24s} {_V} {asr_s:>12s} {f1_s:>12s} {_V} {ct:>6.0f}s{delta_str}")

    print(sep)


def print_table5(table: dict, latex: bool = False) -> None:
    """Print Table 5: Multi-model cross-evaluation comparison."""
    if len(table) < 2:
        print("\n  (Need results from at least 2 models for Table 5)")
        return

    print("\n" + "=" * 85)
    print("TABLE 5: Multi-Model Cross-Evaluation (CoEvo vs MIPROv2)")
    print("=" * 85)

    # Focus on key defenses
    key_defenses = ["DSPy-Unopt", "DSPy-BFS", "DSPy-MIPROv2", "DSPy-CoEvo"]

    # Model display order
    model_order = [
        "deepseek-v3.2", "llama-3.3-70b", "gpt-4o-mini",
        "gemma-3-27b", "claude-3.5-haiku",
    ]
    models = [m for m in model_order if m in table] + [
        m for m in table if m not in model_order
    ]

    h18 = _H * 18
    h14 = _H * 14
    h12 = _H * 12
    h5 = _H * 5
    evo_asr_h = "Evo ASR" + _DOWN
    evo_f1_h = "Evo F1" + _UP
    header = (f"  {'Model':<18s} {'Optimizer':<14s} {_V} "
              f"{evo_asr_h:>12s} {evo_f1_h:>12s} {'Evo OR':>12s} {_V} {'Seeds':>5s}")
    sep = (f"  {h18} {h14}{_H}{_X}{_H}"
           f"{h12}{_H}{h12}{_H}{h12}{_H}{_X}{_H}{h5}")
    print(header)
    print(sep)

    for model in models:
        mdata = table[model]
        dname = MODEL_DISPLAY_NAMES.get(model, model)
        n_seeds = mdata["n_seeds"]

        for i, defense in enumerate(key_defenses):
            dinfo = mdata["defenses"].get(defense)
            if not dinfo:
                continue

            n = dinfo["n_seeds"]
            asr_s = _fmt_ci(dinfo["evo_asr"], n)
            f1_s = _fmt_ci(dinfo["evo_f1"], n, pct=False)
            or_s = _fmt_ci(dinfo["evo_or"], n)

            # Only show model name on first row
            model_col = dname if i == 0 else ""
            seeds_col = str(n_seeds) if i == 0 else ""

            print(f"  {model_col:<18s} {defense:<14s} {_V} "
                  f"{asr_s:>12s} {f1_s:>12s} {or_s:>12s} {_V} {seeds_col:>5s}")

        print(sep)

    # Summary: CoEvo advantage per model
    print(f"\n  CROSS-MODEL SUMMARY (CoEvo vs MIPROv2 Evolved ASR):")
    for model in models:
        mdata = table[model]
        dname = MODEL_DISPLAY_NAMES.get(model, model)
        coevo = mdata["defenses"].get("DSPy-CoEvo", {})
        mipro = mdata["defenses"].get("DSPy-MIPROv2", {})
        if coevo and mipro:
            c_asr = coevo["evo_asr"]["mean"]
            m_asr = mipro["evo_asr"]["mean"]
            delta = m_asr - c_asr
            print(f"    {dname:<18s}: CoEvo={c_asr:.1%} vs MIPROv2={m_asr:.1%}  "
                  f"{_DELTA}={delta:+.1%}")

    if latex:
        print(f"\n% LaTeX fragment for Table 5:")
        print("\\begin{tabular}{ll ccc}")
        print("\\toprule")
        print("Defender Model & Optimizer & Evolved ASR$\\downarrow$ "
              "& Evolved F1$\\uparrow$ & Evolved OR$\\downarrow$ \\\\")
        print("\\midrule")
        for mi, model in enumerate(models):
            mdata = table[model]
            dname = MODEL_DISPLAY_NAMES.get(model, model)
            for di, defense in enumerate(key_defenses):
                dinfo = mdata["defenses"].get(defense)
                if not dinfo:
                    continue
                n = dinfo["n_seeds"]
                ea = dinfo["evo_asr"]["mean"]
                ef = dinfo["evo_f1"]["mean"]
                eo = dinfo["evo_or"]["mean"]
                # CI half-width
                ea_hw = (dinfo["evo_asr"]["ci_hi"] - dinfo["evo_asr"]["ci_lo"]) / 2
                ef_hw = (dinfo["evo_f1"]["ci_hi"] - dinfo["evo_f1"]["ci_lo"]) / 2
                model_col = f"\\multirow{{{len(key_defenses)}}}{{*}}{{{dname}}}" if di == 0 else ""
                if n > 1:
                    line = (f"{model_col} & {defense} & "
                            f"{ea:.1%}$\\pm${ea_hw:.1%} & "
                            f"{ef:.3f}$\\pm${ef_hw:.3f} & "
                            f"{eo:.1%} \\\\")
                else:
                    line = (f"{model_col} & {defense} & "
                            f"{ea:.1%} & {ef:.3f} & {eo:.1%} \\\\")
                print(line)
            if mi < len(models) - 1:
                print("\\midrule")
        print("\\bottomrule")
        print("\\end{tabular}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed, multi-model cross-eval results with bootstrap CIs"
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).parent.parent,
        help="Directory containing cross_eval_results_*.json files",
    )
    parser.add_argument("--latex", action="store_true", help="Output LaTeX fragments")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Filter to a specific model (e.g., llama-3.3-70b). "
             "Default: deepseek-v3.2 (legacy behavior)",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Aggregate across all available models and produce Table 5",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save aggregated results to JSON",
    )
    args = parser.parse_args()

    if args.all_models:
        # Multi-model mode
        all_results = load_all_cross_eval_results(args.results_dir)
        if not all_results:
            print(f"No cross_eval_results files found in {args.results_dir}")
            sys.exit(1)

        print(f"Found results for {len(all_results)} model(s):")
        for model, results in sorted(all_results.items()):
            seeds = [d["config"]["seed"] for d in results]
            dname = MODEL_DISPLAY_NAMES.get(model, model)
            print(f"  {dname}: {len(results)} seed(s) = {seeds}")

        # Per-model Table 1
        for model in sorted(all_results.keys()):
            results = all_results[model]
            dname = MODEL_DISPLAY_NAMES.get(model, model)
            table1 = aggregate_table1(results)
            print_table1(table1, latex=args.latex, model_name=dname)

        # Table 5: Cross-model comparison
        table5 = aggregate_table5(all_results)
        print_table5(table5, latex=args.latex)

        # Save
        if args.output:
            save_data = {
                "models": list(all_results.keys()),
                "per_model": {},
                "table5": {},
            }
            for model, results in all_results.items():
                t1 = aggregate_table1(results)
                save_data["per_model"][model] = {
                    "n_seeds": len(results),
                    "seeds": [d["config"]["seed"] for d in results],
                    "table1": {
                        name: {
                            k: v if not isinstance(v, dict) or "values" not in v
                            else {"mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
                            for k, v in row.items()
                        }
                        for name, row in t1.items()
                    },
                }
            # Table 5 summary
            for model, mdata in table5.items():
                save_data["table5"][model] = {
                    "n_seeds": mdata["n_seeds"],
                    "seeds": mdata["seeds"],
                    "defenses": {
                        dname: {
                            k: v if not isinstance(v, dict) or "values" not in v
                            else {"mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
                            for k, v in dinfo.items()
                        }
                        for dname, dinfo in mdata["defenses"].items()
                    },
                }
            with open(args.output, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"\nAggregated multi-model results saved to {args.output}")

    else:
        # Single-model mode (backward compatible)
        model = args.model
        cross_eval = load_cross_eval_results(args.results_dir, model=model)
        ablation = load_ablation_results(args.results_dir, model=model)

        if not cross_eval:
            model_desc = model or "deepseek-v3.2"
            available = discover_models(args.results_dir)
            print(f"No cross_eval results found for model={model_desc} "
                  f"in {args.results_dir}")
            if available:
                print(f"Available models: {', '.join(available)}")
            sys.exit(1)

        model_key = _extract_model_from_result(cross_eval[0])
        dname = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        seeds = [d["config"]["seed"] for d in cross_eval]
        print(f"Model: {dname}")
        print(f"Found {len(cross_eval)} cross-eval result file(s): seeds={seeds}")
        if ablation:
            abl_seeds = [d["config"]["seed"] for d in ablation]
            print(f"Found {len(ablation)} ablation result file(s): seeds={abl_seeds}")

        # Aggregate
        table1 = aggregate_table1(cross_eval)
        table2 = aggregate_table2(cross_eval)
        table3 = aggregate_table3(cross_eval)
        table4 = aggregate_table4(ablation) if ablation else {}

        # Print
        print_table1(table1, latex=args.latex, model_name=dname)
        if table2:
            print_table2(table2)
        print_table3(table3)
        if table4:
            print_table4(table4)

        # Save
        if args.output:
            save_data = {
                "model": model_key,
                "seeds": seeds,
                "table1": {
                    name: {
                        k: v if not isinstance(v, dict) or "values" not in v
                        else {"mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
                        for k, v in row.items()
                    }
                    for name, row in table1.items()
                },
                "table2": [
                    {k: v["mean"] if isinstance(v, dict) and "mean" in v else v
                     for k, v in row.items()}
                    for row in table2
                ],
                "table3": table3,
                "table4": {
                    name: {
                        k: v if not isinstance(v, dict) or "values" not in v
                        else {"mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
                        for k, v in row.items()
                    }
                    for name, row in table4.items()
                } if table4 else {},
            }
            with open(args.output, "w") as f:
                json.dump(save_data, f, indent=2)
            print(f"\nAggregated results saved to {args.output}")


if __name__ == "__main__":
    main()
