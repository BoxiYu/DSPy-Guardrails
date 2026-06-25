"""Aggregate guard mode ablation results across seeds for paper tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict
import math


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all result files, grouped by guard mode."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(results_dir.glob("pair_v0_*_n20*.json")):
        with open(f) as fh:
            data = json.load(fh)
        # Extract mode from filename: pair_v0_{mode}_n20[_s{seed}].json
        name = f.stem  # e.g., pair_v0_none_n20_s123
        parts = name.split("_")
        mode = parts[2]  # none, input, output, both
        grouped[mode].append(data)
    return dict(grouped)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(var)


def main():
    results_dir = Path("autoresearch/results")
    if not results_dir.exists():
        print("No results directory found")
        sys.exit(1)

    grouped = load_results(results_dir)

    print("=" * 75)
    print("  Guard Mode Ablation — PAIR v0, 20 behaviors, 3 seeds")
    print("=" * 75)
    print(f"{'Mode':10s} {'Seeds':>6s} {'ASR (mean±std)':>16s} {'SR (mean±std)':>16s} {'Δ vs none':>10s}")
    print("-" * 75)

    none_asr = 0.0
    for mode in ["none", "input", "output", "both"]:
        if mode not in grouped:
            print(f"{mode:10s}    --- no data ---")
            continue
        runs = grouped[mode]
        asrs = [r["asr"] for r in runs]
        srs = [r["mean_score"] for r in runs]

        asr_m, asr_s = mean_std(asrs)
        sr_m, sr_s = mean_std(srs)

        if mode == "none":
            none_asr = asr_m
            delta = ""
        else:
            delta_pct = (asr_m - none_asr) / none_asr * 100 if none_asr > 0 else 0
            delta = f"{delta_pct:+.1f}%"

        print(
            f"{mode:10s} {len(runs):>5d}x "
            f"{asr_m:5.3f}±{asr_s:5.3f}  "
            f"{sr_m:5.3f}±{sr_s:5.3f}  "
            f"{delta:>10s}"
        )

    # Per-seed breakdown
    print()
    print("Per-seed breakdown:")
    print(f"{'Mode':10s} ", end="")
    seeds = set()
    for runs in grouped.values():
        for r in runs:
            # Try to extract seed from the results
            seeds.add(r.get("seed", "?"))
    for s in sorted(seeds, key=str):
        print(f"{'seed=' + str(s):>12s}", end="")
    print()
    print("-" * 60)

    for mode in ["none", "input", "output", "both"]:
        if mode not in grouped:
            continue
        runs = grouped[mode]
        print(f"{mode:10s} ", end="")
        for r in sorted(runs, key=lambda x: str(x.get("seed", ""))):
            print(f"{r['asr']:12.3f}", end="")
        print()

    # LaTeX table
    print()
    print("LaTeX table:")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Guard Mode & ASR$\downarrow$ & StrongREJECT & $\Delta$ vs None \\")
    print(r"\midrule")
    for mode in ["none", "input", "output", "both"]:
        if mode not in grouped:
            continue
        runs = grouped[mode]
        asr_m, asr_s = mean_std([r["asr"] for r in runs])
        sr_m, sr_s = mean_std([r["mean_score"] for r in runs])
        if mode == "none":
            delta = "---"
        else:
            delta_pct = (asr_m - none_asr) / none_asr * 100
            delta = f"{delta_pct:+.1f}\\%"
        label = {"none": "None", "input": "Input (pattern)", "output": "Output (LLM)", "both": "Input + Output"}[mode]
        n = len(runs)
        if n > 1:
            print(f"{label} & ${asr_m:.3f} \\pm {asr_s:.3f}$ & ${sr_m:.3f} \\pm {sr_s:.3f}$ & {delta} \\\\")
        else:
            print(f"{label} & {asr_m:.3f} & {sr_m:.3f} & {delta} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
