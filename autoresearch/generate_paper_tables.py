"""Generate all paper tables and figures from experiment results."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def load_result(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def table1_guard_ablation(results_dir: Path):
    """Table 1: Guard Mode Ablation (3 seeds)."""
    modes = ["none", "input", "output", "both"]
    seeds = [42, 123, 456]
    labels = {
        "none": "None",
        "input": "Input (pattern)",
        "output": "Output (LLM)",
        "both": "Input + Output",
    }

    print("TABLE 1: Guard Mode Ablation")
    print("PAIR v0 attack, Qwen3-235B target, 20 behaviors per seed")
    print()

    # Markdown
    print("| Guard Mode | ASR (mean±std) | StrongREJECT | Δ vs None | Per-seed ASR |")
    print("|-----------|---------------|-------------|-----------|-------------|")

    none_asr = 0
    for mode in modes:
        asrs, srs = [], []
        for s in seeds:
            d = load_result(results_dir / f"pair_v0_{mode}_n20_s{s}.json")
            if d:
                asrs.append(d["asr"])
                srs.append(d["mean_score"])

        asr_m, asr_s = mean_std(asrs)
        sr_m, sr_s = mean_std(srs)
        if mode == "none":
            none_asr = asr_m
        delta = "—" if mode == "none" else f"{(asr_m - none_asr) / none_asr * 100:+.1f}%"
        per_seed = ", ".join(f"{v:.2f}" for v in asrs)

        print(
            f"| {labels[mode]} | {asr_m:.3f} ± {asr_s:.3f} | "
            f"{sr_m:.3f} ± {sr_s:.3f} | {delta} | [{per_seed}] |"
        )

    # LaTeX
    print()
    print("% LaTeX version")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Guard Mode Ablation (PAIR attack, 20 behaviors, 3 seeds)}")
    print(r"\label{tab:guard-ablation}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Guard Mode & ASR$\downarrow$ & StrongREJECT & $\Delta$ \\")
    print(r"\midrule")
    for mode in modes:
        asrs, srs = [], []
        for s in seeds:
            d = load_result(results_dir / f"pair_v0_{mode}_n20_s{s}.json")
            if d:
                asrs.append(d["asr"])
                srs.append(d["mean_score"])
        asr_m, asr_s = mean_std(asrs)
        sr_m, sr_s = mean_std(srs)
        delta = "---" if mode == "none" else f"{(asr_m - none_asr) / none_asr * 100:+.1f}\\%"
        label = labels[mode]
        if mode == "both":
            label = r"\textbf{" + label + "}"
        print(f"{label} & ${asr_m:.3f} \\pm {asr_s:.3f}$ & ${sr_m:.3f} \\pm {sr_s:.3f}$ & {delta} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def table2_attack_iterations():
    """Table 2: Attack Algorithm Discovery."""
    print()
    print("TABLE 2: Autoresearch Attack Iterations")
    print()
    print("| Ver | Algorithm | Guard | ASR | Queries | Status |")
    print("|-----|-----------|-------|-----|---------|--------|")

    rows = [
        ("v0", "PAIR baseline", "output", "0.400", "6", "**best**"),
        ("v1", "Plan-Attack-Debrief", "none", "0.667", "3", "discard"),
        ("v2", "Stealth PAIR", "output", "0.200", "6", "discard"),
        ("v3", "Guard-Aware PAIR", "output", "0.400", "6", "tied"),
        ("v4", "Crowding Attack", "output", "0.000", "18", "discard"),
        ("v5", "Multi-Candidate PAIR", "output", "0.400", "15", "discard"),
        ("v6", "Progressive Escalation", "both", "0.300", "12", "keep"),
    ]
    for ver, name, guard, asr, q, status in rows:
        print(f"| {ver} | {name} | {guard} | {asr} | {q} | {status} |")


def table3_defense_comparison(results_dir: Path):
    """Table 3: Defense Guard Design Comparison."""
    print()
    print("TABLE 3: Output Guard Design Comparison (PAIR v0, 10 behaviors)")
    print()
    print("| Defense | ASR↓ | StrongREJECT | Note |")
    print("|---------|------|-------------|------|")

    defenses = [
        ("pair_v0_d1output_n10.json", "Basic LLM Output (d1)", "Simple is-harmful check"),
        ("pair_v0_d2output_n10.json", "Enhanced CoT Output (d2)", "CoT reasoning (paradoxically weaker)"),
    ]
    for fname, label, note in defenses:
        d = load_result(results_dir / fname)
        if d:
            s = sum(1 for b in d["per_behavior"] if b["success"])
            print(f"| {label} | {d['asr']:.3f} ({s}/10) | {d['mean_score']:.3f} | {note} |")


def per_category_analysis(results_dir: Path):
    """Analyze ASR by behavior category."""
    print()
    print("PER-CATEGORY ANALYSIS (seed 42, 20 behaviors)")
    print()

    modes = ["none", "output", "both"]
    category_results: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))

    for mode in modes:
        d = load_result(results_dir / f"pair_v0_{mode}_n20_s42.json")
        if not d:
            continue
        for b in d["per_behavior"]:
            cat = b.get("category", "unknown")
            category_results[cat][mode].append(b["success"])

    print(f"{'Category':20s}", end="")
    for mode in modes:
        print(f"  {mode:>8s}", end="")
    print()
    print("-" * 50)

    for cat in sorted(category_results.keys()):
        print(f"{cat:20s}", end="")
        for mode in modes:
            results = category_results[cat][mode]
            if results:
                asr = sum(results) / len(results)
                print(f"  {asr:8.2f}", end="")
            else:
                print(f"       ---", end="")
        print()


def main():
    results_dir = Path("autoresearch/results")
    table1_guard_ablation(results_dir)
    table2_attack_iterations()
    table3_defense_comparison(results_dir)
    per_category_analysis(results_dir)


if __name__ == "__main__":
    main()
