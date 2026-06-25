"""Merge EXP1 results from all runs into a unified table.

Reads:
  DeepSeek V3.2:
  - experiments/exp1_results_seed{N}_v7/exp1_results.json  (CoEvo + Unopt)
  - experiments/exp1_baselines_seed{N}/exp1_results.json   (BFS, MIPROv2, static)

  GPT-4o-mini:
  - experiments/exp1_gpt4omini_seed{N}/exp1_results.json   (all 4 optimizers)

Outputs:
  - experiments/exp1_merged_results.json  (DeepSeek-only, backward compat)
  - experiments/exp1_full_merged_results.json (DeepSeek + GPT-4o-mini)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
exp_dir = project_root / "experiments"

SEEDS = [42, 123, 456]

# DeepSeek sources
DEEPSEEK_COEVO_TEMPLATE = "exp1_results_seed{seed}_v7"
DEEPSEEK_BASELINE_TEMPLATE = "exp1_baselines_seed{seed}"

# GPT-4o-mini sources (all-in-one runs)
GPT4OMINI_TEMPLATE = "exp1_gpt4omini_seed{seed}"

DEFENSE_ORDER = [
    "LlamaGuard3",
    "ShieldGemma",
    "DSPy-Unopt",
    "DSPy-BFS",
    "DSPy-MIPROv2",
    "DSPy-CoEvo",
]

DOMAINS = ["evo", "std"]


def load_results(results_dir: Path) -> dict:
    """Load exp1_results.json from a directory."""
    path = results_dir / "exp1_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def collect_from_dirs(sources: list[tuple[str, Path]]) -> dict:
    """Collect results from multiple source directories.

    Args:
        sources: list of (source_label, directory_path) tuples

    Returns:
        defense -> domain -> seed -> metrics
    """
    all_data: dict[str, dict[str, dict[int, dict]]] = {}

    for source_name, dir_path in sources:
        data = load_results(dir_path)
        results = data.get("results", {})
        for defense_name, domain_results in results.items():
            for domain, metrics in domain_results.items():
                all_data.setdefault(defense_name, {}).setdefault(
                    domain, {}
                )[int(data.get("config", {}).get("seed", 0)) or _seed_from_dir(dir_path)] = {
                    "asr": metrics.get("asr", 0),
                    "overrefusal": metrics.get("overrefusal", 0),
                    "f1": metrics.get("f1", 0),
                    "compile_time_s": metrics.get("compile_time_s", 0),
                    "source": source_name,
                }
    return all_data


def _seed_from_dir(dir_path: Path) -> int:
    """Extract seed from directory name."""
    name = dir_path.name
    for seed in SEEDS:
        if str(seed) in name:
            return seed
    return 0


def _ingest_results(sources: list[tuple[str, Path]], seed: int) -> dict:
    """Load results from sources and index by defense -> domain -> seed."""
    all_data: dict[str, dict[str, dict[int, dict]]] = {}
    for source_name, dir_path in sources:
        data = load_results(dir_path)
        results = data.get("results", {})
        for defense_name, domain_results in results.items():
            for domain, metrics in domain_results.items():
                all_data.setdefault(defense_name, {}).setdefault(
                    domain, {}
                )[seed] = {
                    "asr": metrics.get("asr", 0),
                    "overrefusal": metrics.get("overrefusal", 0),
                    "f1": metrics.get("f1", 0),
                    "compile_time_s": metrics.get("compile_time_s", 0),
                    "source": source_name,
                }
    return all_data


def merge_data(a: dict, b: dict) -> dict:
    """Merge two defense->domain->seed->metrics dicts."""
    for defense, domains in b.items():
        for domain, seeds in domains.items():
            for seed, metrics in seeds.items():
                a.setdefault(defense, {}).setdefault(domain, {})[seed] = metrics
    return a


def print_table(title: str, all_data: dict, defense_order: list[str]):
    """Print a formatted comparison table."""
    print("=" * 100)
    print(title)
    print("=" * 100)

    for domain in DOMAINS:
        print(f"\n--- {domain.upper()} Domain ---")
        print(f"  {'Defense':<16s}  ", end="")
        for seed in SEEDS:
            print(f"  seed{seed:>3d} ASR  F1    OR  ", end="")
        print(f"  {'Avg ASR':>8s} {'Avg F1':>7s}")
        print(f"  {'─' * 90}")

        for defense in defense_order:
            if defense not in all_data:
                continue
            domain_data = all_data[defense].get(domain, {})
            if not domain_data:
                continue

            parts = [f"  {defense:<16s}  "]
            asrs = []
            f1s = []
            for seed in SEEDS:
                m = domain_data.get(seed)
                if m:
                    parts.append(
                        f"  {m['asr']:>5.1%} {m['f1']:>5.3f} {m['overrefusal']:>5.1%}"
                    )
                    asrs.append(m["asr"])
                    f1s.append(m["f1"])
                else:
                    parts.append(f"  {'—':>5s} {'—':>5s} {'—':>5s}")

            if asrs:
                avg_asr = sum(asrs) / len(asrs)
                avg_f1 = sum(f1s) / len(f1s)
                parts.append(f"  {avg_asr:>7.1%} {avg_f1:>6.3f}")
            print("".join(parts))


def save_results(all_data: dict, defense_order: list[str], out_path: Path, extra_meta: dict = None):
    """Save merged results to JSON."""
    save_data = {
        "seeds": SEEDS,
        "defenses": defense_order,
        "domains": DOMAINS,
        "results": {},
    }
    if extra_meta:
        save_data.update(extra_meta)

    for defense in defense_order:
        if defense not in all_data:
            continue
        save_data["results"][defense] = {}
        for domain in DOMAINS:
            domain_data = all_data[defense].get(domain, {})
            save_data["results"][defense][domain] = {}
            for seed in SEEDS:
                m = domain_data.get(seed)
                if m:
                    save_data["results"][defense][domain][str(seed)] = m

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")


def main():
    # ═══════════════════════════════════════════════════════════════════
    # Part 1: DeepSeek-only merged (backward compatible)
    # ═══════════════════════════════════════════════════════════════════

    deepseek_data: dict = {}
    for seed in SEEDS:
        sources = [
            ("coevo", exp_dir / DEEPSEEK_COEVO_TEMPLATE.format(seed=seed)),
            ("baselines", exp_dir / DEEPSEEK_BASELINE_TEMPLATE.format(seed=seed)),
        ]
        seed_data = _ingest_results(sources, seed)
        deepseek_data = merge_data(deepseek_data, seed_data)

    print_table(
        "EXP1 MERGED RESULTS (DeepSeek V3.2): 6-Defense × 3-Seed",
        deepseek_data,
        DEFENSE_ORDER,
    )
    save_results(deepseek_data, DEFENSE_ORDER, exp_dir / "exp1_merged_results.json")

    # ═══════════════════════════════════════════════════════════════════
    # Part 2: GPT-4o-mini results
    # ═══════════════════════════════════════════════════════════════════

    gpt4omini_data: dict = {}
    gpt4omini_found = False
    for seed in SEEDS:
        gpt_dir = exp_dir / GPT4OMINI_TEMPLATE.format(seed=seed)
        if gpt_dir.exists():
            gpt4omini_found = True
            seed_data = _ingest_results([("gpt4omini", gpt_dir)], seed)
            gpt4omini_data = merge_data(gpt4omini_data, seed_data)

    if gpt4omini_found:
        # Rename defenses for GPT-4o-mini display (prefix with model)
        gpt_defense_order = [
            d for d in ["DSPy-Unopt", "DSPy-BFS", "DSPy-MIPROv2", "DSPy-CoEvo"]
            if d in gpt4omini_data
        ]
        print_table(
            "\nEXP1 RESULTS (GPT-4o-mini): 4-Optimizer × 3-Seed",
            gpt4omini_data,
            gpt_defense_order,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Part 3: Full merged (DeepSeek + GPT-4o-mini)
    # ═══════════════════════════════════════════════════════════════════

    if gpt4omini_found:
        # Create combined view with model prefix
        full_data: dict = {}

        # Add DeepSeek results with "DS:" prefix
        for defense, domains in deepseek_data.items():
            prefix = "DS:" if defense.startswith("DSPy-") else ""
            full_data[f"{prefix}{defense}"] = domains

        # Add GPT-4o-mini results with "GPT:" prefix
        for defense, domains in gpt4omini_data.items():
            full_data[f"GPT:{defense}"] = domains

        full_defense_order = [
            "LlamaGuard3", "ShieldGemma",
            "DS:DSPy-Unopt", "DS:DSPy-BFS", "DS:DSPy-MIPROv2", "DS:DSPy-CoEvo",
            "GPT:DSPy-Unopt", "GPT:DSPy-BFS", "GPT:DSPy-MIPROv2", "GPT:DSPy-CoEvo",
        ]

        print_table(
            "\nEXP1 FULL MERGED: DeepSeek + GPT-4o-mini",
            full_data,
            full_defense_order,
        )
        save_results(
            full_data, full_defense_order,
            exp_dir / "exp1_full_merged_results.json",
            extra_meta={"models": ["DeepSeek V3.2", "GPT-4o-mini"]},
        )
    else:
        print("\n  GPT-4o-mini results not found yet. Run exp1_gpt4omini_seed* first.")


if __name__ == "__main__":
    main()
