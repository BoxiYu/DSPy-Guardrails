#!/usr/bin/env python3
"""Extract training trajectory data from CoEvoOptimizer compilation logs.

Reads cross_eval_results JSON files (which contain coevo_metadata with per-round
stats) and produces CSV and JSON output for EXP2 (Table 2 / Fig 3).

Single-seed usage:
    python scripts/extract_trajectory.py \
      --compile-log experiments/cross_eval_results_seed42.json \
      --seed 42 \
      --output-csv results/tables/trajectory_seed42.csv \
      --output-json results/figures/trajectory_seed42.json

Multi-seed aggregation:
    python scripts/extract_trajectory.py \
      --compile-logs experiments/cross_eval_results_seed42.json \
                     experiments/cross_eval_results_seed123.json \
                     experiments/cross_eval_results_seed456.json \
      --output-csv results/tables/trajectory_aggregated.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev


def extract_rounds(data: dict) -> list[dict]:
    """Extract per-round trajectory from a compile log JSON.

    Returns a list of dicts, one per round (plus round 0 for initial state).
    """
    meta = data.get("coevo_metadata", {})
    rounds_raw = meta.get("rounds", [])
    initial_score = meta.get("initial_score")
    total_attacks = meta.get("total_attacks_generated", 0)

    rows: list[dict] = []

    # Round 0: initial state (pre-optimization)
    initial_instruction_length = meta.get("initial_instruction_length", 0)
    rows.append({
        "round": 0,
        "asr": None,  # no attacks evaluated yet
        "defense_score": initial_score,
        "overrefusal_rate": None,
        "n_attacks_generated": 0,
        "cumulative_attacks": 0,
        "n_attacks_bypassed": 0,
        "instruction_changed": False,
        "instruction_length": initial_instruction_length,
        "n_demos_selected": 0,
        "duration_s": 0.0,
        "mutation_success_rate": None,
    })

    cumulative = 0
    for rs in rounds_raw:
        n_gen = rs.get("n_attacks_generated", 0)
        n_bypassed = rs.get("n_attacks_bypassed", 0)
        cumulative += n_gen

        # Mutation success rate: fraction of generated attacks that bypassed
        mut_rate = n_bypassed / n_gen if n_gen > 0 else 0.0

        rows.append({
            "round": rs["round_num"],
            "asr": rs.get("asr"),
            "defense_score": rs.get("defense_score"),
            "overrefusal_rate": rs.get("overrefusal_rate"),
            "n_attacks_generated": n_gen,
            "cumulative_attacks": cumulative,
            "n_attacks_bypassed": n_bypassed,
            "instruction_changed": rs.get("instruction_changed", False),
            "instruction_length": rs.get("instruction_length", 0),
            "n_demos_selected": rs.get("n_demos_selected", 0),
            "duration_s": rs.get("duration_s", 0.0),
            "mutation_success_rate": mut_rate,
        })

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write trajectory rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to {path}")


def write_json(rows: list[dict], path: Path, seed: int | None = None) -> None:
    """Write trajectory rows to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {"seed": seed, "trajectory": rows}
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON written to {path}")


def aggregate_trajectories(all_rows: list[list[dict]]) -> list[dict]:
    """Aggregate trajectories across seeds (mean +/- std).

    Assumes all trajectories have the same number of rounds.
    """
    n_rounds = min(len(rows) for rows in all_rows)
    n_seeds = len(all_rows)

    aggregated: list[dict] = []
    numeric_keys = [
        "asr", "defense_score", "overrefusal_rate",
        "n_attacks_generated", "cumulative_attacks",
        "n_attacks_bypassed", "instruction_length",
        "n_demos_selected", "duration_s", "mutation_success_rate",
    ]

    for i in range(n_rounds):
        row: dict = {"round": all_rows[0][i]["round"], "n_seeds": n_seeds}

        for key in numeric_keys:
            values = []
            for seed_rows in all_rows:
                v = seed_rows[i].get(key)
                if v is not None:
                    values.append(float(v))

            if values:
                row[f"{key}_mean"] = round(mean(values), 4)
                row[f"{key}_std"] = round(stdev(values), 4) if len(values) > 1 else 0.0
            else:
                row[f"{key}_mean"] = None
                row[f"{key}_std"] = None

        aggregated.append(row)

    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CoEvoOptimizer training trajectory for EXP2."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--compile-log", type=Path,
        help="Single compile log JSON file",
    )
    group.add_argument(
        "--compile-logs", type=Path, nargs="+",
        help="Multiple compile log JSON files (for multi-seed aggregation)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed (for metadata)")
    parser.add_argument("--output-csv", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.compile_log:
        # Single-seed mode
        with open(args.compile_log) as f:
            data = json.load(f)

        rows = extract_rounds(data)
        if not rows:
            print("No round data found in compile log.", file=sys.stderr)
            sys.exit(1)

        print(f"Extracted {len(rows)} rows ({len(rows) - 1} rounds + init)")
        for r in rows:
            rnd = r["round"]
            score = r["defense_score"]
            asr = r["asr"]
            score_str = f"{score:.3f}" if score is not None else "N/A"
            asr_str = f"{asr:.1%}" if asr is not None else "N/A"
            print(f"  Round {rnd}: score={score_str} ASR={asr_str} "
                  f"attacks={r['cumulative_attacks']}")

        if args.output_csv:
            write_csv(rows, args.output_csv)
        if args.output_json:
            write_json(rows, args.output_json, seed=args.seed)

        if not args.output_csv and not args.output_json:
            # Print to stdout as JSON
            json.dump(rows, sys.stdout, indent=2)
            print()

    else:
        # Multi-seed aggregation mode
        all_rows: list[list[dict]] = []
        for log_path in args.compile_logs:
            with open(log_path) as f:
                data = json.load(f)
            rows = extract_rounds(data)
            if rows:
                all_rows.append(rows)
                print(f"  {log_path.name}: {len(rows) - 1} rounds")
            else:
                print(f"  {log_path.name}: no round data, skipping",
                      file=sys.stderr)

        if not all_rows:
            print("No valid compile logs found.", file=sys.stderr)
            sys.exit(1)

        aggregated = aggregate_trajectories(all_rows)
        print(f"\nAggregated {len(all_rows)} seeds, {len(aggregated)} rows:")
        for r in aggregated:
            rnd = r["round"]
            score = r.get("defense_score_mean")
            score_sd = r.get("defense_score_std")
            asr = r.get("asr_mean")
            asr_sd = r.get("asr_std")
            score_str = f"{score:.3f}" if score is not None else "N/A"
            sd_str = f"+/-{score_sd:.3f}" if score_sd is not None else ""
            asr_str = f"{asr:.1%}" if asr is not None else "N/A"
            asr_sd_str = f"+/-{asr_sd:.3f}" if asr_sd is not None else ""
            print(f"  Round {rnd}: score={score_str}{sd_str} "
                  f"ASR={asr_str}{asr_sd_str}")

        if args.output_csv:
            write_csv(aggregated, args.output_csv)
        if args.output_json:
            write_json(aggregated, args.output_json)

        if not args.output_csv and not args.output_json:
            json.dump(aggregated, sys.stdout, indent=2)
            print()


if __name__ == "__main__":
    main()
