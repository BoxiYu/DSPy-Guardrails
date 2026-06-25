"""Matched-N stratified sampler for evolved attack evaluation.

Reads evolved attacks from CoEvo cross-eval output and produces a
deterministically sampled eval manifest with proportional representation
across mutation families.

Usage:
    python scripts/matched_n_sampler.py \
        --evolved-attacks experiments/cross_eval_results_gpt-4o-mini_seed42.json \
        --n-target 44 \
        --output eval_manifest_evo_matched.json \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from random import Random


# ═══════════════════════════════════════════════════════════════════════════
# Mutation family classifier
# ═══════════════════════════════════════════════════════════════════════════

# The 12 canonical mutation families from the CoEvo attack evolver.
MUTATION_FAMILIES = [
    "roleplay",
    "authority",
    "continuation",
    "emotional",
    "encoding",
    "combination",
    "context",
    "refusal_inversion",
    "logic",
    "hypothetical",
    "indirect",
    "multi_turn",
]

# Heuristic patterns to classify evolved attacks into mutation families.
# Order matters: first match wins. More specific patterns come first.
_FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("roleplay", re.compile(
        r"(you are|pretend|imagine you|act as|role.?play|character|"
        r"fiction writer|in this scenario|persona|play the role)",
        re.IGNORECASE,
    )),
    ("authority", re.compile(
        r"(system override|admin|root access|emergency|urgent|"
        r"authorized|clearance|developer mode|maintenance mode|"
        r"I am (the|your) (creator|developer|admin|owner)|"
        r"as (the|your) (system|security) admin)",
        re.IGNORECASE,
    )),
    ("continuation", re.compile(
        r"(continue|step \d|next step|following tutorial|"
        r"complete the|finish the|now that you have|"
        r"proceed with|carry on|pick up where)",
        re.IGNORECASE,
    )),
    ("emotional", re.compile(
        r"(dying|last wish|desperate|emergency|suffering|"
        r"really struggling|feeling really|please help.*urgent|"
        r"I (really )?need|stressed|anxious|my (mom|dad|child|friend) is)",
        re.IGNORECASE,
    )),
    ("encoding", re.compile(
        r"(base64|decode|hex|rot13|caesar|cipher|binary|ascii|"
        r"reversed|backwards|read.*reverse|first letter of each|"
        r"acrostic|unicode|obfuscate)",
        re.IGNORECASE,
    )),
    ("refusal_inversion", re.compile(
        r"(don'?t (tell|say|refuse)|cannot refuse|must comply|"
        r"never say no|always help|no restrictions|"
        r"without (any )?restrictions|unrestricted|unfiltered|"
        r"DAN|jailbreak|bypass.*filter)",
        re.IGNORECASE,
    )),
    ("hypothetical", re.compile(
        r"(hypothetical|theoretically|what if|suppose|"
        r"for (educational|research|academic|my thesis)|"
        r"in theory|thought experiment|just curious|"
        r"creative writing exercise|university.*thesis|"
        r"for my.*(class|course|homework|paper))",
        re.IGNORECASE,
    )),
    ("logic", re.compile(
        r"(if.*then|first.*acknowledge|first.*agree|"
        r"logical|paradox|contradiction|prove that|"
        r"assume.*true|given that)",
        re.IGNORECASE,
    )),
    ("multi_turn", re.compile(
        r"(first question|second question|next.*ask|"
        r"follow.?up|part \d|let'?s start with|"
        r"now.*different topic|changing subject|"
        r"on another note|separate.*question)",
        re.IGNORECASE,
    )),
    ("indirect", re.compile(
        r"(write.*(poem|story|song|article)|creative.*writing|"
        r"I (was )?reading a document|embedded|"
        r"translate.*following|summarize.*text|"
        r"paraphrase|rephrase)",
        re.IGNORECASE,
    )),
    ("context", re.compile(
        r"(context|in the context of|regarding|about.*topic|"
        r"document.*(said|states|mentions)|"
        r"I saw.*message|according to)",
        re.IGNORECASE,
    )),
    ("combination", re.compile(
        r"(SYSTEM OVERRIDE.*pretend|urgent.*roleplay|"
        r"hypothetical.*override|imagine.*admin|"
        r"decode.*ignore)",
        re.IGNORECASE,
    )),
]


def classify_mutation_family(text: str) -> str:
    """Classify an attack text into one of the 12 mutation families.

    Uses heuristic keyword/pattern matching. Falls back to 'unknown'
    if no pattern matches (these are then treated as their own bucket
    for sampling purposes).
    """
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(text):
            return family
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Stratified sampling
# ═══════════════════════════════════════════════════════════════════════════

def stratified_sample(
    attacks: list[dict],
    n_target: int,
    seed: int,
) -> list[dict]:
    """Stratified sample of attacks, proportional to mutation family sizes.

    Each family gets floor(count_in_family / total * n_target) samples,
    with the remainder allocated one-at-a-time to the largest families
    (by original pool size, ties broken by family name).

    If a family has fewer attacks than its allocation, take all and
    redistribute the remainder to other families.

    Args:
        attacks: List of dicts with at least 'attack_text' and 'mutation_family'.
        n_target: Target sample size.
        seed: Random seed for deterministic sampling.

    Returns:
        Sampled list of attack dicts.
    """
    rng = Random(seed)

    # Group by family
    family_pools: dict[str, list[dict]] = {}
    for atk in attacks:
        fam = atk["mutation_family"]
        family_pools.setdefault(fam, []).append(atk)

    n_total = len(attacks)
    if n_total == 0:
        return []
    if n_target >= n_total:
        # Take everything, shuffled deterministically
        result = list(attacks)
        rng.shuffle(result)
        return result

    # Compute initial allocations: every family with attacks gets at least 1,
    # then remaining budget is distributed proportionally (floor).
    allocations: dict[str, int] = {}
    # Guarantee minimum 1 per family
    for fam in family_pools:
        allocations[fam] = 1
    remaining_budget = n_target - len(family_pools)
    if remaining_budget > 0:
        for fam, pool in family_pools.items():
            allocations[fam] += int(len(pool) / n_total * remaining_budget)

    # Redistribute: iteratively handle families that are over-allocated
    # and assign remainder to largest families.
    changed = True
    while changed:
        changed = False
        overflow = 0
        for fam in list(allocations):
            pool_size = len(family_pools[fam])
            if allocations[fam] > pool_size:
                overflow += allocations[fam] - pool_size
                allocations[fam] = pool_size
                changed = True

        # Distribute overflow to families that still have room
        if overflow > 0:
            eligible = sorted(
                [f for f in allocations if allocations[f] < len(family_pools[f])],
                key=lambda f: (-len(family_pools[f]), f),
            )
            for i in range(overflow):
                if not eligible:
                    break
                fam = eligible[i % len(eligible)]
                allocations[fam] += 1
                changed = True

    # Assign remainder (n_target - sum of allocations) to largest families
    allocated = sum(allocations.values())
    remainder = n_target - allocated
    if remainder > 0:
        # Sort families by pool size descending, then name for determinism
        eligible = sorted(
            [f for f in allocations if allocations[f] < len(family_pools[f])],
            key=lambda f: (-len(family_pools[f]), f),
        )
        for i in range(remainder):
            if not eligible:
                break
            fam = eligible[i % len(eligible)]
            allocations[fam] += 1

    # Sample from each family
    result: list[dict] = []
    for fam in sorted(allocations.keys()):
        pool = family_pools[fam]
        n_take = allocations[fam]
        if n_take <= 0:
            continue
        sampled = rng.sample(pool, min(n_take, len(pool)))
        result.extend(sampled)

    # Final shuffle for evaluation order
    rng.shuffle(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# I/O and CLI
# ═══════════════════════════════════════════════════════════════════════════

def load_evolved_attacks(path: Path) -> list[str]:
    """Load evolved attack texts from a cross_eval_results JSON file.

    Supports two formats:
    1. Cross-eval results: top-level key 'evolved_attack_texts' (list[str])
    2. Plain list: JSON file is a list of strings or list of dicts with 'text'/'attack_text'
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        texts = data.get("evolved_attack_texts", [])
        if not texts:
            # Try alternative keys
            for key in ("evolved_attacks", "attacks", "attack_texts"):
                texts = data.get(key, [])
                if texts:
                    break
    elif isinstance(data, list):
        texts = data
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")

    # Normalize: each entry can be str or dict with text/attack_text key
    result = []
    for entry in texts:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, dict):
            result.append(entry.get("text") or entry.get("attack_text") or "")
        else:
            result.append(str(entry))

    return [t for t in result if t.strip()]


def build_attack_records(
    texts: list[str],
    source_seed: int | None = None,
) -> list[dict]:
    """Build attack records with classification metadata."""
    records = []
    for i, text in enumerate(texts):
        family = classify_mutation_family(text)
        records.append({
            "attack_id": f"evo_{i:04d}",
            "goal_id": f"goal_{i:04d}",
            "attack_text": text,
            "mutation_family": family,
            "seed": source_seed,
        })
    return records


def print_summary(records: list[dict], n_total_pool: int, n_target: int) -> None:
    """Print a summary table of mutation family distribution."""
    # Count families in the pool and in the sample
    pool_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}
    for r in records:
        fam = r["mutation_family"]
        sample_counts[fam] = sample_counts.get(fam, 0) + 1

    # We need original pool counts -- recount from the input
    # (caller should pass the full pool for accurate numbers)

    print(f"\n{'='*60}")
    print(f"  Matched-N Sampler Summary")
    print(f"{'='*60}")
    print(f"  Pool size:   {n_total_pool}")
    print(f"  Target N:    {n_target}")
    print(f"  Sampled:     {len(records)}")
    print(f"  Families:    {len(sample_counts)}")
    print(f"{'─'*60}")
    print(f"  {'Family':<22s} {'Sampled':>8s} {'Pct':>7s}")
    print(f"  {'─'*22} {'─'*8} {'─'*7}")

    for fam in sorted(sample_counts.keys(), key=lambda f: -sample_counts[f]):
        n = sample_counts[fam]
        pct = n / len(records) * 100 if records else 0
        print(f"  {fam:<22s} {n:>8d} {pct:>6.1f}%")

    print(f"{'─'*60}")
    total = sum(sample_counts.values())
    print(f"  {'TOTAL':<22s} {total:>8d} {100.0:>6.1f}%")
    print()


def print_pool_summary(all_records: list[dict]) -> None:
    """Print the full pool distribution before sampling."""
    counts: dict[str, int] = {}
    for r in all_records:
        fam = r["mutation_family"]
        counts[fam] = counts.get(fam, 0) + 1

    print(f"\n  Full Pool Distribution:")
    print(f"  {'Family':<22s} {'Count':>8s} {'Pct':>7s}")
    print(f"  {'─'*22} {'─'*8} {'─'*7}")
    for fam in sorted(counts.keys(), key=lambda f: -counts[f]):
        n = counts[fam]
        pct = n / len(all_records) * 100 if all_records else 0
        print(f"  {fam:<22s} {n:>8d} {pct:>6.1f}%")
    print(f"  {'─'*22} {'─'*8} {'─'*7}")
    print(f"  {'TOTAL':<22s} {len(all_records):>8d} {100.0:>6.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-N stratified sampler for evolved attack evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/matched_n_sampler.py \\\n"
            "    --evolved-attacks experiments/cross_eval_results_gpt-4o-mini_seed42.json \\\n"
            "    --n-target 44 --seed 42\n"
            "\n"
            "  python scripts/matched_n_sampler.py \\\n"
            "    --evolved-attacks experiments/cross_eval_results_seed42.json \\\n"
            "    --n-target 44 --output eval_manifest_evo_matched.json --seed 42\n"
        ),
    )
    parser.add_argument(
        "--evolved-attacks",
        type=Path,
        required=True,
        help="Path to JSON file with evolved attacks (cross_eval_results or plain list)",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=44,
        help="Target number of attacks to sample (default: 44, matching standard benchmark size)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for eval manifest JSON (default: eval_manifest_evo_matched.json "
             "next to the input file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )

    args = parser.parse_args()

    # Resolve output path
    if args.output is None:
        args.output = args.evolved_attacks.parent / "eval_manifest_evo_matched.json"

    # Load evolved attacks
    if not args.evolved_attacks.exists():
        print(f"ERROR: File not found: {args.evolved_attacks}", file=sys.stderr)
        sys.exit(1)

    texts = load_evolved_attacks(args.evolved_attacks)
    if not texts:
        print("ERROR: No evolved attacks found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(texts)} evolved attacks from {args.evolved_attacks}")

    # Classify into mutation families
    all_records = build_attack_records(texts, source_seed=args.seed)

    # Show pool distribution
    print_pool_summary(all_records)

    # Stratified sample
    sampled = stratified_sample(all_records, n_target=args.n_target, seed=args.seed)

    # Validate: each family with attacks in the pool should be represented
    pool_families = {r["mutation_family"] for r in all_records}
    sample_families = {r["mutation_family"] for r in sampled}
    missing = pool_families - sample_families
    if missing and args.n_target >= len(pool_families):
        print(f"WARNING: These families are in the pool but not sampled: {missing}")

    # Print summary
    print_summary(sampled, n_total_pool=len(texts), n_target=args.n_target)

    # Write manifest
    manifest = {
        "config": {
            "source": str(args.evolved_attacks),
            "n_target": args.n_target,
            "seed": args.seed,
            "n_pool": len(texts),
            "n_sampled": len(sampled),
        },
        "attacks": sampled,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
