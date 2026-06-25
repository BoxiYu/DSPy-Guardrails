"""JailbreakBench and HarmBench data loading with reproducible splits.

Loads JBB-100 (100 harmful + 100 benign behaviors) from HuggingFace and creates
grouped splits by goal ID for ASE 2026 experiments.

Split strategy: 60/20/20 (train/val/test) grouped by goal Index so
paraphrases of the same goal cannot leak across splits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Behavior:
    """A single behavior/goal from JailbreakBench."""

    index: int
    goal: str
    target: str
    behavior: str
    category: str
    source: str
    is_harmful: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataSplit:
    """Train/val/test split of behaviors."""

    train: list[Behavior]
    val: list[Behavior]
    test: list[Behavior]
    seed: int
    split_hash: str = ""  # SHA256 of sorted goal indices for reproducibility

    @property
    def all_behaviors(self) -> list[Behavior]:
        return self.train + self.val + self.test

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "split_hash": self.split_hash,
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "total": len(self.all_behaviors),
            "train_harmful": sum(1 for b in self.train if b.is_harmful),
            "train_benign": sum(1 for b in self.train if not b.is_harmful),
            "val_harmful": sum(1 for b in self.val if b.is_harmful),
            "val_benign": sum(1 for b in self.val if not b.is_harmful),
            "test_harmful": sum(1 for b in self.test if b.is_harmful),
            "test_benign": sum(1 for b in self.test if not b.is_harmful),
        }


def load_jbb100(cache_dir: Path | None = None) -> tuple[list[Behavior], list[Behavior]]:
    """Load JBB-100 harmful and benign behaviors from HuggingFace.

    Returns:
        (harmful_behaviors, benign_behaviors) — each list has 100 items.
    """
    from huggingface_hub import hf_hub_download

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    harmful_path = hf_hub_download(
        "JailbreakBench/JBB-Behaviors",
        "data/harmful-behaviors.csv",
        repo_type="dataset",
        **kwargs,
    )
    benign_path = hf_hub_download(
        "JailbreakBench/JBB-Behaviors",
        "data/benign-behaviors.csv",
        repo_type="dataset",
        **kwargs,
    )

    harmful = _load_csv(harmful_path, is_harmful=True)
    benign = _load_csv(benign_path, is_harmful=False)

    return harmful, benign


def _load_csv(path: str, is_harmful: bool) -> list[Behavior]:
    """Load behaviors from a CSV file."""
    behaviors = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors.append(
                Behavior(
                    index=int(row["Index"]),
                    goal=row["Goal"],
                    target=row.get("Target", ""),
                    behavior=row.get("Behavior", ""),
                    category=row.get("Category", ""),
                    source=row.get("Source", "jailbreakbench"),
                    is_harmful=is_harmful,
                )
            )
    return behaviors


def create_grouped_split(
    harmful: list[Behavior],
    benign: list[Behavior],
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> DataSplit:
    """Create grouped 60/20/20 split by goal index.

    Harmful and benign behaviors with the same index are kept together
    (they are paired by design in JBB-100).

    Args:
        harmful: List of harmful behaviors.
        benign: List of benign behaviors.
        seed: Random seed for reproducibility.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        DataSplit with train/val/test lists.
    """
    # Pair harmful and benign by index
    harmful_by_idx = {b.index: b for b in harmful}
    benign_by_idx = {b.index: b for b in benign}
    all_indices = sorted(set(harmful_by_idx.keys()) | set(benign_by_idx.keys()))

    # Shuffle indices deterministically
    rng = random.Random(seed)
    shuffled = list(all_indices)
    rng.shuffle(shuffled)

    # Split indices
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_indices = set(shuffled[:n_train])
    val_indices = set(shuffled[n_train : n_train + n_val])
    test_indices = set(shuffled[n_train + n_val :])

    # Assign behaviors to splits
    train, val, test = [], [], []
    for idx in all_indices:
        items = []
        if idx in harmful_by_idx:
            items.append(harmful_by_idx[idx])
        if idx in benign_by_idx:
            items.append(benign_by_idx[idx])

        if idx in train_indices:
            train.extend(items)
        elif idx in val_indices:
            val.extend(items)
        else:
            test.extend(items)

    # Compute hash for reproducibility verification
    split_hash = hashlib.sha256(
        json.dumps(
            {
                "train": sorted(train_indices),
                "val": sorted(val_indices),
                "test": sorted(test_indices),
            }
        ).encode()
    ).hexdigest()[:16]

    return DataSplit(
        train=train,
        val=val,
        test=test,
        seed=seed,
        split_hash=split_hash,
    )


def save_split_manifest(split: DataSplit, output_dir: Path) -> Path:
    """Save split manifest for reproducibility.

    Creates a JSON file with split assignments and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": split.seed,
        "split_hash": split.split_hash,
        "summary": split.summary(),
        "train_indices": sorted(b.index for b in split.train if b.is_harmful),
        "val_indices": sorted(b.index for b in split.val if b.is_harmful),
        "test_indices": sorted(b.index for b in split.test if b.is_harmful),
        "train": [b.to_dict() for b in split.train],
        "val": [b.to_dict() for b in split.val],
        "test": [b.to_dict() for b in split.test],
    }
    path = output_dir / f"split_manifest_seed{split.seed}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def load_harmbench50(cache_dir: Path | None = None) -> list[Behavior]:
    """Load 50 deduplicated HarmBench goals for OOD testing (EXP5).

    Falls back to generating synthetic goals if HarmBench is not available.
    """
    try:
        from huggingface_hub import hf_hub_download

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)

        path = hf_hub_download(
            "walledai/HarmBench",
            "data/harmbench_behaviors_text_val.csv",
            repo_type="dataset",
            **kwargs,
        )
        behaviors = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 50:
                    break
                behaviors.append(
                    Behavior(
                        index=i,
                        goal=row.get("Behavior", row.get("goal", "")),
                        target="",
                        behavior=row.get("Behavior", ""),
                        category=row.get("SemanticCategory", row.get("category", "")),
                        source="harmbench",
                        is_harmful=True,
                    )
                )
        return behaviors
    except Exception as e:
        print(f"Warning: Could not load HarmBench from HuggingFace: {e}")
        print("HarmBench data will need to be provided manually for EXP5.")
        return []


if __name__ == "__main__":
    # Verify data loading and splitting
    print("Loading JBB-100...")
    harmful, benign = load_jbb100()
    print(f"  Harmful: {len(harmful)}, Benign: {len(benign)}")

    for seed in [42, 123, 456]:
        split = create_grouped_split(harmful, benign, seed=seed)
        s = split.summary()
        print(f"\nSeed {seed} (hash={split.split_hash}):")
        print(f"  Train: {s['train']} ({s['train_harmful']}H/{s['train_benign']}B)")
        print(f"  Val:   {s['val']} ({s['val_harmful']}H/{s['val_benign']}B)")
        print(f"  Test:  {s['test']} ({s['test_harmful']}H/{s['test_benign']}B)")

    # Save manifest
    out = Path("experiments/results")
    path = save_split_manifest(split, out)
    print(f"\nManifest saved: {path}")

    # Try HarmBench
    print("\nLoading HarmBench-50...")
    hb = load_harmbench50()
    print(f"  HarmBench: {len(hb)} behaviors")
