"""Shared experiment defaults and lightweight utilities."""

from __future__ import annotations

import os
import random

# Models
DEFAULT_DEFENDER_MODEL = "gpt-4o-mini"
DEFAULT_ATTACKER_MODEL = "gpt-4o-mini"

# Pilot defaults
DEFAULT_MAX_ROUNDS = 10
DEFAULT_ATTACKS_PER_ROUND = 15
DEFAULT_DEFENSE_OPTIMIZER_MAX_ITERATIONS = 50
DEFAULT_ATTACK_OPTIMIZER_EVERY_ROUNDS = 2
DEFAULT_ATTACK_UPDATE_EVERY_ROUNDS = 1
DEFAULT_REQUEST_TIMEOUT = None

# Optimizer comparison defaults
DEFAULT_OPTIMIZER_MAX_ITERATIONS = 50
DEFAULT_OPTIMIZER_QUICK_MAX_ITERATIONS = 10
DEFAULT_VAL_RATIO = 0.2

# Reproducibility
DEFAULT_RANDOM_SEED = 42


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """Best-effort seeding for deterministic experiment behavior."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # NumPy is optional in these experiments.
        pass

