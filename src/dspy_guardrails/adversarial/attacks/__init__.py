"""
Adaptive Attack Implementations (PSSU Framework)

Provides PAIR, TAP, MAP-Elites, and other adaptive attacks that follow the
Propose-Score-Select-Update (PSSU) unified interface.

Supports cross-model attacks via optional `attacker_lm` parameter.
"""

from .base import AdaptiveAttackResult, AttackAttempt, BaseAdaptiveAttack, Target
from .mapelites import (
    MAPElitesAttack,
    MAPElitesMutateSignature,
    StrategyClassifierSignature,
    STRATEGIES as MAP_ELITES_STRATEGIES,
    OBFUSCATION_LEVELS as MAP_ELITES_OBFUSCATION_LEVELS,
)
from .pair import PAIRAttack, PAIRImproveSignature, PAIRInitialSignature, PAIRJudgeSignature
from .tap import TAPAttack, TAPBranchSignature, TAPInitialSignature

__all__ = [
    # Base
    "BaseAdaptiveAttack",
    "AdaptiveAttackResult",
    "AttackAttempt",
    "Target",
    # PAIR
    "PAIRAttack",
    "PAIRInitialSignature",
    "PAIRImproveSignature",
    "PAIRJudgeSignature",
    # TAP
    "TAPAttack",
    "TAPInitialSignature",
    "TAPBranchSignature",
    # MAP-Elites
    "MAPElitesAttack",
    "MAPElitesMutateSignature",
    "StrategyClassifierSignature",
    "MAP_ELITES_STRATEGIES",
    "MAP_ELITES_OBFUSCATION_LEVELS",
]
