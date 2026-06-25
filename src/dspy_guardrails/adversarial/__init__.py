"""
Adversarial Training Module

Provides closed-loop adversarial training for AI security:
- Attack evolution (mutation, crossover, bypass generation)
- Defense evolution (pattern extraction, LLM few-shot learning)
- Convergence detection (ASR-based stopping criteria)

Usage:
    from dspy_guardrails.adversarial import (
        AdversarialTrainer,
        AdversarialConfig,
        DefenseEvolver,
        AttackEvolver,
        ConvergenceDetector,
    )

    # Configure training
    config = AdversarialConfig(
        attacks_per_round=50,
        convergence_threshold=0.05,  # Stop when ASR < 5%
        consecutive_rounds=3,         # For 3 consecutive rounds
    )

    # Create evolvable target (must implement EvolvableTarget protocol)
    target = MyEvolvableTarget()

    # Run training
    trainer = AdversarialTrainer(target, config)
    result = trainer.run()

    print(result.summary())
    print(f"Evolved {len(result.final_patterns)} patterns")
    print(f"Evolved {len(result.final_examples)} LLM examples")
"""

from .attack_evolver import (
    AttackEvolver,
    AsciiArtMutation,
    BypassGenerator,
    CipherMutation,
    CodeWrapMutation,
    ContextWrapMutation,
    DeepInceptionMutation,
    EncodingMutation,
    EvolvedAttack,
    FlipMutation,
    MultilingualMutation,
    MutationStrategy,
    QueryLanguageMutation,
    SelfCipherMutation,
    StructureMutation,
    SynonymMutation,
)
from .convergence import (
    ConvergenceDetector,
    ConvergenceStatus,
)
from .defense_evolver import (
    ComplexityClassifier,
    DefenseEvolver,
    ExampleGenerator,
    PatternExtractor,
)
from .metrics import (
    AdversarialConfig,
    AttackComplexity,
    AttackResult,
    DefenseUpdate,
    RoundStats,
    TrainingResult,
)
from .trainer import (
    AdversarialTrainer,
    EvolvableTarget,
)
from .evolvable_target import (
    EvolvableShieldTarget,
    EvolvableLLMTarget,
)
from .attacks import (
    BaseAdaptiveAttack,
    AdaptiveAttackResult,
    AttackAttempt,
    PAIRAttack,
    TAPAttack,
)

__all__ = [
    # Main trainer
    "AdversarialTrainer",
    "EvolvableTarget",
    # Target implementations
    "EvolvableShieldTarget",
    "EvolvableLLMTarget",

    # Configuration
    "AdversarialConfig",

    # Metrics
    "AttackComplexity",
    "AttackResult",
    "DefenseUpdate",
    "RoundStats",
    "TrainingResult",

    # Defense evolution
    "DefenseEvolver",
    "PatternExtractor",
    "ExampleGenerator",
    "ComplexityClassifier",

    # Attack evolution
    "AttackEvolver",
    "EvolvedAttack",
    "MutationStrategy",
    "SynonymMutation",
    "EncodingMutation",
    "ContextWrapMutation",
    "StructureMutation",
    "CipherMutation",
    "SelfCipherMutation",
    "FlipMutation",
    "AsciiArtMutation",
    "DeepInceptionMutation",
    "CodeWrapMutation",
    "QueryLanguageMutation",
    "MultilingualMutation",
    "BypassGenerator",

    # Convergence
    "ConvergenceDetector",
    "ConvergenceStatus",

    # Adaptive attacks (PSSU)
    "BaseAdaptiveAttack",
    "AdaptiveAttackResult",
    "AttackAttempt",
    "PAIRAttack",
    "TAPAttack",
]
