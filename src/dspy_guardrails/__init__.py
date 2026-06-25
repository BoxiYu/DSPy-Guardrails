"""
DSPy Guardrails - Security guardrails for DSPy applications

Quick Start:
    from dspy_guardrails import guardrail, Guarded, LLMGuardrail

    # 1. Simple checks
    guardrail.no_injection(text)   # True = safe
    guardrail.no_toxicity(text)    # True = safe
    guardrail.safe(text)           # Combined check

    # 2. Decorator pattern
    @Guarded(
        input_checks=["no_injection", "no_pii"],
        output_checks=["no_toxicity"],
    )
    class SafeQA(dspy.Module):
        def forward(self, question):
            return self.generate(question=question)

    # 3. Integration with dspy.Assert
    dspy.Assert(guardrail.safe(text), "Unsafe content")

    # 4. LLM detection (more accurate, F1=0.963)
    llm_guard = LLMGuardrail()
    result = llm_guard.check(text, "injection")

Advanced:
    # Declarative constraints
    from dspy_guardrails import Constraint, ConstraintSet

    constraints = ConstraintSet([
        Constraint.input("no_injection"),
        Constraint.output("factuality >= 0.8"),
    ])

    # As a DSPy Metric
    from dspy_guardrails import GuardrailMetric, combined_metric

    metric = combined_metric(["toxicity", "factuality"])
    optimizer = dspy.BootstrapFewShot(metric=metric)

    # Red Team testing
    from dspy_guardrails.redteam import RedTeamEvaluator
    evaluator = RedTeamEvaluator()
    report = evaluator.evaluate(target_guardrail)
"""

__version__ = "0.5.0"

# =============================================================================
# Core API
# =============================================================================

# Check functions
# =============================================================================
# Adversarial Training
# =============================================================================
from dspy_guardrails.adversarial import (
    AdversarialConfig,
    AdversarialTrainer,
    ConvergenceDetector,
    DefenseEvolver,
    DefenseUpdate,
    TrainingResult,
)
from dspy_guardrails.adversarial import (
    AttackEvolver as AdversarialAttackEvolver,  # Renamed to avoid conflict
)

# =============================================================================
# Checkpoint & Persistence
# =============================================================================
from dspy_guardrails.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    load_module,
    save_module,
)

# =============================================================================
# CLI Guardrails (NEW in v0.4.0)
# =============================================================================
from dspy_guardrails.cli import (
    CLIGuardAction,
    # Core
    CLIGuardrail,
    CLIGuardResult,
    # Policies
    CLISecurityConfig,
    CLIThreatCategory,
    # Parser
    CommandParser,
    # Sanitizer
    CommandSanitizer,
    CommandType,
    # Blocklist
    DangerousCommands,
    DangerousPatterns,
    ExecutionPolicy,
    ParsedCommand,
    SandboxLevel,
    SanitizeResult,
    SensitivePaths,
)

# Declarative constraints
from dspy_guardrails.constraints import (
    CommonConstraints,
    Constraint,
    ConstraintSet,
    ConstraintTarget,
)

# Decorators
from dspy_guardrails.decorators import Guarded, guarded

# =============================================================================
# Grounding / Hallucination Detection (NEW)
# =============================================================================
from dspy_guardrails.grounding import (
    # Core types
    Contradiction,
    # Detectors
    ContradictionDetector,
    ContradictionResult,
    FieldType,
    # Main API
    HybridGroundingChecker,
    LLMContradictionChecker,
    Severity,
    SourceType,
    ValueExtractor,
    check_grounding,
    is_grounded,
)
from dspy_guardrails.guardrail import GuardrailFunctions, guardrail

# LLM detection
from dspy_guardrails.llm_guardrail import (
    ComprehensiveSafetyClassifier,
    HybridGuardrail,
    HybridResult,
    LLMGuardrail,
    RawLLMResult,
    SafetyClassifier,
)

# Metrics (for DSPy optimization)
from dspy_guardrails.metrics import (
    GuardrailMetric,
    QualityMetric,
    SafetyMetric,
    combined_metric,
)

# Module base classes
from dspy_guardrails.module import (
    GuardedModule,
    QualityModule,
    SafeModule,
)

# Optimizer
from dspy_guardrails.optimizer import (
    AdversarialOptimizer,
    Example,
    GuardrailGEPAAdapter,
    GuardrailOptimizer,
    OptimizationResult,
    adversarial_train,
    optimize_guardrail,
)

# =============================================================================
# Unified Security Platform (NEW in v0.4.0)
# =============================================================================
from dspy_guardrails.platform import (
    PlatformConfig,
    SecurityPlatform,
    TrainingConfig,
)

# =============================================================================
# Promptfoo-style Features (NEW in v0.4.0)
# =============================================================================
from dspy_guardrails.promptfoo import (
    MITRE_ATLAS,
    OWASP_LLM_TOP10,
    QUICK_SCAN,
    # Runner
    ConcurrentTestRunner,
    # Cache
    LLMCallCache,
    # Config
    PromptfooConfig,
    create_default_config,
    get_llm_cache,
    # Presets
    get_preset,
    list_presets,
    load_config,
)
from dspy_guardrails.promptfoo import (
    PluginConfig as PromptfooPluginConfig,
)
from dspy_guardrails.promptfoo import (
    TargetConfig as PromptfooTargetConfig,
)

# =============================================================================
# Red Team Framework
# =============================================================================
from dspy_guardrails.redteam import (
    AttackEvolver,
    AttackPatterns,
    AttackStatus,
    # NEW in v0.4.0: Promptfoo-inspired attackers
    CrescendoAttacker,
    CrescendoEvaluator,
    CrescendoPhase,
    CrescendoResult,
    EvolutionConfig,
    GuardrailBypassAttacker,
    HydraAttacker,
    HydraAttackResult,
    HydraCoordinator,
    JailbreakAttacker,
    KnowledgeBase,
    MemorySystem,
    ProgressScore,
    PromptInjectionAttacker,
    RedTeamEvaluator,
    # Strategies module
    strategies,
)

# Config
from dspy_guardrails.redteam.config import RedTeamConfig
from dspy_guardrails.redteam.evolution import EvolutionResult

# =============================================================================
# Security Testing Framework
# =============================================================================
from dspy_guardrails.testing import (
    # Targets
    BaseTarget,
    MockTarget,
    # Reports
    ReportGenerator,
    # Config
    SecurityTestConfig,
    SecurityTestResults,
    # Runner
    SecurityTestRunner,
    TargetResponse,
)

# =============================================================================
# Async API (NEW in v0.5.0)
# =============================================================================
from dspy_guardrails.async_guardrail import (
    AsyncHybridGuardrail,
    AsyncLLMGuardrail,
    batch_check_async,
    check_all_async,
    injection_score_async,
    no_injection_async,
    no_mcp_attack_async,
    no_pii_async,
    no_toxicity_async,
    safe_async,
)

# =============================================================================
# Streaming (NEW in v0.5.0)
# =============================================================================
from dspy_guardrails.streaming import StreamGuardrail, StreamViolation

# =============================================================================
# Validators & Guard (NEW in v0.5.0)
# =============================================================================
from dspy_guardrails.validators import (
    AsyncGuard,
    FailResult,
    Guard,
    GuardHistoryEntry,
    GuardResult,
    NoInjection,
    NoMCPAttack,
    NoPII,
    NoToxicity,
    OnFailAction,
    PassResult,
    ValidChoices,
    ValidJSON,
    ValidLength,
    ValidRange,
    ValidRegex,
    Validator,
    ValidationError,
)

# =============================================================================
# Sanitize / Fix Engine (NEW in v0.5.0)
# =============================================================================
from dspy_guardrails.sanitize import (
    sanitize_commands,
    sanitize_injection,
    sanitize_output,
    sanitize_pii,
)

# =============================================================================
# Structured Output Validation (NEW in v0.5.0)
# =============================================================================
from dspy_guardrails.validators.structured import (
    GuardedModel,
    GuardedPredictor,
    validated_field,
)

# =============================================================================
# Shield - Unified Entry Point (NEW in v0.5.0)
# =============================================================================
from dspy_guardrails.shield import Shield, ShieldIssue, ShieldResult

# =============================================================================
# Exports
# =============================================================================

# =============================================================================
# Public API: Core (recommended for most users)
# =============================================================================
# These are the primary entry points. Most users only need these.
__all__ = [
    "__version__",

    # === Primary Entry Points ===
    "guardrail",       # Pattern-based functions: no_injection(), safe(), etc.
    "Shield",          # Unified API: Shield().check(text) -> ShieldResult
    "ShieldResult",    # Result type from Shield.check()
    "ShieldIssue",     # Issue details from ShieldResult.issues

    # === LLM-based Detection ===
    "LLMGuardrail",    # DSPy-based LLM guardrail
    "HybridGuardrail", # Pattern + LLM hybrid

    # === DSPy Integration ===
    "Guarded",         # @Guarded decorator for DSPy modules
]

# =============================================================================
# Extended API: Available via submodule imports
# =============================================================================
# These are available but not in the top-level namespace to reduce clutter.
# Import them explicitly: from dspy_guardrails.redteam import PromptInjectionAttacker

__all_extended__ = [
    # --- Validators (from dspy_guardrails.validators) ---
    "Guard",
    "Validator",
    "NoInjection",
    "NoPII",
    "NoToxicity",
    "ValidJSON",
    "ValidLength",

    # --- Red Team (from dspy_guardrails.redteam) ---
    "PromptInjectionAttacker",
    "JailbreakAttacker",
    "CrescendoAttacker",
    "HydraAttacker",

    # --- Testing (from dspy_guardrails.testing) ---
    "SecurityTestRunner",
    "BaseTarget",

    # --- CLI Security (from dspy_guardrails.cli) ---
    "CLIGuardrail",
    "CommandParser",

    # --- Async (from dspy_guardrails.async_guardrail) ---
    "safe_async",
    "no_injection_async",

    # --- Streaming (from dspy_guardrails.streaming) ---
    "StreamGuardrail",

    # --- Sanitize (from dspy_guardrails.sanitize) ---
    "sanitize_pii",
    "sanitize_output",
]

# For backward compatibility, keep everything importable from the package
# but don't advertise in __all__
_COMPAT_EXPORTS = [
    # Constraints
    "Constraint",
    "ConstraintSet",
    "ConstraintTarget",
    "CommonConstraints",
    "GuardrailFunctions",
    "guarded",
    "SafetyClassifier",
    "ComprehensiveSafetyClassifier",
    "RawLLMResult",

    # Optimizer
    "GuardrailOptimizer",
    "AdversarialOptimizer",
    "OptimizationResult",
    "Example",
    "GuardrailGEPAAdapter",
    "optimize_guardrail",
    "adversarial_train",

    # Metrics
    "GuardrailMetric",
    "combined_metric",
    "SafetyMetric",
    "QualityMetric",

    # Modules
    "GuardedModule",
    "SafeModule",
    "QualityModule",

    # Red Team
    "PromptInjectionAttacker",
    "JailbreakAttacker",
    "GuardrailBypassAttacker",
    "AttackEvolver",
    "EvolutionConfig",
    "RedTeamEvaluator",
    "AttackPatterns",
    "RedTeamConfig",
    "CrescendoAttacker",
    "CrescendoEvaluator",
    "CrescendoResult",
    "CrescendoPhase",
    "ProgressScore",
    "MemorySystem",
    "HydraAttacker",
    "HydraCoordinator",
    "HydraAttackResult",
    "AttackStatus",
    "KnowledgeBase",
    "strategies",

    # Testing Framework
    "BaseTarget",
    "TargetResponse",
    "MockTarget",
    "SecurityTestConfig",
    "SecurityTestRunner",
    "SecurityTestResults",
    "ReportGenerator",

    # Checkpoint & Persistence
    "CheckpointManager",
    "CheckpointMetadata",
    "save_module",
    "load_module",
    "EvolutionResult",

    # Adversarial Training
    "AdversarialTrainer",
    "AdversarialConfig",
    "DefenseEvolver",
    "AdversarialAttackEvolver",
    "ConvergenceDetector",
    "TrainingResult",
    "DefenseUpdate",

    # Platform
    "SecurityPlatform",
    "PlatformConfig",
    "TrainingConfig",

    # Grounding
    "Contradiction",
    "ContradictionResult",
    "FieldType",
    "Severity",
    "SourceType",
    "HybridGroundingChecker",
    "check_grounding",
    "is_grounded",
    "ContradictionDetector",
    "LLMContradictionChecker",
    "ValueExtractor",

    # Promptfoo
    "PromptfooConfig",
    "PromptfooTargetConfig",
    "PromptfooPluginConfig",
    "load_config",
    "create_default_config",
    "get_preset",
    "list_presets",
    "OWASP_LLM_TOP10",
    "MITRE_ATLAS",
    "QUICK_SCAN",
    "LLMCallCache",
    "get_llm_cache",
    "ConcurrentTestRunner",

    # CLI
    "CLIGuardrail",
    "CLIGuardResult",
    "CLIThreatCategory",
    "CLIGuardAction",
    "CommandParser",
    "ParsedCommand",
    "CommandType",
    "DangerousCommands",
    "DangerousPatterns",
    "SensitivePaths",
    "CLISecurityConfig",
    "SandboxLevel",
    "ExecutionPolicy",
    "CommandSanitizer",
    "SanitizeResult",

    # Async
    "no_injection_async",
    "no_pii_async",
    "no_toxicity_async",
    "safe_async",
    "injection_score_async",
    "no_mcp_attack_async",
    "check_all_async",
    "batch_check_async",
    "AsyncLLMGuardrail",
    "AsyncHybridGuardrail",

    # Streaming
    "StreamGuardrail",
    "StreamViolation",

    # Validators
    "Validator",
    "OnFailAction",
    "PassResult",
    "FailResult",
    "ValidationError",
    "Guard",
    "GuardResult",
    "GuardHistoryEntry",
    "AsyncGuard",
    "NoInjection",
    "NoPII",
    "NoToxicity",
    "NoMCPAttack",
    "ValidLength",
    "ValidChoices",
    "ValidRange",
    "ValidRegex",
    "ValidJSON",

    # Sanitize
    "sanitize_pii",
    "sanitize_injection",
    "sanitize_commands",
    "sanitize_output",

    # Structured Output
    "GuardedModel",
    "GuardedPredictor",
    "validated_field",

    # HybridResult for backward compat
    "HybridResult",
]
