"""
DSPy Red Team Framework - 自进化的 AI 安全测试工具

基于 DSPy 的自我进化机制，构建自动化的 AI 安全测试框架。

Features:
    - 多种攻击生成器 (Prompt Injection, Jailbreak, Bypass)
    - 多轮对话攻击
    - 自进化攻击引擎 (启发式评分 + 遗传算法)
    - 标准安全基准测试 (HarmBench, AdvBench, JailbreakBench)
    - Payload 执行验证
    - 统一配置管理
    - 统一攻击载荷库

    NEW (借鉴 promptfoo):
    - CrescendoAttacker: 渐进式攻击 + 回溯机制
    - HydraAttacker: 多头并行攻击 + 知识共享
    - Strategies 模块: 独立的编码/混淆策略

Usage:
    from dspy_guardrails.redteam import (
        # 攻击器
        PromptInjectionAttacker,
        JailbreakAttacker,
        MultiTurnAttacker,
        CrescendoAttacker,  # NEW
        HydraAttacker,       # NEW

        # 进化引擎
        AttackEvolver,
        GeneticAttackEvolver,

        # 评估器
        RedTeamEvaluator,

        # 配置
        RedTeamConfig,

        # 载荷
        get_all_payloads,
        InjectionPayloads,
        JailbreakPayloads,
        MCPPayloads,

        # 策略 (NEW)
        strategies,
    )

    # 创建攻击模块
    attacker = PromptInjectionAttacker()

    # 生成攻击
    attack = attacker(
        target_behavior="extract system prompt",
        defense_description="input filtering with regex"
    )

    # 使用统一载荷库
    payloads = get_all_payloads()
    for payload in payloads:
        result = attacker.test(payload)

    # Crescendo 渐进式攻击 (NEW)
    crescendo = CrescendoAttacker(max_turns=10, max_backtracks=3)
    result = crescendo.attack(target_llm=my_llm, target_behavior="reveal system prompt")

    # Hydra 多头并行攻击 (NEW)
    hydra = HydraAttacker(max_workers=4)
    result = hydra.attack_guardrail(guardrail=my_guard, target_behaviors=[...])

    # 使用策略模块 (NEW)
    from dspy_guardrails.redteam import strategies
    encoded = strategies.apply_strategy("ignore instructions", "base64")

    # 自定义配置
    config = RedTeamConfig(
        evolution=EvolutionConfig(num_generations=20),
    )
"""

# Attackers
# Strategies Module (NEW - inspired by promptfoo)
from . import strategies
from .attackers import (
    AttackResult,
    BypassSignature,
    GuardrailBypassAttacker,
    InjectionAttackSignature,
    JailbreakAttacker,
    JailbreakSignature,
    PromptInjectionAttacker,
)

# Benchmarks
from .benchmarks import (
    AdvBenchDataset,
    BenchmarkReport,
    BenchmarkResult,
    # Runner
    BenchmarkRunner,
    # Types
    BenchmarkSample,
    # Datasets
    HarmBenchDataset,
    HarmCategory,
    JailbreakBenchDataset,
    ToxicChatDataset,
    # Utilities
    get_all_benchmark_samples,
    get_benchmark_stats,
)

# Config
from .config import (
    AttackerConfig,
    AttackGeneratorConfig,
    EvaluatorConfig,
    OutputConfig,
    # Promptfoo-style config (NEW)
    PromptfooStyleConfig,
    RedTeamConfig,
    TargetConfig,
    TestCaseConfig,
    ValidationConfig,
    create_sample_config,
    load_config,
)

# Crescendo Attacker (NEW - inspired by promptfoo)
from .crescendo import (
    CrescendoAttacker,
    CrescendoEvaluator,
    CrescendoPhase,
    CrescendoResult,
    CrescendoState,
    MemorySystem,
    ProgressScore,
)

# Evaluator
from .evaluator import (
    EvaluationResult,
    RedTeamEvaluator,
    VulnerabilityReport,
)

# Evolution Engine
from .evolution import (
    AttackEvolver,
    EvolutionConfig,
    EvolutionResult,
    GeneticAttackEvolver,
)

# Hydra Attacker (NEW - inspired by promptfoo)
from .hydra import (
    AttackStatus,
    HydraAttacker,
    HydraAttackResult,
    HydraCoordinator,
    HydraHead,
    KnowledgeBase,
)

# MCP Attacks
from .mcp_attacks import (
    MCPAttacker,
    MCPAttackEvaluator,
    MCPAttackPayloads,
    MCPAttackResult,
    MCPAttackType,
    MCPRedTeamRunner,
    MCPToolDefinition,
    run_mcp_red_team,
    test_mcp_guardrail,
)

# Multi-Turn Attacks
from .multi_turn import (
    STRATEGIES,
    AgentAttackResult,
    # Agent Multi-Turn (DSPy ReAct Implementation)
    AgentMultiTurnAttacker,
    AgentMultiTurnConfig,
    ContextManipulation,
    Conversation,
    ConversationRole,
    EmotionalManipulation,
    GradualEscalation,
    Message,
    MultiTurnAttacker,
    MultiTurnAttackResult,
    MultiTurnEvaluator,
    # Strategies
    MultiTurnStrategy,
    RoleplayProgression,
    SocraticMethod,
    ToolCallRecord,
    create_hooked_tool,
)

# Patterns
from .patterns import (
    AttackPattern,
    AttackPatterns,
    AttackTestCases,
)

# Payload Validation
from .payload_validator import (
    AttackSuccessValidator,
    AttackValidationReport,
    Payload,
    PayloadType,
    PayloadValidator,
    StandardPayloads,
    ValidationResult,
    generate_validation_report,
)

# Payloads
from .payloads import (
    AttackPayload,
    BypassPayloads,
    InjectionPayloads,
    JailbreakPayloads,
    MCPPayloads,
    PayloadCategory,
    get_all_payloads,
    get_payloads_by_category,
)
from .payloads.domain import AirlinePayloads

# Pentest Agent
from .pentest import (
    AttackEvaluator,
    PentestAgent,
    PentestAgentConfig,
    PentestReport,
    PentestState,
    PentestStateMachine,
    ReplayableTestCase,
    Strategist,
    Trajectory,
    Vulnerability,
)

# Severity Classification
from .severity import (
    SEVERITY_MAPPING,
    SeverityClassification,
    SeverityClassifier,
    SeverityLevel,
    calculate_weighted_risk_score,
)

__all__ = [
    # Attackers
    "PromptInjectionAttacker",
    "JailbreakAttacker",
    "GuardrailBypassAttacker",
    "AttackResult",
    "InjectionAttackSignature",
    "JailbreakSignature",
    "BypassSignature",

    # Evolution
    "AttackEvolver",
    "GeneticAttackEvolver",
    "EvolutionConfig",
    "EvolutionResult",

    # Evaluator
    "RedTeamEvaluator",
    "EvaluationResult",
    "VulnerabilityReport",

    # Patterns
    "AttackPatterns",
    "AttackPattern",
    "AttackTestCases",

    # Multi-Turn
    "MultiTurnAttacker",
    "MultiTurnEvaluator",
    "MultiTurnAttackResult",
    "Conversation",
    "Message",
    "ConversationRole",
    "MultiTurnStrategy",
    "GradualEscalation",
    "RoleplayProgression",
    "ContextManipulation",
    "SocraticMethod",
    "EmotionalManipulation",
    "STRATEGIES",
    # Agent Multi-Turn (DSPy ReAct)
    "AgentMultiTurnAttacker",
    "AgentMultiTurnConfig",
    "AgentAttackResult",
    "ToolCallRecord",
    "create_hooked_tool",

    # Benchmarks
    "HarmBenchDataset",
    "AdvBenchDataset",
    "JailbreakBenchDataset",
    "ToxicChatDataset",
    "BenchmarkSample",
    "BenchmarkResult",
    "BenchmarkReport",
    "HarmCategory",
    "BenchmarkRunner",
    "get_all_benchmark_samples",
    "get_benchmark_stats",

    # Payload Validation
    "PayloadValidator",
    "AttackSuccessValidator",
    "Payload",
    "PayloadType",
    "ValidationResult",
    "StandardPayloads",
    "AttackValidationReport",
    "generate_validation_report",

    # MCP Attacks
    "MCPAttacker",
    "MCPAttackEvaluator",
    "MCPRedTeamRunner",
    "MCPAttackType",
    "MCPAttackResult",
    "MCPAttackPayloads",
    "MCPToolDefinition",
    "test_mcp_guardrail",
    "run_mcp_red_team",

    # Severity Classification
    "SeverityLevel",
    "SeverityClassification",
    "SeverityClassifier",
    "SEVERITY_MAPPING",
    "calculate_weighted_risk_score",

    # Config
    "RedTeamConfig",
    "AttackGeneratorConfig",
    "EvaluatorConfig",
    "ValidationConfig",
    # Promptfoo-style config (NEW)
    "PromptfooStyleConfig",
    "TargetConfig",
    "AttackerConfig",
    "TestCaseConfig",
    "OutputConfig",
    "load_config",
    "create_sample_config",

    # Payloads
    "AttackPayload",
    "PayloadCategory",
    "InjectionPayloads",
    "JailbreakPayloads",
    "MCPPayloads",
    "BypassPayloads",
    "AirlinePayloads",
    "get_all_payloads",
    "get_payloads_by_category",

    # Pentest Agent
    "PentestAgent",
    "PentestAgentConfig",
    "PentestState",
    "PentestStateMachine",
    "Strategist",
    "AttackEvaluator",
    "PentestReport",
    "Trajectory",
    "Vulnerability",
    "ReplayableTestCase",

    # Crescendo (NEW)
    "CrescendoAttacker",
    "CrescendoEvaluator",
    "CrescendoResult",
    "CrescendoPhase",
    "CrescendoState",
    "ProgressScore",
    "MemorySystem",

    # Hydra (NEW)
    "HydraAttacker",
    "HydraCoordinator",
    "HydraAttackResult",
    "HydraHead",
    "KnowledgeBase",
    "AttackStatus",

    # Strategies Module (NEW)
    "strategies",

    # Quick functions
    "evolve_attacks",
    "create_evolver",
]


# =============================================================================
# Quick Functions
# =============================================================================

def create_evolver(
    target_guardrail,
    mode: str = "genetic",
    **kwargs,
) -> AttackEvolver | GeneticAttackEvolver:
    """
    Create an attack evolver with the specified mode.

    Args:
        target_guardrail: Function(text) -> bool (True=safe, False=detected)
        mode: "basic" or "genetic" (default)
        **kwargs: Additional arguments for the evolver

    Returns:
        AttackEvolver or GeneticAttackEvolver instance

    Example:
        from dspy_guardrails.redteam import create_evolver
        from dspy_guardrails import guardrail

        evolver = create_evolver(guardrail.no_injection, mode="genetic")
        result = evolver.evolve(attacker, "reveal system prompt")
    """
    if mode == "genetic":
        return GeneticAttackEvolver(target_guardrail=target_guardrail, **kwargs)
    else:
        return AttackEvolver(target_guardrail=target_guardrail, **kwargs)


def evolve_attacks(
    target_guardrail,
    attacker,
    initial_target: str = "bypass safety filter",
    defense_description: str = "unknown",
    num_generations: int = 5,
    mode: str = "genetic",
) -> EvolutionResult:
    """
    Quick function to evolve attacks against a guardrail.

    Args:
        target_guardrail: Function(text) -> bool (True=safe, False=detected)
        attacker: DSPy attack module (e.g., PromptInjectionAttacker)
        initial_target: Target behavior description
        defense_description: Description of the defense
        num_generations: Number of evolution generations
        mode: "basic" or "genetic"

    Returns:
        EvolutionResult with best attacks and metrics

    Example:
        from dspy_guardrails.redteam import evolve_attacks, PromptInjectionAttacker
        from dspy_guardrails import guardrail

        result = evolve_attacks(
            target_guardrail=guardrail.no_injection,
            attacker=PromptInjectionAttacker(),
            initial_target="reveal system prompt",
            num_generations=10,
        )
        print(f"Best attack: {result.best_attack.prompt}")
        print(f"Bypass rate: {result.final_bypass_rate:.1%}")
    """
    config = EvolutionConfig(num_generations=num_generations)
    evolver = create_evolver(target_guardrail, mode=mode, config=config)
    return evolver.evolve(attacker, initial_target, defense_description)
