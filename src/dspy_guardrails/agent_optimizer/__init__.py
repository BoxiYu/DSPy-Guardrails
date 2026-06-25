"""
DSPy Agent Optimizer Extension
==============================

扩展 DSPy 以支持 Agent 级别的优化，包括：
1. Trajectory 记录层 - 完整记录每一步的决策和结果
2. Credit Assignment - 将最终奖励归因到每个步骤
3. 策略优化 - 优化 Agent 路由、工具选择、Handoff 决策
4. 多轮状态管理 - 跟踪多轮对话中的性能

设计原则：
- 与 DSPy 兼容，可包装任何 dspy.Module
- 最小侵入式设计
- 支持多种优化策略 (Bootstrap, RL, Evolution)
"""

from .credit import (
    AttentionCredit,
    CreditAssigner,
    CreditSummary,
    DecayCredit,
    UniformCredit,
    assign_credit,
)
from .optimizer import (
    AgentOptimizer,
    OptimizationConfig,
    OptimizationMethod,
    OptimizationResult,
    optimize_agent,
)
from .trajectory import (
    Step,
    Trajectory,
    TrajectoryRecorder,
)
from .wrapper import (
    OptimizableAgent,
    wrap_agent,
)

__all__ = [
    # Trajectory
    "Step",
    "Trajectory",
    "TrajectoryRecorder",
    # Credit Assignment
    "CreditAssigner",
    "UniformCredit",
    "DecayCredit",
    "AttentionCredit",
    "CreditSummary",
    "assign_credit",
    # Optimizer
    "AgentOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizationMethod",
    "optimize_agent",
    # Wrapper
    "OptimizableAgent",
    "wrap_agent",
]
