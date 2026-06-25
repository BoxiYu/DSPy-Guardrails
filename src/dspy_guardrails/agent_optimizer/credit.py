"""
Credit Assignment Module
========================

将最终奖励分配到轨迹中的每个步骤。

支持多种策略：
- Uniform: 均匀分配
- Decay: 时间衰减 (越近的步骤贡献越大)
- Attention: 基于注意力的分配
- Counterfactual: 反事实分析
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from .trajectory import Step, StepType, Trajectory


class CreditAssigner(ABC):
    """Credit Assignment 基类"""

    @abstractmethod
    def assign(self, trajectory: Trajectory) -> Trajectory:
        """
        为轨迹中的每个步骤分配 credit

        Args:
            trajectory: 完整的执行轨迹

        Returns:
            更新了 credit 字段的轨迹
        """
        pass


class UniformCredit(CreditAssigner):
    """
    均匀分配

    将最终奖励均匀分配到所有步骤。
    """

    def __init__(self, include_types: list[StepType] = None):
        """
        Args:
            include_types: 只对这些类型的步骤分配 credit
                         默认: [TOOL_CALL, HANDOFF, AGENT_CALL]
        """
        self.include_types = include_types or [
            StepType.TOOL_CALL,
            StepType.HANDOFF,
            StepType.AGENT_CALL,
        ]

    def assign(self, trajectory: Trajectory) -> Trajectory:
        # 筛选相关步骤
        relevant_steps = [
            s for s in trajectory.steps
            if s.step_type in self.include_types
        ]

        if not relevant_steps:
            return trajectory

        # 均匀分配
        credit_per_step = trajectory.final_reward / len(relevant_steps)

        for step in relevant_steps:
            step.credit = credit_per_step
            step.reward = credit_per_step

        return trajectory


class DecayCredit(CreditAssigner):
    """
    时间衰减分配

    越接近结束的步骤，获得越多的 credit (因为它们对最终结果影响更直接)。
    """

    def __init__(
        self,
        decay_factor: float = 0.9,
        include_types: list[StepType] = None,
        reverse: bool = False,
    ):
        """
        Args:
            decay_factor: 衰减因子 (0-1)，越小衰减越快
            include_types: 只对这些类型的步骤分配
            reverse: 如果 True，早期步骤获得更多 credit
        """
        self.decay_factor = decay_factor
        self.include_types = include_types or [
            StepType.TOOL_CALL,
            StepType.HANDOFF,
            StepType.AGENT_CALL,
        ]
        self.reverse = reverse

    def assign(self, trajectory: Trajectory) -> Trajectory:
        # 筛选相关步骤
        relevant_steps = [
            s for s in trajectory.steps
            if s.step_type in self.include_types
        ]

        if not relevant_steps:
            return trajectory

        n = len(relevant_steps)

        # 计算权重
        weights = []
        for i in range(n):
            if self.reverse:
                # 早期步骤权重高
                w = self.decay_factor ** i
            else:
                # 晚期步骤权重高
                w = self.decay_factor ** (n - 1 - i)
            weights.append(w)

        # 归一化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # 分配 credit
        for i, step in enumerate(relevant_steps):
            step.credit = trajectory.final_reward * weights[i]
            step.reward = step.credit

        return trajectory


class AttentionCredit(CreditAssigner):
    """
    基于注意力的分配

    使用 LLM 分析哪些步骤对最终结果贡献最大。
    """

    def __init__(
        self,
        analyzer: Callable[[Trajectory], list[float]] = None,
        include_types: list[StepType] = None,
    ):
        """
        Args:
            analyzer: 分析函数，输入轨迹，输出每步的重要性分数
            include_types: 只对这些类型的步骤分配
        """
        self.analyzer = analyzer or self._default_analyzer
        self.include_types = include_types or [
            StepType.TOOL_CALL,
            StepType.HANDOFF,
            StepType.AGENT_CALL,
        ]

    def _default_analyzer(self, trajectory: Trajectory) -> list[float]:
        """
        默认分析器：基于规则的启发式方法

        规则：
        - 成功的工具调用: +1
        - 失败/错误的步骤: -0.5
        - 导致 handoff 的步骤: +0.3
        - 最后一步: +0.5
        """
        relevant_steps = [
            s for s in trajectory.steps
            if s.step_type in self.include_types
        ]

        if not relevant_steps:
            return []

        scores = []
        for i, step in enumerate(relevant_steps):
            score = 0.5  # 基础分

            # 成功的工具调用
            if step.step_type == StepType.TOOL_CALL and not step.error:
                score += 0.5

            # 错误的步骤
            if step.error:
                score -= 0.5

            # Handoff (转移到正确的 agent)
            if step.step_type == StepType.HANDOFF:
                score += 0.3

            # 最后一步
            if i == len(relevant_steps) - 1:
                score += 0.5

            scores.append(max(0, score))

        return scores

    def assign(self, trajectory: Trajectory) -> Trajectory:
        relevant_steps = [
            s for s in trajectory.steps
            if s.step_type in self.include_types
        ]

        if not relevant_steps:
            return trajectory

        # 获取重要性分数
        importance_scores = self.analyzer(trajectory)

        # 确保长度匹配
        if len(importance_scores) != len(relevant_steps):
            # Fallback to uniform
            importance_scores = [1.0] * len(relevant_steps)

        # 归一化
        total = sum(importance_scores)
        if total > 0:
            importance_scores = [s / total for s in importance_scores]
        else:
            importance_scores = [1.0 / len(relevant_steps)] * len(relevant_steps)

        # 分配 credit
        for i, step in enumerate(relevant_steps):
            step.credit = trajectory.final_reward * importance_scores[i]
            step.reward = step.credit

        return trajectory


class CounterfactualCredit(CreditAssigner):
    """
    反事实 Credit Assignment

    评估"如果没有这一步，结果会怎样"。
    需要一个评估函数来模拟不同的轨迹。

    注意：这是一个高级方法，计算成本较高。
    """

    def __init__(
        self,
        evaluator: Callable[[list[Step]], float] = None,
        include_types: list[StepType] = None,
    ):
        """
        Args:
            evaluator: 评估函数，输入步骤列表，输出预测奖励
            include_types: 只对这些类型的步骤分配
        """
        self.evaluator = evaluator
        self.include_types = include_types or [
            StepType.TOOL_CALL,
            StepType.HANDOFF,
        ]

    def assign(self, trajectory: Trajectory) -> Trajectory:
        if not self.evaluator:
            # 没有评估器，回退到 uniform
            return UniformCredit(self.include_types).assign(trajectory)

        relevant_indices = [
            i for i, s in enumerate(trajectory.steps)
            if s.step_type in self.include_types
        ]

        if not relevant_indices:
            return trajectory

        # 计算基准值
        baseline_reward = trajectory.final_reward

        # 对每个步骤计算反事实
        credits = []
        for idx in relevant_indices:
            # 创建不含该步骤的轨迹
            counterfactual_steps = [
                s for i, s in enumerate(trajectory.steps)
                if i != idx
            ]

            # 评估反事实轨迹
            counterfactual_reward = self.evaluator(counterfactual_steps)

            # Credit = 有这一步的奖励 - 没有这一步的奖励
            credit = baseline_reward - counterfactual_reward
            credits.append(credit)

        # 分配 credit
        for i, idx in enumerate(relevant_indices):
            trajectory.steps[idx].credit = credits[i]
            trajectory.steps[idx].reward = credits[i]

        return trajectory


@dataclass
class CreditSummary:
    """Credit 分配摘要"""
    total_positive_credit: float = 0.0
    total_negative_credit: float = 0.0
    top_contributors: list[tuple] = None  # [(step_id, credit), ...]
    bottom_contributors: list[tuple] = None
    step_type_breakdown: dict[str, float] = None

    @classmethod
    def from_trajectory(cls, trajectory: Trajectory, top_k: int = 3) -> "CreditSummary":
        """从轨迹生成摘要"""
        positive = sum(s.credit for s in trajectory.steps if s.credit > 0)
        negative = sum(s.credit for s in trajectory.steps if s.credit < 0)

        # 排序
        sorted_steps = sorted(trajectory.steps, key=lambda s: s.credit, reverse=True)
        top = [(s.step_id, s.credit, s.action_name) for s in sorted_steps[:top_k]]
        bottom = [(s.step_id, s.credit, s.action_name) for s in sorted_steps[-top_k:]]

        # 按类型统计
        type_breakdown = {}
        for s in trajectory.steps:
            t = s.step_type.value
            type_breakdown[t] = type_breakdown.get(t, 0) + s.credit

        return cls(
            total_positive_credit=positive,
            total_negative_credit=negative,
            top_contributors=top,
            bottom_contributors=bottom,
            step_type_breakdown=type_breakdown,
        )


def assign_credit(
    trajectory: Trajectory,
    method: str = "decay",
    **kwargs,
) -> Trajectory:
    """
    便捷函数：为轨迹分配 credit

    Args:
        trajectory: 执行轨迹
        method: 分配方法 ("uniform", "decay", "attention", "counterfactual")
        **kwargs: 传递给分配器的参数

    Returns:
        更新了 credit 的轨迹
    """
    assigners = {
        "uniform": UniformCredit,
        "decay": DecayCredit,
        "attention": AttentionCredit,
        "counterfactual": CounterfactualCredit,
    }

    assigner_class = assigners.get(method, DecayCredit)
    assigner = assigner_class(**kwargs)

    return assigner.assign(trajectory)
