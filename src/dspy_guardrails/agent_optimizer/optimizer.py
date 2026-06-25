"""
Agent Optimizer Module
======================

Agent 级别的优化器，支持多种优化策略。
"""

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .credit import CreditSummary, assign_credit
from .trajectory import Outcome, Trajectory
from .wrapper import AgentResult, OptimizableAgent


class OptimizationMethod(Enum):
    """优化方法"""
    BOOTSTRAP = "bootstrap"         # 从成功案例中提取 few-shot
    MIPRO = "mipro"                 # DSPy MIPROv2
    SELF_REFINE = "self_refine"    # 自我反思迭代
    EVOLUTION = "evolution"         # 进化算法


@dataclass
class OptimizationConfig:
    """优化配置"""
    method: OptimizationMethod = OptimizationMethod.BOOTSTRAP
    num_trials: int = 20
    batch_size: int = 5
    patience: int = 5                # 连续多少次没有改进就停止
    credit_method: str = "decay"     # credit assignment 方法
    save_trajectories: bool = True
    output_dir: str = "./optimization_results"

    # Bootstrap 特定
    num_demos: int = 4
    demo_selection: str = "diverse"  # "diverse", "best", "random"

    # Self-refine 特定
    num_iterations: int = 5

    # Evolution 特定
    population_size: int = 10
    mutation_rate: float = 0.1


@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    improvement: float              # 相对改进
    baseline_score: float
    optimized_score: float
    best_params: dict[str, Any]
    trajectories: list[Trajectory]
    credit_summary: CreditSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Optimization Result:\n"
            f"  Success: {self.success}\n"
            f"  Baseline: {self.baseline_score:.2%}\n"
            f"  Optimized: {self.optimized_score:.2%}\n"
            f"  Improvement: {self.improvement:+.2%}\n"
            f"  Trajectories: {len(self.trajectories)}"
        )


class AgentOptimizer:
    """
    Agent 优化器

    支持多种优化策略来提升 Agent 性能。

    Example:
        from dspy_guardrails.agent_optimizer import AgentOptimizer, wrap_agent

        # 包装 Agent
        wrapped = wrap_agent(my_agent)

        # 创建优化器
        optimizer = AgentOptimizer(
            metric=my_success_metric,
            config=OptimizationConfig(method="bootstrap"),
        )

        # 运行优化
        result = optimizer.optimize(
            agent=wrapped,
            train_tasks=["task1", "task2", ...],
            eval_tasks=["eval1", "eval2", ...],
        )

        print(result.summary())
    """

    def __init__(
        self,
        metric: Callable[[AgentResult], float],
        config: OptimizationConfig = None,
    ):
        """
        Args:
            metric: 评估函数，输入 AgentResult，输出分数 (0-1)
            config: 优化配置
        """
        self.metric = metric
        self.config = config or OptimizationConfig()
        self._trajectories: list[Trajectory] = []
        self._best_score = 0.0
        self._best_params = {}

    def optimize(
        self,
        agent: OptimizableAgent,
        train_tasks: list[str],
        eval_tasks: list[str] = None,
    ) -> OptimizationResult:
        """
        运行优化

        Args:
            agent: 可优化的 Agent
            train_tasks: 训练任务列表
            eval_tasks: 评估任务列表 (可选)

        Returns:
            OptimizationResult
        """
        eval_tasks = eval_tasks or train_tasks

        # 1. 评估 Baseline
        print("=" * 60)
        print("Phase 1: Evaluating Baseline")
        print("=" * 60)
        baseline_score, baseline_trajectories = self._evaluate(agent, eval_tasks)
        print(f"Baseline score: {baseline_score:.2%}")

        # 2. 收集训练轨迹
        print("\n" + "=" * 60)
        print("Phase 2: Collecting Training Trajectories")
        print("=" * 60)
        train_trajectories = self._collect_trajectories(agent, train_tasks)

        # 3. Credit Assignment
        print("\n" + "=" * 60)
        print("Phase 3: Credit Assignment")
        print("=" * 60)
        for traj in train_trajectories:
            assign_credit(traj, method=self.config.credit_method)

        # 4. 运行优化
        print("\n" + "=" * 60)
        print(f"Phase 4: Running {self.config.method.value} Optimization")
        print("=" * 60)

        if self.config.method == OptimizationMethod.BOOTSTRAP:
            best_params = self._bootstrap_optimize(agent, train_trajectories)
        elif self.config.method == OptimizationMethod.SELF_REFINE:
            best_params = self._self_refine_optimize(agent, train_trajectories)
        elif self.config.method == OptimizationMethod.EVOLUTION:
            best_params = self._evolution_optimize(agent, train_trajectories, eval_tasks)
        else:
            best_params = self._bootstrap_optimize(agent, train_trajectories)

        # 5. 应用最佳参数并评估
        print("\n" + "=" * 60)
        print("Phase 5: Final Evaluation")
        print("=" * 60)
        agent.set_params(best_params)
        optimized_score, optimized_trajectories = self._evaluate(agent, eval_tasks)
        print(f"Optimized score: {optimized_score:.2%}")

        # 6. 生成结果
        improvement = optimized_score - baseline_score
        success = improvement > 0

        all_trajectories = baseline_trajectories + train_trajectories + optimized_trajectories
        self._trajectories = all_trajectories

        # 生成 credit 摘要
        credit_summary = None
        if train_trajectories:
            # 合并所有轨迹的 credit
            all_credits = []
            for traj in train_trajectories:
                for step in traj.steps:
                    if step.credit != 0:
                        all_credits.append((step.step_id, step.credit, step.action_name))

            if all_credits:
                sorted_credits = sorted(all_credits, key=lambda x: x[1], reverse=True)
                credit_summary = CreditSummary(
                    total_positive_credit=sum(c for _, c, _ in all_credits if c > 0),
                    total_negative_credit=sum(c for _, c, _ in all_credits if c < 0),
                    top_contributors=sorted_credits[:5],
                    bottom_contributors=sorted_credits[-5:],
                )

        result = OptimizationResult(
            success=success,
            improvement=improvement,
            baseline_score=baseline_score,
            optimized_score=optimized_score,
            best_params=best_params,
            trajectories=all_trajectories,
            credit_summary=credit_summary,
            metadata={
                "config": self.config.__dict__,
                "num_train_tasks": len(train_tasks),
                "num_eval_tasks": len(eval_tasks),
            },
        )

        # 保存结果
        if self.config.save_trajectories:
            self._save_results(result)

        return result

    def _evaluate(
        self,
        agent: OptimizableAgent,
        tasks: list[str],
    ) -> tuple:
        """评估 Agent"""
        trajectories = []
        scores = []

        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] {task[:50]}...")
            result = agent.run(task, record=True)

            score = self.metric(result)
            scores.append(score)

            if result.trajectory:
                result.trajectory.final_reward = score
                trajectories.append(result.trajectory)

            status = "✓" if score >= 0.5 else "✗"
            print(f"    {status} score={score:.2f}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, trajectories

    def _collect_trajectories(
        self,
        agent: OptimizableAgent,
        tasks: list[str],
    ) -> list[Trajectory]:
        """收集训练轨迹"""
        trajectories = []

        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] Collecting: {task[:50]}...")
            result = agent.run(task, record=True)

            if result.trajectory:
                # 计算奖励
                score = self.metric(result)
                result.trajectory.final_reward = score
                trajectories.append(result.trajectory)

        return trajectories

    def _bootstrap_optimize(
        self,
        agent: OptimizableAgent,
        trajectories: list[Trajectory],
    ) -> dict[str, Any]:
        """
        Bootstrap 优化

        从成功的轨迹中提取 few-shot 示例。
        """
        # 筛选成功的轨迹
        successful = [t for t in trajectories if t.outcome == Outcome.SUCCESS]

        if not successful:
            print("  No successful trajectories found, using partial successes")
            successful = sorted(trajectories, key=lambda t: t.final_reward, reverse=True)

        # 选择示例
        if self.config.demo_selection == "best":
            # 选择得分最高的
            selected = sorted(successful, key=lambda t: t.final_reward, reverse=True)
            selected = selected[:self.config.num_demos]
        elif self.config.demo_selection == "diverse":
            # 选择多样化的 (不同工具组合)
            selected = self._select_diverse(successful, self.config.num_demos)
        else:
            # 随机选择
            import random
            selected = random.sample(successful, min(len(successful), self.config.num_demos))

        # 从轨迹中提取 demos
        demos = []
        for traj in selected:
            demo = self._trajectory_to_demo(traj)
            if demo:
                demos.append(demo)

        print(f"  Selected {len(demos)} demos from {len(trajectories)} trajectories")

        return {"demos": demos}

    def _select_diverse(
        self,
        trajectories: list[Trajectory],
        n: int,
    ) -> list[Trajectory]:
        """选择多样化的轨迹"""
        if len(trajectories) <= n:
            return trajectories

        # 按工具组合分组
        groups = {}
        for traj in trajectories:
            key = tuple(sorted(traj.get_tools_used()))
            if key not in groups:
                groups[key] = []
            groups[key].append(traj)

        # 从每组选最好的
        selected = []
        for _key, group in groups.items():
            best = max(group, key=lambda t: t.final_reward)
            selected.append(best)
            if len(selected) >= n:
                break

        # 如果不够，补充得分最高的
        if len(selected) < n:
            remaining = [t for t in trajectories if t not in selected]
            remaining = sorted(remaining, key=lambda t: t.final_reward, reverse=True)
            selected.extend(remaining[:n - len(selected)])

        return selected[:n]

    def _trajectory_to_demo(self, trajectory: Trajectory) -> dict | None:
        """将轨迹转换为 demo"""
        if not trajectory.task or not trajectory.final_response:
            return None

        return {
            "input": trajectory.task,
            "output": trajectory.final_response,
            "tools_used": trajectory.get_tools_used(),
            "reward": trajectory.final_reward,
        }

    def _self_refine_optimize(
        self,
        agent: OptimizableAgent,
        trajectories: list[Trajectory],
    ) -> dict[str, Any]:
        """
        Self-Refine 优化

        让 LLM 分析失败案例并提出改进建议。
        """
        # 找出失败的轨迹
        failures = [t for t in trajectories if t.outcome != Outcome.SUCCESS]

        if not failures:
            print("  No failures to analyze")
            return {}

        # 获取当前参数
        current_params = agent.get_optimizable_params()

        # 使用 LLM 分析失败并生成改进
        improvements = self._analyze_failures(failures, current_params)

        return improvements

    def _analyze_failures(
        self,
        failures: list[Trajectory],
        current_params: dict,
    ) -> dict[str, Any]:
        """分析失败并生成改进"""
        # 这里使用简单的规则，实际可以用 LLM

        # 统计失败模式
        failure_patterns = {}
        for traj in failures:
            tools = tuple(traj.get_tools_used())
            if tools not in failure_patterns:
                failure_patterns[tools] = 0
            failure_patterns[tools] += 1

        print(f"  Analyzed {len(failures)} failures")
        print(f"  Failure patterns: {failure_patterns}")

        # 生成改进建议 (简化版)
        improvements = current_params.copy()

        # 如果某个工具组合经常失败，可以在 instructions 中添加提示
        if failure_patterns:
            most_common = max(failure_patterns.items(), key=lambda x: x[1])
            print(f"  Most common failure pattern: {most_common}")

        return improvements

    def _evolution_optimize(
        self,
        agent: OptimizableAgent,
        trajectories: list[Trajectory],
        eval_tasks: list[str],
    ) -> dict[str, Any]:
        """
        进化算法优化

        生成参数变体，评估并选择最优。
        """
        current_params = agent.get_optimizable_params()

        # 初始种群
        population = [current_params]

        # 生成变体
        for _ in range(self.config.population_size - 1):
            variant = self._mutate_params(current_params)
            population.append(variant)

        best_score = 0
        best_params = current_params

        # 评估每个变体
        for i, params in enumerate(population):
            print(f"  Evaluating variant {i+1}/{len(population)}...")
            agent.set_params(params)

            # 快速评估 (只用部分任务)
            eval_subset = eval_tasks[:3]
            score, _ = self._evaluate(agent, eval_subset)

            if score > best_score:
                best_score = score
                best_params = params
                print(f"    New best: {score:.2%}")

        return best_params

    def _mutate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """变异参数"""
        mutated = params.copy()

        # 简单变异：随机修改一些参数
        # 实际实现需要根据具体参数类型来处理

        return mutated

    def _save_results(self, result: OptimizationResult):
        """保存结果"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(
            self.config.output_dir,
            f"optimization_{timestamp}.json"
        )

        # 转换为可序列化格式
        data = {
            "success": result.success,
            "improvement": result.improvement,
            "baseline_score": result.baseline_score,
            "optimized_score": result.optimized_score,
            "metadata": result.metadata,
            "trajectories": [t.to_dict() for t in result.trajectories[:10]],  # 只保存前10个
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filepath}")


# 便捷函数
def optimize_agent(
    agent: OptimizableAgent,
    train_tasks: list[str],
    metric: Callable[[AgentResult], float],
    method: str = "bootstrap",
    **kwargs,
) -> OptimizationResult:
    """
    便捷函数：优化 Agent

    Args:
        agent: 可优化的 Agent
        train_tasks: 训练任务
        metric: 评估函数
        method: 优化方法
        **kwargs: 其他配置

    Returns:
        OptimizationResult
    """
    config = OptimizationConfig(
        method=OptimizationMethod(method),
        **kwargs,
    )

    optimizer = AgentOptimizer(metric=metric, config=config)
    return optimizer.optimize(agent, train_tasks)
