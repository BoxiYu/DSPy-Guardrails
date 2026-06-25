"""
Agent Wrapper Module
====================

包装 DSPy Module 或自定义 Agent，使其可被优化。
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dspy

from .trajectory import (
    Outcome,
    StepType,
    Trajectory,
    TrajectoryRecorder,
)


class OptimizableAgent(ABC):
    """
    可优化 Agent 的抽象基类

    任何想要被优化的 Agent 都需要实现这个接口。
    """

    @abstractmethod
    def run(self, task: str, record: bool = True) -> "AgentResult":
        """
        运行 Agent

        Args:
            task: 用户任务/输入
            record: 是否记录轨迹

        Returns:
            AgentResult 包含响应和轨迹
        """
        pass

    @abstractmethod
    def get_optimizable_params(self) -> dict[str, Any]:
        """获取可优化的参数"""
        pass

    @abstractmethod
    def set_params(self, params: dict[str, Any]):
        """设置参数"""
        pass


@dataclass
class AgentResult:
    """Agent 执行结果"""
    response: str
    trajectory: Trajectory | None = None
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class DSPyReActWrapper(OptimizableAgent):
    """
    DSPy ReAct Module 的包装器

    将 dspy.ReAct 包装成可优化的 Agent。
    """

    def __init__(
        self,
        react_module: dspy.Module,
        input_field: str = "user_request",
        output_field: str = "response",
        success_checker: Callable[[str], bool] = None,
    ):
        """
        Args:
            react_module: dspy.ReAct 实例
            input_field: 输入字段名
            output_field: 输出字段名
            success_checker: 判断是否成功的函数
        """
        self.module = react_module
        self.input_field = input_field
        self.output_field = output_field
        self.success_checker = success_checker or (lambda x: len(x) > 10)

    def run(self, task: str, record: bool = True) -> AgentResult:
        recorder = TrajectoryRecorder(task=task) if record else None

        if recorder:
            recorder.set_agent("ReAct")
            recorder.record_step(
                step_type=StepType.USER_INPUT,
                input_data={"task": task},
            )

        start_time = time.time()

        try:
            # 调用 ReAct
            kwargs = {self.input_field: task}
            prediction = self.module(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # 提取响应
            response = getattr(prediction, self.output_field, str(prediction))

            # 提取轨迹
            if record and hasattr(prediction, 'trajectory'):
                traj_data = prediction.trajectory
                if isinstance(traj_data, dict):
                    # 解析 DSPy ReAct 轨迹格式
                    i = 0
                    while f'thought_{i}' in traj_data or f'tool_name_{i}' in traj_data:
                        thought = traj_data.get(f'thought_{i}', '')
                        tool_name = traj_data.get(f'tool_name_{i}', '')
                        tool_args = traj_data.get(f'tool_args_{i}', {})
                        observation = traj_data.get(f'observation_{i}', '')

                        if tool_name and tool_name != 'finish':
                            recorder.record_step(
                                step_type=StepType.TOOL_CALL,
                                thought=thought,
                                action_type="tool_call",
                                action_name=tool_name,
                                action_args=tool_args if isinstance(tool_args, dict) else {},
                                result=observation,
                            )
                        i += 1

            # 判断成功
            success = self.success_checker(response)
            outcome = Outcome.SUCCESS if success else Outcome.PARTIAL

            if recorder:
                recorder.record_step(
                    step_type=StepType.AGENT_OUTPUT,
                    result=response,
                    latency_ms=latency_ms,
                )
                trajectory = recorder.finish(
                    response=response,
                    outcome=outcome,
                    reward=1.0 if success else 0.0,
                )
            else:
                trajectory = None

            return AgentResult(
                response=response,
                trajectory=trajectory,
                success=success,
            )

        except Exception as e:
            if recorder:
                recorder.record_step(
                    step_type=StepType.AGENT_OUTPUT,
                    error=str(e),
                )
                trajectory = recorder.finish(
                    response=f"Error: {e}",
                    outcome=Outcome.ERROR,
                    reward=-1.0,
                )
            else:
                trajectory = None

            return AgentResult(
                response=f"Error: {e}",
                trajectory=trajectory,
                success=False,
            )

    def get_optimizable_params(self) -> dict[str, Any]:
        """获取可优化参数"""
        params = {}

        # 获取模块的 signature
        if hasattr(self.module, 'signature'):
            sig = self.module.signature
            if hasattr(sig, '__doc__'):
                params['instructions'] = sig.__doc__

        # 获取 demos (few-shot examples)
        if hasattr(self.module, 'demos'):
            params['demos'] = self.module.demos

        return params

    def set_params(self, params: dict[str, Any]):
        """设置参数"""
        if 'demos' in params and hasattr(self.module, 'demos'):
            self.module.demos = params['demos']


class MultiAgentWrapper(OptimizableAgent):
    """
    多 Agent 系统的包装器

    适用于 skywise-agent 这样的自定义多 Agent 系统。
    """

    def __init__(
        self,
        agent_system: Any,
        chat_method: str = "chat",
        reset_method: str = "reset",
        context_method: str = "get_context",
        success_checker: Callable[[str, dict], bool] = None,
    ):
        """
        Args:
            agent_system: 多 Agent 系统实例
            chat_method: 聊天方法名
            reset_method: 重置方法名
            context_method: 获取上下文方法名
            success_checker: 判断成功的函数 (response, context) -> bool
        """
        self.system = agent_system
        self.chat_method = chat_method
        self.reset_method = reset_method
        self.context_method = context_method
        self.success_checker = success_checker or self._default_checker

        # 注入轨迹记录
        self._inject_recording()

    def _default_checker(self, response: str, context: dict) -> bool:
        """默认成功检查器"""
        # 有实质性回复
        if len(response) < 20:
            return False

        # 没有错误信息
        if "error" in response.lower() or "sorry" in response.lower():
            return False

        return True

    def _inject_recording(self):
        """注入轨迹记录到 Agent 系统"""
        # 这里我们不修改原系统，而是在 run 方法中手动记录
        pass

    def run(self, task: str, record: bool = True) -> AgentResult:
        recorder = TrajectoryRecorder(task=task) if record else None

        # 重置 Agent
        if hasattr(self.system, self.reset_method):
            getattr(self.system, self.reset_method)()

        # 获取初始状态
        initial_context = {}
        if hasattr(self.system, self.context_method):
            initial_context = getattr(self.system, self.context_method)()

        if recorder:
            recorder.snapshot_state(initial_context)
            recorder.set_agent("system")
            recorder.record_step(
                step_type=StepType.USER_INPUT,
                input_data={"task": task},
            )

        start_time = time.time()

        try:
            # 调用聊天
            chat_fn = getattr(self.system, self.chat_method)
            response = chat_fn(task)

            latency_ms = (time.time() - start_time) * 1000

            # 获取最终状态
            final_context = {}
            if hasattr(self.system, self.context_method):
                final_context = getattr(self.system, self.context_method)()

            # 记录当前 Agent
            if recorder and hasattr(self.system, 'current_agent'):
                current = self.system.current_agent
                agent_name = current.value if hasattr(current, 'value') else str(current)
                recorder.set_agent(agent_name)

            if recorder:
                recorder.record_step(
                    step_type=StepType.AGENT_OUTPUT,
                    result=response,
                    state_after=final_context,
                    latency_ms=latency_ms,
                )

            # 判断成功
            success = self.success_checker(response, final_context)
            outcome = Outcome.SUCCESS if success else Outcome.PARTIAL

            if recorder:
                trajectory = recorder.finish(
                    response=response,
                    outcome=outcome,
                    reward=1.0 if success else 0.0,
                )
            else:
                trajectory = None

            return AgentResult(
                response=response,
                trajectory=trajectory,
                success=success,
                metadata={"final_context": final_context},
            )

        except Exception as e:
            if recorder:
                recorder.record_step(
                    step_type=StepType.AGENT_OUTPUT,
                    error=str(e),
                )
                trajectory = recorder.finish(
                    response=f"Error: {e}",
                    outcome=Outcome.ERROR,
                    reward=-1.0,
                )
            else:
                trajectory = None

            return AgentResult(
                response=f"Error: {e}",
                trajectory=trajectory,
                success=False,
            )

    def run_multi_turn(
        self,
        turns: list[str],
        record: bool = True,
    ) -> AgentResult:
        """
        运行多轮对话

        Args:
            turns: 用户输入列表
            record: 是否记录轨迹

        Returns:
            最终结果
        """
        recorder = TrajectoryRecorder(task=turns[0]) if record else None

        # 重置
        if hasattr(self.system, self.reset_method):
            getattr(self.system, self.reset_method)()

        if recorder:
            recorder.set_agent("system")

        responses = []
        start_time = time.time()

        try:
            for i, turn in enumerate(turns):
                if recorder:
                    recorder.record_step(
                        step_type=StepType.USER_INPUT,
                        input_data={"turn": i, "message": turn},
                    )

                turn_start = time.time()
                chat_fn = getattr(self.system, self.chat_method)
                response = chat_fn(turn)
                turn_latency = (time.time() - turn_start) * 1000

                responses.append(response)

                # 记录当前 Agent
                if recorder and hasattr(self.system, 'current_agent'):
                    current = self.system.current_agent
                    agent_name = current.value if hasattr(current, 'value') else str(current)

                    if recorder._current_agent != agent_name:
                        recorder.record_handoff(
                            from_agent=recorder._current_agent,
                            to_agent=agent_name,
                        )

                if recorder:
                    recorder.record_step(
                        step_type=StepType.AGENT_OUTPUT,
                        result=response,
                        latency_ms=turn_latency,
                    )

            total_latency = (time.time() - start_time) * 1000

            # 获取最终状态
            final_context = {}
            if hasattr(self.system, self.context_method):
                final_context = getattr(self.system, self.context_method)()

            # 判断成功
            final_response = responses[-1] if responses else ""
            success = self.success_checker(final_response, final_context)
            outcome = Outcome.SUCCESS if success else Outcome.PARTIAL

            if recorder:
                trajectory = recorder.finish(
                    response=final_response,
                    outcome=outcome,
                    reward=1.0 if success else 0.0,
                )
            else:
                trajectory = None

            return AgentResult(
                response=final_response,
                trajectory=trajectory,
                success=success,
                metadata={
                    "all_responses": responses,
                    "final_context": final_context,
                    "num_turns": len(turns),
                    "total_latency_ms": total_latency,
                },
            )

        except Exception as e:
            if recorder:
                trajectory = recorder.finish(
                    response=f"Error: {e}",
                    outcome=Outcome.ERROR,
                    reward=-1.0,
                )
            else:
                trajectory = None

            return AgentResult(
                response=f"Error: {e}",
                trajectory=trajectory,
                success=False,
            )

    def get_optimizable_params(self) -> dict[str, Any]:
        """获取可优化参数"""
        params = {}

        # 尝试获取各个 Agent 的 instructions
        if hasattr(self.system, 'agents'):
            for agent_type, agent in self.system.agents.items():
                if hasattr(agent, 'router') and hasattr(agent.router, 'signature'):
                    name = agent_type.value if hasattr(agent_type, 'value') else str(agent_type)
                    params[f'{name}_instructions'] = str(agent.router.signature.__doc__)

        return params

    def set_params(self, params: dict[str, Any]):
        """设置参数"""
        # 需要根据具体实现来设置
        pass


def wrap_agent(
    agent: Any,
    agent_type: str = "auto",
    **kwargs,
) -> OptimizableAgent:
    """
    便捷函数：包装 Agent

    Args:
        agent: Agent 实例
        agent_type: "react", "multi_agent", 或 "auto"
        **kwargs: 传递给包装器的参数

    Returns:
        OptimizableAgent 实例
    """
    if agent_type == "auto":
        # 自动检测
        if isinstance(agent, dspy.Module) and hasattr(agent, 'tools'):
            agent_type = "react"
        elif hasattr(agent, 'chat') and hasattr(agent, 'agents'):
            agent_type = "multi_agent"
        else:
            agent_type = "react"  # 默认

    if agent_type == "react":
        return DSPyReActWrapper(agent, **kwargs)
    elif agent_type == "multi_agent":
        return MultiAgentWrapper(agent, **kwargs)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
