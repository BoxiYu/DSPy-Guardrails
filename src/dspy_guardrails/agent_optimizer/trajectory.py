"""
Trajectory Recording Layer
==========================

记录 Agent 执行的完整轨迹，用于后续分析和优化。
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(Enum):
    """步骤类型"""
    AGENT_CALL = "agent_call"       # Agent 被调用
    TOOL_CALL = "tool_call"         # 工具被调用
    HANDOFF = "handoff"             # Agent 间转移
    LLM_CALL = "llm_call"           # LLM 推理
    USER_INPUT = "user_input"       # 用户输入
    AGENT_OUTPUT = "agent_output"   # Agent 输出


@dataclass
class Step:
    """
    单步执行记录

    记录 Agent 执行过程中的每一个关键决策点。
    """
    # 基础信息
    step_id: int
    step_type: StepType
    timestamp: float = field(default_factory=time.time)

    # 上下文
    agent_name: str = ""
    state_before: dict[str, Any] = field(default_factory=dict)

    # 输入
    input_data: dict[str, Any] = field(default_factory=dict)

    # 决策 (如果是 LLM 调用)
    thought: str = ""
    reasoning: str = ""

    # 动作
    action_type: str = ""           # 如: "tool_call", "handoff", "respond"
    action_name: str = ""           # 如: "flight_status", "booking_agent"
    action_args: dict[str, Any] = field(default_factory=dict)

    # 结果
    result: Any = None
    state_after: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    # 性能
    latency_ms: float = 0.0
    token_count: int = 0

    # 优化相关 (后续填充)
    reward: float = 0.0             # 这一步的奖励
    advantage: float = 0.0          # 相对优势
    credit: float = 0.0             # Credit assignment 分配的分数

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "input": self.input_data,
            "thought": self.thought,
            "action": {
                "type": self.action_type,
                "name": self.action_name,
                "args": self.action_args,
            },
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "reward": self.reward,
            "credit": self.credit,
        }


class Outcome(Enum):
    """任务结果"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Trajectory:
    """
    完整执行轨迹

    记录从用户输入到最终输出的完整过程。
    """
    # 基础信息
    trajectory_id: str = ""
    task: str = ""                          # 原始任务/用户输入
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    # 步骤记录
    steps: list[Step] = field(default_factory=list)

    # 结果
    final_response: str = ""
    outcome: Outcome = Outcome.FAILURE
    final_reward: float = 0.0

    # 元数据
    metadata: dict[str, Any] = field(default_factory=dict)

    # 统计
    total_steps: int = 0
    total_tool_calls: int = 0
    total_handoffs: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    def add_step(self, step: Step):
        """添加步骤"""
        self.steps.append(step)
        self.total_steps += 1
        self.total_latency_ms += step.latency_ms
        self.total_tokens += step.token_count

        if step.step_type == StepType.TOOL_CALL:
            self.total_tool_calls += 1
        elif step.step_type == StepType.HANDOFF:
            self.total_handoffs += 1

    def finish(self, response: str, outcome: Outcome, reward: float = 0.0):
        """完成轨迹记录"""
        self.end_time = time.time()
        self.final_response = response
        self.outcome = outcome
        self.final_reward = reward

    def get_tools_used(self) -> list[str]:
        """获取使用的工具列表"""
        return [
            s.action_name for s in self.steps
            if s.step_type == StepType.TOOL_CALL and s.action_name
        ]

    def get_agents_visited(self) -> list[str]:
        """获取经过的 Agent 列表"""
        agents = []
        for s in self.steps:
            if s.agent_name and (not agents or agents[-1] != s.agent_name):
                agents.append(s.agent_name)
        return agents

    def get_handoff_chain(self) -> list[tuple]:
        """获取 Handoff 链"""
        handoffs = []
        for s in self.steps:
            if s.step_type == StepType.HANDOFF:
                handoffs.append((s.agent_name, s.action_name))
        return handoffs

    def get_critical_steps(self, threshold: float = 0.5) -> list[Step]:
        """获取关键步骤 (高 credit)"""
        return [s for s in self.steps if abs(s.credit) >= threshold]

    def get_failure_point(self) -> Step | None:
        """获取失败点 (如果有)"""
        for s in reversed(self.steps):
            if s.error:
                return s
            if s.credit < -0.3:  # 负向贡献
                return s
        return None

    def duration_seconds(self) -> float:
        """执行时长"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "trajectory_id": self.trajectory_id,
            "task": self.task,
            "outcome": self.outcome.value,
            "final_reward": self.final_reward,
            "duration_seconds": self.duration_seconds(),
            "stats": {
                "total_steps": self.total_steps,
                "total_tool_calls": self.total_tool_calls,
                "total_handoffs": self.total_handoffs,
                "total_latency_ms": self.total_latency_ms,
                "total_tokens": self.total_tokens,
            },
            "tools_used": self.get_tools_used(),
            "agents_visited": self.get_agents_visited(),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """生成摘要"""
        return (
            f"Trajectory[{self.trajectory_id[:8]}]: "
            f"{self.outcome.value} | "
            f"{self.total_steps} steps | "
            f"{self.total_tool_calls} tools | "
            f"{self.total_handoffs} handoffs | "
            f"{self.duration_seconds():.2f}s | "
            f"reward={self.final_reward:.2f}"
        )


class TrajectoryRecorder:
    """
    轨迹记录器

    用于在 Agent 执行过程中记录轨迹。
    """

    def __init__(self, task: str = "", trajectory_id: str = ""):
        self.trajectory = Trajectory(
            trajectory_id=trajectory_id or self._generate_id(),
            task=task,
        )
        self._step_counter = 0
        self._current_agent = ""
        self._state_snapshot: dict[str, Any] = {}

    def _generate_id(self) -> str:
        """生成轨迹 ID"""
        import random
        import string
        return "traj_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

    def set_agent(self, agent_name: str):
        """设置当前 Agent"""
        self._current_agent = agent_name

    def snapshot_state(self, state: dict[str, Any]):
        """保存状态快照"""
        self._state_snapshot = state.copy() if state else {}

    def record_step(
        self,
        step_type: StepType,
        input_data: dict[str, Any] = None,
        thought: str = "",
        reasoning: str = "",
        action_type: str = "",
        action_name: str = "",
        action_args: dict[str, Any] = None,
        result: Any = None,
        error: str = None,
        latency_ms: float = 0.0,
        token_count: int = 0,
        state_after: dict[str, Any] = None,
    ) -> Step:
        """记录一个步骤"""
        step = Step(
            step_id=self._step_counter,
            step_type=step_type,
            agent_name=self._current_agent,
            state_before=self._state_snapshot.copy(),
            input_data=input_data or {},
            thought=thought,
            reasoning=reasoning,
            action_type=action_type,
            action_name=action_name,
            action_args=action_args or {},
            result=result,
            state_after=state_after or {},
            error=error,
            latency_ms=latency_ms,
            token_count=token_count,
        )

        self.trajectory.add_step(step)
        self._step_counter += 1

        # 更新状态快照
        if state_after:
            self._state_snapshot = state_after.copy()

        return step

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        latency_ms: float = 0.0,
        error: str = None,
    ) -> Step:
        """记录工具调用"""
        return self.record_step(
            step_type=StepType.TOOL_CALL,
            action_type="tool_call",
            action_name=tool_name,
            action_args=args,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )

    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        reason: str = "",
    ) -> Step:
        """记录 Agent Handoff"""
        self._current_agent = to_agent
        return self.record_step(
            step_type=StepType.HANDOFF,
            action_type="handoff",
            action_name=to_agent,
            action_args={"from": from_agent, "reason": reason},
        )

    def record_llm_call(
        self,
        input_prompt: str,
        thought: str,
        reasoning: str,
        output: str,
        latency_ms: float = 0.0,
        token_count: int = 0,
    ) -> Step:
        """记录 LLM 调用"""
        return self.record_step(
            step_type=StepType.LLM_CALL,
            input_data={"prompt": input_prompt[:500]},
            thought=thought,
            reasoning=reasoning,
            result=output,
            latency_ms=latency_ms,
            token_count=token_count,
        )

    def finish(
        self,
        response: str,
        outcome: Outcome,
        reward: float = 0.0,
    ) -> Trajectory:
        """完成记录"""
        self.trajectory.finish(response, outcome, reward)
        return self.trajectory

    def get_trajectory(self) -> Trajectory:
        """获取轨迹"""
        return self.trajectory


# 便捷函数
def create_recorder(task: str = "") -> TrajectoryRecorder:
    """创建轨迹记录器"""
    return TrajectoryRecorder(task=task)
