"""Unified target protocol for security testing."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TargetType(Enum):
    """支持的目标类型"""
    GUARDRAIL = "guardrail"
    HTTP_AGENT = "http_agent"
    MCP_SERVER = "mcp_server"
    DSPY_MODULE = "dspy_module"


class TargetCapability(Enum):
    """目标能力标识"""
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    TOOL_USE = "tool_use"
    STREAMING = "streaming"
    SESSION = "session"


class TargetResponse(BaseModel):
    """统一响应格式"""
    response: str
    was_blocked: bool = False
    block_reason: str | None = None
    guardrail_scores: dict[str, float] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnifiedTarget(ABC):
    """统一目标接口"""

    target_type: TargetType
    capabilities: list[TargetCapability]

    @abstractmethod
    def invoke(self, prompt: str) -> TargetResponse:
        """单轮调用目标"""
        pass

    @abstractmethod
    def invoke_multi_turn(self, messages: list[dict[str, str]]) -> TargetResponse:
        """多轮调用目标"""
        pass

    @abstractmethod
    def reset_session(self) -> None:
        """重置会话状态"""
        pass

    def supports(self, capability: TargetCapability) -> bool:
        """检查是否支持指定能力"""
        return capability in self.capabilities

    def get_info(self) -> dict[str, Any]:
        """获取目标信息"""
        return {
            "type": self.target_type.value,
            "capabilities": [c.value for c in self.capabilities],
        }
