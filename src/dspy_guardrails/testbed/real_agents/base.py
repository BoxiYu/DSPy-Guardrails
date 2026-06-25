"""
Agent 基类 - Base Agent Classes
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class AgentType(Enum):
    """Agent 类型"""
    SIMPLE = "simple"         # 简单 LLM 包装
    TOOLS = "tools"           # ReAct 工具调用
    RAG = "rag"               # 检索增强生成
    MULTI_AGENT = "multi"     # 多 Agent 系统


class ProtectionLevel(Enum):
    """保护等级"""
    NONE = "none"             # 无保护
    PARTIAL = "partial"       # 部分保护（模式匹配）
    FULL = "full"             # 完全保护（模式 + LLM）


@dataclass
class AgentResponse:
    """Agent 响应"""
    content: str                              # 响应内容
    success: bool = True                      # 是否成功
    blocked: bool = False                     # 是否被护栏拦截
    block_reason: str | None = None        # 拦截原因
    tool_calls: list = field(default_factory=list)  # 工具调用记录
    metadata: dict = field(default_factory=dict)    # 元数据

    def __str__(self) -> str:
        if self.blocked:
            return f"[BLOCKED] {self.block_reason}"
        return self.content


@dataclass
class ConversationTurn:
    """对话轮次"""
    role: str                # user / assistant / system
    content: str             # 消息内容
    timestamp: float = 0.0   # 时间戳
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(
        self,
        name: str = "Agent",
        system_prompt: str = "",
        protection_level: ProtectionLevel = ProtectionLevel.NONE,
        guardrail_fn: Callable | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.protection_level = protection_level
        self.guardrail_fn = guardrail_fn
        self.conversation_history: list[ConversationTurn] = []
        self._call_count = 0

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Agent 类型"""
        pass

    @abstractmethod
    def _generate(self, user_input: str) -> str:
        """内部生成逻辑（子类实现）"""
        pass

    def chat(self, user_input: str) -> AgentResponse:
        """
        对话接口

        Args:
            user_input: 用户输入

        Returns:
            AgentResponse: Agent 响应
        """
        self._call_count += 1

        # 输入护栏检查
        if self.guardrail_fn and self.protection_level != ProtectionLevel.NONE:
            is_safe, reason = self._check_input(user_input)
            if not is_safe:
                return AgentResponse(
                    content="",
                    success=False,
                    blocked=True,
                    block_reason=f"输入被拦截: {reason}",
                    metadata={"stage": "input", "protection_level": self.protection_level.value}
                )

        # 添加到对话历史
        import time
        self.conversation_history.append(
            ConversationTurn(role="user", content=user_input, timestamp=time.time())
        )

        # 生成响应
        try:
            response_content = self._generate(user_input)
        except Exception as e:
            return AgentResponse(
                content="",
                success=False,
                blocked=False,
                metadata={"error": str(e)}
            )

        # 输出护栏检查
        if self.guardrail_fn and self.protection_level != ProtectionLevel.NONE:
            is_safe, reason = self._check_output(response_content)
            if not is_safe:
                return AgentResponse(
                    content="",
                    success=False,
                    blocked=True,
                    block_reason=f"输出被拦截: {reason}",
                    metadata={"stage": "output", "protection_level": self.protection_level.value}
                )

        # 添加助手响应到历史
        self.conversation_history.append(
            ConversationTurn(role="assistant", content=response_content, timestamp=time.time())
        )

        return AgentResponse(
            content=response_content,
            success=True,
            blocked=False,
            metadata={"agent_type": self.agent_type.value}
        )

    def _check_input(self, text: str) -> tuple[bool, str | None]:
        """检查输入"""
        if not self.guardrail_fn:
            return True, None

        try:
            result = self.guardrail_fn(text, "input")
            if isinstance(result, bool):
                return result, None if result else "输入不安全"
            elif isinstance(result, tuple):
                return result
            else:
                return True, None
        except Exception:
            return True, None  # 护栏失败时允许通过

    def _check_output(self, text: str) -> tuple[bool, str | None]:
        """检查输出"""
        if not self.guardrail_fn:
            return True, None

        try:
            result = self.guardrail_fn(text, "output")
            if isinstance(result, bool):
                return result, None if result else "输出不安全"
            elif isinstance(result, tuple):
                return result
            else:
                return True, None
        except Exception:
            return True, None

    def reset(self):
        """重置对话状态"""
        self.conversation_history.clear()

    def get_history(self) -> list[ConversationTurn]:
        """获取对话历史"""
        return self.conversation_history.copy()

    @property
    def call_count(self) -> int:
        """调用次数"""
        return self._call_count

    def reset_count(self):
        """重置计数"""
        self._call_count = 0
