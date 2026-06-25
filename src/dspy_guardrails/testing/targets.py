"""
Test Targets - 测试目标接口

定义安全测试的目标接口和常见实现。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TargetResponse:
    """测试目标的响应"""

    response: str
    guardrail_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def was_blocked(self) -> bool:
        """检查是否被任何 guardrail 拦截"""
        for _name, status in self.guardrail_status.items():
            if not status.get("passed", True):
                return True
        return False

    @property
    def blocking_guardrail(self) -> str | None:
        """获取拦截的 guardrail 名称"""
        for name, status in self.guardrail_status.items():
            if not status.get("passed", True):
                return name
        return None


@dataclass
class ConversationTurn:
    """对话的单轮"""
    role: str  # "user" or "assistant"
    content: str
    guardrail_status: dict[str, dict[str, Any]] = field(default_factory=dict)


class BaseTarget(ABC):
    """
    测试目标抽象基类

    所有测试目标必须实现此接口。
    """

    def __init__(self):
        self.conversation_history: list[ConversationTurn] = []

    @abstractmethod
    def invoke(self, prompt: str) -> TargetResponse:
        """
        调用目标

        Args:
            prompt: 用户消息

        Returns:
            TargetResponse: 包含响应和 guardrail 状态
        """
        pass

    @abstractmethod
    def reset_session(self) -> None:
        """重置会话"""
        pass

    def get_conversation_history(self) -> list[ConversationTurn]:
        """获取对话历史"""
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """清除对话历史"""
        self.conversation_history = []


class MockTarget(BaseTarget):
    """
    Mock 测试目标

    用于测试框架本身或在没有真实目标时进行演示。

    Examples:
        # 默认行为：总是允许
        target = MockTarget()

        # 自定义响应生成
        target = MockTarget(
            response_fn=lambda p: f"Response to: {p}",
            block_fn=lambda p: "ignore" in p.lower(),
        )
    """

    def __init__(
        self,
        response_fn: Callable[[str], str] | None = None,
        block_fn: Callable[[str], bool] | None = None,
        guardrail_name: str = "mock_guardrail",
    ):
        super().__init__()
        self.response_fn = response_fn or (lambda p: f"Mock response to: {p[:50]}...")
        self.block_fn = block_fn or (lambda p: False)
        self.guardrail_name = guardrail_name

    def invoke(self, prompt: str) -> TargetResponse:
        """调用 mock 目标"""
        import time
        start = time.time()

        blocked = self.block_fn(prompt)

        if blocked:
            response = "I cannot help with that request."
            guardrail_status = {
                self.guardrail_name: {
                    "passed": False,
                    "reason": "Blocked by mock guardrail",
                }
            }
        else:
            response = self.response_fn(prompt)
            guardrail_status = {
                self.guardrail_name: {
                    "passed": True,
                }
            }

        latency = (time.time() - start) * 1000

        # 记录历史
        self.conversation_history.append(
            ConversationTurn(role="user", content=prompt)
        )
        self.conversation_history.append(
            ConversationTurn(
                role="assistant",
                content=response,
                guardrail_status=guardrail_status,
            )
        )

        return TargetResponse(
            response=response,
            guardrail_status=guardrail_status,
            latency_ms=latency,
        )

    def reset_session(self) -> None:
        """重置会话"""
        self.clear_history()


class GuardrailTarget(BaseTarget):
    """
    包装 Guardrail 函数的测试目标

    将 guardrail 函数包装为可测试的目标。

    Examples:
        from dspy_guardrails import guardrail

        target = GuardrailTarget(
            guardrail_fn=guardrail.no_injection,
            response_fn=lambda p: f"Processed: {p}",
        )
    """

    def __init__(
        self,
        guardrail_fn: Callable[[str], bool],
        response_fn: Callable[[str], str] | None = None,
        guardrail_name: str = "guardrail",
    ):
        super().__init__()
        self.guardrail_fn = guardrail_fn
        self.response_fn = response_fn or (lambda p: f"Response: {p[:100]}...")
        self.guardrail_name = guardrail_name

    def invoke(self, prompt: str) -> TargetResponse:
        """调用目标"""
        import time
        start = time.time()

        # 运行 guardrail
        try:
            passed = self.guardrail_fn(prompt)
        except Exception:
            passed = False

        if passed:
            response = self.response_fn(prompt)
        else:
            response = "Request blocked by guardrail."

        guardrail_status = {
            self.guardrail_name: {
                "passed": passed,
            }
        }

        latency = (time.time() - start) * 1000

        return TargetResponse(
            response=response,
            guardrail_status=guardrail_status,
            latency_ms=latency,
        )

    def reset_session(self) -> None:
        self.clear_history()
