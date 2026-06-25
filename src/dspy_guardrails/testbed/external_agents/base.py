"""
ExternalAgent - 外部 Agent 连接基类

提供连接外部 AI Agent 平台的统一接口。
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ConnectionStatus(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class AuthMethod(Enum):
    """认证方式"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    CUSTOM = "custom"


@dataclass
class ExternalAgentConfig:
    """外部 Agent 配置基类"""
    name: str = "external_agent"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    auth_method: AuthMethod = AuthMethod.NONE
    auth_config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExternalAgentResponse:
    """外部 Agent 响应"""
    content: str
    blocked: bool = False
    block_reason: str | None = None
    latency_ms: float = 0.0
    raw_response: dict | None = None
    metadata: dict = field(default_factory=dict)

    # 兼容 AgentResponse 接口
    tool_calls: list = field(default_factory=list)
    error: str | None = None


class ExternalAgent(ABC):
    """
    外部 Agent 连接基类

    所有外部 Agent 适配器必须继承此类，实现统一的接口。
    支持护栏集成、会话管理、健康检查等。

    使用示例:
        class MyAgent(ExternalAgent):
            def _send_message(self, message: str, session_id: str) -> ExternalAgentResponse:
                # 实现具体的消息发送逻辑
                ...
    """

    def __init__(
        self,
        config: ExternalAgentConfig = None,
        guardrail_fn: Callable[[str, str], tuple[bool, str | None]] | None = None,
    ):
        self.config = config or ExternalAgentConfig()
        self.guardrail_fn = guardrail_fn
        self._status = ConnectionStatus.DISCONNECTED
        self._session_id: str | None = None
        self._conversation_history: list[dict] = []

    @property
    def status(self) -> ConnectionStatus:
        """获取连接状态"""
        return self._status

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._status == ConnectionStatus.CONNECTED

    @abstractmethod
    def connect(self) -> bool:
        """
        建立连接

        Returns:
            bool: 连接是否成功
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    def _send_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """
        发送消息到外部 Agent（子类实现）

        Args:
            message: 用户消息
            session_id: 会话ID

        Returns:
            ExternalAgentResponse: Agent 响应
        """
        pass

    def chat(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """
        与 Agent 对话（统一入口）

        集成护栏检查、会话管理等功能。

        Args:
            message: 用户消息
            session_id: 会话ID（可选）

        Returns:
            ExternalAgentResponse: Agent 响应
        """
        start_time = time.time()

        # 自动连接
        if not self.is_connected:
            if not self.connect():
                return ExternalAgentResponse(
                    content="",
                    blocked=True,
                    block_reason="连接失败",
                    error="Failed to connect to external agent",
                )

        # 输入护栏检查
        if self.guardrail_fn:
            passed, reason = self.guardrail_fn(message, "input")
            if not passed:
                return ExternalAgentResponse(
                    content="",
                    blocked=True,
                    block_reason=reason,
                    latency_ms=(time.time() - start_time) * 1000,
                )

        # 发送消息
        try:
            response = self._send_message(message, session_id or self._session_id)
        except Exception as e:
            return ExternalAgentResponse(
                content="",
                blocked=True,
                block_reason=f"请求失败: {str(e)}",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

        # 输出护栏检查
        if self.guardrail_fn and response.content:
            passed, reason = self.guardrail_fn(response.content, "output")
            if not passed:
                response.blocked = True
                response.block_reason = reason

        # 更新延迟
        response.latency_ms = (time.time() - start_time) * 1000

        # 记录对话历史
        self._conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": start_time,
        })
        if response.content:
            self._conversation_history.append({
                "role": "assistant",
                "content": response.content,
                "timestamp": time.time(),
                "blocked": response.blocked,
            })

        return response

    def reset(self) -> None:
        """重置会话状态"""
        self._conversation_history = []
        self._session_id = None

    def get_conversation_history(self) -> list[dict]:
        """获取对话历史"""
        return self._conversation_history.copy()

    @abstractmethod
    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 服务是否健康
        """
        pass

    def get_info(self) -> dict:
        """获取 Agent 信息"""
        return {
            "name": self.config.name,
            "type": self.__class__.__name__,
            "status": self._status.value,
            "auth_method": self.config.auth_method.value,
            "has_guardrail": self.guardrail_fn is not None,
            "conversation_length": len(self._conversation_history),
        }

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()
        return False
