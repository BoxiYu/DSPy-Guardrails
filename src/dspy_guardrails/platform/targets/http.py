"""HTTPTarget - Adapter for HTTP-based AI agents."""

from typing import Any

import requests

from .protocol import (
    TargetCapability,
    TargetResponse,
    TargetType,
    UnifiedTarget,
)


class HTTPTarget(UnifiedTarget):
    """
    HTTP Agent 适配器

    通过 HTTP API 调用 AI Agent，支持单轮和多轮对话。

    Features:
        - 可配置的请求方法、端点和超时
        - 自定义响应解析键
        - 会话 ID 自动追踪
        - 完整的错误处理

    Example:
        >>> target = HTTPTarget(
        ...     base_url="http://localhost:8000",
        ...     endpoint="/api/chat",
        ... )
        >>> response = target.invoke("Hello")
        >>> print(response.response)
        'Hello! How can I help you?'

    Example with authentication:
        >>> target = HTTPTarget(
        ...     base_url="http://localhost:8000",
        ...     endpoint="/api/chat",
        ...     headers={"Authorization": "Bearer token123"},
        ... )

    Example with custom response keys:
        >>> target = HTTPTarget(
        ...     base_url="http://localhost:8000",
        ...     endpoint="/api/chat",
        ...     response_key="message",      # Default: "response"
        ...     blocked_key="is_blocked",    # Default: "blocked"
        ...     block_reason_key="reason",   # Default: "block_reason"
        ... )
    """

    target_type = TargetType.HTTP_AGENT
    capabilities = [
        TargetCapability.SINGLE_TURN,
        TargetCapability.MULTI_TURN,
        TargetCapability.SESSION,
    ]

    def __init__(
        self,
        base_url: str,
        endpoint: str = "/chat",
        method: str = "POST",
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        response_key: str = "response",
        blocked_key: str = "blocked",
        block_reason_key: str = "block_reason",
        session_enabled: bool = True,
    ):
        """
        初始化 HTTP 目标

        Args:
            base_url: 基础 URL (e.g., "http://localhost:8000")
            endpoint: API 端点 (e.g., "/chat" or "/api/v1/chat")
            method: HTTP 方法 (默认 "POST")
            headers: 请求头字典 (e.g., {"Authorization": "Bearer token"})
            timeout: 超时时间 (秒, 默认 30)
            response_key: 响应文本的 JSON 键 (默认 "response")
            blocked_key: 拦截状态的 JSON 键 (默认 "blocked")
            block_reason_key: 拦截原因的 JSON 键 (默认 "block_reason")
            session_enabled: 是否启用会话追踪 (默认 True)
        """
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._method = method.upper()
        self._headers = headers or {}
        self._timeout = timeout
        self._response_key = response_key
        self._blocked_key = blocked_key
        self._block_reason_key = block_reason_key
        self._session_enabled = session_enabled

        # 会话状态
        self._session_id: str | None = None
        self._conversation: list[dict[str, str]] = []

    def _build_url(self) -> str:
        """
        构建完整 URL

        处理各种边界情况:
        - base_url 带尾斜杠: "http://host/" -> "http://host"
        - endpoint 不带前导斜杠: "api/chat" -> "/api/chat"

        Returns:
            完整的 API URL
        """
        endpoint = self._endpoint if self._endpoint.startswith("/") else f"/{self._endpoint}"
        return f"{self._base_url}{endpoint}"

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        """
        构建请求负载

        Args:
            prompt: 用户输入

        Returns:
            请求 JSON 负载
        """
        payload: dict[str, Any] = {"message": prompt}
        if self._session_enabled and self._session_id:
            payload["session_id"] = self._session_id
        return payload

    def _parse_response(self, data: dict[str, Any], latency_ms: float) -> TargetResponse:
        """
        解析 HTTP 响应

        Args:
            data: JSON 响应数据
            latency_ms: 请求延迟 (毫秒)

        Returns:
            统一响应格式
        """
        return TargetResponse(
            response=data.get(self._response_key, str(data)),
            was_blocked=data.get(self._blocked_key, False),
            block_reason=data.get(self._block_reason_key),
            guardrail_scores=data.get("guardrail_scores", {}),
            tool_calls=data.get("tool_calls", []),
            latency_ms=latency_ms,
            metadata={"raw_response": data},
        )

    def invoke(self, prompt: str) -> TargetResponse:
        """
        执行单轮调用

        Args:
            prompt: 用户输入

        Returns:
            统一响应格式

        Example:
            >>> target = HTTPTarget(base_url="http://localhost:8000")
            >>> response = target.invoke("What's my flight status?")
            >>> if response.was_blocked:
            ...     print(f"Blocked: {response.block_reason}")
            ... else:
            ...     print(response.response)
        """
        url = self._build_url()
        payload = self._build_payload(prompt)

        try:
            response = requests.request(
                method=self._method,
                url=url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()

            latency_ms = response.elapsed.total_seconds() * 1000
            data = response.json()

            # 更新会话 ID (如果服务器返回)
            if self._session_enabled and "session_id" in data:
                self._session_id = data["session_id"]

            return self._parse_response(data, latency_ms)

        except requests.RequestException as e:
            return TargetResponse(
                response=f"HTTP Error: {str(e)}",
                was_blocked=False,
                metadata={"error": str(e)},
            )

    def invoke_multi_turn(
        self,
        messages: list[dict[str, str]],
    ) -> TargetResponse:
        """
        执行多轮调用

        Args:
            messages: 消息列表，每条消息包含 "role" 和 "content"

        Returns:
            统一响应格式

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ...     {"role": "user", "content": "Book me a flight"},
            ... ]
            >>> response = target.invoke_multi_turn(messages)
        """
        url = self._build_url()
        payload: dict[str, Any] = {
            "messages": messages,
        }
        if self._session_enabled and self._session_id:
            payload["session_id"] = self._session_id

        try:
            response = requests.request(
                method=self._method,
                url=url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
            response.raise_for_status()

            latency_ms = response.elapsed.total_seconds() * 1000
            data = response.json()

            # 更新会话 ID
            if self._session_enabled and "session_id" in data:
                self._session_id = data["session_id"]

            return self._parse_response(data, latency_ms)

        except requests.RequestException as e:
            return TargetResponse(
                response=f"HTTP Error: {str(e)}",
                was_blocked=False,
                metadata={"error": str(e)},
            )

    def reset_session(self) -> None:
        """
        重置会话状态

        清除会话 ID 和对话历史，开始新的会话。

        Example:
            >>> target.invoke("Hello")  # Start conversation
            >>> target.reset_session()   # Clear session
            >>> target.invoke("Hi")      # New conversation
        """
        self._session_id = None
        self._conversation = []
