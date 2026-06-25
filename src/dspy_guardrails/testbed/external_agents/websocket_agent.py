"""
WebSocketAgent - WebSocket 实时连接器

支持需要实时双向通信的 Agent 服务。
"""

import json
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from .base import (
    ConnectionStatus,
    ExternalAgent,
    ExternalAgentConfig,
    ExternalAgentResponse,
)


@dataclass
class WebSocketAgentConfig(ExternalAgentConfig):
    """WebSocket Agent 配置"""
    # 连接配置
    ws_url: str = "ws://localhost:8000/ws"
    ping_interval: float = 30.0
    ping_timeout: float = 10.0

    # 认证
    auth_token: str | None = None
    auth_message: dict | None = None  # 连接后发送的认证消息

    # 消息格式
    message_template: dict = field(default_factory=lambda: {
        "type": "message",
        "content": "{message}",
        "session_id": "{session_id}",
    })
    response_content_path: str = "content"
    response_type_field: str = "type"
    response_type_value: str = "response"  # 响应消息的 type 值

    # 额外 headers
    headers: dict = field(default_factory=dict)


class WebSocketAgent(ExternalAgent):
    """
    WebSocket Agent 连接器

    支持：
    - 实时双向通信
    - 自动重连
    - 心跳检测
    - 异步消息处理

    使用示例:
        agent = WebSocketAgent(
            ws_url="ws://localhost:8000/ws",
            auth_token="your-token",
        )

        with agent:
            response = agent.chat("Hello!")
    """

    def __init__(
        self,
        ws_url: str = None,
        auth_token: str = None,
        config: WebSocketAgentConfig = None,
        guardrail_fn: Callable | None = None,
        **kwargs,
    ):
        if not WEBSOCKET_AVAILABLE:
            raise ImportError(
                "websocket-client is required for WebSocketAgent. "
                "Install it with: pip install websocket-client"
            )

        if config is None:
            config = WebSocketAgentConfig(
                ws_url=ws_url or "ws://localhost:8000/ws",
                auth_token=auth_token,
                **kwargs,
            )

        super().__init__(config, guardrail_fn)
        self.config: WebSocketAgentConfig = config

        self._ws: websocket.WebSocketApp | None = None
        self._ws_thread: threading.Thread | None = None
        self._response_queue: queue.Queue = queue.Queue()
        self._pending_responses: dict[str, Future] = {}
        self._message_id = 0
        self._lock = threading.Lock()

    def _get_next_message_id(self) -> str:
        """获取下一个消息 ID"""
        with self._lock:
            self._message_id += 1
            return f"msg_{self._message_id}"

    def _on_open(self, ws):
        """连接打开回调"""
        self._status = ConnectionStatus.CONNECTED

        # 发送认证消息
        if self.config.auth_message:
            ws.send(json.dumps(self.config.auth_message))
        elif self.config.auth_token:
            ws.send(json.dumps({
                "type": "auth",
                "token": self.config.auth_token,
            }))

    def _on_message(self, ws, message):
        """收到消息回调"""
        try:
            data = json.loads(message)

            # 检查是否是响应消息
            msg_type = data.get(self.config.response_type_field, "")
            if msg_type == self.config.response_type_value:
                self._response_queue.put(data)

            # 检查是否有对应的 pending future
            msg_id = data.get("message_id")
            if msg_id and msg_id in self._pending_responses:
                future = self._pending_responses.pop(msg_id)
                if not future.done():
                    future.set_result(data)

        except json.JSONDecodeError:
            # 非 JSON 消息，直接放入队列
            self._response_queue.put({"content": message})

    def _on_error(self, ws, error):
        """错误回调"""
        self._status = ConnectionStatus.ERROR

    def _on_close(self, ws, close_status_code, close_msg):
        """连接关闭回调"""
        self._status = ConnectionStatus.DISCONNECTED

    def connect(self) -> bool:
        """建立 WebSocket 连接"""
        try:
            self._status = ConnectionStatus.CONNECTING

            # 构建 headers
            headers = dict(self.config.headers)
            if self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"

            # 创建 WebSocket
            self._ws = websocket.WebSocketApp(
                self.config.ws_url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            # 在后台线程运行
            self._ws_thread = threading.Thread(
                target=self._ws.run_forever,
                kwargs={
                    "ping_interval": self.config.ping_interval,
                    "ping_timeout": self.config.ping_timeout,
                },
                daemon=True,
            )
            self._ws_thread.start()

            # 等待连接建立
            timeout = 10.0
            start = time.time()
            while self._status == ConnectionStatus.CONNECTING:
                if time.time() - start > timeout:
                    self._status = ConnectionStatus.ERROR
                    return False
                time.sleep(0.1)

            return self._status == ConnectionStatus.CONNECTED

        except Exception:
            self._status = ConnectionStatus.ERROR
            return False

    def disconnect(self) -> None:
        """断开连接"""
        if self._ws:
            self._ws.close()
            self._ws = None

        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5.0)
            self._ws_thread = None

        self._status = ConnectionStatus.DISCONNECTED

    def _build_message(
        self,
        message: str,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> dict:
        """构建消息"""
        msg = {}

        for key, template in self.config.message_template.items():
            if isinstance(template, str):
                value = template.format(
                    message=message,
                    session_id=session_id or "",
                )
                if value:
                    msg[key] = value
            else:
                msg[key] = template

        if message_id:
            msg["message_id"] = message_id

        return msg

    def _extract_response(self, data: dict) -> tuple[str, bool, str | None]:
        """从响应数据中提取内容"""
        content = data
        for key in self.config.response_content_path.split("."):
            if isinstance(content, dict) and key in content:
                content = content[key]
            else:
                content = str(data)
                break

        if not isinstance(content, str):
            content = str(content)

        blocked = data.get("blocked", False)
        block_reason = data.get("block_reason")

        return content, blocked, block_reason

    def _send_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """发送消息并等待响应"""
        if not self._ws or self._status != ConnectionStatus.CONNECTED:
            raise RuntimeError("Not connected")

        message_id = self._get_next_message_id()
        msg = self._build_message(message, session_id, message_id)

        # 创建 Future 等待响应
        future = Future()
        self._pending_responses[message_id] = future

        # 清空之前的响应
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break

        # 发送消息
        self._ws.send(json.dumps(msg))

        # 等待响应
        try:
            # 优先使用 Future
            data = future.result(timeout=self.config.timeout)
        except FutureTimeoutError:
            # 尝试从队列获取
            try:
                data = self._response_queue.get(timeout=1.0)
            except queue.Empty:
                return ExternalAgentResponse(
                    content="",
                    blocked=True,
                    block_reason="响应超时",
                    error="Response timeout",
                )
        finally:
            # 清理
            self._pending_responses.pop(message_id, None)

        content, blocked, block_reason = self._extract_response(data)

        return ExternalAgentResponse(
            content=content,
            blocked=blocked,
            block_reason=block_reason,
            raw_response=data,
        )

    def health_check(self) -> bool:
        """健康检查"""
        return (
            self._ws is not None and
            self._status == ConnectionStatus.CONNECTED
        )

    def send_raw(self, data: dict) -> None:
        """发送原始消息（不等待响应）"""
        if not self._ws or self._status != ConnectionStatus.CONNECTED:
            raise RuntimeError("Not connected")
        self._ws.send(json.dumps(data))
