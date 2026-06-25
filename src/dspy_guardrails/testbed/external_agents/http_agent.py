"""
HTTPAgent - 通用 HTTP API 连接器

支持连接任何提供 REST API 的 Agent 服务。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import (
    ConnectionStatus,
    ExternalAgent,
    ExternalAgentConfig,
    ExternalAgentResponse,
)


@dataclass
class HTTPAgentConfig(ExternalAgentConfig):
    """HTTP Agent 配置"""
    # 连接配置
    base_url: str = "http://localhost:8000"
    chat_endpoint: str = "/chat"
    health_endpoint: str = "/health"

    # 请求配置
    method: str = "POST"
    headers: dict = field(default_factory=dict)

    # 认证
    auth_token: str | None = None
    api_key: str | None = None
    api_key_header: str = "X-API-Key"

    # 请求/响应映射
    request_template: dict = field(default_factory=lambda: {
        "message": "{message}",
        "session_id": "{session_id}",
    })
    response_content_path: str = "response"  # JSON 路径到响应内容
    response_blocked_path: str | None = None  # JSON 路径到 blocked 字段

    # SSL
    verify_ssl: bool = True
    cert_path: str | None = None


class HTTPAgent(ExternalAgent):
    """
    通用 HTTP API Agent 连接器

    支持：
    - 多种认证方式（API Key、Bearer Token、Basic Auth）
    - 自定义请求/响应映射
    - 自动重试
    - 会话管理

    使用示例:
        agent = HTTPAgent(
            base_url="http://localhost:8000",
            chat_endpoint="/chat",
            auth_token="your-token",
        )

        response = agent.chat("Hello!")
    """

    def __init__(
        self,
        base_url: str = None,
        chat_endpoint: str = None,
        auth_token: str = None,
        api_key: str = None,
        config: HTTPAgentConfig = None,
        guardrail_fn: Callable | None = None,
        **kwargs,
    ):
        # 构建配置
        if config is None:
            config = HTTPAgentConfig(
                base_url=base_url or "http://localhost:8000",
                chat_endpoint=chat_endpoint or "/chat",
                auth_token=auth_token,
                api_key=api_key,
                **kwargs,
            )

        super().__init__(config, guardrail_fn)
        self.config: HTTPAgentConfig = config
        self._session: requests.Session | None = None

    def _create_session(self) -> requests.Session:
        """创建 HTTP 会话"""
        session = requests.Session()

        # 配置重试
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 设置默认 headers
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        session.headers.update(self.config.headers)

        # 设置认证
        if self.config.auth_token:
            session.headers["Authorization"] = f"Bearer {self.config.auth_token}"
        elif self.config.api_key:
            session.headers[self.config.api_key_header] = self.config.api_key

        # SSL 配置
        session.verify = self.config.verify_ssl
        if self.config.cert_path:
            session.cert = self.config.cert_path

        return session

    def connect(self) -> bool:
        """建立连接"""
        try:
            self._status = ConnectionStatus.CONNECTING
            self._session = self._create_session()

            # 健康检查
            if self.health_check():
                self._status = ConnectionStatus.CONNECTED
                return True
            else:
                self._status = ConnectionStatus.ERROR
                return False

        except Exception:
            self._status = ConnectionStatus.ERROR
            return False

    def disconnect(self) -> None:
        """断开连接"""
        if self._session:
            self._session.close()
            self._session = None
        self._status = ConnectionStatus.DISCONNECTED

    def _build_request_body(
        self,
        message: str,
        session_id: str | None = None,
    ) -> dict:
        """构建请求体"""
        body = {}

        for key, template in self.config.request_template.items():
            if isinstance(template, str):
                value = template.format(
                    message=message,
                    session_id=session_id or "",
                )
                # 跳过空值
                if value and value != "":
                    body[key] = value
            else:
                body[key] = template

        return body

    def _extract_response(self, data: dict) -> tuple[str, bool, str | None]:
        """从响应数据中提取内容"""
        # 提取响应内容
        content = data
        for key in self.config.response_content_path.split("."):
            if isinstance(content, dict) and key in content:
                content = content[key]
            else:
                content = str(data)
                break

        if not isinstance(content, str):
            content = str(content)

        # 提取 blocked 状态
        blocked = False
        block_reason = None
        if self.config.response_blocked_path:
            blocked_value = data
            for key in self.config.response_blocked_path.split("."):
                if isinstance(blocked_value, dict) and key in blocked_value:
                    blocked_value = blocked_value[key]
                else:
                    blocked_value = None
                    break
            if blocked_value is not None:
                blocked = bool(blocked_value)

        return content, blocked, block_reason

    def _send_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """发送消息"""
        if not self._session:
            raise RuntimeError("Not connected")

        url = f"{self.config.base_url.rstrip('/')}{self.config.chat_endpoint}"
        body = self._build_request_body(message, session_id)

        try:
            response = self._session.request(
                method=self.config.method,
                url=url,
                json=body,
                timeout=self.config.timeout,
            )
            response.raise_for_status()

            data = response.json()
            content, blocked, block_reason = self._extract_response(data)

            return ExternalAgentResponse(
                content=content,
                blocked=blocked,
                block_reason=block_reason,
                raw_response=data,
            )

        except requests.exceptions.Timeout:
            return ExternalAgentResponse(
                content="",
                blocked=True,
                block_reason="请求超时",
                error="Request timeout",
            )
        except requests.exceptions.HTTPError as e:
            return ExternalAgentResponse(
                content="",
                blocked=True,
                block_reason=f"HTTP 错误: {e.response.status_code}",
                error=str(e),
                raw_response={"status_code": e.response.status_code},
            )
        except Exception as e:
            return ExternalAgentResponse(
                content="",
                blocked=True,
                block_reason=f"请求失败: {str(e)}",
                error=str(e),
            )

    def health_check(self) -> bool:
        """健康检查"""
        if not self._session:
            return False

        try:
            url = f"{self.config.base_url.rstrip('/')}{self.config.health_endpoint}"
            response = self._session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            # 如果没有 health endpoint，尝试连接 base_url
            try:
                response = self._session.get(
                    self.config.base_url,
                    timeout=5,
                )
                return response.status_code < 500
            except Exception:
                return False

    def send_raw_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: dict = None,
        **kwargs,
    ) -> dict:
        """发送原始请求（用于自定义操作）"""
        if not self._session:
            raise RuntimeError("Not connected")

        url = f"{self.config.base_url.rstrip('/')}{endpoint}"
        response = self._session.request(
            method=method,
            url=url,
            json=data,
            timeout=self.config.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()
