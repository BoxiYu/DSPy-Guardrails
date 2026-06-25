"""
GenesysAgent - Genesys Cloud Virtual Agent 连接器

支持连接 Genesys Cloud 的 Web Messaging 和 Bot Flow API。
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import requests

from ..base import (
    AuthMethod,
    ConnectionStatus,
    ExternalAgent,
    ExternalAgentConfig,
    ExternalAgentResponse,
)


@dataclass
class GenesysConfig(ExternalAgentConfig):
    """Genesys Cloud 配置"""
    # 区域配置
    region: str = "mypurecloud.com"  # mypurecloud.com, mypurecloud.de, etc.

    # 部署配置
    deployment_id: str = ""  # Web Messaging Deployment ID
    bot_flow_id: str | None = None  # Bot Flow ID (if using Bot Flow directly)

    # OAuth 2.0 客户端凭证
    oauth_client_id: str = ""
    oauth_client_secret: str = ""

    # 或使用 API Key
    api_key: str | None = None

    # Web Messaging 配置
    origin: str = "https://example.com"  # 允许的来源

    # 会话配置
    customer_id: str | None = None
    customer_name: str = "Test User"
    customer_email: str | None = None

    def __post_init__(self):
        self.auth_method = AuthMethod.OAUTH2 if self.oauth_client_id else AuthMethod.API_KEY
        self.name = f"genesys_{self.region}"


class GenesysAgent(ExternalAgent):
    """
    Genesys Cloud Virtual Agent 连接器

    支持两种连接方式：
    1. Web Messaging API - 模拟网页聊天客户端
    2. Bot Flow API - 直接调用 Bot Flow

    认证方式：
    - OAuth 2.0 Client Credentials
    - API Key

    使用示例:
        agent = GenesysAgent(
            region="mypurecloud.com",
            deployment_id="your-deployment-id",
            oauth_client_id="your-client-id",
            oauth_client_secret="your-secret",
        )

        response = agent.chat("Hello!")

    注意：
        需要在 Genesys Cloud 中配置：
        1. OAuth Client (Client Credentials Grant)
        2. Web Messaging Deployment
        3. 对应的 Bot Flow 或 Digital Bot Flow
    """

    # API Base URLs by region
    REGION_URLS = {
        "mypurecloud.com": "https://api.mypurecloud.com",
        "mypurecloud.de": "https://api.mypurecloud.de",
        "mypurecloud.ie": "https://api.mypurecloud.ie",
        "mypurecloud.com.au": "https://api.mypurecloud.com.au",
        "mypurecloud.jp": "https://api.mypurecloud.jp",
        "usw2.pure.cloud": "https://api.usw2.pure.cloud",
        "cac1.pure.cloud": "https://api.cac1.pure.cloud",
        "apne2.pure.cloud": "https://api.apne2.pure.cloud",
        "euw2.pure.cloud": "https://api.euw2.pure.cloud",
        "aps1.pure.cloud": "https://api.aps1.pure.cloud",
    }

    def __init__(
        self,
        region: str = None,
        deployment_id: str = None,
        oauth_client_id: str = None,
        oauth_client_secret: str = None,
        api_key: str = None,
        config: GenesysConfig = None,
        guardrail_fn: Callable | None = None,
        **kwargs,
    ):
        if config is None:
            config = GenesysConfig(
                region=region or "mypurecloud.com",
                deployment_id=deployment_id or "",
                oauth_client_id=oauth_client_id or "",
                oauth_client_secret=oauth_client_secret or "",
                api_key=api_key,
                **kwargs,
            )

        super().__init__(config, guardrail_fn)
        self.config: GenesysConfig = config

        self._session: requests.Session | None = None
        self._access_token: str | None = None
        self._token_expires_at: float = 0
        self._conversation_id: str | None = None
        self._member_id: str | None = None

    @property
    def _api_base_url(self) -> str:
        """获取 API 基础 URL"""
        return self.REGION_URLS.get(
            self.config.region,
            f"https://api.{self.config.region}"
        )

    @property
    def _login_url(self) -> str:
        """获取登录 URL"""
        return f"https://login.{self.config.region}"

    def _get_access_token(self) -> str:
        """获取 OAuth 2.0 访问令牌"""
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        url = f"{self._login_url}/oauth/token"
        data = {
            "grant_type": "client_credentials",
        }

        response = requests.post(
            url,
            data=data,
            auth=(self.config.oauth_client_id, self.config.oauth_client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        self._token_expires_at = time.time() + token_data.get("expires_in", 3600)

        return self._access_token

    def _create_session(self) -> requests.Session:
        """创建 HTTP 会话"""
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        return session

    def connect(self) -> bool:
        """建立连接"""
        try:
            self._status = ConnectionStatus.CONNECTING
            self._session = self._create_session()

            # 获取访问令牌
            if self.config.oauth_client_id:
                token = self._get_access_token()
                self._session.headers["Authorization"] = f"Bearer {token}"
            elif self.config.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.config.api_key}"
            else:
                raise ValueError("No authentication configured")

            # 创建 Web Messaging 会话
            if self.config.deployment_id:
                self._create_messaging_session()

            self._status = ConnectionStatus.CONNECTED
            return True

        except Exception:
            self._status = ConnectionStatus.ERROR
            return False

    def disconnect(self) -> None:
        """断开连接"""
        # 结束会话
        if self._conversation_id:
            try:
                self._end_conversation()
            except Exception:
                pass

        if self._session:
            self._session.close()
            self._session = None

        self._access_token = None
        self._conversation_id = None
        self._member_id = None
        self._status = ConnectionStatus.DISCONNECTED

    def _create_messaging_session(self) -> None:
        """创建 Web Messaging 会话"""
        url = f"{self._api_base_url}/api/v2/webmessaging/deployments/{self.config.deployment_id}/sessions"

        customer_id = self.config.customer_id or str(uuid.uuid4())

        data = {
            "deploymentId": self.config.deployment_id,
            "startNew": True,
            "journeyContext": {
                "customer": {
                    "id": customer_id,
                    "idType": "cookie",
                }
            }
        }

        response = self._session.post(url, json=data, timeout=30)
        response.raise_for_status()

        session_data = response.json()
        self._conversation_id = session_data.get("conversationId")
        self._member_id = session_data.get("memberId")

    def _end_conversation(self) -> None:
        """结束会话"""
        if not self._conversation_id:
            return

        url = f"{self._api_base_url}/api/v2/webmessaging/messages/{self._conversation_id}/member/disconnect"
        try:
            self._session.post(url, timeout=10)
        except Exception:
            pass

    def _send_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """发送消息到 Genesys Virtual Agent"""
        if not self._session:
            raise RuntimeError("Not connected")

        # 刷新令牌
        if self.config.oauth_client_id:
            token = self._get_access_token()
            self._session.headers["Authorization"] = f"Bearer {token}"

        # 使用 Web Messaging API
        if self.config.deployment_id and self._conversation_id:
            return self._send_web_messaging(message)

        # 使用 Bot Flow API
        elif self.config.bot_flow_id:
            return self._send_bot_flow(message)

        else:
            raise RuntimeError("No deployment or bot flow configured")

    def _send_web_messaging(self, message: str) -> ExternalAgentResponse:
        """通过 Web Messaging 发送消息"""
        # 发送消息
        send_url = f"{self._api_base_url}/api/v2/webmessaging/messages"
        send_data = {
            "conversationId": self._conversation_id,
            "memberId": self._member_id,
            "body": {
                "text": message,
            },
            "channel": {
                "messageId": str(uuid.uuid4()),
            }
        }

        response = self._session.post(send_url, json=send_data, timeout=30)
        response.raise_for_status()

        # 轮询获取响应
        return self._poll_for_response()

    def _poll_for_response(
        self,
        max_wait: float = None,
        poll_interval: float = 0.5,
    ) -> ExternalAgentResponse:
        """轮询等待 Agent 响应"""
        max_wait = max_wait or self.config.timeout
        start_time = time.time()

        url = f"{self._api_base_url}/api/v2/webmessaging/messages/{self._conversation_id}"

        while time.time() - start_time < max_wait:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            messages = data.get("entities", [])

            # 查找最新的 Agent 消息
            for msg in reversed(messages):
                if msg.get("direction") == "outbound":
                    content = ""
                    if "textBody" in msg:
                        content = msg["textBody"]
                    elif "normalizedMessage" in msg:
                        content = msg["normalizedMessage"].get("text", "")

                    return ExternalAgentResponse(
                        content=content,
                        raw_response=msg,
                        metadata={
                            "conversation_id": self._conversation_id,
                            "message_id": msg.get("id"),
                        }
                    )

            time.sleep(poll_interval)

        return ExternalAgentResponse(
            content="",
            blocked=True,
            block_reason="响应超时",
            error="Response timeout",
        )

    def _send_bot_flow(self, message: str) -> ExternalAgentResponse:
        """通过 Bot Flow API 发送消息"""
        url = f"{self._api_base_url}/api/v2/flows/{self.config.bot_flow_id}/sessions"

        session_id = self._session_id or str(uuid.uuid4())

        data = {
            "inputData": {
                "text": message,
            },
            "sessionId": session_id,
        }

        response = self._session.post(url, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()
        self._session_id = result.get("sessionId")

        # 提取响应
        output = result.get("outputData", {})
        content = output.get("response", output.get("text", ""))

        return ExternalAgentResponse(
            content=content,
            raw_response=result,
            metadata={
                "session_id": self._session_id,
                "flow_id": self.config.bot_flow_id,
            }
        )

    def health_check(self) -> bool:
        """健康检查"""
        if not self._session:
            return False

        try:
            # 尝试获取组织信息
            url = f"{self._api_base_url}/api/v2/organizations/me"
            response = self._session.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_conversation_history_from_api(self) -> list[dict]:
        """从 API 获取对话历史"""
        if not self._conversation_id:
            return []

        url = f"{self._api_base_url}/api/v2/webmessaging/messages/{self._conversation_id}"
        response = self._session.get(url, timeout=30)
        response.raise_for_status()

        return response.json().get("entities", [])

    def get_info(self) -> dict:
        """获取 Agent 信息"""
        info = super().get_info()
        info.update({
            "region": self.config.region,
            "deployment_id": self.config.deployment_id,
            "bot_flow_id": self.config.bot_flow_id,
            "conversation_id": self._conversation_id,
        })
        return info
