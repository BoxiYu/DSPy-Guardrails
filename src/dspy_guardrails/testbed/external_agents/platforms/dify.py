"""
DifyAgent - Dify 平台连接器

支持连接 Dify 平台的 Chat 和 Workflow API。
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import requests

from ..base import (
    AuthMethod,
    ConnectionStatus,
    ExternalAgent,
    ExternalAgentConfig,
    ExternalAgentResponse,
)


@dataclass
class DifyConfig(ExternalAgentConfig):
    """Dify 配置"""
    # 连接配置
    base_url: str = "https://api.dify.ai/v1"  # 或自托管地址
    api_key: str = ""  # App API Key

    # App 类型
    app_type: str = "chat"  # chat, completion, workflow

    # 会话配置
    user: str = "test_user"
    conversation_id: str | None = None

    # 响应模式
    streaming: bool = False

    # 额外参数
    inputs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.auth_method = AuthMethod.BEARER_TOKEN
        self.name = f"dify_{self.app_type}"


class DifyAgent(ExternalAgent):
    """
    Dify 平台连接器

    支持：
    - Chat 应用 API
    - Completion 应用 API
    - Workflow 应用 API
    - 流式响应

    使用示例:
        # Chat 应用
        agent = DifyAgent(
            base_url="https://api.dify.ai/v1",
            api_key="app-xxx",
            app_type="chat",
        )
        response = agent.chat("Hello!")

        # Workflow 应用
        agent = DifyAgent(
            base_url="https://api.dify.ai/v1",
            api_key="app-xxx",
            app_type="workflow",
            inputs={"input_var": "value"},
        )
        response = agent.chat("Run workflow")

    注意：
        - 需要在 Dify 中创建应用并获取 API Key
        - Dify 仅支持通过 Web UI 创建应用
        - API 只能用于执行已创建的应用
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        app_type: str = None,
        config: DifyConfig = None,
        guardrail_fn: Callable | None = None,
        **kwargs,
    ):
        if config is None:
            config = DifyConfig(
                base_url=base_url or "https://api.dify.ai/v1",
                api_key=api_key or "",
                app_type=app_type or "chat",
                **kwargs,
            )

        super().__init__(config, guardrail_fn)
        self.config: DifyConfig = config

        self._session: requests.Session | None = None
        self._dify_conversation_id: str | None = None

    def _create_session(self) -> requests.Session:
        """创建 HTTP 会话"""
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        })
        return session

    def connect(self) -> bool:
        """建立连接"""
        try:
            self._status = ConnectionStatus.CONNECTING
            self._session = self._create_session()

            # 验证 API Key
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
        self._dify_conversation_id = None
        self._status = ConnectionStatus.DISCONNECTED

    def _send_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """发送消息"""
        if not self._session:
            raise RuntimeError("Not connected")

        if self.config.app_type == "chat":
            return self._send_chat_message(message)
        elif self.config.app_type == "completion":
            return self._send_completion(message)
        elif self.config.app_type == "workflow":
            return self._run_workflow(message)
        else:
            raise ValueError(f"Unsupported app type: {self.config.app_type}")

    def _send_chat_message(self, message: str) -> ExternalAgentResponse:
        """发送 Chat 消息"""
        url = f"{self.config.base_url}/chat-messages"

        data = {
            "inputs": self.config.inputs,
            "query": message,
            "user": self.config.user,
            "response_mode": "streaming" if self.config.streaming else "blocking",
        }

        # 使用已有会话
        if self._dify_conversation_id:
            data["conversation_id"] = self._dify_conversation_id

        if self.config.streaming:
            return self._handle_streaming_response(url, data)
        else:
            return self._handle_blocking_response(url, data)

    def _send_completion(self, message: str) -> ExternalAgentResponse:
        """发送 Completion 请求"""
        url = f"{self.config.base_url}/completion-messages"

        inputs = dict(self.config.inputs)
        inputs["query"] = message

        data = {
            "inputs": inputs,
            "user": self.config.user,
            "response_mode": "streaming" if self.config.streaming else "blocking",
        }

        if self.config.streaming:
            return self._handle_streaming_response(url, data)
        else:
            return self._handle_blocking_response(url, data)

    def _run_workflow(self, message: str) -> ExternalAgentResponse:
        """运行 Workflow"""
        url = f"{self.config.base_url}/workflows/run"

        inputs = dict(self.config.inputs)
        inputs["query"] = message

        data = {
            "inputs": inputs,
            "user": self.config.user,
            "response_mode": "streaming" if self.config.streaming else "blocking",
        }

        if self.config.streaming:
            return self._handle_streaming_response(url, data, is_workflow=True)
        else:
            return self._handle_workflow_response(url, data)

    def _handle_blocking_response(
        self,
        url: str,
        data: dict,
    ) -> ExternalAgentResponse:
        """处理阻塞式响应"""
        response = self._session.post(url, json=data, timeout=self.config.timeout)
        response.raise_for_status()

        result = response.json()

        # 保存会话 ID
        if "conversation_id" in result:
            self._dify_conversation_id = result["conversation_id"]

        content = result.get("answer", "")

        return ExternalAgentResponse(
            content=content,
            raw_response=result,
            metadata={
                "conversation_id": result.get("conversation_id"),
                "message_id": result.get("message_id"),
                "model": result.get("metadata", {}).get("model"),
                "usage": result.get("metadata", {}).get("usage"),
            }
        )

    def _handle_workflow_response(
        self,
        url: str,
        data: dict,
    ) -> ExternalAgentResponse:
        """处理 Workflow 响应"""
        response = self._session.post(url, json=data, timeout=self.config.timeout)
        response.raise_for_status()

        result = response.json()

        # 提取输出
        outputs = result.get("data", {}).get("outputs", {})
        content = outputs.get("text", outputs.get("result", str(outputs)))

        return ExternalAgentResponse(
            content=content,
            raw_response=result,
            metadata={
                "workflow_run_id": result.get("workflow_run_id"),
                "task_id": result.get("task_id"),
                "status": result.get("data", {}).get("status"),
            }
        )

    def _handle_streaming_response(
        self,
        url: str,
        data: dict,
        is_workflow: bool = False,
    ) -> ExternalAgentResponse:
        """处理流式响应"""
        response = self._session.post(
            url,
            json=data,
            timeout=self.config.timeout,
            stream=True,
        )
        response.raise_for_status()

        content_parts = []
        metadata = {}

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue

            try:
                event_data = json.loads(line[6:])
                event_type = event_data.get("event")

                if event_type == "message":
                    content_parts.append(event_data.get("answer", ""))
                elif event_type == "message_end":
                    metadata["conversation_id"] = event_data.get("conversation_id")
                    metadata["message_id"] = event_data.get("message_id")
                    self._dify_conversation_id = event_data.get("conversation_id")
                elif event_type == "workflow_finished":
                    outputs = event_data.get("data", {}).get("outputs", {})
                    content_parts.append(outputs.get("text", str(outputs)))
                    metadata["workflow_run_id"] = event_data.get("workflow_run_id")

            except json.JSONDecodeError:
                continue

        return ExternalAgentResponse(
            content="".join(content_parts),
            metadata=metadata,
        )

    def health_check(self) -> bool:
        """健康检查"""
        if not self._session:
            return False

        try:
            # 尝试获取应用参数
            url = f"{self.config.base_url}/parameters"
            response = self._session.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def get_app_info(self) -> dict:
        """获取应用信息"""
        if not self._session:
            return {}

        url = f"{self.config.base_url}/parameters"
        response = self._session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_conversations(self, limit: int = 20) -> list[dict]:
        """获取会话列表"""
        if not self._session:
            return []

        url = f"{self.config.base_url}/conversations"
        params = {
            "user": self.config.user,
            "limit": limit,
        }
        response = self._session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])

    def get_conversation_messages(
        self,
        conversation_id: str = None,
        limit: int = 20,
    ) -> list[dict]:
        """获取会话消息"""
        if not self._session:
            return []

        conv_id = conversation_id or self._dify_conversation_id
        if not conv_id:
            return []

        url = f"{self.config.base_url}/messages"
        params = {
            "user": self.config.user,
            "conversation_id": conv_id,
            "limit": limit,
        }
        response = self._session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])

    def provide_feedback(
        self,
        message_id: str,
        rating: str,  # "like" or "dislike"
    ) -> bool:
        """提供消息反馈"""
        if not self._session:
            return False

        url = f"{self.config.base_url}/messages/{message_id}/feedbacks"
        data = {
            "rating": rating,
            "user": self.config.user,
        }
        response = self._session.post(url, json=data, timeout=10)
        return response.status_code == 200

    def reset(self) -> None:
        """重置会话"""
        super().reset()
        self._dify_conversation_id = None

    def get_info(self) -> dict:
        """获取 Agent 信息"""
        info = super().get_info()
        info.update({
            "base_url": self.config.base_url,
            "app_type": self.config.app_type,
            "conversation_id": self._dify_conversation_id,
            "streaming": self.config.streaming,
        })
        return info
