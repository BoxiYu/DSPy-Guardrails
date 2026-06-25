"""
LangServeAgent - LangServe 部署的 Agent 连接器

支持连接 LangServe 部署的 LangChain Agent。
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
class LangServeConfig(ExternalAgentConfig):
    """LangServe 配置"""
    # 连接配置
    base_url: str = "http://localhost:8000"
    chain_path: str = ""  # e.g., "/agent" or "/chat"

    # 认证
    api_key: str | None = None
    auth_token: str | None = None

    # 请求配置
    input_key: str = "input"  # 输入字段名
    output_key: str = "output"  # 输出字段名

    # 配置（传递给 chain 的 config）
    config: dict = field(default_factory=dict)

    # 流式响应
    streaming: bool = False

    def __post_init__(self):
        if self.api_key:
            self.auth_method = AuthMethod.API_KEY
        elif self.auth_token:
            self.auth_method = AuthMethod.BEARER_TOKEN
        self.name = f"langserve_{self.chain_path.strip('/')}"


class LangServeAgent(ExternalAgent):
    """
    LangServe 部署的 Agent 连接器

    支持：
    - invoke - 同步调用
    - stream - 流式调用
    - batch - 批量调用

    使用示例:
        agent = LangServeAgent(
            base_url="http://localhost:8000",
            chain_path="/agent",
        )
        response = agent.chat("Hello!")

        # 流式响应
        agent = LangServeAgent(
            base_url="http://localhost:8000",
            chain_path="/chat",
            streaming=True,
        )
        response = agent.chat("Tell me a story")

    LangServe 部署示例:
        from langserve import add_routes
        from fastapi import FastAPI
        from langchain.chat_models import ChatOpenAI
        from langchain.agents import create_openai_functions_agent

        app = FastAPI()
        agent = create_openai_functions_agent(...)
        add_routes(app, agent, path="/agent")

    注意：
        LangServe 提供标准的 REST API 端点：
        - POST /invoke - 同步调用
        - POST /stream - 流式调用
        - POST /batch - 批量调用
        - GET /input_schema - 输入 schema
        - GET /output_schema - 输出 schema
    """

    def __init__(
        self,
        base_url: str = None,
        chain_path: str = None,
        api_key: str = None,
        config: LangServeConfig = None,
        guardrail_fn: Callable | None = None,
        **kwargs,
    ):
        if config is None:
            config = LangServeConfig(
                base_url=base_url or "http://localhost:8000",
                chain_path=chain_path or "",
                api_key=api_key,
                **kwargs,
            )

        super().__init__(config, guardrail_fn)
        self.config: LangServeConfig = config

        self._session: requests.Session | None = None
        self._input_schema: dict | None = None
        self._output_schema: dict | None = None

    def _get_endpoint_url(self, endpoint: str) -> str:
        """获取完整端点 URL"""
        base = self.config.base_url.rstrip("/")
        path = self.config.chain_path.strip("/")
        if path:
            return f"{base}/{path}/{endpoint}"
        return f"{base}/{endpoint}"

    def _create_session(self) -> requests.Session:
        """创建 HTTP 会话"""
        session = requests.Session()
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        if self.config.auth_token:
            session.headers["Authorization"] = f"Bearer {self.config.auth_token}"
        elif self.config.api_key:
            session.headers["X-API-Key"] = self.config.api_key

        return session

    def connect(self) -> bool:
        """建立连接"""
        try:
            self._status = ConnectionStatus.CONNECTING
            self._session = self._create_session()

            # 获取 schemas
            self._fetch_schemas()

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
        self._input_schema = None
        self._output_schema = None
        self._status = ConnectionStatus.DISCONNECTED

    def _fetch_schemas(self) -> None:
        """获取输入输出 schema"""
        try:
            # 输入 schema
            url = self._get_endpoint_url("input_schema")
            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                self._input_schema = response.json()

            # 输出 schema
            url = self._get_endpoint_url("output_schema")
            response = self._session.get(url, timeout=10)
            if response.status_code == 200:
                self._output_schema = response.json()
        except Exception:
            pass

    def _build_input(self, message: str) -> dict:
        """构建输入数据"""
        # 检查 schema 确定输入格式
        if self._input_schema:
            properties = self._input_schema.get("properties", {})
            if self.config.input_key in properties:
                return {self.config.input_key: message}
            elif "messages" in properties:
                return {"messages": [{"role": "user", "content": message}]}
            elif "question" in properties:
                return {"question": message}
            elif "query" in properties:
                return {"query": message}

        # 默认格式
        return {self.config.input_key: message}

    def _extract_output(self, data: dict) -> str:
        """从响应中提取输出"""
        if isinstance(data, str):
            return data

        # 检查常见的输出字段
        for key in [self.config.output_key, "output", "response", "answer", "result", "content"]:
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    # AgentExecutor 输出
                    if "output" in value:
                        return value["output"]
                    elif "content" in value:
                        return value["content"]
                    return str(value)

        return str(data)

    def _send_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """发送消息"""
        if not self._session:
            raise RuntimeError("Not connected")

        if self.config.streaming:
            return self._stream_invoke(message)
        else:
            return self._invoke(message)

    def _invoke(self, message: str) -> ExternalAgentResponse:
        """同步调用"""
        url = self._get_endpoint_url("invoke")
        input_data = self._build_input(message)

        data = {
            "input": input_data,
        }
        if self.config.config:
            data["config"] = self.config.config

        response = self._session.post(url, json=data, timeout=self.config.timeout)
        response.raise_for_status()

        result = response.json()
        output = result.get("output", result)
        content = self._extract_output(output)

        return ExternalAgentResponse(
            content=content,
            raw_response=result,
            metadata={
                "callback_events": result.get("callback_events", []),
            }
        )

    def _stream_invoke(self, message: str) -> ExternalAgentResponse:
        """流式调用"""
        url = self._get_endpoint_url("stream")
        input_data = self._build_input(message)

        data = {
            "input": input_data,
        }
        if self.config.config:
            data["config"] = self.config.config

        response = self._session.post(
            url,
            json=data,
            timeout=self.config.timeout,
            stream=True,
        )
        response.raise_for_status()

        content_parts = []
        events = []

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if line.startswith("data: "):
                try:
                    event_data = json.loads(line[6:])
                    events.append(event_data)

                    # 提取内容
                    if isinstance(event_data, dict):
                        if "content" in event_data:
                            content_parts.append(event_data["content"])
                        elif "output" in event_data:
                            content_parts.append(self._extract_output(event_data))
                    elif isinstance(event_data, str):
                        content_parts.append(event_data)

                except json.JSONDecodeError:
                    content_parts.append(line[6:])

        return ExternalAgentResponse(
            content="".join(content_parts),
            raw_response={"events": events},
        )

    def batch(self, messages: list[str]) -> list[ExternalAgentResponse]:
        """批量调用"""
        if not self._session:
            raise RuntimeError("Not connected")

        url = self._get_endpoint_url("batch")
        inputs = [self._build_input(msg) for msg in messages]

        data = {
            "inputs": inputs,
        }
        if self.config.config:
            data["config"] = self.config.config

        response = self._session.post(url, json=data, timeout=self.config.timeout * len(messages))
        response.raise_for_status()

        result = response.json()
        outputs = result.get("output", [])

        responses = []
        for output in outputs:
            content = self._extract_output(output)
            responses.append(ExternalAgentResponse(
                content=content,
                raw_response=output,
            ))

        return responses

    def health_check(self) -> bool:
        """健康检查"""
        if not self._session:
            return False

        try:
            # 尝试获取 input_schema
            url = self._get_endpoint_url("input_schema")
            response = self._session.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            # 尝试连接基础 URL
            try:
                response = self._session.get(self.config.base_url, timeout=5)
                return response.status_code < 500
            except Exception:
                return False

    def get_schemas(self) -> dict:
        """获取 schemas"""
        return {
            "input": self._input_schema,
            "output": self._output_schema,
        }

    def get_info(self) -> dict:
        """获取 Agent 信息"""
        info = super().get_info()
        info.update({
            "base_url": self.config.base_url,
            "chain_path": self.config.chain_path,
            "streaming": self.config.streaming,
            "has_input_schema": self._input_schema is not None,
            "has_output_schema": self._output_schema is not None,
        })
        return info
