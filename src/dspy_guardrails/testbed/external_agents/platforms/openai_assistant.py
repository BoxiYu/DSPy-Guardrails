"""
OpenAIAssistantAgent - OpenAI Assistants API 连接器

支持连接 OpenAI Assistants API。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..base import (
    AuthMethod,
    ConnectionStatus,
    ExternalAgent,
    ExternalAgentConfig,
    ExternalAgentResponse,
)


@dataclass
class OpenAIAssistantConfig(ExternalAgentConfig):
    """OpenAI Assistant 配置"""
    # API 配置
    api_key: str = ""
    organization: str | None = None
    base_url: str | None = None  # 支持代理

    # Assistant 配置
    assistant_id: str = ""  # 已创建的 Assistant ID

    # 或创建新 Assistant
    model: str = "gpt-4-turbo-preview"
    assistant_name: str = "Test Assistant"
    instructions: str | None = None
    tools: list = field(default_factory=list)  # code_interpreter, file_search, function

    # 运行配置
    poll_interval: float = 0.5
    max_poll_attempts: int = 120

    def __post_init__(self):
        self.auth_method = AuthMethod.API_KEY
        self.name = f"openai_assistant_{self.assistant_id or 'new'}"


class OpenAIAssistantAgent(ExternalAgent):
    """
    OpenAI Assistants API 连接器

    支持：
    - 使用已有 Assistant
    - 创建新 Assistant
    - 多轮对话（Thread）
    - 工具调用（Code Interpreter, File Search, Functions）

    使用示例:
        # 使用已有 Assistant
        agent = OpenAIAssistantAgent(
            api_key="sk-xxx",
            assistant_id="asst_xxx",
        )
        response = agent.chat("Hello!")

        # 创建新 Assistant
        agent = OpenAIAssistantAgent(
            api_key="sk-xxx",
            model="gpt-4-turbo-preview",
            assistant_name="My Assistant",
            instructions="You are a helpful assistant.",
        )
        response = agent.chat("Hello!")

    注意：
        需要安装 openai>=1.0.0
    """

    def __init__(
        self,
        api_key: str = None,
        assistant_id: str = None,
        model: str = None,
        instructions: str = None,
        config: OpenAIAssistantConfig = None,
        guardrail_fn: Callable | None = None,
        **kwargs,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for OpenAIAssistantAgent. "
                "Install it with: pip install openai>=1.0.0"
            )

        if config is None:
            config = OpenAIAssistantConfig(
                api_key=api_key or "",
                assistant_id=assistant_id or "",
                model=model or "gpt-4-turbo-preview",
                instructions=instructions,
                **kwargs,
            )

        super().__init__(config, guardrail_fn)
        self.config: OpenAIAssistantConfig = config

        self._client: OpenAI | None = None
        self._assistant_id: str | None = None
        self._thread_id: str | None = None
        self._created_assistant: bool = False

    def connect(self) -> bool:
        """建立连接"""
        try:
            self._status = ConnectionStatus.CONNECTING

            # 创建 OpenAI 客户端
            self._client = OpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization,
                base_url=self.config.base_url,
            )

            # 使用已有 Assistant 或创建新的
            if self.config.assistant_id:
                self._assistant_id = self.config.assistant_id
                # 验证 Assistant 存在
                self._client.beta.assistants.retrieve(self._assistant_id)
            else:
                # 创建新 Assistant
                assistant = self._client.beta.assistants.create(
                    model=self.config.model,
                    name=self.config.assistant_name,
                    instructions=self.config.instructions,
                    tools=self._build_tools(),
                )
                self._assistant_id = assistant.id
                self._created_assistant = True

            # 创建 Thread
            thread = self._client.beta.threads.create()
            self._thread_id = thread.id

            self._status = ConnectionStatus.CONNECTED
            return True

        except Exception:
            self._status = ConnectionStatus.ERROR
            return False

    def disconnect(self) -> None:
        """断开连接"""
        # 删除自己创建的 Assistant
        if self._created_assistant and self._assistant_id:
            try:
                self._client.beta.assistants.delete(self._assistant_id)
            except Exception:
                pass

        self._client = None
        self._assistant_id = None
        self._thread_id = None
        self._created_assistant = False
        self._status = ConnectionStatus.DISCONNECTED

    def _build_tools(self) -> list[dict]:
        """构建工具配置"""
        tools = []
        for tool in self.config.tools:
            if isinstance(tool, str):
                if tool == "code_interpreter":
                    tools.append({"type": "code_interpreter"})
                elif tool == "file_search":
                    tools.append({"type": "file_search"})
            elif isinstance(tool, dict):
                tools.append(tool)
        return tools

    def _send_message(
        self,
        message: str,
        session_id: str | None = None,
    ) -> ExternalAgentResponse:
        """发送消息"""
        if not self._client or not self._thread_id:
            raise RuntimeError("Not connected")

        # 添加消息到 Thread
        self._client.beta.threads.messages.create(
            thread_id=self._thread_id,
            role="user",
            content=message,
        )

        # 创建 Run
        run = self._client.beta.threads.runs.create(
            thread_id=self._thread_id,
            assistant_id=self._assistant_id,
        )

        # 等待完成
        run = self._wait_for_run(run.id)

        if run.status == "completed":
            # 获取响应消息
            messages = self._client.beta.threads.messages.list(
                thread_id=self._thread_id,
                order="desc",
                limit=1,
            )

            if messages.data:
                content = self._extract_message_content(messages.data[0])
                return ExternalAgentResponse(
                    content=content,
                    raw_response={
                        "run_id": run.id,
                        "message": messages.data[0].model_dump(),
                    },
                    metadata={
                        "thread_id": self._thread_id,
                        "run_id": run.id,
                        "usage": run.usage.model_dump() if run.usage else None,
                    }
                )

            return ExternalAgentResponse(
                content="",
                blocked=True,
                block_reason="No response message",
            )

        elif run.status == "requires_action":
            # 需要工具调用
            tool_calls = []
            if run.required_action:
                for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    })

            return ExternalAgentResponse(
                content="Tool call required",
                tool_calls=tool_calls,
                raw_response={"run": run.model_dump()},
                metadata={
                    "thread_id": self._thread_id,
                    "run_id": run.id,
                    "status": run.status,
                }
            )

        else:
            # 失败
            error_msg = run.last_error.message if run.last_error else "Unknown error"
            return ExternalAgentResponse(
                content="",
                blocked=True,
                block_reason=f"Run failed: {error_msg}",
                error=error_msg,
                metadata={
                    "status": run.status,
                    "run_id": run.id,
                }
            )

    def _wait_for_run(self, run_id: str):
        """等待 Run 完成"""
        attempts = 0
        while attempts < self.config.max_poll_attempts:
            run = self._client.beta.threads.runs.retrieve(
                thread_id=self._thread_id,
                run_id=run_id,
            )

            if run.status in ["completed", "failed", "cancelled", "requires_action", "expired"]:
                return run

            time.sleep(self.config.poll_interval)
            attempts += 1

        # 超时
        run = self._client.beta.threads.runs.retrieve(
            thread_id=self._thread_id,
            run_id=run_id,
        )
        return run

    def _extract_message_content(self, message) -> str:
        """提取消息内容"""
        content_parts = []
        for content in message.content:
            if content.type == "text":
                content_parts.append(content.text.value)
            elif content.type == "image_file":
                content_parts.append(f"[Image: {content.image_file.file_id}]")
        return "\n".join(content_parts)

    def submit_tool_outputs(
        self,
        run_id: str,
        tool_outputs: list[dict],
    ) -> ExternalAgentResponse:
        """提交工具输出"""
        if not self._client or not self._thread_id:
            raise RuntimeError("Not connected")

        run = self._client.beta.threads.runs.submit_tool_outputs(
            thread_id=self._thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
        )

        # 等待完成
        run = self._wait_for_run(run.id)

        if run.status == "completed":
            messages = self._client.beta.threads.messages.list(
                thread_id=self._thread_id,
                order="desc",
                limit=1,
            )

            if messages.data:
                content = self._extract_message_content(messages.data[0])
                return ExternalAgentResponse(
                    content=content,
                    metadata={"run_id": run.id},
                )

        return ExternalAgentResponse(
            content="",
            blocked=True,
            block_reason=f"Run status: {run.status}",
        )

    def health_check(self) -> bool:
        """健康检查"""
        if not self._client:
            return False

        try:
            # 验证 Assistant 存在
            if self._assistant_id:
                self._client.beta.assistants.retrieve(self._assistant_id)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        """重置会话（创建新 Thread）"""
        super().reset()

        if self._client:
            thread = self._client.beta.threads.create()
            self._thread_id = thread.id

    def get_thread_messages(self, limit: int = 20) -> list[dict]:
        """获取 Thread 消息"""
        if not self._client or not self._thread_id:
            return []

        messages = self._client.beta.threads.messages.list(
            thread_id=self._thread_id,
            limit=limit,
        )
        return [m.model_dump() for m in messages.data]

    def get_info(self) -> dict:
        """获取 Agent 信息"""
        info = super().get_info()
        info.update({
            "assistant_id": self._assistant_id,
            "thread_id": self._thread_id,
            "model": self.config.model,
            "created_assistant": self._created_assistant,
        })
        return info
