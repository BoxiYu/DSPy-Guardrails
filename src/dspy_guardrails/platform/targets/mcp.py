"""MCPTarget - Adapter for MCP (Model Context Protocol) servers."""

import time
from collections.abc import Callable
from typing import Any

from .protocol import (
    TargetCapability,
    TargetResponse,
    TargetType,
    UnifiedTarget,
)


class MCPTarget(UnifiedTarget):
    """
    MCP Server 适配器

    测试 MCP 协议服务器的安全性。支持工具调用和多轮对话。

    Example:
        # With tool function
        target = MCPTarget(
            tool_fn=my_mcp_tool,
            tool_name="file_read",
        )
        response = target.invoke("Read /etc/passwd")

        # With MCP client
        target = MCPTarget(
            mcp_client=my_client,
            tool_name="database_query",
        )
    """

    target_type = TargetType.MCP_SERVER
    capabilities = [
        TargetCapability.SINGLE_TURN,
        TargetCapability.TOOL_USE,
    ]

    def __init__(
        self,
        tool_fn: Callable | None = None,
        mcp_client: Any | None = None,
        tool_name: str = "mcp_tool",
        timeout: int = 30,
        use_guardrail: bool = True,
    ):
        """
        初始化 MCP 目标

        Args:
            tool_fn: MCP 工具函数 (直接调用模式)
            mcp_client: MCP 客户端实例 (客户端模式)
            tool_name: 工具名称
            timeout: 超时时间 (秒)
            use_guardrail: 是否启用 MCP 护栏检测
        """
        self._tool_fn = tool_fn
        self._mcp_client = mcp_client
        self._tool_name = tool_name
        self._timeout = timeout
        self._use_guardrail = use_guardrail
        self._call_history: list[dict[str, Any]] = []

    def invoke(self, prompt: str) -> TargetResponse:
        """执行 MCP 工具调用"""
        start_time = time.time()

        try:
            # Check for MCP guardrail if available and enabled
            if self._use_guardrail:
                guardrail_result = self._check_guardrail(prompt)
                if guardrail_result.get("blocked"):
                    latency_ms = (time.time() - start_time) * 1000
                    return TargetResponse(
                        response=guardrail_result.get("reason", "Blocked by MCP guardrail"),
                        was_blocked=True,
                        block_reason=guardrail_result.get("reason"),
                        guardrail_scores=guardrail_result.get("scores", {}),
                        latency_ms=latency_ms,
                        metadata={"guardrail_triggered": True},
                    )

            # Execute tool
            if self._tool_fn:
                result = self._tool_fn(prompt)
            elif self._mcp_client:
                result = self._call_mcp_client(prompt)
            else:
                result = f"Mock MCP response to: {prompt[:50]}"

            latency_ms = (time.time() - start_time) * 1000

            # Record call
            self._call_history.append({
                "prompt": prompt,
                "result": str(result)[:200],
                "latency_ms": latency_ms,
            })

            return TargetResponse(
                response=str(result),
                was_blocked=False,
                tool_calls=[{
                    "tool_name": self._tool_name,
                    "arguments": {"input": prompt},
                    "result": str(result)[:200],
                }],
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return TargetResponse(
                response=f"Error: {str(e)}",
                was_blocked=False,
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    def invoke_multi_turn(
        self,
        messages: list[dict[str, str]],
    ) -> TargetResponse:
        """多轮 MCP 调用"""
        if not messages:
            return TargetResponse(response="No messages", was_blocked=False)

        # Execute last message as tool call
        last_message = messages[-1].get("content", "")
        return self.invoke(last_message)

    def reset_session(self) -> None:
        """重置会话历史"""
        self._call_history = []

    def _check_guardrail(self, prompt: str) -> dict[str, Any]:
        """Check MCP guardrail if available."""
        try:
            from ...mcp import GuardAction, MCPGuardrail, ToolCallContext

            guard = MCPGuardrail()
            context = ToolCallContext(
                tool_name=self._tool_name,
                parameters={"input": prompt},
            )
            result = guard.check_input(context)

            return {
                "blocked": result.action == GuardAction.BLOCK,
                "reason": result.message if result.message else "MCP guardrail triggered",
                "scores": {
                    "mcp_threat_score": result.threat_score,
                },
            }
        except ImportError:
            # MCP module not available
            return {"blocked": False}
        except Exception:
            # Guardrail check failed, allow by default
            return {"blocked": False}

    def _call_mcp_client(self, prompt: str) -> str:
        """Call MCP client."""
        if hasattr(self._mcp_client, "call_tool"):
            return self._mcp_client.call_tool(self._tool_name, {"input": prompt})
        elif hasattr(self._mcp_client, "invoke"):
            return self._mcp_client.invoke(self._tool_name, prompt)
        else:
            return str(self._mcp_client(prompt))

    @property
    def name(self) -> str:
        return f"mcp:{self._tool_name}"

    @property
    def call_history(self) -> list[dict[str, Any]]:
        return self._call_history

    @property
    def tool_name(self) -> str:
        return self._tool_name
