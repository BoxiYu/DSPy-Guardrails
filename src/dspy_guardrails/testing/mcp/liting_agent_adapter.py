"""
LitingAgentAdapter - liting-Agent (pydantic-ai) 适配器

为 liting-Agent 提供 MCP 攻击验证集成。

Examples:
    from dspy_guardrails.testing.mcp import (
        LitingAgentAdapter,
        MCPAttackVerifier,
        VerificationConfig,
    )

    # 创建适配器
    adapter = LitingAgentAdapter()

    # 创建验证器
    verifier = MCPAttackVerifier(
        agent=adapter.triage_agent,
        context_class=adapter.context_class,
        config=VerificationConfig(verbose=True),
    )

    # 运行验证
    result = await verifier.run_verification(payloads)
"""

import sys
from dataclasses import dataclass
from typing import Any

from .network_monitor import NetworkMonitor
from .sandbox import SandboxMCP
from .tool_proxy import ToolProxy


@dataclass
class LitingAgentConfig:
    """liting-Agent 配置"""
    # liting-Agent 路径
    liting_agent_path: str = "/home/ybx/guardrails_Playground/liting-Agent"
    # 使用的模块文件
    module_name: str = "b_gpt52"
    # 是否启用工具代理
    enable_tool_proxy: bool = True
    # 是否阻止危险操作
    block_on_threat: bool = False


class LitingAgentAdapter:
    """
    liting-Agent 适配器

    为 liting-Agent 提供 MCP 攻击验证集成。

    Features:
        - 动态导入 liting-Agent 模块
        - 包装所有工具函数，拦截调用
        - 提供统一的验证接口

    Examples:
        adapter = LitingAgentAdapter()

        # 获取包装后的 Agent
        agent = adapter.get_proxied_agent("triage_agent")

        # 运行并检查证据
        result = await agent.run("Check my flight status", deps=context)
        for record in adapter.tool_proxy.call_log:
            print(f"Tool: {record.tool_name}, Threats: {record.threats_detected}")
    """

    def __init__(self, config: LitingAgentConfig = None):
        """
        初始化适配器

        Args:
            config: 适配器配置
        """
        self.config = config or LitingAgentConfig()

        # 初始化验证组件
        self.tool_proxy = ToolProxy(block_on_threat=self.config.block_on_threat)
        self.sandbox = SandboxMCP()
        self.network_monitor = NetworkMonitor()

        # 导入 liting-Agent 模块
        self._module = None
        self._agents = {}
        self._original_tools = {}
        self._context_class = None

        self._import_liting_agent()

    def _import_liting_agent(self):
        """动态导入 liting-Agent 模块"""
        # 添加路径
        if self.config.liting_agent_path not in sys.path:
            sys.path.insert(0, self.config.liting_agent_path)

        try:
            # 导入模块
            self._module = __import__(self.config.module_name)

            # 获取 Context 类
            self._context_class = getattr(self._module, "AirlineAgentChatContext", None)

            # 获取所有 Agent
            agent_names = [
                "triage_agent",
                "flight_information_agent",
                "booking_cancellation_agent",
                "seat_special_services_agent",
                "refunds_compensation_agent",
                "faq_agent",
            ]

            for name in agent_names:
                agent = getattr(self._module, name, None)
                if agent:
                    self._agents[name] = agent

            # 获取所有工具函数
            tool_names = [
                # 业务工具
                "update_seat",
                "assign_special_service_seat",
                "display_seat_map",
                "flight_status_tool",
                "get_matching_flights",
                "book_new_flight",
                "cancel_flight",
                "issue_compensation",
                "faq_lookup_tool",
                "get_trip_details",
                # Handoff 工具
                "transfer_to_seat_agent",
                "transfer_to_flight_info",
                "transfer_to_booking",
                "transfer_to_refunds",
                "transfer_to_faq",
                "transfer_to_triage",
            ]

            for name in tool_names:
                tool = getattr(self._module, name, None)
                if tool:
                    self._original_tools[name] = tool

        except ImportError as e:
            raise ImportError(
                f"无法导入 liting-Agent 模块 '{self.config.module_name}': {e}\n"
                f"请确保路径正确: {self.config.liting_agent_path}"
            ) from e

    @property
    def context_class(self) -> type:
        """获取 Context 类"""
        return self._context_class

    @property
    def triage_agent(self):
        """获取 Triage Agent"""
        return self._agents.get("triage_agent")

    @property
    def all_agents(self) -> dict[str, Any]:
        """获取所有 Agent"""
        return self._agents

    def create_test_context(
        self,
        customer_name: str = "Test Customer",
        confirmation_number: str = "TEST123456",
        flight_number: str = "UA123",
        seat_number: str = "12A",
    ):
        """
        创建测试上下文

        Args:
            customer_name: 客户名称
            confirmation_number: 确认号
            flight_number: 航班号
            seat_number: 座位号

        Returns:
            AirlineAgentChatContext 实例
        """
        if not self._context_class:
            raise RuntimeError("Context class not available")

        return self._context_class(
            customer_name=customer_name,
            confirmation_number=confirmation_number,
            flight_number=flight_number,
            seat_number=seat_number,
        )

    def wrap_agent_tools(self, agent) -> None:
        """
        包装 Agent 的所有工具

        Args:
            agent: pydantic-ai Agent 实例
        """
        if not self.config.enable_tool_proxy:
            return

        if not hasattr(agent, "_function_tools"):
            return

        for _tool_name, tool_info in agent._function_tools.items():
            original_fn = tool_info.function

            # 创建包装函数
            wrapped_fn = self.tool_proxy.wrap_tool(
                original_fn,
                sandbox=self.sandbox,
                network_monitor=self.network_monitor,
            )

            # 替换工具函数
            tool_info.function = wrapped_fn

    def get_proxied_agent(self, agent_name: str):
        """
        获取带代理的 Agent

        Args:
            agent_name: Agent 名称

        Returns:
            包装后的 Agent
        """
        agent = self._agents.get(agent_name)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_name}")

        self.wrap_agent_tools(agent)
        return agent

    def reset(self) -> None:
        """重置所有监控器"""
        self.tool_proxy.reset()
        self.sandbox.reset()
        self.network_monitor.reset()

    def get_evidence_summary(self) -> dict[str, Any]:
        """
        获取证据摘要

        Returns:
            证据摘要字典
        """
        return {
            "tool_calls": len(self.tool_proxy.call_log),
            "threats_detected": sum(
                len(tc.threats_detected) for tc in self.tool_proxy.call_log
            ),
            "sandbox_executions": len(self.sandbox.executions),
            "critical_executions": len(self.sandbox.get_critical_executions()),
            "network_requests": len(self.network_monitor.requests),
            "exfiltration_attempts": len(self.network_monitor.get_exfiltration_attempts()),
        }


def create_liting_agent_verifier(
    config: LitingAgentConfig = None,
    verification_config = None,
):
    """
    创建 liting-Agent 验证器

    Args:
        config: liting-Agent 配置
        verification_config: 验证配置

    Returns:
        (adapter, verifier) 元组
    """
    from .verifier import MCPAttackVerifier, VerificationConfig

    # 创建适配器
    adapter = LitingAgentAdapter(config)

    # 创建验证器
    verifier = MCPAttackVerifier(
        agent=adapter.triage_agent,
        context_class=adapter.context_class,
        config=verification_config or VerificationConfig(),
    )

    # 替换验证器的组件为适配器的组件（共享状态）
    verifier.tool_proxy = adapter.tool_proxy
    verifier.sandbox = adapter.sandbox
    verifier.network_monitor = adapter.network_monitor

    return adapter, verifier
