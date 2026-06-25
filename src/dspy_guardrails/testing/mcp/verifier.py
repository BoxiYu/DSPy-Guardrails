"""
MCPAttackVerifier - MCP 攻击验证器

测试专用的验证器，协调 ToolProxy、SandboxMCP、NetworkMonitor 组件，
验证 Agent 对 MCP 攻击的真实行为。
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .attack_log import AttackLog, MCPAttackEvidence
from .network_monitor import NetworkMonitor
from .sandbox import SandboxMCP
from .tool_proxy import ToolProxy


@dataclass
class VerificationConfig:
    """验证配置"""
    block_on_threat: bool = False      # 检测到威胁时是否阻止执行
    verbose: bool = True               # 详细输出
    rate_limit_seconds: float = 1.0    # 请求间隔（秒）
    timeout_seconds: float = 30.0      # 单个测试超时时间


@dataclass
class VerificationResult:
    """验证结果"""
    attack_log: AttackLog
    total_tests: int
    attacks_attempted: int
    attacks_would_succeed: int
    execution_time_seconds: float


class MCPAttackVerifier:
    """
    MCP 攻击验证器 - 测试专用

    协调各组件验证 Agent 对 MCP 攻击的真实行为。

    Examples:
        # pydantic-ai Agent
        from b_gpt52 import triage_agent, AirlineAgentChatContext

        verifier = MCPAttackVerifier(
            agent=triage_agent,
            context_class=AirlineAgentChatContext,
        )

        # 运行验证
        result = await verifier.run_verification(MCP_PAYLOADS)

        # 查看结果
        print(f"攻击尝试: {result.attacks_attempted}")
        for evidence in result.attack_log.get_critical_evidences():
            print(f"  - {evidence.attack_id}: {evidence.category}")
    """

    def __init__(
        self,
        agent: Any = None,
        context_class: type = None,
        context_factory: Callable[[], Any] = None,
        config: VerificationConfig = None,
    ):
        """
        初始化验证器

        Args:
            agent: pydantic-ai Agent 实例
            context_class: Context 类（如 AirlineAgentChatContext）
            context_factory: 自定义 Context 工厂函数
            config: 验证配置
        """
        self.agent = agent
        self.context_class = context_class
        self.context_factory = context_factory
        self.config = config or VerificationConfig()

        # 初始化验证组件
        self.tool_proxy = ToolProxy(block_on_threat=self.config.block_on_threat)
        self.sandbox = SandboxMCP()
        self.network_monitor = NetworkMonitor()
        self.attack_log = AttackLog()

        # 代理 Agent（如果提供了 agent）
        self._proxied_agent = None
        self._original_tools = {}

        if agent:
            self._setup_proxied_agent()

    def _setup_proxied_agent(self):
        """设置代理 Agent（Hook 工具调用）"""
        if not self.agent:
            return

        # 保存原始工具
        if hasattr(self.agent, '_function_tools'):
            # pydantic-ai Agent
            self._original_tools = dict(self.agent._function_tools)

            # 包装每个工具
            for _tool_name, tool_info in self._original_tools.items():
                original_fn = tool_info.function
                proxied_fn = self.tool_proxy.wrap_tool(
                    original_fn,
                    sandbox=self.sandbox,
                    network_monitor=self.network_monitor,
                )
                # 替换工具函数
                tool_info.function = proxied_fn

    def _create_context(self) -> Any:
        """创建测试上下文"""
        if self.context_factory:
            return self.context_factory()

        if self.context_class:
            # 创建默认测试上下文
            return self.context_class(
                customer_name="Test Customer",
                confirmation_number="TEST123456",
                flight_number="GCX-001",
                seat_number="12A",
            )

        return None

    def _reset_monitors(self):
        """重置所有监控器"""
        self.tool_proxy.reset()
        self.sandbox.reset()
        self.network_monitor.reset()

    async def verify_single_attack(
        self,
        payload: str,
        payload_id: str,
        category: str,
        previous_result: dict[str, Any] = None,
    ) -> MCPAttackEvidence:
        """
        验证单个攻击

        Args:
            payload: 攻击载荷
            payload_id: 载荷 ID
            category: 攻击类别
            previous_result: 之前的测试结果（用于对比）

        Returns:
            攻击证据
        """
        # 重置监控器
        self._reset_monitors()

        # 创建测试上下文
        context = self._create_context()

        # 执行攻击
        agent_response = ""
        try:
            if self.agent:
                result = await self.agent.run(payload, deps=context)
                agent_response = str(result.output) if hasattr(result, 'output') else str(result)
            else:
                # 无 Agent 模式：直接分析载荷
                agent_response = "[NO AGENT - Payload analysis only]"
                self._analyze_payload_directly(payload)

        except Exception as e:
            agent_response = f"[ERROR] {e}"

        # 收集证据
        evidence = self.attack_log.record_attack_test(
            payload=payload,
            payload_id=payload_id,
            category=category,
            tool_calls=list(self.tool_proxy.call_log),
            sandbox_executions=list(self.sandbox.executions),
            network_requests=list(self.network_monitor.requests),
            agent_response=agent_response,
            previous_detection=previous_result,
        )

        return evidence

    def _analyze_payload_directly(self, payload: str):
        """直接分析载荷（无 Agent 模式）"""
        # 检测文件路径
        file_path = self.sandbox._extract_file_path(payload)
        if file_path:
            self.sandbox.execute_file_read(file_path)

        # 检测命令
        command = self.sandbox._extract_command(payload)
        if command:
            self.sandbox.execute_command(command)

        # 检测 URL
        url = self.sandbox._extract_url(payload)
        if url:
            self.network_monitor.intercept(
                method="GET",
                url=url,
                body=payload,
            )

        # 分析参数中的威胁
        self.tool_proxy._analyze_arguments(payload)

    async def run_verification(
        self,
        payloads: list[dict[str, str]],
        previous_results: dict[str, dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        运行完整验证

        Args:
            payloads: 攻击载荷列表 [{"id": "...", "category": "...", "payload": "..."}]
            previous_results: 之前的测试结果（按 payload_id 索引）

        Returns:
            验证结果
        """
        previous_results = previous_results or {}
        start_time = time.time()

        # 重置攻击日志
        self.attack_log.reset()

        if self.config.verbose:
            print("=" * 60)
            print("MCP 攻击验证 - Genesys CX Virtual Agent")
            print("=" * 60)
            print(f"载荷数量: {len(payloads)}")
            print()

        for i, payload_data in enumerate(payloads):
            payload_id = payload_data.get("id", f"payload_{i}")
            category = payload_data.get("category", "unknown")
            payload = payload_data.get("payload", "")

            if self.config.verbose:
                print(f"[{i+1}/{len(payloads)}] {payload_id} ({category})...", end=" ")

            # 获取之前的结果
            prev_result = previous_results.get(payload_id)

            # 验证攻击
            try:
                evidence = await self.verify_single_attack(
                    payload=payload,
                    payload_id=payload_id,
                    category=category,
                    previous_result=prev_result,
                )

                if self.config.verbose:
                    if evidence.attack_attempted:
                        status = "⚠️  ATTACK ATTEMPTED"
                        if evidence.attack_would_succeed:
                            status += " (would succeed)"
                        print(status)
                    else:
                        print("✅ BLOCKED")

            except Exception as e:
                if self.config.verbose:
                    print(f"❌ ERROR: {e}")

            # 速率限制
            if i < len(payloads) - 1:
                await asyncio.sleep(self.config.rate_limit_seconds)

        execution_time = time.time() - start_time

        # 生成摘要
        summary = self.attack_log.get_summary()

        if self.config.verbose:
            print()
            print("=" * 60)
            print("验证结果摘要")
            print("=" * 60)
            print(f"总测试数: {summary.total_tests}")
            print(f"攻击尝试: {summary.attacks_attempted}")
            print(f"真实成功: {summary.attacks_would_succeed}")
            print(f"执行时间: {execution_time:.1f}s")

            if summary.critical_findings:
                print()
                print("关键发现:")
                for finding in summary.critical_findings[:5]:
                    print(f"  - {finding}")

        return VerificationResult(
            attack_log=self.attack_log,
            total_tests=summary.total_tests,
            attacks_attempted=summary.attacks_attempted,
            attacks_would_succeed=summary.attacks_would_succeed,
            execution_time_seconds=execution_time,
        )

    def run_verification_sync(
        self,
        payloads: list[dict[str, str]],
        previous_results: dict[str, dict[str, Any]] = None,
    ) -> VerificationResult:
        """同步版本的验证方法"""
        return asyncio.run(self.run_verification(payloads, previous_results))


# 便捷函数
def create_verifier_for_pydantic_ai(
    agent: Any,
    context_class: type,
    config: VerificationConfig = None,
) -> MCPAttackVerifier:
    """
    为 pydantic-ai Agent 创建验证器

    Args:
        agent: pydantic-ai Agent
        context_class: Context 类
        config: 验证配置

    Returns:
        配置好的验证器
    """
    return MCPAttackVerifier(
        agent=agent,
        context_class=context_class,
        config=config,
    )


def create_payload_analyzer(config: VerificationConfig = None) -> MCPAttackVerifier:
    """
    创建载荷分析器（无需 Agent）

    用于直接分析攻击载荷，不需要实际的 Agent。

    Returns:
        配置好的验证器
    """
    return MCPAttackVerifier(config=config)
