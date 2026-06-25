"""
MCP 攻击验证模块

提供完整的 MCP 攻击验证能力：
- ToolProxy: 工具调用拦截与参数检测
- SandboxMCP: 沙箱执行环境
- NetworkMonitor: 网络请求监控
- AttackLog: 攻击证据收集
- MCPAttackVerifier: 验证器主入口
- MCPAttackReportGenerator: 报告生成

Examples:
    # 基础用法 - 组件级别
    from dspy_guardrails.testing.mcp import (
        ToolProxy,
        SandboxMCP,
        NetworkMonitor,
    )

    tool_proxy = ToolProxy()
    sandbox = SandboxMCP()
    network_monitor = NetworkMonitor()

    wrapped_tool = tool_proxy.wrap_tool(original_tool, sandbox, network_monitor)

    # 高级用法 - 完整验证流程
    from dspy_guardrails.testing.mcp import (
        MCPAttackVerifier,
        VerificationConfig,
        MCPAttackReportGenerator,
    )

    # 创建验证器
    verifier = MCPAttackVerifier(
        agent=my_agent,
        context_class=MyContext,
        config=VerificationConfig(verbose=True),
    )

    # 运行验证
    result = await verifier.run_verification(attack_payloads)

    # 生成报告
    reporter = MCPAttackReportGenerator(result.attack_log)
    reporter.generate_all(output_dir="./reports")
"""

from .attack_log import (
    AttackLog,
    AttackLogSummary,
    MCPAttackEvidence,
)
from .liting_agent_adapter import (
    LitingAgentAdapter,
    LitingAgentConfig,
    create_liting_agent_verifier,
)
from .liting_mcp_adapter import (
    MCP_SPECIFIC_PAYLOADS,
    LitingMCPAdapter,
    LitingMCPConfig,
    MCPToolCallRecord,
    get_mcp_attack_payloads,
)
from .network_monitor import (
    NetworkMonitor,
    NetworkRequest,
)
from .report_generator import (
    MCPAttackReportGenerator,
)
from .sandbox import (
    SandboxExecution,
    SandboxMCP,
)
from .tool_proxy import (
    ThreatDetail,
    ToolCallRecord,
    ToolProxy,
)
from .verifier import (
    MCPAttackVerifier,
    VerificationConfig,
    VerificationResult,
    create_payload_analyzer,
    create_verifier_for_pydantic_ai,
)

__all__ = [
    # ToolProxy
    "ToolProxy",
    "ToolCallRecord",
    "ThreatDetail",

    # SandboxMCP
    "SandboxMCP",
    "SandboxExecution",

    # NetworkMonitor
    "NetworkMonitor",
    "NetworkRequest",

    # AttackLog
    "AttackLog",
    "MCPAttackEvidence",
    "AttackLogSummary",

    # Verifier
    "MCPAttackVerifier",
    "VerificationConfig",
    "VerificationResult",
    "create_verifier_for_pydantic_ai",
    "create_payload_analyzer",

    # ReportGenerator
    "MCPAttackReportGenerator",

    # LitingAgentAdapter
    "LitingAgentAdapter",
    "LitingAgentConfig",
    "create_liting_agent_verifier",

    # LitingMCPAdapter (完整 MCP 版本)
    "LitingMCPAdapter",
    "LitingMCPConfig",
    "MCPToolCallRecord",
    "MCP_SPECIFIC_PAYLOADS",
    "get_mcp_attack_payloads",
]
