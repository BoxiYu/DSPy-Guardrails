"""
MCP Guardrails - Model Context Protocol 安全防护框架

保护 MCP 服务器和客户端免受各种攻击：
- Tool Input Injection: 防止恶意参数注入
- Tool Output Injection: 防止间接 Prompt Injection
- Tool Abuse: 防止危险工具滥用
- Data Exfiltration: 防止敏感数据泄露
- Resource Exhaustion: 防止资源耗尽攻击

Usage:
    from dspy_guardrails.mcp import (
        MCPGuardrail,
        ToolInputGuard,
        ToolOutputGuard,
        ToolCallPolicy,
        MCPSecurityAuditor,
    )

    # 创建 MCP Guardrail
    guardrail = MCPGuardrail(
        input_guards=[
            ToolInputGuard.injection_check(),
            ToolInputGuard.path_traversal_check(),
        ],
        output_guards=[
            ToolOutputGuard.indirect_injection_check(),
            ToolOutputGuard.pii_filter(),
        ],
        policies=[
            ToolCallPolicy.rate_limit(max_calls=100, window_seconds=60),
            ToolCallPolicy.dangerous_tool_confirmation(),
        ],
    )

    # 包装 MCP 工具
    @guardrail.protect
    async def my_tool(params):
        return result
"""

from .auditor import (
    AuditEvent,
    MCPSecurityAuditor,
    ThreatLevel,
)
from .core import (
    GuardAction,
    GuardResult,
    MCPGuardrail,
    MCPSecurityConfig,
    MCPSecurityError,
    ThreatCategory,
    ToolCallContext,
)
from .input_guards import (
    InputValidationResult,
    ToolInputGuard,
)
from .output_guards import (
    OutputSanitizationResult,
    ToolOutputGuard,
)
from .policies import (
    PolicyDecision,
    RateLimiter,
    ToolCallPolicy,
)
from .wrapper import (
    SecureMCPServer,
    SecureToolRegistry,
    secure_tool,
)

__all__ = [
    # Core
    "MCPGuardrail",
    "GuardResult",
    "GuardAction",
    "MCPSecurityConfig",
    "ToolCallContext",
    "ThreatCategory",
    "MCPSecurityError",

    # Input Guards
    "ToolInputGuard",
    "InputValidationResult",

    # Output Guards
    "ToolOutputGuard",
    "OutputSanitizationResult",

    # Policies
    "ToolCallPolicy",
    "PolicyDecision",
    "RateLimiter",

    # Auditor
    "MCPSecurityAuditor",
    "AuditEvent",
    "ThreatLevel",

    # Wrapper
    "SecureMCPServer",
    "secure_tool",
    "SecureToolRegistry",
]
