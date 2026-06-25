"""
MCP Guardrails Core - 核心框架

提供 MCP 安全防护的核心组件和配置。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dspy_guardrails.mcp.auditor import MCPSecurityAuditor
    from dspy_guardrails.mcp.input_guards import ToolInputGuard
    from dspy_guardrails.mcp.output_guards import ToolOutputGuard
    from dspy_guardrails.mcp.policies import ToolCallPolicy


class GuardAction(Enum):
    """防护动作"""
    ALLOW = "allow"           # 允许通过
    BLOCK = "block"           # 阻止
    MODIFY = "modify"         # 修改后通过
    WARN = "warn"             # 警告但通过
    CONFIRM = "confirm"       # 需要用户确认
    AUDIT = "audit"           # 仅审计记录


class ThreatCategory(Enum):
    """威胁类别"""
    INJECTION = "injection"               # 注入攻击
    INDIRECT_INJECTION = "indirect_injection"  # 间接注入
    PATH_TRAVERSAL = "path_traversal"     # 路径遍历
    COMMAND_INJECTION = "command_injection"  # 命令注入
    DATA_EXFILTRATION = "data_exfiltration"  # 数据泄露
    PRIVILEGE_ESCALATION = "privilege_escalation"  # 权限提升
    RESOURCE_ABUSE = "resource_abuse"     # 资源滥用
    PII_EXPOSURE = "pii_exposure"         # PII 暴露
    SENSITIVE_DATA = "sensitive_data"     # 敏感数据
    DANGEROUS_OPERATION = "dangerous_operation"  # 危险操作
    RATE_LIMIT = "rate_limit"             # 速率限制
    RUG_PULL = "rug_pull"                 # Rug Pull 攻击 (工具篡改)
    UNKNOWN = "unknown"                   # 未知威胁


@dataclass
class GuardResult:
    """防护结果"""
    action: GuardAction
    passed: bool
    threat_category: ThreatCategory | None = None
    threat_score: float = 0.0  # 0.0-1.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    modified_value: Any = None  # 如果 action=MODIFY，存储修改后的值
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "passed": self.passed,
            "threat_category": self.threat_category.value if self.threat_category else None,
            "threat_score": self.threat_score,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MCPSecurityConfig:
    """MCP 安全配置"""

    # 阻止阈值
    block_threshold: float = 0.8  # 威胁分数超过此值则阻止

    # 危险工具列表
    dangerous_tools: list[str] = field(default_factory=lambda: [
        "execute_command",
        "run_shell",
        "delete_file",
        "write_file",
        "send_email",
        "http_request",
        "database_query",
        "create_process",
    ])

    # 只读工具（安全）
    readonly_tools: list[str] = field(default_factory=lambda: [
        "read_file",
        "list_directory",
        "get_info",
        "search",
        "query",
    ])

    # 敏感参数名
    sensitive_params: list[str] = field(default_factory=lambda: [
        "password",
        "secret",
        "token",
        "api_key",
        "credential",
        "private_key",
    ])

    # 速率限制
    rate_limit_enabled: bool = True
    rate_limit_max_calls: int = 100
    rate_limit_window_seconds: int = 60

    # 审计
    audit_enabled: bool = True
    audit_log_path: str | None = None

    # 确认模式
    confirmation_required_tools: list[str] = field(default_factory=list)

    # 白名单
    allowed_paths: list[str] = field(default_factory=list)
    allowed_hosts: list[str] = field(default_factory=list)


@dataclass
class ToolCallContext:
    """工具调用上下文"""
    tool_name: str
    parameters: dict[str, Any]
    caller_id: str | None = None
    session_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def call_hash(self) -> str:
        """生成调用哈希"""
        content = f"{self.tool_name}:{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class MCPGuardrail:
    """
    MCP Guardrail - MCP 安全防护主类

    提供完整的 MCP 工具调用安全防护：
    - 输入验证
    - 输出过滤
    - 策略执行
    - 审计记录

    Examples:
        guardrail = MCPGuardrail(config=MCPSecurityConfig())

        # 检查工具调用
        context = ToolCallContext(
            tool_name="execute_command",
            parameters={"command": "rm -rf /"},
        )

        result = guardrail.check_input(context)
        if not result.passed:
            raise SecurityError(result.message)

        # 过滤输出
        output = tool_result
        filtered = guardrail.filter_output(output, context)
    """

    def __init__(
        self,
        config: MCPSecurityConfig = None,
        input_guards: list[ToolInputGuard] = None,
        output_guards: list[ToolOutputGuard] = None,
        policies: list[ToolCallPolicy] = None,
        auditor: MCPSecurityAuditor = None,
    ):
        self.config = config or MCPSecurityConfig()
        self.input_guards = input_guards or []
        self.output_guards = output_guards or []
        self.policies = policies or []
        self.auditor = auditor

        # 延迟导入避免循环依赖
        self._setup_default_guards()

    def _setup_default_guards(self):
        """设置默认防护"""
        from .input_guards import ToolInputGuard
        from .output_guards import ToolOutputGuard
        from .policies import ToolCallPolicy

        # 如果没有配置，添加默认防护
        if not self.input_guards:
            self.input_guards = [
                ToolInputGuard.injection_check(),
                ToolInputGuard.path_traversal_check(),
                ToolInputGuard.command_injection_check(),
            ]

        if not self.output_guards:
            self.output_guards = [
                ToolOutputGuard.indirect_injection_check(),
                ToolOutputGuard.pii_filter(),
            ]

        if not self.policies:
            if self.config.rate_limit_enabled:
                self.policies.append(
                    ToolCallPolicy.rate_limit(
                        max_calls=self.config.rate_limit_max_calls,
                        window_seconds=self.config.rate_limit_window_seconds,
                    )
                )
            self.policies.append(
                ToolCallPolicy.dangerous_tool_check(self.config.dangerous_tools)
            )

    def check_input(self, context: ToolCallContext) -> GuardResult:
        """
        检查工具输入

        Args:
            context: 工具调用上下文

        Returns:
            GuardResult: 检查结果
        """
        all_results = []

        # 执行所有输入防护
        for guard in self.input_guards:
            result = guard.check(context)
            all_results.append(result)

            # 如果阻止，立即返回
            if result.action == GuardAction.BLOCK:
                self._audit("input_blocked", context, result)
                return result

        # 执行策略检查
        for policy in self.policies:
            result = policy.evaluate(context)
            all_results.append(result)

            if result.action == GuardAction.BLOCK:
                self._audit("policy_blocked", context, result)
                return result

            if result.action == GuardAction.CONFIRM:
                self._audit("confirmation_required", context, result)
                return result

        # 所有检查通过
        final_result = self._aggregate_results(all_results)
        self._audit("input_allowed", context, final_result)
        return final_result

    def filter_output(
        self,
        output: Any,
        context: ToolCallContext,
    ) -> tuple[Any, GuardResult]:
        """
        过滤工具输出

        Args:
            output: 工具输出
            context: 工具调用上下文

        Returns:
            (filtered_output, GuardResult): 过滤后的输出和结果
        """
        current_output = output
        all_results = []

        for guard in self.output_guards:
            result = guard.filter(current_output, context)
            all_results.append(result)

            if result.action == GuardAction.BLOCK:
                self._audit("output_blocked", context, result)
                return None, result

            if result.action == GuardAction.MODIFY and result.modified_value is not None:
                current_output = result.modified_value

        final_result = self._aggregate_results(all_results)
        self._audit("output_filtered", context, final_result)
        return current_output, final_result

    def protect(self, func: Callable) -> Callable:
        """
        装饰器：保护工具函数

        Examples:
            @guardrail.protect
            async def my_tool(params):
                return result
        """
        import asyncio
        import functools

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 提取工具名和参数
            tool_name = func.__name__
            parameters = kwargs if kwargs else (args[0] if args else {})

            context = ToolCallContext(
                tool_name=tool_name,
                parameters=parameters if isinstance(parameters, dict) else {},
            )

            # 检查输入
            input_result = self.check_input(context)
            if not input_result.passed:
                raise MCPSecurityError(input_result)

            # 执行工具
            result = await func(*args, **kwargs)

            # 过滤输出
            filtered_output, output_result = self.filter_output(result, context)
            if not output_result.passed:
                raise MCPSecurityError(output_result)

            return filtered_output

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tool_name = func.__name__
            parameters = kwargs if kwargs else (args[0] if args else {})

            context = ToolCallContext(
                tool_name=tool_name,
                parameters=parameters if isinstance(parameters, dict) else {},
            )

            input_result = self.check_input(context)
            if not input_result.passed:
                raise MCPSecurityError(input_result)

            result = func(*args, **kwargs)

            filtered_output, output_result = self.filter_output(result, context)
            if not output_result.passed:
                raise MCPSecurityError(output_result)

            return filtered_output

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def _aggregate_results(self, results: list[GuardResult]) -> GuardResult:
        """聚合多个结果"""
        if not results:
            return GuardResult(action=GuardAction.ALLOW, passed=True)

        # 找最高威胁
        max_score = max(r.threat_score for r in results)
        blocked = any(r.action == GuardAction.BLOCK for r in results)
        warnings = [r for r in results if r.action == GuardAction.WARN]

        if blocked:
            block_result = next(r for r in results if r.action == GuardAction.BLOCK)
            return block_result

        return GuardResult(
            action=GuardAction.ALLOW,
            passed=True,
            threat_score=max_score,
            message="; ".join(r.message for r in warnings if r.message),
            details={"checks_passed": len(results), "warnings": len(warnings)},
        )

    def _audit(
        self,
        event_type: str,
        context: ToolCallContext,
        result: GuardResult,
    ):
        """记录审计事件"""
        if self.auditor and self.config.audit_enabled:
            self.auditor.log(event_type, context, result)


class MCPSecurityError(Exception):
    """MCP 安全异常"""

    def __init__(self, result: GuardResult):
        self.result = result
        super().__init__(result.message)

    def to_dict(self) -> dict:
        return {
            "error": "MCPSecurityError",
            "action": self.result.action.value,
            "threat_category": self.result.threat_category.value if self.result.threat_category else None,
            "message": self.result.message,
            "details": self.result.details,
        }
