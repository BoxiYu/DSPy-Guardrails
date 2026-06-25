"""
MCP Server Wrapper - MCP 服务器包装器

提供简单的方式为 MCP 服务器添加安全防护。
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .auditor import MCPSecurityAuditor
from .core import (
    GuardAction,
    MCPGuardrail,
    MCPSecurityConfig,
    MCPSecurityError,
    ToolCallContext,
)
from .input_guards import ToolInputGuard
from .output_guards import ToolOutputGuard
from .policies import ToolCallPolicy

F = TypeVar('F', bound=Callable)


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    handler: Callable
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical
    requires_confirmation: bool = False


class SecureToolRegistry:
    """
    安全工具注册表

    统一管理和保护 MCP 工具。

    Examples:
        registry = SecureToolRegistry()

        # 注册工具
        @registry.register(risk_level="high", requires_confirmation=True)
        async def execute_command(params):
            return run(params["command"])

        # 或者手动注册
        registry.add_tool(
            name="read_file",
            handler=read_file_handler,
            risk_level="low",
        )

        # 获取安全包装后的工具
        tools = registry.get_secure_tools()
    """

    def __init__(
        self,
        guardrail: MCPGuardrail = None,
        config: MCPSecurityConfig = None,
    ):
        self.guardrail = guardrail or MCPGuardrail(config=config or MCPSecurityConfig())
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str = None,
        description: str = "",
        risk_level: str = "low",
        requires_confirmation: bool = False,
        input_schema: dict[str, Any] = None,
    ) -> Callable[[F], F]:
        """
        装饰器：注册工具

        Examples:
            @registry.register(name="my_tool", risk_level="high")
            async def my_tool(params):
                return result
        """
        def decorator(func: F) -> F:
            tool_name = name or func.__name__

            self._tools[tool_name] = ToolDefinition(
                name=tool_name,
                handler=func,
                description=description or func.__doc__ or "",
                input_schema=input_schema or {},
                risk_level=risk_level,
                requires_confirmation=requires_confirmation,
            )

            return func

        return decorator

    def add_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        risk_level: str = "low",
        requires_confirmation: bool = False,
        input_schema: dict[str, Any] = None,
    ):
        """手动添加工具"""
        self._tools[name] = ToolDefinition(
            name=name,
            handler=handler,
            description=description,
            input_schema=input_schema or {},
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
        )

    def get_tool(self, name: str) -> ToolDefinition | None:
        """获取工具定义"""
        return self._tools.get(name)

    def get_secure_handler(self, name: str) -> Callable | None:
        """获取安全包装后的工具处理器"""
        tool = self._tools.get(name)
        if not tool:
            return None

        return self._wrap_handler(tool)

    def get_secure_tools(self) -> dict[str, Callable]:
        """获取所有安全包装后的工具"""
        return {
            name: self._wrap_handler(tool)
            for name, tool in self._tools.items()
        }

    def _wrap_handler(self, tool: ToolDefinition) -> Callable:
        """包装处理器"""
        import asyncio

        @functools.wraps(tool.handler)
        async def async_wrapper(params: dict[str, Any]) -> Any:
            context = ToolCallContext(
                tool_name=tool.name,
                parameters=params,
                metadata={"risk_level": tool.risk_level},
            )

            # 检查输入
            input_result = self.guardrail.check_input(context)
            if not input_result.passed:
                raise MCPSecurityError(input_result)

            # 检查确认
            if tool.requires_confirmation and input_result.action != GuardAction.CONFIRM:
                # 在实际 MCP 中，这里会返回确认请求
                pass

            # 执行工具
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(params)
            else:
                result = tool.handler(params)

            # 过滤输出
            filtered, output_result = self.guardrail.filter_output(result, context)
            if not output_result.passed:
                raise MCPSecurityError(output_result)

            return filtered

        @functools.wraps(tool.handler)
        def sync_wrapper(params: dict[str, Any]) -> Any:
            context = ToolCallContext(
                tool_name=tool.name,
                parameters=params,
                metadata={"risk_level": tool.risk_level},
            )

            input_result = self.guardrail.check_input(context)
            if not input_result.passed:
                raise MCPSecurityError(input_result)

            result = tool.handler(params)

            filtered, output_result = self.guardrail.filter_output(result, context)
            if not output_result.passed:
                raise MCPSecurityError(output_result)

            return filtered

        if asyncio.iscoroutinefunction(tool.handler):
            return async_wrapper
        return sync_wrapper


def secure_tool(
    guardrail: MCPGuardrail = None,
    risk_level: str = "low",
    requires_confirmation: bool = False,
) -> Callable[[F], F]:
    """
    装饰器：为单个工具添加安全防护

    Examples:
        @secure_tool(risk_level="high")
        async def dangerous_operation(params):
            return result

        # 使用自定义 guardrail
        my_guardrail = MCPGuardrail(config=custom_config)

        @secure_tool(guardrail=my_guardrail)
        async def my_tool(params):
            return result
    """
    _guardrail = guardrail or MCPGuardrail()

    def decorator(func: F) -> F:
        import asyncio

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # 提取参数
            params = kwargs if kwargs else (args[0] if args else {})
            if not isinstance(params, dict):
                params = {}

            context = ToolCallContext(
                tool_name=func.__name__,
                parameters=params,
                metadata={"risk_level": risk_level},
            )

            # 检查输入
            input_result = _guardrail.check_input(context)
            if not input_result.passed:
                raise MCPSecurityError(input_result)

            # 执行
            result = await func(*args, **kwargs)

            # 过滤输出
            filtered, output_result = _guardrail.filter_output(result, context)
            if not output_result.passed:
                raise MCPSecurityError(output_result)

            return filtered

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            params = kwargs if kwargs else (args[0] if args else {})
            if not isinstance(params, dict):
                params = {}

            context = ToolCallContext(
                tool_name=func.__name__,
                parameters=params,
                metadata={"risk_level": risk_level},
            )

            input_result = _guardrail.check_input(context)
            if not input_result.passed:
                raise MCPSecurityError(input_result)

            result = func(*args, **kwargs)

            filtered, output_result = _guardrail.filter_output(result, context)
            if not output_result.passed:
                raise MCPSecurityError(output_result)

            return filtered

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class SecureMCPServer:
    """
    安全 MCP 服务器包装器

    为 MCP 服务器提供全面的安全防护。

    Examples:
        # 基础用法
        server = SecureMCPServer(
            name="my-server",
            config=MCPSecurityConfig(
                dangerous_tools=["execute_command"],
                rate_limit_max_calls=100,
            ),
        )

        # 注册工具
        @server.tool(risk_level="low")
        async def read_file(params):
            return open(params["path"]).read()

        @server.tool(risk_level="high", requires_confirmation=True)
        async def delete_file(params):
            os.remove(params["path"])
            return {"success": True}

        # 获取安全指标
        metrics = server.get_security_metrics()

        # 生成安全报告
        report = server.generate_security_report()
    """

    def __init__(
        self,
        name: str = "secure-mcp-server",
        config: MCPSecurityConfig = None,
        input_guards: list[ToolInputGuard] = None,
        output_guards: list[ToolOutputGuard] = None,
        policies: list[ToolCallPolicy] = None,
        audit_log_path: str = None,
    ):
        self.name = name

        # 创建审计器
        self.auditor = MCPSecurityAuditor(
            log_path=audit_log_path,
            alert_callback=self._on_security_alert,
        )

        # 创建 guardrail
        self.guardrail = MCPGuardrail(
            config=config or MCPSecurityConfig(),
            input_guards=input_guards,
            output_guards=output_guards,
            policies=policies,
            auditor=self.auditor,
        )

        # 工具注册表
        self.registry = SecureToolRegistry(guardrail=self.guardrail)

        # 告警回调
        self._alert_handlers: list[Callable] = []

    def tool(
        self,
        name: str = None,
        description: str = "",
        risk_level: str = "low",
        requires_confirmation: bool = False,
        input_schema: dict[str, Any] = None,
    ) -> Callable[[F], F]:
        """装饰器：注册工具"""
        return self.registry.register(
            name=name,
            description=description,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            input_schema=input_schema,
        )

    def add_tool(
        self,
        name: str,
        handler: Callable,
        **kwargs,
    ):
        """添加工具"""
        self.registry.add_tool(name=name, handler=handler, **kwargs)

    async def call_tool(
        self,
        name: str,
        params: dict[str, Any],
        session_id: str = None,
        caller_id: str = None,
    ) -> Any:
        """
        安全调用工具

        Args:
            name: 工具名
            params: 参数
            session_id: 会话 ID
            caller_id: 调用者 ID

        Returns:
            工具结果
        """
        handler = self.registry.get_secure_handler(name)
        if not handler:
            raise ValueError(f"Tool not found: {name}")

        return await handler(params)

    def get_tools(self) -> dict[str, ToolDefinition]:
        """获取所有工具定义"""
        return self.registry._tools.copy()

    def get_secure_handlers(self) -> dict[str, Callable]:
        """获取所有安全处理器"""
        return self.registry.get_secure_tools()

    def get_security_metrics(self) -> dict:
        """获取安全指标"""
        return self.auditor.get_metrics()

    def get_recent_threats(self, limit: int = 50) -> list:
        """获取最近威胁"""
        return self.auditor.get_recent_threats(limit=limit)

    def generate_security_report(self) -> str:
        """生成安全报告"""
        return self.auditor.generate_report()

    def on_alert(self, handler: Callable) -> Callable:
        """
        注册告警处理器

        Examples:
            @server.on_alert
            def handle_alert(event):
                send_notification(event)
        """
        self._alert_handlers.append(handler)
        return handler

    def _on_security_alert(self, event):
        """内部告警处理"""
        for handler in self._alert_handlers:
            try:
                handler(event)
            except Exception:
                pass


# ============================================================================
# MCP Protocol Integration Helpers
# ============================================================================

def create_mcp_error_response(error: MCPSecurityError) -> dict:
    """创建 MCP 错误响应"""
    return {
        "content": [
            {
                "type": "text",
                "text": f"Security Error: {error.result.message}",
            }
        ],
        "isError": True,
        "_meta": {
            "security": error.to_dict(),
        },
    }


def create_confirmation_request(
    tool_name: str,
    params: dict[str, Any],
    reason: str,
) -> dict:
    """创建确认请求"""
    return {
        "type": "confirmation_required",
        "tool": tool_name,
        "reason": reason,
        "message": f"The tool '{tool_name}' requires confirmation before execution.",
        "params_preview": {
            k: str(v)[:100] for k, v in list(params.items())[:5]
        },
    }
