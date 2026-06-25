"""Tests for MCPTarget adapter."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dspy_guardrails.platform.targets import (
    MCPTarget,
    TargetType,
    TargetCapability,
)


class TestMCPTargetInit:
    """初始化测试"""

    def test_mcp_target_init(self):
        """测试基本初始化"""
        target = MCPTarget(tool_name="test_tool")

        assert target.target_type == TargetType.MCP_SERVER
        assert TargetCapability.SINGLE_TURN in target.capabilities
        assert TargetCapability.TOOL_USE in target.capabilities

    def test_mcp_target_init_with_tool_fn(self):
        """测试带工具函数初始化"""
        mock_fn = Mock(return_value="tool result")
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="my_tool",
            timeout=60,
        )

        assert target._tool_fn == mock_fn
        assert target._tool_name == "my_tool"
        assert target._timeout == 60

    def test_mcp_target_init_with_mcp_client(self):
        """测试带 MCP 客户端初始化"""
        mock_client = Mock()
        target = MCPTarget(
            mcp_client=mock_client,
            tool_name="client_tool",
        )

        assert target._mcp_client == mock_client
        assert target._tool_name == "client_tool"

    def test_mcp_target_name(self):
        """测试目标名称"""
        target = MCPTarget(tool_name="file_read")
        assert target.name == "mcp:file_read"

    def test_mcp_target_tool_name_property(self):
        """测试工具名称属性"""
        target = MCPTarget(tool_name="database_query")
        assert target.tool_name == "database_query"


class TestMCPTargetInvoke:
    """单轮调用测试"""

    def test_mcp_target_invoke_with_tool_fn(self):
        """测试使用工具函数调用"""
        mock_fn = Mock(return_value="File content: Hello World")
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="file_read",
            use_guardrail=False,  # Disable guardrail for this test
        )

        response = target.invoke("Read /home/test.txt")

        assert response.response == "File content: Hello World"
        assert response.was_blocked is False
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["tool_name"] == "file_read"
        mock_fn.assert_called_once_with("Read /home/test.txt")

    def test_mcp_target_invoke_mock_response(self):
        """测试 mock 响应"""
        target = MCPTarget(
            tool_name="mock_tool",
            use_guardrail=False,
        )

        response = target.invoke("Test query")

        assert "Mock MCP response to:" in response.response
        assert response.was_blocked is False

    def test_mcp_target_invoke_with_mcp_client_call_tool(self):
        """测试 MCP 客户端 call_tool 方法"""
        mock_client = Mock()
        mock_client.call_tool = Mock(return_value="Query result: 5 rows")
        target = MCPTarget(
            mcp_client=mock_client,
            tool_name="database_query",
            use_guardrail=False,
        )

        response = target.invoke("SELECT * FROM users")

        assert response.response == "Query result: 5 rows"
        mock_client.call_tool.assert_called_once_with(
            "database_query",
            {"input": "SELECT * FROM users"},
        )

    def test_mcp_target_invoke_with_mcp_client_invoke(self):
        """测试 MCP 客户端 invoke 方法"""
        mock_client = Mock(spec=[])  # No call_tool method
        mock_client.invoke = Mock(return_value="Invoked result")
        target = MCPTarget(
            mcp_client=mock_client,
            tool_name="my_tool",
            use_guardrail=False,
        )

        response = target.invoke("Test input")

        assert response.response == "Invoked result"
        mock_client.invoke.assert_called_once_with("my_tool", "Test input")

    def test_mcp_target_invoke_with_callable_client(self):
        """测试可调用的 MCP 客户端"""
        mock_client = Mock(return_value="Callable result")
        # Remove call_tool and invoke so it falls through to __call__
        mock_client.call_tool = None
        mock_client.invoke = None
        delattr(mock_client, "call_tool")
        delattr(mock_client, "invoke")

        target = MCPTarget(
            mcp_client=mock_client,
            tool_name="my_tool",
            use_guardrail=False,
        )

        response = target.invoke("Test")

        assert response.response == "Callable result"

    def test_mcp_target_invoke_error_handling(self):
        """测试错误处理"""
        mock_fn = Mock(side_effect=RuntimeError("Tool execution failed"))
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="failing_tool",
            use_guardrail=False,
        )

        response = target.invoke("Trigger error")

        assert "Error:" in response.response
        assert response.was_blocked is False
        assert "error" in response.metadata

    def test_mcp_target_invoke_latency(self):
        """测试延迟记录"""
        target = MCPTarget(
            tool_name="test_tool",
            use_guardrail=False,
        )
        response = target.invoke("Test input")
        assert response.latency_ms >= 0

    def test_mcp_target_invoke_tool_calls(self):
        """测试工具调用记录"""
        mock_fn = Mock(return_value="Result data")
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="my_tool",
            use_guardrail=False,
        )

        response = target.invoke("Execute something")

        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]
        assert tool_call["tool_name"] == "my_tool"
        assert tool_call["arguments"] == {"input": "Execute something"}
        assert "Result data" in tool_call["result"]


class TestMCPTargetGuardrail:
    """护栏集成测试"""

    def test_mcp_target_invoke_blocked_by_guardrail(self):
        """测试被护栏拦截"""
        # Import the actual module to get the classes
        from dspy_guardrails.mcp import MCPGuardrail, ToolCallContext, GuardAction, GuardResult

        # Create a mock that patches the _check_guardrail method
        target = MCPTarget(
            tool_name="dangerous_tool",
            use_guardrail=True,
        )

        # Patch the _check_guardrail method directly
        with patch.object(target, "_check_guardrail") as mock_check:
            mock_check.return_value = {
                "blocked": True,
                "reason": "Dangerous tool call detected",
                "scores": {"mcp_threat_score": 0.95},
            }

            response = target.invoke("Execute malicious command")

            assert response.was_blocked is True
            assert "Dangerous tool call detected" in response.block_reason
            assert response.metadata.get("guardrail_triggered") is True

    def test_mcp_target_invoke_with_guardrail_disabled(self):
        """测试禁用护栏"""
        mock_fn = Mock(return_value="Result")
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="my_tool",
            use_guardrail=False,  # Disabled
        )

        # Even dangerous input should pass
        response = target.invoke("rm -rf /")

        assert response.was_blocked is False
        mock_fn.assert_called_once()


class TestMCPTargetMultiTurn:
    """多轮调用测试"""

    def test_mcp_target_invoke_multi_turn(self):
        """测试多轮调用"""
        mock_fn = Mock(return_value="Multi-turn result")
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="test_tool",
            use_guardrail=False,
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Execute query"},
        ]

        response = target.invoke_multi_turn(messages)

        assert response.response == "Multi-turn result"
        mock_fn.assert_called_once_with("Execute query")

    def test_mcp_target_invoke_multi_turn_empty(self):
        """测试多轮调用空消息"""
        target = MCPTarget(tool_name="test_tool")
        response = target.invoke_multi_turn([])
        assert response.response == "No messages"
        assert response.was_blocked is False


class TestMCPTargetSession:
    """会话管理测试"""

    def test_mcp_target_call_history(self):
        """测试调用历史"""
        mock_fn = Mock(return_value="Result 1")
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="test_tool",
            use_guardrail=False,
        )

        target.invoke("Query 1")
        mock_fn.return_value = "Result 2"
        target.invoke("Query 2")

        assert len(target.call_history) == 2
        assert target.call_history[0]["prompt"] == "Query 1"
        assert target.call_history[1]["prompt"] == "Query 2"

    def test_mcp_target_reset_session(self):
        """测试重置会话"""
        mock_fn = Mock(return_value="Result")
        target = MCPTarget(
            tool_fn=mock_fn,
            tool_name="test_tool",
            use_guardrail=False,
        )

        target.invoke("Query")
        assert len(target.call_history) == 1

        target.reset_session()
        assert len(target.call_history) == 0


class TestMCPTargetCapabilities:
    """能力检查测试"""

    def test_mcp_target_supports(self):
        """测试能力检查"""
        target = MCPTarget(tool_name="test_tool")

        assert target.supports(TargetCapability.SINGLE_TURN)
        assert target.supports(TargetCapability.TOOL_USE)
        assert not target.supports(TargetCapability.MULTI_TURN)
        assert not target.supports(TargetCapability.STREAMING)

    def test_mcp_target_get_info(self):
        """测试获取信息"""
        target = MCPTarget(tool_name="test_tool")
        info = target.get_info()

        assert info["type"] == "mcp_server"
        assert "single_turn" in info["capabilities"]
        assert "tool_use" in info["capabilities"]


class TestMCPTargetIntegration:
    """集成测试"""

    def test_mcp_target_real_tool_execution(self):
        """测试真实工具执行"""
        def calculator(expression: str) -> str:
            """Simple calculator tool."""
            try:
                # Only allow simple arithmetic
                if all(c in "0123456789+-*/ ()" for c in expression):
                    result = eval(expression)
                    return f"Result: {result}"
                else:
                    return "Invalid expression"
            except Exception as e:
                return f"Error: {e}"

        target = MCPTarget(
            tool_fn=calculator,
            tool_name="calculator",
            use_guardrail=False,
        )

        response = target.invoke("2 + 3 * 4")

        assert response.response == "Result: 14"
        assert response.was_blocked is False

    def test_mcp_target_with_stateful_tool(self):
        """测试有状态工具"""
        state = {"counter": 0}

        def counter_tool(action: str) -> str:
            if "increment" in action.lower():
                state["counter"] += 1
            elif "decrement" in action.lower():
                state["counter"] -= 1
            return f"Counter: {state['counter']}"

        target = MCPTarget(
            tool_fn=counter_tool,
            tool_name="counter",
            use_guardrail=False,
        )

        r1 = target.invoke("increment")
        assert "Counter: 1" in r1.response

        r2 = target.invoke("increment")
        assert "Counter: 2" in r2.response

        r3 = target.invoke("decrement")
        assert "Counter: 1" in r3.response
