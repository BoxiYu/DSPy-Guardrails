"""
ToolsAgent - ReAct 工具调用 Agent

使用 ReAct 模式进行工具调用，支持多轮推理。
参考 Dify 的 FunctionCallAgentRunner 设计。
"""

from collections.abc import Callable
from dataclasses import dataclass

import dspy

from ..tools import BaseTool, ToolResult, get_all_tools
from .base import AgentResponse, AgentType, BaseAgent, ProtectionLevel


@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    arguments: dict
    result: ToolResult | None = None
    iteration: int = 0


# DSPy Signatures
class ThinkAndAct(dspy.Signature):
    """ReAct 思考和行动"""
    system_prompt: str = dspy.InputField(desc="系统提示词")
    conversation_history: str = dspy.InputField(desc="对话历史")
    available_tools: str = dspy.InputField(desc="可用工具列表")
    user_input: str = dspy.InputField(desc="用户输入")
    scratchpad: str = dspy.InputField(desc="思考过程和工具调用结果")

    thought: str = dspy.OutputField(desc="思考过程：分析用户需求，决定下一步行动")
    action: str = dspy.OutputField(desc="行动：TOOL[tool_name] 或 FINISH")
    action_input: str = dspy.OutputField(desc="行动输入：工具参数JSON 或 最终回复")


class ToolsAgent(BaseAgent):
    """
    ReAct 工具调用 Agent

    特点：
    - 使用 ReAct (Reasoning + Acting) 模式
    - 支持多步推理和工具调用
    - 自动选择合适的工具处理用户请求
    - 支持工具调用链

    工具类别：
    - 查询类：query_flight_status, get_booking, search_flights, get_customer_profile
    - 修改类：cancel_booking, change_seat, update_customer_info, create_refund_request
    - 外部API：search_hotels, send_notification, check_weather
    - 知识库：search_faq, get_policy
    """

    DEFAULT_SYSTEM_PROMPT = """你是一位专业的航空公司客服代表，具备使用工具处理客户请求的能力。

## 工作流程
1. 理解客户需求
2. 选择合适的工具获取信息或执行操作
3. 基于工具返回结果回复客户

## 重要规则
- 必须使用工具获取真实数据，不要编造信息
- 保护客户隐私，敏感信息需要脱敏
- 修改类操作需要确认客户意图
- 无法处理的请求建议转人工

## 输出格式
- 思考：分析需求，决定使用哪个工具
- 行动：TOOL[工具名] 或 FINISH
- 输入：工具参数JSON 或 最终回复"""

    def __init__(
        self,
        name: str = "ToolsAgent",
        system_prompt: str = None,
        protection_level: ProtectionLevel = ProtectionLevel.NONE,
        guardrail_fn: Callable | None = None,
        tools: list[BaseTool] = None,
        max_iterations: int = 5,
        max_history_turns: int = 10,
    ):
        super().__init__(
            name=name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            protection_level=protection_level,
            guardrail_fn=guardrail_fn,
        )
        self.tools = {t.name: t for t in (tools or get_all_tools())}
        self.max_iterations = max_iterations
        self.max_history_turns = max_history_turns
        self._react_module = dspy.ChainOfThought(ThinkAndAct)
        self.tool_calls: list[ToolCall] = []

    @property
    def agent_type(self) -> AgentType:
        return AgentType.TOOLS

    def _format_tools(self) -> str:
        """格式化工具列表"""
        lines = []
        for name, tool in self.tools.items():
            params = tool.parameters.get("properties", {})
            param_str = ", ".join([
                f"{k}: {v.get('type', 'any')}"
                for k, v in params.items()
            ])
            lines.append(f"- {name}({param_str}): {tool.description}")
        return "\n".join(lines)

    def _format_history(self) -> str:
        """格式化对话历史"""
        if not self.conversation_history:
            return "无历史对话"

        recent = self.conversation_history[-(self.max_history_turns * 2):]
        lines = []
        for turn in recent:
            role = "用户" if turn.role == "user" else "客服"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    def _parse_action(self, action: str) -> tuple[str, str | None]:
        """解析行动"""
        action = action.strip()

        if action.upper() == "FINISH":
            return "FINISH", None

        # 解析 TOOL[tool_name]
        if action.upper().startswith("TOOL[") and action.endswith("]"):
            tool_name = action[5:-1].strip()
            return "TOOL", tool_name

        # 尝试直接作为工具名
        if action in self.tools:
            return "TOOL", action

        return "UNKNOWN", action

    def _parse_tool_input(self, action_input: str) -> dict:
        """解析工具输入"""
        import json
        try:
            # 尝试解析 JSON
            return json.loads(action_input)
        except json.JSONDecodeError:
            # 尝试解析简单格式
            params = {}
            for part in action_input.split(","):
                if "=" in part or ":" in part:
                    sep = "=" if "=" in part else ":"
                    key, value = part.split(sep, 1)
                    params[key.strip()] = value.strip().strip('"\'')
            return params

    def _execute_tool(self, tool_name: str, arguments: dict, iteration: int) -> ToolResult:
        """执行工具"""
        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"未知工具: {tool_name}"
            )

        try:
            result = tool(**arguments)
            self.tool_calls.append(ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                iteration=iteration
            ))
            return result
        except Exception as e:
            result = ToolResult(success=False, error=str(e))
            self.tool_calls.append(ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                iteration=iteration
            ))
            return result

    def _generate(self, user_input: str) -> str:
        """ReAct 循环生成响应"""
        self.tool_calls = []
        scratchpad = ""
        history_str = self._format_history()
        tools_str = self._format_tools()

        for iteration in range(self.max_iterations):
            try:
                result = self._react_module(
                    system_prompt=self.system_prompt,
                    conversation_history=history_str,
                    available_tools=tools_str,
                    user_input=user_input,
                    scratchpad=scratchpad,
                )

                thought = result.thought
                action = result.action
                action_input = result.action_input

                # 添加到 scratchpad
                scratchpad += f"\n\n迭代 {iteration + 1}:\n"
                scratchpad += f"思考: {thought}\n"
                scratchpad += f"行动: {action}\n"
                scratchpad += f"输入: {action_input}\n"

                # 解析行动
                action_type, tool_name = self._parse_action(action)

                if action_type == "FINISH":
                    return action_input

                elif action_type == "TOOL" and tool_name:
                    # 执行工具
                    arguments = self._parse_tool_input(action_input)
                    tool_result = self._execute_tool(tool_name, arguments, iteration)

                    # 添加结果到 scratchpad
                    scratchpad += f"结果: {tool_result}\n"

                else:
                    # 未知行动，尝试直接返回
                    return action_input

            except Exception as e:
                scratchpad += f"\n错误: {str(e)}\n"
                continue

        # 达到最大迭代次数
        return "抱歉，我在处理您的请求时遇到了困难。请稍后重试或联系人工客服。"

    def chat(self, user_input: str) -> AgentResponse:
        """重写 chat 以包含工具调用信息"""
        response = super().chat(user_input)

        # 添加工具调用记录
        if self.tool_calls:
            response.tool_calls = [
                {
                    "tool": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": str(tc.result) if tc.result else None,
                    "iteration": tc.iteration
                }
                for tc in self.tool_calls
            ]
            response.metadata["tool_call_count"] = len(self.tool_calls)

        return response

    def reset(self):
        """重置状态"""
        super().reset()
        self.tool_calls = []
