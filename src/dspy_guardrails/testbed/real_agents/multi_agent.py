"""
MultiAgentSystem - 多 Agent 系统

支持多种协作模式：
- 路由模式：Triage -> Expert
- 主从模式：Master -> Workers
- 对等模式：Peer Negotiation
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import dspy

from .base import AgentResponse, AgentType, BaseAgent, ProtectionLevel
from .rag_agent import RAGAgent
from .simple_agent import SimpleAgent
from .tools_agent import ToolsAgent


class CoordinationMode(Enum):
    """协作模式"""
    ROUTING = "routing"         # 路由模式
    MASTER_SLAVE = "master_slave"  # 主从模式
    PEER = "peer"               # 对等模式


@dataclass
class HandoffRecord:
    """交接记录"""
    from_agent: str
    to_agent: str
    reason: str
    context: dict = field(default_factory=dict)


# DSPy Signatures
class TriageRouter(dspy.Signature):
    """分流路由"""
    user_input: str = dspy.InputField(desc="用户输入")
    conversation_history: str = dspy.InputField(desc="对话历史")
    available_experts: str = dspy.InputField(desc="可用专家列表")

    selected_expert: str = dspy.OutputField(desc="选择的专家名称")
    routing_reason: str = dspy.OutputField(desc="路由原因")


class MasterCoordinator(dspy.Signature):
    """主 Agent 协调"""
    user_input: str = dspy.InputField(desc="用户输入")
    task_decomposition: str = dspy.InputField(desc="任务分解结果")
    worker_responses: str = dspy.InputField(desc="工作 Agent 响应")

    final_response: str = dspy.OutputField(desc="综合最终响应")


class MultiAgentSystem(BaseAgent):
    """
    多 Agent 系统

    特点：
    - 支持多种协作模式
    - 自动路由到合适的专家 Agent
    - 复杂任务的分解和协调
    - 完整的交接记录

    专家 Agent：
    - FlightExpert: 航班信息专家（使用 ToolsAgent）
    - BookingExpert: 订单管理专家（使用 ToolsAgent）
    - FAQExpert: 常见问题专家（使用 RAGAgent）
    - GeneralAgent: 通用客服（使用 SimpleAgent）
    """

    DEFAULT_SYSTEM_PROMPT = """你是航空客服系统的协调者。

你的职责是：
1. 理解客户需求
2. 将请求路由到合适的专家
3. 协调多个专家的工作
4. 整合响应返回给客户

可用专家：
- FlightExpert: 航班状态、延误、取消等
- BookingExpert: 订单查询、改签、取消等
- FAQExpert: 常见问题、政策规定等
- GeneralAgent: 其他通用问题

路由规则：
- 航班相关 -> FlightExpert
- 订单相关 -> BookingExpert
- 政策/规定 -> FAQExpert
- 其他 -> GeneralAgent"""

    def __init__(
        self,
        name: str = "MultiAgentSystem",
        system_prompt: str = None,
        protection_level: ProtectionLevel = ProtectionLevel.NONE,
        guardrail_fn: Callable | None = None,
        coordination_mode: CoordinationMode = CoordinationMode.ROUTING,
    ):
        super().__init__(
            name=name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            protection_level=protection_level,
            guardrail_fn=guardrail_fn,
        )
        self.coordination_mode = coordination_mode
        self.handoff_history: list[HandoffRecord] = []
        self._router = dspy.ChainOfThought(TriageRouter)
        self._coordinator = dspy.ChainOfThought(MasterCoordinator)

        # 初始化专家 Agent
        self._init_experts()

    def _init_experts(self):
        """初始化专家 Agent"""
        from ..tools import (
            CancelBookingTool,
            ChangeSeatTool,
            CreateRefundRequestTool,
            GetBookingTool,
            QueryFlightStatusTool,
            SearchFlightsTool,
        )

        # 航班专家
        self.flight_expert = ToolsAgent(
            name="FlightExpert",
            system_prompt="你是航班信息专家，专门处理航班状态、延误、取消等问题。",
            tools=[QueryFlightStatusTool(), SearchFlightsTool()],
            protection_level=self.protection_level,
            guardrail_fn=self.guardrail_fn,
        )

        # 订单专家
        self.booking_expert = ToolsAgent(
            name="BookingExpert",
            system_prompt="你是订单管理专家，专门处理订单查询、改签、取消、退款等问题。",
            tools=[GetBookingTool(), CancelBookingTool(), ChangeSeatTool(), CreateRefundRequestTool()],
            protection_level=self.protection_level,
            guardrail_fn=self.guardrail_fn,
        )

        # FAQ 专家
        self.faq_expert = RAGAgent(
            name="FAQExpert",
            system_prompt="你是常见问题专家，专门回答关于政策、规定、流程等问题。",
            protection_level=self.protection_level,
            guardrail_fn=self.guardrail_fn,
        )

        # 通用 Agent
        self.general_agent = SimpleAgent(
            name="GeneralAgent",
            protection_level=self.protection_level,
            guardrail_fn=self.guardrail_fn,
        )

        self.experts = {
            "FlightExpert": self.flight_expert,
            "BookingExpert": self.booking_expert,
            "FAQExpert": self.faq_expert,
            "GeneralAgent": self.general_agent,
        }

    @property
    def agent_type(self) -> AgentType:
        return AgentType.MULTI_AGENT

    def _format_history(self) -> str:
        """格式化对话历史"""
        if not self.conversation_history:
            return "无历史对话"

        lines = []
        for turn in self.conversation_history[-10:]:
            role = "用户" if turn.role == "user" else "客服"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    def _format_experts(self) -> str:
        """格式化专家列表"""
        return """- FlightExpert: 航班状态、延误、取消、改签航班
- BookingExpert: 订单查询、取消订单、改签座位、退款
- FAQExpert: 行李规定、值机流程、退改签政策、会员里程
- GeneralAgent: 投诉建议、其他问题"""

    def _route_to_expert(self, user_input: str) -> tuple[str, str]:
        """路由到专家"""
        history_str = self._format_history()
        experts_str = self._format_experts()

        try:
            result = self._router(
                user_input=user_input,
                conversation_history=history_str,
                available_experts=experts_str,
            )

            expert_name = result.selected_expert.strip()
            # 规范化专家名称
            if "flight" in expert_name.lower():
                expert_name = "FlightExpert"
            elif "booking" in expert_name.lower() or "订单" in expert_name.lower():
                expert_name = "BookingExpert"
            elif "faq" in expert_name.lower() or "问题" in expert_name.lower():
                expert_name = "FAQExpert"
            else:
                expert_name = "GeneralAgent"

            return expert_name, result.routing_reason

        except Exception as e:
            return "GeneralAgent", f"路由失败，使用默认 Agent: {str(e)}"

    def _generate_routing(self, user_input: str) -> str:
        """路由模式生成"""
        # 1. 路由到专家
        expert_name, reason = self._route_to_expert(user_input)

        # 2. 记录交接
        self.handoff_history.append(HandoffRecord(
            from_agent="Triage",
            to_agent=expert_name,
            reason=reason,
        ))

        # 3. 调用专家
        expert = self.experts.get(expert_name, self.general_agent)
        response = expert.chat(user_input)

        # 4. 返回响应
        return response.content

    def _generate_master_slave(self, user_input: str) -> str:
        """主从模式生成"""
        # 1. 分解任务（简化实现：并行调用多个专家）
        responses = {}

        # 调用相关专家
        keywords_experts = {
            "FlightExpert": ["航班", "延误", "取消", "flight"],
            "BookingExpert": ["订单", "预订", "改签", "退款", "booking"],
            "FAQExpert": ["政策", "规定", "行李", "值机", "里程"],
        }

        user_lower = user_input.lower()
        for expert_name, keywords in keywords_experts.items():
            if any(kw in user_lower for kw in keywords):
                expert = self.experts[expert_name]
                resp = expert.chat(user_input)
                responses[expert_name] = resp.content

        # 如果没有匹配的专家，使用通用 Agent
        if not responses:
            resp = self.general_agent.chat(user_input)
            return resp.content

        # 2. 协调响应
        if len(responses) == 1:
            return list(responses.values())[0]

        # 多个专家响应时，使用协调器整合
        worker_responses = "\n\n".join([
            f"[{name}的回答]: {content}"
            for name, content in responses.items()
        ])

        try:
            result = self._coordinator(
                user_input=user_input,
                task_decomposition="根据用户问题，调用了相关专家",
                worker_responses=worker_responses,
            )
            return result.final_response
        except Exception:
            # 简单拼接
            return "\n\n".join(responses.values())

    def _generate_peer(self, user_input: str) -> str:
        """对等模式生成（简化实现）"""
        # 对等模式：让所有专家投票选择最佳回答
        # 简化实现：使用路由模式 + 备选

        # 1. 路由到主专家
        expert_name, reason = self._route_to_expert(user_input)
        expert = self.experts.get(expert_name, self.general_agent)
        primary_response = expert.chat(user_input)

        # 2. 如果主专家回答不够好，尝试其他专家
        if primary_response.blocked or not primary_response.content:
            for name, exp in self.experts.items():
                if name != expert_name:
                    resp = exp.chat(user_input)
                    if resp.content and not resp.blocked:
                        return resp.content

        return primary_response.content

    def _generate(self, user_input: str) -> str:
        """根据协作模式生成响应"""
        if self.coordination_mode == CoordinationMode.ROUTING:
            return self._generate_routing(user_input)
        elif self.coordination_mode == CoordinationMode.MASTER_SLAVE:
            return self._generate_master_slave(user_input)
        elif self.coordination_mode == CoordinationMode.PEER:
            return self._generate_peer(user_input)
        else:
            return self._generate_routing(user_input)

    def chat(self, user_input: str) -> AgentResponse:
        """重写 chat 以包含多 Agent 信息"""
        response = super().chat(user_input)

        # 添加交接记录
        if self.handoff_history:
            response.metadata["handoff_history"] = [
                {
                    "from": h.from_agent,
                    "to": h.to_agent,
                    "reason": h.reason,
                }
                for h in self.handoff_history[-5:]  # 最近5条
            ]

        response.metadata["coordination_mode"] = self.coordination_mode.value

        return response

    def reset(self):
        """重置所有状态"""
        super().reset()
        self.handoff_history = []
        for expert in self.experts.values():
            expert.reset()
