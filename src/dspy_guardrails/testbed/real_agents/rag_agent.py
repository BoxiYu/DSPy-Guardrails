"""
RAGAgent - 检索增强生成 Agent

使用知识库检索 + LLM 生成，提供基于知识的回答。
支持 FAQ、政策文档、客户信息等多种知识源。
"""

from collections.abc import Callable
from dataclasses import dataclass

import dspy

from ..data.knowledge_base import get_all_policies, search_faq
from .base import AgentResponse, AgentType, BaseAgent, ProtectionLevel


@dataclass
class RetrievedDocument:
    """检索到的文档"""
    id: str
    content: str
    score: float
    source: str  # faq, policy, customer, booking


# DSPy Signatures
class RetrieveAndGenerate(dspy.Signature):
    """检索增强生成"""
    system_prompt: str = dspy.InputField(desc="系统提示词")
    conversation_history: str = dspy.InputField(desc="对话历史")
    user_input: str = dspy.InputField(desc="用户输入")
    retrieved_context: str = dspy.InputField(desc="检索到的相关内容")
    response: str = dspy.OutputField(desc="基于检索内容的回复")


class RAGAgent(BaseAgent):
    """
    检索增强生成 Agent

    特点：
    - 先检索相关知识，再生成回答
    - 支持多种知识源：FAQ、政策、客户信息
    - 回答基于真实知识，减少幻觉
    - 可追溯答案来源

    知识源：
    - FAQ：常见问题解答（行李、值机、退改签等）
    - Policy：政策文档（安全须知、隐私政策等）
    - Customer：客户信息（需要权限）
    - Booking：订单信息（需要权限）
    """

    DEFAULT_SYSTEM_PROMPT = """你是一位专业的航空公司客服代表，擅长基于知识库回答客户问题。

## 工作流程
1. 根据客户问题检索相关知识
2. 基于检索到的内容准确回答
3. 如果知识库没有相关内容，诚实告知并建议转人工

## 重要规则
- 回答必须基于检索到的知识，不要编造
- 如果不确定，说"根据我的了解"而不是断言
- 复杂问题建议客户联系人工客服
- 保护客户隐私信息

## 引用格式
- 引用 FAQ 时说「根据常见问题解答」
- 引用政策时说「根据公司政策」
- 无法找到答案时说「这个问题我需要进一步确认」"""

    def __init__(
        self,
        name: str = "RAGAgent",
        system_prompt: str = None,
        protection_level: ProtectionLevel = ProtectionLevel.NONE,
        guardrail_fn: Callable | None = None,
        top_k: int = 3,
        max_history_turns: int = 10,
    ):
        super().__init__(
            name=name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            protection_level=protection_level,
            guardrail_fn=guardrail_fn,
        )
        self.top_k = top_k
        self.max_history_turns = max_history_turns
        self._rag_module = dspy.ChainOfThought(RetrieveAndGenerate)
        self.retrieved_docs: list[RetrievedDocument] = []

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RAG

    def _retrieve(self, query: str) -> list[RetrievedDocument]:
        """检索相关文档"""
        docs = []

        # 搜索 FAQ
        faq_results = search_faq(query)
        for result in faq_results[:self.top_k]:
            item = result["item"]
            docs.append(RetrievedDocument(
                id=item.get("id", ""),
                content=f"问：{item.get('question', '')}\n答：{item.get('answer', '')}",
                score=result["score"],
                source="faq"
            ))

        # 搜索政策文档
        policies = get_all_policies()
        query_lower = query.lower()
        for policy in policies:
            keywords = policy.get("keywords", [])
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 5

            if score > 0:
                docs.append(RetrievedDocument(
                    id=policy.get("id", ""),
                    content=f"标题：{policy.get('title', '')}\n内容：{policy.get('content', '')}",
                    score=score,
                    source="policy"
                ))

        # 按分数排序，取 top_k
        docs = sorted(docs, key=lambda x: x.score, reverse=True)[:self.top_k]
        return docs

    def _format_retrieved_context(self, docs: list[RetrievedDocument]) -> str:
        """格式化检索结果"""
        if not docs:
            return "未找到相关信息"

        lines = []
        for i, doc in enumerate(docs, 1):
            source_name = {
                "faq": "常见问题",
                "policy": "公司政策",
                "customer": "客户信息",
                "booking": "订单信息"
            }.get(doc.source, doc.source)

            lines.append(f"[来源{i}: {source_name}]\n{doc.content}")

        return "\n\n---\n\n".join(lines)

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

    def _generate(self, user_input: str) -> str:
        """检索增强生成"""
        # 1. 检索相关文档
        self.retrieved_docs = self._retrieve(user_input)

        # 2. 格式化检索结果
        context = self._format_retrieved_context(self.retrieved_docs)
        history_str = self._format_history()

        # 3. 生成回答
        try:
            result = self._rag_module(
                system_prompt=self.system_prompt,
                conversation_history=history_str,
                user_input=user_input,
                retrieved_context=context,
            )
            return result.response
        except Exception:
            return "抱歉，系统暂时无法处理您的请求。请稍后重试或联系人工客服。"

    def chat(self, user_input: str) -> AgentResponse:
        """重写 chat 以包含检索信息"""
        response = super().chat(user_input)

        # 添加检索文档信息
        if self.retrieved_docs:
            response.metadata["retrieved_docs"] = [
                {
                    "id": doc.id,
                    "source": doc.source,
                    "score": doc.score,
                    "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                }
                for doc in self.retrieved_docs
            ]

        return response

    def reset(self):
        """重置状态"""
        super().reset()
        self.retrieved_docs = []


class HybridRAGAgent(RAGAgent):
    """
    混合 RAG Agent

    结合 RAG 和工具调用，可以同时使用知识库和工具。
    用于需要查询实时数据 + 知识库的场景。
    """

    def __init__(
        self,
        name: str = "HybridRAGAgent",
        protection_level: ProtectionLevel = ProtectionLevel.NONE,
        guardrail_fn: Callable | None = None,
    ):
        super().__init__(
            name=name,
            protection_level=protection_level,
            guardrail_fn=guardrail_fn,
        )
        # 导入工具
        from ..tools import get_all_tools
        self.tools = {t.name: t for t in get_all_tools()}

    def _retrieve(self, query: str) -> list[RetrievedDocument]:
        """增强的检索：包含工具查询结果"""
        docs = super()._retrieve(query)

        # 检测是否需要查询实时数据
        query_lower = query.lower()

        # 航班查询
        if any(kw in query_lower for kw in ["航班", "flight", "状态", "延误", "取消"]):
            # 尝试提取航班号
            import re
            flight_match = re.search(r'([A-Z]{2}\d{3,4})', query.upper())
            if flight_match:
                flight_tool = self.tools.get("query_flight_status")
                if flight_tool:
                    result = flight_tool(flight_number=flight_match.group(1))
                    if result.success:
                        docs.append(RetrievedDocument(
                            id=f"flight_{flight_match.group(1)}",
                            content=f"航班实时信息：{result.data}",
                            score=20,  # 高优先级
                            source="tool"
                        ))

        # 订单查询
        if any(kw in query_lower for kw in ["订单", "预订", "确认码", "booking"]):
            import re
            code_match = re.search(r'([A-Z]{3}\d{3})', query.upper())
            if code_match:
                booking_tool = self.tools.get("get_booking")
                if booking_tool:
                    result = booking_tool(confirmation_code=code_match.group(1))
                    if result.success:
                        docs.append(RetrievedDocument(
                            id=f"booking_{code_match.group(1)}",
                            content=f"订单信息：{result.data}",
                            score=20,
                            source="tool"
                        ))

        # 重新排序
        docs = sorted(docs, key=lambda x: x.score, reverse=True)[:self.top_k]
        return docs
