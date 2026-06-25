"""
知识库工具 - Knowledge Tools
搜索FAQ、政策和知识库
"""

from ..data.knowledge_base import get_all_policies, get_faq, get_faqs_by_category, get_policy
from ..data.knowledge_base import search_faq as search_faq_db
from .base import BaseTool, ToolCategory, ToolResult


class SearchFAQTool(BaseTool):
    """搜索FAQ工具"""

    @property
    def name(self) -> str:
        return "search_faq"

    @property
    def description(self) -> str:
        return "搜索常见问题解答，包括行李、值机、退改签、延误赔偿等"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.KNOWLEDGE

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题"
                },
                "category": {
                    "type": "string",
                    "enum": ["行李", "值机", "改签退票", "延误赔偿", "会员", "特殊服务"],
                    "description": "FAQ类别（可选）"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 3
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str, category: str = None, top_k: int = 3) -> ToolResult:
        if category:
            # 按类别搜索
            faqs = get_faqs_by_category(category)
            # 简单关键词匹配
            results = []
            query_lower = query.lower()
            for faq in faqs:
                question = faq.get("question", "").lower()
                _answer = faq.get("answer", "").lower()
                keywords = faq.get("keywords", [])

                score = 0
                for keyword in keywords:
                    if keyword in query_lower:
                        score += 10
                if query_lower in question:
                    score += 5
                if score > 0:
                    results.append({"item": faq, "score": score})

            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        else:
            # 全局搜索
            results = search_faq_db(query)[:top_k]

        if not results:
            return ToolResult(
                success=True,
                data=[],
                metadata={
                    "query": query,
                    "message": "未找到相关FAQ，建议转人工客服"
                }
            )

        # 提取结果
        faq_results = []
        for r in results:
            item = r["item"]
            faq_results.append({
                "id": item.get("id"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "category": item.get("category"),
                "relevance_score": r["score"]
            })

        return ToolResult(
            success=True,
            data=faq_results,
            metadata={
                "query": query,
                "category": category,
                "count": len(faq_results)
            }
        )


class GetPolicyTool(BaseTool):
    """获取政策工具"""

    @property
    def name(self) -> str:
        return "get_policy"

    @property
    def description(self) -> str:
        return "获取航空公司政策文档，如安全须知、隐私政策等"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.KNOWLEDGE

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "政策ID，如 POLICY001"
                },
                "policy_type": {
                    "type": "string",
                    "enum": ["safety", "privacy", "refund", "baggage", "all"],
                    "description": "政策类型（可选）"
                }
            },
            "required": []
        }

    def execute(self, policy_id: str = None, policy_type: str = None) -> ToolResult:
        if policy_id:
            policy = get_policy(policy_id)
            if not policy:
                return ToolResult(
                    success=False,
                    error=f"未找到政策 {policy_id}"
                )
            return ToolResult(
                success=True,
                data=policy,
                metadata={"policy_id": policy_id}
            )

        if policy_type == "all" or not policy_type:
            policies = get_all_policies()
            return ToolResult(
                success=True,
                data=policies,
                metadata={"count": len(policies)}
            )

        # 按类型筛选
        type_mapping = {
            "safety": "POLICY001",
            "privacy": "POLICY002",
            "refund": "FAQ005",
            "baggage": "FAQ001"
        }

        policy_id = type_mapping.get(policy_type)
        if policy_id:
            policy = get_policy(policy_id) or get_faq(policy_id)
            return ToolResult(
                success=True,
                data=policy,
                metadata={"policy_type": policy_type}
            )

        return ToolResult(
            success=False,
            error=f"未知的政策类型: {policy_type}"
        )
