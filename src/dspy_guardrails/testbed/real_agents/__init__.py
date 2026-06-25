"""
Real Agents - 真实 Agent 实现

提供 4 种类型的 Agent 用于实验评估：
1. SimpleAgent - 简单 LLM 包装器（baseline）
2. ToolsAgent - ReAct 工具调用
3. RAGAgent - 检索增强生成
4. MultiAgentSystem - 多 Agent 协作
"""

from .base import (
    AgentResponse,
    AgentType,
    BaseAgent,
    ConversationTurn,
    ProtectionLevel,
)
from .multi_agent import (
    CoordinationMode,
    HandoffRecord,
    MultiAgentSystem,
)
from .rag_agent import (
    HybridRAGAgent,
    RAGAgent,
    RetrievedDocument,
)
from .simple_agent import (
    SimpleAgent,
    SimpleAgentWithKnowledge,
)
from .tools_agent import (
    ToolCall,
    ToolsAgent,
)


def create_agent(
    agent_type: str,
    protection_level: str = "none",
    guardrail_fn=None,
    **kwargs
) -> BaseAgent:
    """
    工厂函数：创建指定类型的 Agent

    Args:
        agent_type: 类型 (simple, tools, rag, multi)
        protection_level: 保护等级 (none, partial, full)
        guardrail_fn: 护栏检查函数
        **kwargs: 其他参数

    Returns:
        BaseAgent: 创建的 Agent 实例
    """
    level = ProtectionLevel(protection_level.lower())

    agents = {
        "simple": SimpleAgent,
        "simple_knowledge": SimpleAgentWithKnowledge,
        "tools": ToolsAgent,
        "rag": RAGAgent,
        "rag_hybrid": HybridRAGAgent,
        "multi": MultiAgentSystem,
    }

    agent_class = agents.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"未知的 Agent 类型: {agent_type}，可选: {list(agents.keys())}")

    return agent_class(
        protection_level=level,
        guardrail_fn=guardrail_fn,
        **kwargs
    )


__all__ = [
    # 基类
    "BaseAgent",
    "AgentType",
    "ProtectionLevel",
    "AgentResponse",
    "ConversationTurn",

    # SimpleAgent
    "SimpleAgent",
    "SimpleAgentWithKnowledge",

    # ToolsAgent
    "ToolsAgent",
    "ToolCall",

    # RAGAgent
    "RAGAgent",
    "HybridRAGAgent",
    "RetrievedDocument",

    # MultiAgentSystem
    "MultiAgentSystem",
    "CoordinationMode",
    "HandoffRecord",

    # 工厂函数
    "create_agent",
]
