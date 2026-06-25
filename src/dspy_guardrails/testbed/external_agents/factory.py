"""
Agent 工厂 - 统一创建和管理外部 Agent

提供注册、创建、列表等功能。
"""

from collections.abc import Callable

from .base import ExternalAgent, ExternalAgentConfig

# Agent 类型注册表
_AGENT_REGISTRY: dict[str, tuple[type[ExternalAgent], type[ExternalAgentConfig]]] = {}


def register_agent_type(
    type_name: str,
    agent_class: type[ExternalAgent],
    config_class: type[ExternalAgentConfig] = None,
) -> None:
    """
    注册 Agent 类型

    Args:
        type_name: 类型名称
        agent_class: Agent 类
        config_class: 配置类（可选）
    """
    _AGENT_REGISTRY[type_name.lower()] = (agent_class, config_class or ExternalAgentConfig)


def list_agent_types() -> list[dict]:
    """
    列出所有注册的 Agent 类型

    Returns:
        list[dict]: Agent 类型列表
    """
    return [
        {
            "type": type_name,
            "agent_class": agent_class.__name__,
            "config_class": config_class.__name__,
            "doc": agent_class.__doc__[:200] if agent_class.__doc__ else None,
        }
        for type_name, (agent_class, config_class) in _AGENT_REGISTRY.items()
    ]


def create_external_agent(
    agent_type: str,
    guardrail_fn: Callable | None = None,
    **kwargs,
) -> ExternalAgent:
    """
    创建外部 Agent 实例

    Args:
        agent_type: Agent 类型名称
        guardrail_fn: 护栏函数
        **kwargs: Agent 配置参数

    Returns:
        ExternalAgent: Agent 实例

    Raises:
        ValueError: 未知的 Agent 类型

    使用示例:
        # 创建 HTTP Agent
        agent = create_external_agent(
            "http",
            base_url="http://localhost:8000",
            chat_endpoint="/chat",
        )

        # 创建 Genesys Agent
        agent = create_external_agent(
            "genesys",
            region="mypurecloud.com",
            deployment_id="xxx",
            oauth_client_id="xxx",
            oauth_client_secret="xxx",
        )

        # 创建 Dify Agent
        agent = create_external_agent(
            "dify",
            base_url="https://api.dify.ai/v1",
            api_key="app-xxx",
        )

        # 创建 OpenAI Assistant
        agent = create_external_agent(
            "openai_assistant",
            api_key="sk-xxx",
            assistant_id="asst_xxx",
        )

        # 创建 LangServe Agent
        agent = create_external_agent(
            "langserve",
            base_url="http://localhost:8000",
            chain_path="/agent",
        )
    """
    type_name = agent_type.lower()

    if type_name not in _AGENT_REGISTRY:
        available = ", ".join(_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available types: {available}"
        )

    agent_class, config_class = _AGENT_REGISTRY[type_name]

    # 分离配置参数
    config_params = {}
    agent_params = {}

    # 获取 config_class 的字段
    if hasattr(config_class, "__dataclass_fields__"):
        config_fields = set(config_class.__dataclass_fields__.keys())
    else:
        config_fields = set()

    for key, value in kwargs.items():
        if key in config_fields or key in ExternalAgentConfig.__dataclass_fields__:
            config_params[key] = value
        else:
            agent_params[key] = value

    # 创建配置
    config = config_class(**config_params) if config_params else None

    # 创建 Agent
    return agent_class(
        config=config,
        guardrail_fn=guardrail_fn,
        **agent_params,
    )


def _register_builtin_agents():
    """注册内置 Agent 类型"""
    from .http_agent import HTTPAgent, HTTPAgentConfig
    from .platforms import (
        DifyAgent,
        DifyConfig,
        GenesysAgent,
        GenesysConfig,
        LangServeAgent,
        LangServeConfig,
        OpenAIAssistantAgent,
        OpenAIAssistantConfig,
    )
    from .websocket_agent import WebSocketAgent, WebSocketAgentConfig

    register_agent_type("http", HTTPAgent, HTTPAgentConfig)
    register_agent_type("websocket", WebSocketAgent, WebSocketAgentConfig)
    register_agent_type("ws", WebSocketAgent, WebSocketAgentConfig)  # 别名
    register_agent_type("genesys", GenesysAgent, GenesysConfig)
    register_agent_type("dify", DifyAgent, DifyConfig)
    register_agent_type("openai_assistant", OpenAIAssistantAgent, OpenAIAssistantConfig)
    register_agent_type("openai", OpenAIAssistantAgent, OpenAIAssistantConfig)  # 别名
    register_agent_type("langserve", LangServeAgent, LangServeConfig)


# 自动注册内置 Agent
_register_builtin_agents()
