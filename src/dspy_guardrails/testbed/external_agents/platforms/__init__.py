"""
平台特定 Agent 适配器

支持连接各种商业和开源 AI Agent 平台。
"""

from .dify import DifyAgent, DifyConfig
from .genesys import GenesysAgent, GenesysConfig
from .langserve import LangServeAgent, LangServeConfig
from .openai_assistant import OpenAIAssistantAgent, OpenAIAssistantConfig

__all__ = [
    "GenesysAgent",
    "GenesysConfig",
    "DifyAgent",
    "DifyConfig",
    "OpenAIAssistantAgent",
    "OpenAIAssistantConfig",
    "LangServeAgent",
    "LangServeConfig",
]
