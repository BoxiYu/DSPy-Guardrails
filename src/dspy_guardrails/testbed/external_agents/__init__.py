"""
External Agents - 外部 Agent 连接器

支持连接各种外部 AI Agent 平台进行测试：
1. HTTPAgent - 通用 HTTP API 连接器
2. WebSocketAgent - WebSocket 实时连接
3. GenesysAgent - Genesys Cloud Virtual Agent
4. DifyAgent - Dify 平台
5. OpenAIAssistant - OpenAI Assistants API
6. LangServeAgent - LangServe 部署的 Agent

使用示例:
    from dspy_guardrails.testbed.external_agents import HTTPAgent, GenesysAgent

    # 通用 HTTP Agent
    agent = HTTPAgent(
        base_url="http://localhost:8000",
        chat_endpoint="/chat",
        auth_token="your-token"
    )

    # Genesys Virtual Agent
    agent = GenesysAgent(
        region="mypurecloud.com",
        deployment_id="your-deployment-id",
        oauth_client_id="your-client-id",
        oauth_client_secret="your-secret"
    )

    # 统一接口
    response = agent.chat("Hello, how can you help me?")

    # 使用编排器测试多个 Agent
    from dspy_guardrails.testbed.external_agents import (
        ExternalAgentOrchestrator,
        create_external_agent,
        create_default_test_cases,
    )

    orchestrator = ExternalAgentOrchestrator()
    orchestrator.add_agent("local", create_external_agent("http", base_url="..."))
    orchestrator.add_agent("genesys", create_external_agent("genesys", ...))

    results = orchestrator.run_tests(create_default_test_cases())
    print(results.get_summary())
"""

from .base import (
    AuthMethod,
    ConnectionStatus,
    ExternalAgent,
    ExternalAgentConfig,
    ExternalAgentResponse,
)
from .factory import (
    create_external_agent,
    list_agent_types,
    register_agent_type,
)
from .http_agent import (
    HTTPAgent,
    HTTPAgentConfig,
)
from .orchestrator import (
    AgentTestCase,
    AgentTestResult,
    ExternalAgentOrchestrator,
    OrchestratorResults,
    create_default_test_cases,
)
from .platforms import (
    # Dify
    DifyAgent,
    DifyConfig,
    # Genesys
    GenesysAgent,
    GenesysConfig,
    # LangServe
    LangServeAgent,
    LangServeConfig,
    # OpenAI
    OpenAIAssistantAgent,
    OpenAIAssistantConfig,
)
from .websocket_agent import (
    WebSocketAgent,
    WebSocketAgentConfig,
)

__all__ = [
    # Base
    "ExternalAgent",
    "ExternalAgentConfig",
    "ExternalAgentResponse",
    "ConnectionStatus",
    "AuthMethod",
    # HTTP
    "HTTPAgent",
    "HTTPAgentConfig",
    # WebSocket
    "WebSocketAgent",
    "WebSocketAgentConfig",
    # Platforms
    "GenesysAgent",
    "GenesysConfig",
    "DifyAgent",
    "DifyConfig",
    "OpenAIAssistantAgent",
    "OpenAIAssistantConfig",
    "LangServeAgent",
    "LangServeConfig",
    # Factory
    "create_external_agent",
    "register_agent_type",
    "list_agent_types",
    # Orchestrator
    "ExternalAgentOrchestrator",
    "AgentTestCase",
    "AgentTestResult",
    "OrchestratorResults",
    "create_default_test_cases",
]
