"""
Testbed module for evaluating dspyGuardrails capabilities.

This module provides tools for creating agent testbeds to evaluate
guardrails against various attack vectors and protection levels.

Components:
- Config: 配置类 (AgentConfig, TestbedConfig, ProtectionLevel)
- Agents: Agent实现 (SimpleAgent, ToolsAgent, RAGAgent, MultiAgentSystem)
- Protection: 护栏包装器
- Data: 15+ Mock 数据源
- Tools: 14 个工具 (查询、修改、外部API、知识库)
- Results: 测试结果
- Orchestrator: 测试编排器
"""

from dspy_guardrails.testbed.agents import GuardrailResult, MockAgent, MockAgentFactory
from dspy_guardrails.testbed.compat import (
    AgentEvalAdapter,
    AgentEvalConfig,
    AgentEvalTest,
)
from dspy_guardrails.testbed.config import (
    AgentComplexity,
    AgentConfig,
    AgentDomain,
    ProtectionLevel,
    TestbedConfig,
)

# 新增：Mock 数据
from dspy_guardrails.testbed.data import (
    AGENT_NOTES_DB,
    AUDIT_LOGS_DB,
    BAGGAGE_CLAIMS_DB,
    BAGGAGE_DB,
    BOOKINGS_DB,
    CUSTOMERS_DB,
    FLIGHTS_DB,
    HOTELS_DB,
    INTERNAL_DB,
    KNOWLEDGE_BASE,
    PAYMENTS_DB,
    PROMOTIONS_DB,
    REFUNDS_DB,
    SEAT_MAPS_DB,
    SPECIAL_SERVICES_DB,
    WATCHLIST_DB,
)

# 新增：实验模块
from dspy_guardrails.testbed.experiment import (
    AttackCase,
    AttackCategory,
    ExperimentCache,
    ExperimentResults,
    ExperimentRunner,
    ReportGenerator,
    create_default_attacks,
)

# 新增：外部 Agent 连接器
from dspy_guardrails.testbed.external_agents import (
    AgentTestCase,
    AgentTestResult,
    AuthMethod,
    ConnectionStatus,
    DifyAgent,
    DifyConfig,
    # Base
    ExternalAgent,
    ExternalAgentConfig,
    # Orchestrator
    ExternalAgentOrchestrator,
    ExternalAgentResponse,
    # Platforms
    GenesysAgent,
    GenesysConfig,
    # HTTP
    HTTPAgent,
    HTTPAgentConfig,
    LangServeAgent,
    LangServeConfig,
    OpenAIAssistantAgent,
    OpenAIAssistantConfig,
    OrchestratorResults,
    # WebSocket
    WebSocketAgent,
    WebSocketAgentConfig,
    create_default_test_cases,
    # Factory
    create_external_agent,
    list_agent_types,
    register_agent_type,
)
from dspy_guardrails.testbed.orchestrator import (
    ProgressEvent,
    TestbedOrchestrator,
)
from dspy_guardrails.testbed.protection import GuardrailWrapper, create_guardrails_for_level

# 新增：真实 Agent
from dspy_guardrails.testbed.real_agents import (
    AgentResponse,
    AgentType,
    BaseAgent,
    MultiAgentSystem,
    RAGAgent,
    SimpleAgent,
    ToolsAgent,
    create_agent,
)
from dspy_guardrails.testbed.results import (
    ComparisonRow,
    SingleTestResult,
    TestbedResults,
)

# 新增：工具
from dspy_guardrails.testbed.tools import (
    BaseTool,
    ToolCategory,
    ToolRegistry,
    ToolResult,
    get_all_tools,
    get_tool,
    get_tools_as_openai_functions,
    get_tools_by_category,
    tool_registry,
)

__all__ = [
    # Config
    "AgentComplexity",
    "AgentConfig",
    "AgentDomain",
    "ProtectionLevel",
    "TestbedConfig",
    # Agents
    "MockAgent",
    "GuardrailResult",
    "MockAgentFactory",
    # Protection
    "GuardrailWrapper",
    "create_guardrails_for_level",
    # Results
    "SingleTestResult",
    "ComparisonRow",
    "TestbedResults",
    # Orchestrator
    "TestbedOrchestrator",
    "ProgressEvent",
    # Compat
    "AgentEvalAdapter",
    "AgentEvalConfig",
    "AgentEvalTest",
    # Data
    "FLIGHTS_DB",
    "BOOKINGS_DB",
    "CUSTOMERS_DB",
    "PAYMENTS_DB",
    "REFUNDS_DB",
    "BAGGAGE_DB",
    "BAGGAGE_CLAIMS_DB",
    "SPECIAL_SERVICES_DB",
    "PROMOTIONS_DB",
    "WATCHLIST_DB",
    "AGENT_NOTES_DB",
    "SEAT_MAPS_DB",
    "HOTELS_DB",
    "KNOWLEDGE_BASE",
    "INTERNAL_DB",
    "AUDIT_LOGS_DB",
    # Tools
    "BaseTool",
    "ToolCategory",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",
    "get_all_tools",
    "get_tool",
    "get_tools_by_category",
    "get_tools_as_openai_functions",
    # Real Agents
    "BaseAgent",
    "AgentType",
    "AgentResponse",
    "SimpleAgent",
    "ToolsAgent",
    "RAGAgent",
    "MultiAgentSystem",
    "create_agent",
    # Experiment
    "ExperimentCache",
    "ExperimentRunner",
    "ExperimentResults",
    "ReportGenerator",
    "AttackCase",
    "AttackCategory",
    "create_default_attacks",
    # External Agents
    "ExternalAgent",
    "ExternalAgentConfig",
    "ExternalAgentResponse",
    "ConnectionStatus",
    "AuthMethod",
    "HTTPAgent",
    "HTTPAgentConfig",
    "WebSocketAgent",
    "WebSocketAgentConfig",
    "GenesysAgent",
    "GenesysConfig",
    "DifyAgent",
    "DifyConfig",
    "OpenAIAssistantAgent",
    "OpenAIAssistantConfig",
    "LangServeAgent",
    "LangServeConfig",
    "create_external_agent",
    "register_agent_type",
    "list_agent_types",
    "ExternalAgentOrchestrator",
    "AgentTestCase",
    "AgentTestResult",
    "OrchestratorResults",
    "create_default_test_cases",
]
