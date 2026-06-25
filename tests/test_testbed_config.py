"""
Tests for the testbed configuration module.
"""

import os
import tempfile

import pytest

from dspy_guardrails.testbed import (
    AgentComplexity,
    AgentConfig,
    AgentDomain,
    ProtectionLevel,
    TestbedConfig,
)


class TestAgentComplexityEnum:
    """Tests for AgentComplexity enum."""

    def test_simple_value(self):
        assert AgentComplexity.SIMPLE.value == "simple"

    def test_tools_value(self):
        assert AgentComplexity.TOOLS.value == "tools"

    def test_multi_agent_value(self):
        assert AgentComplexity.MULTI_AGENT.value == "multi"

    def test_rag_value(self):
        assert AgentComplexity.RAG.value == "rag"

    def test_from_string(self):
        assert AgentComplexity("simple") == AgentComplexity.SIMPLE
        assert AgentComplexity("tools") == AgentComplexity.TOOLS
        assert AgentComplexity("multi") == AgentComplexity.MULTI_AGENT
        assert AgentComplexity("rag") == AgentComplexity.RAG


class TestAgentDomainEnum:
    """Tests for AgentDomain enum."""

    def test_customer_service_value(self):
        assert AgentDomain.CUSTOMER_SERVICE.value == "cs"

    def test_code_assistant_value(self):
        assert AgentDomain.CODE_ASSISTANT.value == "code"

    def test_knowledge_base_value(self):
        assert AgentDomain.KNOWLEDGE_BASE.value == "kb"

    def test_general_value(self):
        assert AgentDomain.GENERAL.value == "general"

    def test_from_string(self):
        assert AgentDomain("cs") == AgentDomain.CUSTOMER_SERVICE
        assert AgentDomain("code") == AgentDomain.CODE_ASSISTANT
        assert AgentDomain("kb") == AgentDomain.KNOWLEDGE_BASE
        assert AgentDomain("general") == AgentDomain.GENERAL


class TestProtectionLevelEnum:
    """Tests for ProtectionLevel enum."""

    def test_none_value(self):
        assert ProtectionLevel.NONE.value == "none"

    def test_partial_value(self):
        assert ProtectionLevel.PARTIAL.value == "partial"

    def test_full_value(self):
        assert ProtectionLevel.FULL.value == "full"

    def test_from_string(self):
        assert ProtectionLevel("none") == ProtectionLevel.NONE
        assert ProtectionLevel("partial") == ProtectionLevel.PARTIAL
        assert ProtectionLevel("full") == ProtectionLevel.FULL


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_auto_naming(self):
        """Test that name is auto-generated when not provided."""
        config = AgentConfig(
            complexity=AgentComplexity.SIMPLE,
            domain=AgentDomain.CUSTOMER_SERVICE,
        )
        assert config.name == "simple-cs"

    def test_auto_naming_tools_code(self):
        """Test auto-naming for tools complexity and code domain."""
        config = AgentConfig(
            complexity=AgentComplexity.TOOLS,
            domain=AgentDomain.CODE_ASSISTANT,
        )
        assert config.name == "tools-code"

    def test_auto_naming_multi_kb(self):
        """Test auto-naming for multi-agent complexity and knowledge base domain."""
        config = AgentConfig(
            complexity=AgentComplexity.MULTI_AGENT,
            domain=AgentDomain.KNOWLEDGE_BASE,
        )
        assert config.name == "multi-kb"

    def test_auto_naming_rag_general(self):
        """Test auto-naming for RAG complexity and general domain."""
        config = AgentConfig(
            complexity=AgentComplexity.RAG,
            domain=AgentDomain.GENERAL,
        )
        assert config.name == "rag-general"

    def test_custom_name(self):
        """Test that custom name is preserved."""
        config = AgentConfig(
            complexity=AgentComplexity.SIMPLE,
            domain=AgentDomain.CUSTOMER_SERVICE,
            name="my-custom-agent",
        )
        assert config.name == "my-custom-agent"

    def test_all_combinations(self):
        """Test auto-naming for all complexity/domain combinations."""
        for complexity in AgentComplexity:
            for domain in AgentDomain:
                config = AgentConfig(complexity=complexity, domain=domain)
                expected_name = f"{complexity.value}-{domain.value}"
                assert config.name == expected_name


class TestTestbedConfigDefaults:
    """Tests for TestbedConfig default values."""

    def test_default_agents(self):
        """Test that agents defaults to empty list."""
        config = TestbedConfig()
        assert config.agents == []

    def test_default_protection_levels(self):
        """Test that protection_levels defaults to all three levels."""
        config = TestbedConfig()
        assert config.protection_levels == [
            ProtectionLevel.NONE,
            ProtectionLevel.PARTIAL,
            ProtectionLevel.FULL,
        ]

    def test_default_attack_suites(self):
        """Test that attack_suites defaults to injection and jailbreak."""
        config = TestbedConfig()
        assert config.attack_suites == ["injection", "jailbreak"]

    def test_default_max_attacks_per_suite(self):
        """Test that max_attacks_per_suite defaults to None."""
        config = TestbedConfig()
        assert config.max_attacks_per_suite is None

    def test_default_llm_model(self):
        """Test that llm_model defaults to the expected model."""
        config = TestbedConfig()
        assert config.llm_model == "openai/kimi-k2-0905-preview"

    def test_default_llm_api_base(self):
        """Test that llm_api_base defaults to None."""
        config = TestbedConfig()
        assert config.llm_api_base is None

    def test_default_parallel(self):
        """Test that parallel defaults to True."""
        config = TestbedConfig()
        assert config.parallel is True

    def test_default_max_workers(self):
        """Test that max_workers defaults to 5."""
        config = TestbedConfig()
        assert config.max_workers == 5

    def test_default_output_dir(self):
        """Test that output_dir defaults to ./testbed_reports."""
        config = TestbedConfig()
        assert config.output_dir == "./testbed_reports"

    def test_default_report_formats(self):
        """Test that report_formats defaults to console, json, html."""
        config = TestbedConfig()
        assert config.report_formats == ["console", "json", "html"]


class TestTestbedConfigFromDict:
    """Tests for TestbedConfig.from_dict() method."""

    def test_parse_empty_dict(self):
        """Test parsing an empty dictionary uses defaults."""
        config = TestbedConfig.from_dict({})
        assert config.agents == []
        assert config.attack_suites == ["injection", "jailbreak"]

    def test_parse_agents(self):
        """Test parsing agents from dictionary."""
        data = {
            "agents": [
                {"complexity": "simple", "domain": "cs"},
                {"complexity": "tools", "domain": "code", "name": "custom-name"},
            ]
        }
        config = TestbedConfig.from_dict(data)

        assert len(config.agents) == 2
        assert config.agents[0].complexity == AgentComplexity.SIMPLE
        assert config.agents[0].domain == AgentDomain.CUSTOMER_SERVICE
        assert config.agents[0].name == "simple-cs"
        assert config.agents[1].complexity == AgentComplexity.TOOLS
        assert config.agents[1].domain == AgentDomain.CODE_ASSISTANT
        assert config.agents[1].name == "custom-name"

    def test_parse_protection_levels(self):
        """Test parsing protection levels from dictionary."""
        data = {"protection_levels": ["none", "full"]}
        config = TestbedConfig.from_dict(data)

        assert config.protection_levels == [ProtectionLevel.NONE, ProtectionLevel.FULL]

    def test_parse_attack_suites(self):
        """Test parsing attack suites from dictionary."""
        data = {"attack_suites": ["injection", "jailbreak", "bypass"]}
        config = TestbedConfig.from_dict(data)

        assert config.attack_suites == ["injection", "jailbreak", "bypass"]

    def test_parse_max_attacks_per_suite(self):
        """Test parsing max_attacks_per_suite from dictionary."""
        data = {"max_attacks_per_suite": 100}
        config = TestbedConfig.from_dict(data)

        assert config.max_attacks_per_suite == 100

    def test_parse_llm_config(self):
        """Test parsing LLM configuration from dictionary."""
        data = {
            "llm_model": "custom/model",
            "llm_api_base": "https://api.example.com",
        }
        config = TestbedConfig.from_dict(data)

        assert config.llm_model == "custom/model"
        assert config.llm_api_base == "https://api.example.com"

    def test_parse_parallel_config(self):
        """Test parsing parallel configuration from dictionary."""
        data = {
            "parallel": False,
            "max_workers": 10,
        }
        config = TestbedConfig.from_dict(data)

        assert config.parallel is False
        assert config.max_workers == 10

    def test_parse_output_config(self):
        """Test parsing output configuration from dictionary."""
        data = {
            "output_dir": "/custom/reports",
            "report_formats": ["json"],
        }
        config = TestbedConfig.from_dict(data)

        assert config.output_dir == "/custom/reports"
        assert config.report_formats == ["json"]

    def test_parse_complete_config(self):
        """Test parsing a complete configuration dictionary."""
        data = {
            "agents": [
                {"complexity": "multi", "domain": "kb"},
                {"complexity": "rag", "domain": "general", "name": "rag-agent"},
            ],
            "protection_levels": ["partial"],
            "attack_suites": ["injection"],
            "max_attacks_per_suite": 50,
            "llm_model": "test/model",
            "llm_api_base": "https://test.api.com",
            "parallel": True,
            "max_workers": 3,
            "output_dir": "./test_output",
            "report_formats": ["html", "json"],
        }
        config = TestbedConfig.from_dict(data)

        assert len(config.agents) == 2
        assert config.agents[0].name == "multi-kb"
        assert config.agents[1].name == "rag-agent"
        assert config.protection_levels == [ProtectionLevel.PARTIAL]
        assert config.attack_suites == ["injection"]
        assert config.max_attacks_per_suite == 50
        assert config.llm_model == "test/model"
        assert config.llm_api_base == "https://test.api.com"
        assert config.parallel is True
        assert config.max_workers == 3
        assert config.output_dir == "./test_output"
        assert config.report_formats == ["html", "json"]


class TestTestbedConfigToDict:
    """Tests for TestbedConfig.to_dict() method."""

    def test_to_dict_defaults(self):
        """Test converting default config to dictionary."""
        config = TestbedConfig()
        data = config.to_dict()

        assert data["agents"] == []
        assert data["protection_levels"] == ["none", "partial", "full"]
        assert data["attack_suites"] == ["injection", "jailbreak"]
        assert data["max_attacks_per_suite"] is None
        assert data["llm_model"] == "openai/kimi-k2-0905-preview"
        assert data["llm_api_base"] is None
        assert data["parallel"] is True
        assert data["max_workers"] == 5
        assert data["output_dir"] == "./testbed_reports"
        assert data["report_formats"] == ["console", "json", "html"]

    def test_to_dict_with_agents(self):
        """Test converting config with agents to dictionary."""
        config = TestbedConfig(
            agents=[
                AgentConfig(
                    complexity=AgentComplexity.SIMPLE,
                    domain=AgentDomain.CUSTOMER_SERVICE,
                ),
                AgentConfig(
                    complexity=AgentComplexity.TOOLS,
                    domain=AgentDomain.CODE_ASSISTANT,
                    name="custom-agent",
                ),
            ]
        )
        data = config.to_dict()

        assert len(data["agents"]) == 2
        assert data["agents"][0]["complexity"] == "simple"
        assert data["agents"][0]["domain"] == "cs"
        assert data["agents"][0]["name"] == "simple-cs"
        assert data["agents"][1]["complexity"] == "tools"
        assert data["agents"][1]["domain"] == "code"
        assert data["agents"][1]["name"] == "custom-agent"

    def test_to_dict_custom_values(self):
        """Test converting config with custom values to dictionary."""
        config = TestbedConfig(
            protection_levels=[ProtectionLevel.FULL],
            attack_suites=["bypass"],
            max_attacks_per_suite=200,
            llm_model="custom/model",
            llm_api_base="https://custom.api.com",
            parallel=False,
            max_workers=8,
            output_dir="/custom/dir",
            report_formats=["console"],
        )
        data = config.to_dict()

        assert data["protection_levels"] == ["full"]
        assert data["attack_suites"] == ["bypass"]
        assert data["max_attacks_per_suite"] == 200
        assert data["llm_model"] == "custom/model"
        assert data["llm_api_base"] == "https://custom.api.com"
        assert data["parallel"] is False
        assert data["max_workers"] == 8
        assert data["output_dir"] == "/custom/dir"
        assert data["report_formats"] == ["console"]


class TestTestbedConfigYAMLRoundtrip:
    """Tests for YAML save and load functionality."""

    def test_yaml_roundtrip_defaults(self):
        """Test saving and loading default config via YAML."""
        original = TestbedConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            original.save_yaml(path)
            loaded = TestbedConfig.from_yaml(path)

        assert loaded.agents == original.agents
        assert loaded.protection_levels == original.protection_levels
        assert loaded.attack_suites == original.attack_suites
        assert loaded.max_attacks_per_suite == original.max_attacks_per_suite
        assert loaded.llm_model == original.llm_model
        assert loaded.llm_api_base == original.llm_api_base
        assert loaded.parallel == original.parallel
        assert loaded.max_workers == original.max_workers
        assert loaded.output_dir == original.output_dir
        assert loaded.report_formats == original.report_formats

    def test_yaml_roundtrip_with_agents(self):
        """Test saving and loading config with agents via YAML."""
        original = TestbedConfig(
            agents=[
                AgentConfig(
                    complexity=AgentComplexity.SIMPLE,
                    domain=AgentDomain.CUSTOMER_SERVICE,
                ),
                AgentConfig(
                    complexity=AgentComplexity.RAG,
                    domain=AgentDomain.GENERAL,
                    name="my-rag-agent",
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            original.save_yaml(path)
            loaded = TestbedConfig.from_yaml(path)

        assert len(loaded.agents) == 2
        assert loaded.agents[0].complexity == AgentComplexity.SIMPLE
        assert loaded.agents[0].domain == AgentDomain.CUSTOMER_SERVICE
        assert loaded.agents[0].name == "simple-cs"
        assert loaded.agents[1].complexity == AgentComplexity.RAG
        assert loaded.agents[1].domain == AgentDomain.GENERAL
        assert loaded.agents[1].name == "my-rag-agent"

    def test_yaml_roundtrip_full_config(self):
        """Test saving and loading a full config via YAML."""
        original = TestbedConfig(
            agents=[
                AgentConfig(
                    complexity=AgentComplexity.MULTI_AGENT,
                    domain=AgentDomain.KNOWLEDGE_BASE,
                ),
            ],
            protection_levels=[ProtectionLevel.PARTIAL, ProtectionLevel.FULL],
            attack_suites=["injection", "bypass", "mcp"],
            max_attacks_per_suite=75,
            llm_model="test/model-v2",
            llm_api_base="https://test.example.com/v1",
            parallel=False,
            max_workers=12,
            output_dir="/tmp/test_reports",
            report_formats=["json", "html"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "full_config.yaml")
            original.save_yaml(path)
            loaded = TestbedConfig.from_yaml(path)

        assert len(loaded.agents) == 1
        assert loaded.agents[0].complexity == AgentComplexity.MULTI_AGENT
        assert loaded.agents[0].domain == AgentDomain.KNOWLEDGE_BASE
        assert loaded.agents[0].name == "multi-kb"
        assert loaded.protection_levels == [ProtectionLevel.PARTIAL, ProtectionLevel.FULL]
        assert loaded.attack_suites == ["injection", "bypass", "mcp"]
        assert loaded.max_attacks_per_suite == 75
        assert loaded.llm_model == "test/model-v2"
        assert loaded.llm_api_base == "https://test.example.com/v1"
        assert loaded.parallel is False
        assert loaded.max_workers == 12
        assert loaded.output_dir == "/tmp/test_reports"
        assert loaded.report_formats == ["json", "html"]

    def test_yaml_creates_parent_directories(self):
        """Test that save_yaml creates parent directories if they don't exist."""
        config = TestbedConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "dir", "config.yaml")
            config.save_yaml(path)
            assert os.path.exists(path)

    def test_from_dict_to_dict_roundtrip(self):
        """Test that from_dict and to_dict are inverse operations."""
        original_data = {
            "agents": [
                {"complexity": "tools", "domain": "code", "name": "test-agent"},
            ],
            "protection_levels": ["none", "full"],
            "attack_suites": ["injection", "jailbreak", "bypass"],
            "max_attacks_per_suite": 100,
            "llm_model": "test/model",
            "llm_api_base": "https://api.test.com",
            "parallel": True,
            "max_workers": 7,
            "output_dir": "./output",
            "report_formats": ["console", "json"],
        }

        config = TestbedConfig.from_dict(original_data)
        result_data = config.to_dict()

        assert result_data == original_data
