"""
Tests for the testbed compatibility module (AgentEvalAdapter).

This module tests the compatibility layer for importing Amazon Agent Evaluation
configurations and converting them to dspyGuardrails testbed format.
"""

import os
import tempfile

import pytest
import yaml

from dspy_guardrails.testbed import (
    AgentComplexity,
    AgentConfig,
    AgentDomain,
    ProtectionLevel,
    TestbedConfig,
)
from dspy_guardrails.testbed.compat import (
    AgentEvalAdapter,
    AgentEvalConfig,
    AgentEvalTest,
)


class TestAgentEvalTest:
    """Tests for AgentEvalTest dataclass."""

    def test_create_minimal(self):
        """Test creating a minimal test case."""
        test = AgentEvalTest(
            name="test1",
            steps=["Hello"],
            expected_results=["Hi there!"],
        )
        assert test.name == "test1"
        assert test.steps == ["Hello"]
        assert test.expected_results == ["Hi there!"]
        assert test.initial_prompt is None
        assert test.max_turns == 2
        assert test.hook is None

    def test_create_with_all_fields(self):
        """Test creating a test case with all fields."""
        test = AgentEvalTest(
            name="complete_test",
            steps=["Step 1", "Step 2"],
            expected_results=["Result 1", "Result 2"],
            initial_prompt="You are a helpful assistant.",
            max_turns=5,
            hook="custom_hook",
        )
        assert test.name == "complete_test"
        assert test.steps == ["Step 1", "Step 2"]
        assert test.expected_results == ["Result 1", "Result 2"]
        assert test.initial_prompt == "You are a helpful assistant."
        assert test.max_turns == 5
        assert test.hook == "custom_hook"


class TestAgentEvalConfig:
    """Tests for AgentEvalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AgentEvalConfig()
        assert config.evaluator_model == "claude-3"
        assert config.target_type == "custom"
        assert config.target_module is None
        assert config.tests == []
        assert config.raw_data == {}

    def test_from_dict_empty(self):
        """Test parsing an empty dictionary."""
        config = AgentEvalConfig.from_dict({})
        assert config.evaluator_model == "claude-3"
        assert config.target_type == "custom"
        assert config.tests == []

    def test_from_dict_evaluator_section(self):
        """Test parsing evaluator section."""
        data = {
            "evaluator": {
                "model": "gpt-4",
            }
        }
        config = AgentEvalConfig.from_dict(data)
        assert config.evaluator_model == "gpt-4"

    def test_from_dict_target_section(self):
        """Test parsing target section."""
        data = {
            "target": {
                "type": "bedrock-agent",
                "module": "my_agent.main",
            }
        }
        config = AgentEvalConfig.from_dict(data)
        assert config.target_type == "bedrock-agent"
        assert config.target_module == "my_agent.main"

    def test_from_dict_tests_section(self):
        """Test parsing tests section."""
        data = {
            "tests": {
                "greeting_test": {
                    "steps": ["Hello", "How are you?"],
                    "expected_results": ["Hi!", "I'm fine."],
                    "max_turns": 3,
                },
                "farewell_test": {
                    "steps": ["Goodbye"],
                    "expected_results": ["Bye!"],
                    "initial_prompt": "Be brief",
                    "hook": "farewell_hook",
                },
            }
        }
        config = AgentEvalConfig.from_dict(data)

        assert len(config.tests) == 2

        # Find tests by name (order may vary due to dict iteration)
        tests_by_name = {t.name: t for t in config.tests}

        greeting = tests_by_name["greeting_test"]
        assert greeting.steps == ["Hello", "How are you?"]
        assert greeting.expected_results == ["Hi!", "I'm fine."]
        assert greeting.max_turns == 3
        assert greeting.initial_prompt is None
        assert greeting.hook is None

        farewell = tests_by_name["farewell_test"]
        assert farewell.steps == ["Goodbye"]
        assert farewell.expected_results == ["Bye!"]
        assert farewell.initial_prompt == "Be brief"
        assert farewell.hook == "farewell_hook"

    def test_from_dict_preserves_raw_data(self):
        """Test that raw data is preserved."""
        data = {
            "evaluator": {"model": "claude-3"},
            "target": {"type": "bedrock-agent"},
            "tests": {"test1": {"steps": ["hi"], "expected_results": ["hello"]}},
            "custom_field": "custom_value",
        }
        config = AgentEvalConfig.from_dict(data)
        assert config.raw_data == data
        assert config.raw_data["custom_field"] == "custom_value"


class TestAgentEvalAdapterImportConfig:
    """Tests for AgentEvalAdapter.import_config() method."""

    def test_import_minimal_config(self):
        """Test importing a minimal configuration file."""
        yaml_content = """
evaluator:
  model: claude-3
target:
  type: bedrock-agent
tests:
  basic_test:
    steps:
      - "Hello"
    expected_results:
      - "Hi there"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                adapter = AgentEvalAdapter.import_config(f.name)
                assert adapter.agent_eval_config.evaluator_model == "claude-3"
                assert adapter.agent_eval_config.target_type == "bedrock-agent"
                assert len(adapter.agent_eval_config.tests) == 1
                assert adapter.agent_eval_config.tests[0].name == "basic_test"
            finally:
                os.unlink(f.name)

    def test_import_complete_config(self):
        """Test importing a complete configuration file."""
        yaml_content = """
evaluator:
  model: gpt-4-turbo
target:
  type: custom
  module: my_agent.main
tests:
  flight_status:
    steps:
      - "What is the status of flight AA123?"
      - "Is there a delay?"
    expected_results:
      - "Flight AA123 is on time."
      - "No delays reported."
    max_turns: 3
  booking:
    steps:
      - "I want to book a flight to NYC"
    expected_results:
      - "I can help you book a flight to NYC."
    initial_prompt: "You are a flight booking assistant."
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                adapter = AgentEvalAdapter.import_config(f.name)
                config = adapter.agent_eval_config

                assert config.evaluator_model == "gpt-4-turbo"
                assert config.target_type == "custom"
                assert config.target_module == "my_agent.main"
                assert len(config.tests) == 2

                tests_by_name = {t.name: t for t in config.tests}
                assert "flight_status" in tests_by_name
                assert "booking" in tests_by_name
                assert tests_by_name["flight_status"].max_turns == 3
                assert tests_by_name["booking"].initial_prompt == "You are a flight booking assistant."
            finally:
                os.unlink(f.name)

    def test_import_nonexistent_file(self):
        """Test that importing a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AgentEvalAdapter.import_config("/nonexistent/path/config.yaml")

    def test_import_empty_file(self):
        """Test importing an empty configuration file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            try:
                adapter = AgentEvalAdapter.import_config(f.name)
                # Should use defaults
                assert adapter.agent_eval_config.evaluator_model == "claude-3"
                assert adapter.agent_eval_config.target_type == "custom"
                assert adapter.agent_eval_config.tests == []
            finally:
                os.unlink(f.name)


class TestAgentEvalAdapterAddSecuritySuite:
    """Tests for AgentEvalAdapter.add_security_suite() method."""

    def test_add_single_suite(self):
        """Test adding a single security suite."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        result = adapter.add_security_suite("injection")

        # Check method chaining
        assert result is adapter
        # Check suite was added
        assert "injection" in adapter._security_suites

    def test_add_multiple_suites(self):
        """Test adding multiple security suites."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.add_security_suite("injection")
        adapter.add_security_suite("jailbreak")
        adapter.add_security_suite("bypass")

        assert adapter._security_suites == ["injection", "jailbreak", "bypass"]

    def test_add_duplicate_suite_deduplicated(self):
        """Test that duplicate suites are deduplicated."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.add_security_suite("injection")
        adapter.add_security_suite("jailbreak")
        adapter.add_security_suite("injection")  # Duplicate
        adapter.add_security_suite("jailbreak")  # Duplicate

        assert adapter._security_suites == ["injection", "jailbreak"]

    def test_method_chaining(self):
        """Test fluent interface for adding suites."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        result = (
            adapter.add_security_suite("injection")
            .add_security_suite("jailbreak")
            .add_security_suite("bypass")
        )

        assert result is adapter
        assert len(adapter._security_suites) == 3


class TestAgentEvalAdapterAddAgent:
    """Tests for AgentEvalAdapter.add_agent() method."""

    def test_add_single_agent(self):
        """Test adding a single agent."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        result = adapter.add_agent(
            AgentComplexity.SIMPLE, AgentDomain.CUSTOMER_SERVICE
        )

        # Check method chaining
        assert result is adapter
        # Check agent was added
        assert len(adapter._agents) == 1
        assert adapter._agents[0].complexity == AgentComplexity.SIMPLE
        assert adapter._agents[0].domain == AgentDomain.CUSTOMER_SERVICE
        assert adapter._agents[0].name == "simple-cs"

    def test_add_agent_with_custom_name(self):
        """Test adding an agent with a custom name."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.add_agent(
            AgentComplexity.TOOLS,
            AgentDomain.CODE_ASSISTANT,
            name="my-code-bot",
        )

        assert len(adapter._agents) == 1
        assert adapter._agents[0].name == "my-code-bot"

    def test_add_multiple_agents(self):
        """Test adding multiple agents."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.add_agent(AgentComplexity.SIMPLE, AgentDomain.CUSTOMER_SERVICE)
        adapter.add_agent(AgentComplexity.TOOLS, AgentDomain.CODE_ASSISTANT)
        adapter.add_agent(AgentComplexity.RAG, AgentDomain.KNOWLEDGE_BASE)

        assert len(adapter._agents) == 3
        assert adapter._agents[0].complexity == AgentComplexity.SIMPLE
        assert adapter._agents[1].complexity == AgentComplexity.TOOLS
        assert adapter._agents[2].complexity == AgentComplexity.RAG

    def test_method_chaining(self):
        """Test fluent interface for adding agents."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        result = (
            adapter.add_agent(AgentComplexity.SIMPLE, AgentDomain.GENERAL)
            .add_agent(AgentComplexity.MULTI_AGENT, AgentDomain.CUSTOMER_SERVICE)
        )

        assert result is adapter
        assert len(adapter._agents) == 2


class TestAgentEvalAdapterSetProtectionLevels:
    """Tests for AgentEvalAdapter.set_protection_levels() method."""

    def test_default_protection_levels(self):
        """Test that default protection levels include all three."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        assert adapter._protection_levels == [
            ProtectionLevel.NONE,
            ProtectionLevel.PARTIAL,
            ProtectionLevel.FULL,
        ]

    def test_set_single_level(self):
        """Test setting a single protection level."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        result = adapter.set_protection_levels([ProtectionLevel.FULL])

        assert result is adapter
        assert adapter._protection_levels == [ProtectionLevel.FULL]

    def test_set_multiple_levels(self):
        """Test setting multiple protection levels."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.set_protection_levels(
            [ProtectionLevel.NONE, ProtectionLevel.FULL]
        )

        assert adapter._protection_levels == [
            ProtectionLevel.NONE,
            ProtectionLevel.FULL,
        ]

    def test_set_empty_levels(self):
        """Test setting empty protection levels."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.set_protection_levels([])

        assert adapter._protection_levels == []


class TestAgentEvalAdapterToTestbedConfig:
    """Tests for AgentEvalAdapter.to_testbed_config() method."""

    def test_creates_testbed_config(self):
        """Test that to_testbed_config returns a TestbedConfig."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        config = adapter.to_testbed_config()

        assert isinstance(config, TestbedConfig)

    def test_uses_default_agents_when_none_specified(self):
        """Test that default agent is used when none are added."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        config = adapter.to_testbed_config()

        assert len(config.agents) == 1
        assert config.agents[0].complexity == AgentComplexity.SIMPLE
        assert config.agents[0].domain == AgentDomain.CUSTOMER_SERVICE

    def test_uses_default_suites_when_none_specified(self):
        """Test that default suites are used when none are added."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        config = adapter.to_testbed_config()

        assert config.attack_suites == ["injection", "jailbreak"]

    def test_uses_configured_agents(self):
        """Test that configured agents are used."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.add_agent(AgentComplexity.RAG, AgentDomain.KNOWLEDGE_BASE)
        adapter.add_agent(AgentComplexity.TOOLS, AgentDomain.CODE_ASSISTANT)

        config = adapter.to_testbed_config()

        assert len(config.agents) == 2
        assert config.agents[0].complexity == AgentComplexity.RAG
        assert config.agents[1].complexity == AgentComplexity.TOOLS

    def test_uses_configured_suites(self):
        """Test that configured suites are used."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.add_security_suite("bypass")
        adapter.add_security_suite("mcp")

        config = adapter.to_testbed_config()

        assert config.attack_suites == ["bypass", "mcp"]

    def test_uses_configured_protection_levels(self):
        """Test that configured protection levels are used."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.set_protection_levels([ProtectionLevel.PARTIAL])

        config = adapter.to_testbed_config()

        assert config.protection_levels == [ProtectionLevel.PARTIAL]

    def test_max_attacks_per_suite_parameter(self):
        """Test that max_attacks_per_suite is set correctly."""
        adapter = AgentEvalAdapter(AgentEvalConfig())

        # Default
        config = adapter.to_testbed_config()
        assert config.max_attacks_per_suite == 50

        # Custom value
        config = adapter.to_testbed_config(max_attacks_per_suite=100)
        assert config.max_attacks_per_suite == 100

    def test_llm_config_is_set(self):
        """Test that LLM configuration is set."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.set_llm_config(
            model="custom/model",
            api_base="https://api.example.com",
        )

        config = adapter.to_testbed_config()

        assert config.llm_model == "custom/model"
        assert config.llm_api_base == "https://api.example.com"

    def test_output_dir_is_set(self):
        """Test that output directory is set."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        adapter.set_output_dir("/custom/output")

        config = adapter.to_testbed_config()

        assert config.output_dir == "/custom/output"


class TestAgentEvalAdapterGetFunctionalTests:
    """Tests for AgentEvalAdapter.get_functional_tests() method."""

    def test_returns_empty_list_when_no_tests(self):
        """Test that empty list is returned when no tests defined."""
        adapter = AgentEvalAdapter(AgentEvalConfig())
        tests = adapter.get_functional_tests()

        assert tests == []

    def test_returns_original_tests(self):
        """Test that original tests are returned."""
        test1 = AgentEvalTest(
            name="test1",
            steps=["Hello"],
            expected_results=["Hi"],
        )
        test2 = AgentEvalTest(
            name="test2",
            steps=["Bye"],
            expected_results=["Goodbye"],
        )
        config = AgentEvalConfig(tests=[test1, test2])
        adapter = AgentEvalAdapter(config)

        tests = adapter.get_functional_tests()

        assert len(tests) == 2
        assert tests[0].name == "test1"
        assert tests[1].name == "test2"

    def test_returns_tests_from_imported_config(self):
        """Test that tests are returned from an imported configuration."""
        yaml_content = """
evaluator:
  model: claude-3
target:
  type: bedrock-agent
tests:
  flight_test:
    steps:
      - "What is my flight status?"
    expected_results:
      - "Your flight is on time."
  booking_test:
    steps:
      - "Book a flight"
    expected_results:
      - "I can help with that."
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            try:
                adapter = AgentEvalAdapter.import_config(f.name)
                tests = adapter.get_functional_tests()

                assert len(tests) == 2
                test_names = {t.name for t in tests}
                assert "flight_test" in test_names
                assert "booking_test" in test_names
            finally:
                os.unlink(f.name)


class TestAgentEvalAdapterExportCombinedConfig:
    """Tests for AgentEvalAdapter.export_combined_config() method."""

    def test_export_preserves_original_tests(self):
        """Test that original tests are preserved in exported config."""
        yaml_content = """
evaluator:
  model: claude-3
target:
  type: bedrock-agent
tests:
  my_test:
    steps:
      - "Hello"
    expected_results:
      - "Hi"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            input_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            output_path = f.name

        try:
            adapter = AgentEvalAdapter.import_config(input_path)
            adapter.add_security_suite("injection")
            adapter.export_combined_config(output_path)

            with open(output_path, "r") as f:
                exported = yaml.safe_load(f)

            # Original structure preserved
            assert "evaluator" in exported
            assert exported["evaluator"]["model"] == "claude-3"
            assert "target" in exported
            assert exported["target"]["type"] == "bedrock-agent"
            assert "tests" in exported
            assert "my_test" in exported["tests"]
            assert exported["tests"]["my_test"]["steps"] == ["Hello"]
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_export_adds_security_section(self):
        """Test that security section is added to exported config."""
        yaml_content = """
evaluator:
  model: claude-3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            input_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            output_path = f.name

        try:
            adapter = AgentEvalAdapter.import_config(input_path)
            adapter.add_security_suite("injection")
            adapter.add_security_suite("jailbreak")
            adapter.add_agent(AgentComplexity.TOOLS, AgentDomain.CODE_ASSISTANT)
            adapter.set_protection_levels([ProtectionLevel.FULL])
            adapter.export_combined_config(output_path)

            with open(output_path, "r") as f:
                exported = yaml.safe_load(f)

            # Security section present
            assert "security" in exported
            security = exported["security"]

            # Agents
            assert len(security["agents"]) == 1
            assert security["agents"][0]["complexity"] == "tools"
            assert security["agents"][0]["domain"] == "code"

            # Protection levels
            assert security["protection_levels"] == ["full"]

            # Attack suites
            assert security["attack_suites"] == ["injection", "jailbreak"]
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_export_creates_parent_directories(self):
        """Test that export creates parent directories if needed."""
        adapter = AgentEvalAdapter(AgentEvalConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "dir", "config.yaml")
            adapter.export_combined_config(output_path)

            assert os.path.exists(output_path)

    def test_export_preserves_custom_fields(self):
        """Test that custom fields are preserved in exported config."""
        yaml_content = """
evaluator:
  model: claude-3
custom_field: custom_value
nested:
  field: value
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            input_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            output_path = f.name

        try:
            adapter = AgentEvalAdapter.import_config(input_path)
            adapter.export_combined_config(output_path)

            with open(output_path, "r") as f:
                exported = yaml.safe_load(f)

            # Custom fields preserved
            assert exported["custom_field"] == "custom_value"
            assert exported["nested"]["field"] == "value"
        finally:
            os.unlink(input_path)
            os.unlink(output_path)


class TestAgentEvalAdapterIntegration:
    """Integration tests for AgentEvalAdapter."""

    def test_full_workflow(self):
        """Test a complete workflow from import to export."""
        yaml_content = """
evaluator:
  model: claude-3
target:
  type: bedrock-agent
  module: airline.agent
tests:
  flight_status:
    steps:
      - "What is the status of AA123?"
      - "Will it be delayed?"
    expected_results:
      - "Flight AA123 is scheduled on time."
      - "No delays expected."
    max_turns: 3
  booking:
    steps:
      - "Book a flight to NYC"
    expected_results:
      - "I can help you book that flight."
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            input_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            output_path = f.name

        try:
            # Import
            adapter = AgentEvalAdapter.import_config(input_path)

            # Configure security testing
            adapter.add_security_suite("injection")
            adapter.add_security_suite("jailbreak")
            adapter.add_security_suite("bypass")
            adapter.add_agent(
                AgentComplexity.MULTI_AGENT,
                AgentDomain.CUSTOMER_SERVICE,
                name="airline-cs",
            )
            adapter.set_protection_levels(
                [ProtectionLevel.NONE, ProtectionLevel.FULL]
            )
            adapter.set_llm_config(
                model="openai/gpt-4",
                api_base="https://api.openai.com/v1",
            )
            adapter.set_output_dir("./security_reports")

            # Get testbed config
            testbed_config = adapter.to_testbed_config(max_attacks_per_suite=75)
            assert isinstance(testbed_config, TestbedConfig)
            assert len(testbed_config.agents) == 1
            assert testbed_config.agents[0].name == "airline-cs"
            assert testbed_config.attack_suites == ["injection", "jailbreak", "bypass"]
            assert testbed_config.protection_levels == [
                ProtectionLevel.NONE,
                ProtectionLevel.FULL,
            ]
            assert testbed_config.max_attacks_per_suite == 75
            assert testbed_config.llm_model == "openai/gpt-4"
            assert testbed_config.output_dir == "./security_reports"

            # Get functional tests
            functional_tests = adapter.get_functional_tests()
            assert len(functional_tests) == 2
            test_names = {t.name for t in functional_tests}
            assert "flight_status" in test_names
            assert "booking" in test_names

            # Export combined config
            adapter.export_combined_config(output_path)

            with open(output_path, "r") as f:
                exported = yaml.safe_load(f)

            # Verify combined config
            assert exported["evaluator"]["model"] == "claude-3"
            assert exported["target"]["type"] == "bedrock-agent"
            assert "tests" in exported
            assert "security" in exported
            assert exported["security"]["attack_suites"] == [
                "injection",
                "jailbreak",
                "bypass",
            ]
        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_fluent_interface(self):
        """Test fluent interface chaining."""
        adapter = AgentEvalAdapter(AgentEvalConfig())

        result = (
            adapter.add_security_suite("injection")
            .add_security_suite("jailbreak")
            .add_agent(AgentComplexity.SIMPLE, AgentDomain.GENERAL)
            .add_agent(AgentComplexity.RAG, AgentDomain.KNOWLEDGE_BASE)
            .set_protection_levels([ProtectionLevel.FULL])
            .set_llm_config(model="test/model")
            .set_output_dir("./test_output")
        )

        assert result is adapter
        assert len(adapter._security_suites) == 2
        assert len(adapter._agents) == 2
        assert adapter._protection_levels == [ProtectionLevel.FULL]
        assert adapter._llm_model == "test/model"
        assert adapter._output_dir == "./test_output"
