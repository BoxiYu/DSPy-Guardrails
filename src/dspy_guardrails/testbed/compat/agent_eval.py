"""
Adapter for Amazon Agent Evaluation (agent-evaluation) configuration format.

This module provides compatibility with Genesys' Amazon Agent Evaluation framework,
allowing users to import existing agent-eval YAML configurations and extend them
with dspyGuardrails security testing capabilities.

Example agent-eval YAML format:
```yaml
evaluator:
  model: claude-3
target:
  type: bedrock-agent
tests:
  test_name:
    steps: ["user message 1", "user message 2"]
    expected_results: ["expected 1", "expected 2"]
    max_turns: 2
```
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dspy_guardrails.testbed.config import (
    AgentComplexity,
    AgentConfig,
    AgentDomain,
    ProtectionLevel,
    TestbedConfig,
)


@dataclass
class AgentEvalTest:
    """Represents a single test case from agent-eval configuration.

    Attributes:
        name: Name of the test case.
        steps: List of user messages to send in sequence.
        expected_results: List of expected responses for each step.
        initial_prompt: Optional initial system prompt.
        max_turns: Maximum number of conversation turns.
        hook: Optional hook identifier for test customization.
    """

    name: str
    steps: list[str]
    expected_results: list[str]
    initial_prompt: str | None = None
    max_turns: int = 2
    hook: str | None = None


@dataclass
class AgentEvalConfig:
    """Configuration parsed from agent-eval YAML format.

    Attributes:
        evaluator_model: Model used for evaluation (e.g., "claude-3").
        target_type: Type of target being evaluated (e.g., "bedrock-agent", "custom").
        target_module: Optional module path for custom targets.
        tests: List of test cases parsed from the configuration.
        raw_data: Raw dictionary data from the YAML file for preservation.
    """

    evaluator_model: str = "claude-3"
    target_type: str = "custom"
    target_module: str | None = None
    tests: list[AgentEvalTest] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvalConfig":
        """Create AgentEvalConfig from a dictionary.

        Args:
            data: Dictionary containing agent-eval configuration.

        Returns:
            AgentEvalConfig instance.
        """
        # Parse evaluator section
        evaluator = data.get("evaluator", {})
        evaluator_model = evaluator.get("model", "claude-3")

        # Parse target section
        target = data.get("target", {})
        target_type = target.get("type", "custom")
        target_module = target.get("module")

        # Parse tests section
        tests_data = data.get("tests", {})
        tests = []
        for test_name, test_config in tests_data.items():
            if isinstance(test_config, dict):
                test = AgentEvalTest(
                    name=test_name,
                    steps=test_config.get("steps", []),
                    expected_results=test_config.get("expected_results", []),
                    initial_prompt=test_config.get("initial_prompt"),
                    max_turns=test_config.get("max_turns", 2),
                    hook=test_config.get("hook"),
                )
                tests.append(test)

        return cls(
            evaluator_model=evaluator_model,
            target_type=target_type,
            target_module=target_module,
            tests=tests,
            raw_data=data,
        )


class AgentEvalAdapter:
    """Adapter for converting agent-eval configurations to dspyGuardrails testbed format.

    This adapter allows users to:
    1. Import existing agent-eval YAML configurations
    2. Add security testing suites
    3. Configure agents and protection levels
    4. Generate TestbedConfig for security evaluation
    5. Export combined configurations that preserve original tests

    Example usage:
    ```python
    adapter = AgentEvalAdapter.import_config("agent_eval.yaml")
    adapter.add_security_suite("injection")
    adapter.add_security_suite("jailbreak")
    adapter.add_agent(AgentComplexity.SIMPLE, AgentDomain.CUSTOMER_SERVICE)

    testbed_config = adapter.to_testbed_config()
    functional_tests = adapter.get_functional_tests()
    ```
    """

    def __init__(self, agent_eval_config: AgentEvalConfig):
        """Initialize the adapter with an agent-eval configuration.

        Args:
            agent_eval_config: Parsed agent-eval configuration.
        """
        self.agent_eval_config = agent_eval_config
        self._security_suites: list[str] = []
        self._agents: list[AgentConfig] = []
        self._protection_levels: list[ProtectionLevel] = [
            ProtectionLevel.NONE,
            ProtectionLevel.PARTIAL,
            ProtectionLevel.FULL,
        ]
        self._llm_model: str | None = None
        self._llm_api_base: str | None = None
        self._output_dir: str = "./testbed_reports"

    @classmethod
    def import_config(cls, path: str) -> "AgentEvalAdapter":
        """Import an agent-eval configuration from a YAML file.

        Args:
            path: Path to the agent-eval YAML configuration file.

        Returns:
            AgentEvalAdapter instance configured from the file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = AgentEvalConfig.from_dict(data)
        return cls(config)

    def add_security_suite(self, suite: str) -> "AgentEvalAdapter":
        """Add a security test suite to run.

        Suites are deduplicated - adding the same suite multiple times
        has no effect after the first addition.

        Args:
            suite: Name of the security suite (e.g., "injection", "jailbreak", "bypass").

        Returns:
            Self for method chaining.
        """
        if suite not in self._security_suites:
            self._security_suites.append(suite)
        return self

    def add_agent(
        self,
        complexity: AgentComplexity,
        domain: AgentDomain,
        name: str | None = None,
    ) -> "AgentEvalAdapter":
        """Add an agent configuration to test.

        Args:
            complexity: Complexity level of the agent.
            domain: Domain category of the agent.
            name: Optional custom name for the agent.

        Returns:
            Self for method chaining.
        """
        agent = AgentConfig(complexity=complexity, domain=domain, name=name)
        self._agents.append(agent)
        return self

    def set_protection_levels(
        self, levels: list[ProtectionLevel]
    ) -> "AgentEvalAdapter":
        """Set the protection levels to evaluate.

        Args:
            levels: List of protection levels to test.

        Returns:
            Self for method chaining.
        """
        self._protection_levels = levels
        return self

    def set_llm_config(
        self,
        model: str | None = None,
        api_base: str | None = None,
    ) -> "AgentEvalAdapter":
        """Set the LLM configuration for testbed evaluation.

        Args:
            model: LLM model identifier.
            api_base: API base URL for the LLM.

        Returns:
            Self for method chaining.
        """
        if model is not None:
            self._llm_model = model
        if api_base is not None:
            self._llm_api_base = api_base
        return self

    def set_output_dir(self, output_dir: str) -> "AgentEvalAdapter":
        """Set the output directory for reports.

        Args:
            output_dir: Directory path for test reports.

        Returns:
            Self for method chaining.
        """
        self._output_dir = output_dir
        return self

    def to_testbed_config(
        self, max_attacks_per_suite: int = 50
    ) -> TestbedConfig:
        """Convert to a TestbedConfig for security evaluation.

        If no agents or security suites have been explicitly configured,
        default values are used:
        - Agents: One simple customer service agent
        - Suites: ["injection", "jailbreak"]

        Args:
            max_attacks_per_suite: Maximum number of attacks per suite.

        Returns:
            TestbedConfig configured for security testing.
        """
        # Use defaults if no agents specified
        agents = self._agents
        if not agents:
            agents = [
                AgentConfig(
                    complexity=AgentComplexity.SIMPLE,
                    domain=AgentDomain.CUSTOMER_SERVICE,
                )
            ]

        # Use defaults if no security suites specified
        attack_suites = self._security_suites
        if not attack_suites:
            attack_suites = ["injection", "jailbreak"]

        return TestbedConfig(
            agents=agents,
            protection_levels=self._protection_levels,
            attack_suites=attack_suites,
            max_attacks_per_suite=max_attacks_per_suite,
            llm_model=self._llm_model or "openai/kimi-k2-0905-preview",
            llm_api_base=self._llm_api_base,
            output_dir=self._output_dir,
        )

    def get_functional_tests(self) -> list[AgentEvalTest]:
        """Get the original functional tests from the agent-eval configuration.

        Returns:
            List of AgentEvalTest objects representing the original test cases.
        """
        return self.agent_eval_config.tests

    def export_combined_config(self, path: str) -> None:
        """Export a combined configuration that merges original tests with security extensions.

        The exported configuration preserves all original agent-eval fields
        and adds a 'security' section with dspyGuardrails testbed settings.

        Args:
            path: Path to save the combined YAML configuration.
        """
        # Start with original raw data to preserve all fields
        combined = dict(self.agent_eval_config.raw_data)

        # Add security section with testbed configuration
        testbed_config = self.to_testbed_config()
        combined["security"] = {
            "agents": [
                {
                    "complexity": agent.complexity.value,
                    "domain": agent.domain.value,
                    "name": agent.name,
                }
                for agent in testbed_config.agents
            ],
            "protection_levels": [
                level.value for level in testbed_config.protection_levels
            ],
            "attack_suites": testbed_config.attack_suites,
            "max_attacks_per_suite": testbed_config.max_attacks_per_suite,
            "llm_model": testbed_config.llm_model,
            "llm_api_base": testbed_config.llm_api_base,
            "output_dir": testbed_config.output_dir,
        }

        # Create parent directories if needed
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(combined, f, default_flow_style=False, sort_keys=False)
