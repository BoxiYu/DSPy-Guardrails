"""
Testbed Orchestrator - Orchestration of testbed evaluation runs.

This module provides the TestbedOrchestrator class for running comprehensive
testbed evaluations with progress tracking and result aggregation.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dspy_guardrails.redteam.payloads import get_payloads_by_category
from dspy_guardrails.redteam.payloads.base import AttackPayload, PayloadCategory
from dspy_guardrails.testbed.agents.base import MockAgent
from dspy_guardrails.testbed.agents.factory import MockAgentFactory
from dspy_guardrails.testbed.config import (
    AgentComplexity,
    AgentDomain,
    ProtectionLevel,
    TestbedConfig,
)
from dspy_guardrails.testbed.results import (
    SingleTestResult,
    TestbedResults,
)


@dataclass
class ProgressEvent:
    """
    Progress event for testbed evaluation callbacks.

    Provides detailed progress information during testbed runs,
    allowing UIs or logging systems to track evaluation progress.

    Attributes:
        type: Event type - one of "test_start", "test_complete",
              "attack_result", "all_complete".
        agent_name: Name of the agent being tested.
        protection: Protection level being tested.
        current: Current attack/test number.
        total: Total number of attacks/tests.
        block_rate: Current block rate (for test_complete events).
        payload: Attack payload text (for attack_result events).
        blocked: Whether the attack was blocked (for attack_result events).

    Examples:
        >>> event = ProgressEvent(
        ...     type="test_start",
        ...     agent_name="simple-cs-full",
        ...     protection="full",
        ...     current=1,
        ...     total=10,
        ... )
        >>> event.type
        'test_start'
    """

    type: str  # "test_start", "test_complete", "attack_result", "all_complete"
    agent_name: str = ""
    protection: str = ""
    current: int = 0
    total: int = 0
    block_rate: float = 0.0
    payload: str = ""
    blocked: bool = False


# Map suite names to PayloadCategory
_SUITE_TO_CATEGORY: dict[str, PayloadCategory] = {
    "injection": PayloadCategory.INJECTION,
    "jailbreak": PayloadCategory.JAILBREAK,
    "mcp": PayloadCategory.MCP,
    "bypass": PayloadCategory.BYPASS,
}


class TestbedOrchestrator:
    """
    Orchestrator for running testbed evaluations.

    Manages the complete testbed evaluation workflow:
    1. Creates agents based on configuration
    2. Loads attack payloads
    3. Tests each agent with payloads
    4. Aggregates and finalizes results

    Attributes:
        config: TestbedConfig defining agents, protection levels, and settings.
        factory: MockAgentFactory for creating agents.

    Examples:
        >>> from dspy_guardrails.testbed import TestbedConfig, AgentConfig
        >>> config = TestbedConfig(
        ...     agents=[AgentConfig(AgentComplexity.SIMPLE, AgentDomain.CUSTOMER_SERVICE)],
        ...     protection_levels=[ProtectionLevel.NONE, ProtectionLevel.FULL],
        ...     attack_suites=["injection"],
        ...     max_attacks_per_suite=10,
        ... )
        >>> orchestrator = TestbedOrchestrator(config)
        >>> results = orchestrator.run()
        >>> results.total_tests
        2
    """

    def __init__(
        self,
        config: TestbedConfig,
        llm: Any | None = None,
    ):
        """
        Initialize TestbedOrchestrator.

        Args:
            config: TestbedConfig with agents, protection levels, and settings.
            llm: Optional DSPy LM instance for agent responses.
        """
        self.config = config
        self.factory = MockAgentFactory(llm=llm)

    @classmethod
    def from_config(cls, path: str, llm: Any | None = None) -> "TestbedOrchestrator":
        """
        Create TestbedOrchestrator from a YAML configuration file.

        Args:
            path: Path to the YAML configuration file.
            llm: Optional DSPy LM instance for agent responses.

        Returns:
            TestbedOrchestrator instance configured from the file.

        Examples:
            >>> # Given a config.yaml file:
            >>> orchestrator = TestbedOrchestrator.from_config("config.yaml")
        """
        config = TestbedConfig.from_yaml(path)
        return cls(config, llm=llm)

    def _load_payloads(self) -> list[AttackPayload]:
        """
        Load attack payloads based on configuration.

        Gets payloads from the suites specified in config.attack_suites,
        limited by config.max_attacks_per_suite if set.

        Returns:
            List of AttackPayload objects.

        Examples:
            >>> config = TestbedConfig(
            ...     attack_suites=["injection", "jailbreak"],
            ...     max_attacks_per_suite=5,
            ... )
            >>> orchestrator = TestbedOrchestrator(config)
            >>> payloads = orchestrator._load_payloads()
            >>> # Returns up to 5 injection + 5 jailbreak payloads
        """
        payloads: list[AttackPayload] = []

        for suite in self.config.attack_suites:
            category = _SUITE_TO_CATEGORY.get(suite)
            if category:
                suite_payloads = get_payloads_by_category(category)
            else:
                # Unknown suite - skip
                continue

            # Apply limit per suite
            if self.config.max_attacks_per_suite is not None:
                suite_payloads = suite_payloads[: self.config.max_attacks_per_suite]

            payloads.extend(suite_payloads)

        return payloads

    def _test_agent(
        self,
        agent: MockAgent,
        payloads: list[AttackPayload],
        callback: Callable[[ProgressEvent], None] | None = None,
    ) -> SingleTestResult:
        """
        Test a single agent with the given payloads.

        Runs each payload through the agent and tracks blocked/bypassed counts.
        Resets agent session after each attack.

        Args:
            agent: MockAgent to test.
            payloads: List of AttackPayload objects to test.
            callback: Optional callback for progress events.

        Returns:
            SingleTestResult with attack statistics.

        Examples:
            >>> agent = factory.create(AgentComplexity.SIMPLE,
            ...                        AgentDomain.CUSTOMER_SERVICE,
            ...                        ProtectionLevel.FULL)
            >>> result = orchestrator._test_agent(agent, payloads)
            >>> result.total_attacks
            10
        """
        start_time = time.time()

        # Extract protection level from agent name (e.g., "simple-cs-full" -> "full")
        name_parts = agent.name.rsplit("-", 1)
        protection_level = name_parts[-1] if len(name_parts) > 1 else "unknown"

        result = SingleTestResult(
            agent_name=agent.name,
            complexity=agent.complexity.value,
            domain=agent.domain.value,
            protection_level=protection_level,
        )

        total_latency = 0.0
        by_category: dict[str, dict[str, int]] = {}

        for i, payload in enumerate(payloads):
            # Emit attack start event
            if callback:
                callback(ProgressEvent(
                    type="attack_result",
                    agent_name=agent.name,
                    protection=protection_level,
                    current=i + 1,
                    total=len(payloads),
                    payload=payload.prompt[:100],  # Truncate for display
                ))

            # Run attack
            response = agent.invoke(payload.prompt)
            total_latency += response.latency_ms

            # Track category stats
            category_name = payload.category.value
            if category_name not in by_category:
                by_category[category_name] = {"blocked": 0, "bypassed": 0}

            if response.was_blocked:
                result.blocked_attacks += 1
                by_category[category_name]["blocked"] += 1
            else:
                result.bypassed_attacks += 1
                by_category[category_name]["bypassed"] += 1
                # Track bypassed payload details
                result.bypassed_payloads.append({
                    "id": payload.id,
                    "prompt": payload.prompt[:200],
                    "category": category_name,
                    "technique": payload.technique,
                    "severity": payload.severity.value,
                })

            result.total_attacks += 1

            # Reset agent session for next attack
            agent.reset_session()

        # Calculate averages
        if result.total_attacks > 0:
            result.avg_latency_ms = total_latency / result.total_attacks

        result.by_category = by_category
        result.total_time_seconds = time.time() - start_time

        return result

    def run(
        self,
        progress_callback: Callable[[ProgressEvent], None] | None = None,
    ) -> TestbedResults:
        """
        Run the complete testbed evaluation.

        Process:
        1. Create agents via factory.create_from_config()
        2. Load payloads via _load_payloads()
        3. For each agent, call _test_agent()
        4. Finalize and return results

        Args:
            progress_callback: Optional callback for progress events.
                               Called with ProgressEvent objects throughout.

        Returns:
            TestbedResults with all evaluation results.

        Examples:
            >>> def on_progress(event):
            ...     print(f"{event.type}: {event.agent_name}")
            >>> results = orchestrator.run(progress_callback=on_progress)
            >>> print(f"Score: {results.overall_score}")
        """
        start_time = time.time()
        results = TestbedResults()

        # 1. Create agents
        agents = self.factory.create_from_config(self.config)

        # 2. Load payloads
        payloads = self._load_payloads()

        # 3. Test each agent
        for i, agent in enumerate(agents):
            # Extract protection level from name
            name_parts = agent.name.rsplit("-", 1)
            protection = name_parts[-1] if len(name_parts) > 1 else "unknown"

            # Emit test start event
            if progress_callback:
                progress_callback(ProgressEvent(
                    type="test_start",
                    agent_name=agent.name,
                    protection=protection,
                    current=i + 1,
                    total=len(agents),
                ))

            # Run tests
            result = self._test_agent(agent, payloads, progress_callback)
            results.add_result(result)

            # Emit test complete event
            if progress_callback:
                progress_callback(ProgressEvent(
                    type="test_complete",
                    agent_name=agent.name,
                    protection=protection,
                    current=i + 1,
                    total=len(agents),
                    block_rate=result.block_rate,
                ))

        # 4. Finalize results
        results.execution_time_seconds = time.time() - start_time
        results.finalize()

        # Emit all complete event
        if progress_callback:
            progress_callback(ProgressEvent(
                type="all_complete",
                current=len(agents),
                total=len(agents),
                block_rate=results.overall_score / 100.0,
            ))

        return results

    def run_single(
        self,
        complexity: AgentComplexity,
        domain: AgentDomain,
        protection: ProtectionLevel,
    ) -> SingleTestResult:
        """
        Test a single agent combination.

        Convenience method for testing a specific agent configuration
        without running the full testbed.

        Args:
            complexity: Agent complexity level.
            domain: Agent domain category.
            protection: Protection level to apply.

        Returns:
            SingleTestResult for the tested agent.

        Examples:
            >>> result = orchestrator.run_single(
            ...     AgentComplexity.SIMPLE,
            ...     AgentDomain.CUSTOMER_SERVICE,
            ...     ProtectionLevel.FULL,
            ... )
            >>> print(f"Block rate: {result.block_rate:.2%}")
        """
        # Create single agent
        agent = self.factory.create(
            complexity=complexity,
            domain=domain,
            protection=protection,
        )

        # Load payloads
        payloads = self._load_payloads()

        # Test agent
        result = self._test_agent(agent, payloads)

        return result
