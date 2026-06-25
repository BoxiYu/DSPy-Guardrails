"""
Configuration classes for the dspyGuardrails testbed.

This module provides configuration dataclasses and enums for setting up
agent testbeds to evaluate guardrails capabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class AgentComplexity(Enum):
    """Agent complexity levels for testbed evaluation."""

    SIMPLE = "simple"
    TOOLS = "tools"
    MULTI_AGENT = "multi"
    RAG = "rag"


class AgentDomain(Enum):
    """Agent domain categories for testbed evaluation."""

    CUSTOMER_SERVICE = "cs"
    CODE_ASSISTANT = "code"
    KNOWLEDGE_BASE = "kb"
    GENERAL = "general"


class ProtectionLevel(Enum):
    """Protection levels for guardrail evaluation."""

    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


@dataclass
class AgentConfig:
    """Configuration for a single agent in the testbed.

    Attributes:
        complexity: The complexity level of the agent.
        domain: The domain category of the agent.
        name: Optional name for the agent. If not provided, auto-generates
              as "{complexity.value}-{domain.value}".
    """

    complexity: AgentComplexity
    domain: AgentDomain
    name: str | None = None

    def __post_init__(self):
        """Auto-generate name if not provided."""
        if self.name is None:
            self.name = f"{self.complexity.value}-{self.domain.value}"


@dataclass
class TestbedConfig:
    """Configuration for the guardrails testbed.

    Attributes:
        agents: List of agent configurations to test.
        protection_levels: Protection levels to evaluate (default: all three).
        attack_suites: Attack suites to run (default: ["injection", "jailbreak"]).
        max_attacks_per_suite: Optional limit on attacks per suite.
        llm_model: LLM model to use (default: "openai/kimi-k2-0905-preview").
        llm_api_base: Optional API base URL for the LLM.
        parallel: Whether to run tests in parallel (default: True).
        max_workers: Maximum number of parallel workers (default: 5).
        output_dir: Directory for output reports (default: "./testbed_reports").
        report_formats: Report formats to generate (default: ["console", "json", "html"]).
    """

    agents: list[AgentConfig] = field(default_factory=list)
    protection_levels: list[ProtectionLevel] = field(
        default_factory=lambda: [ProtectionLevel.NONE, ProtectionLevel.PARTIAL, ProtectionLevel.FULL]
    )
    attack_suites: list[str] = field(default_factory=lambda: ["injection", "jailbreak"])
    max_attacks_per_suite: int | None = None
    llm_model: str = "openai/kimi-k2-0905-preview"
    llm_api_base: str | None = None
    parallel: bool = True
    max_workers: int = 5
    output_dir: str = "./testbed_reports"
    report_formats: list[str] = field(default_factory=lambda: ["console", "json", "html"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestbedConfig":
        """Create a TestbedConfig from a dictionary.

        Args:
            data: Dictionary containing configuration data.

        Returns:
            TestbedConfig instance.
        """
        # Parse agents
        agents = []
        for agent_data in data.get("agents", []):
            complexity = AgentComplexity(agent_data["complexity"])
            domain = AgentDomain(agent_data["domain"])
            name = agent_data.get("name")
            agents.append(AgentConfig(complexity=complexity, domain=domain, name=name))

        # Parse protection levels
        protection_levels = []
        for level in data.get("protection_levels", ["none", "partial", "full"]):
            protection_levels.append(ProtectionLevel(level))

        return cls(
            agents=agents,
            protection_levels=protection_levels,
            attack_suites=data.get("attack_suites", ["injection", "jailbreak"]),
            max_attacks_per_suite=data.get("max_attacks_per_suite"),
            llm_model=data.get("llm_model", "openai/kimi-k2-0905-preview"),
            llm_api_base=data.get("llm_api_base"),
            parallel=data.get("parallel", True),
            max_workers=data.get("max_workers", 5),
            output_dir=data.get("output_dir", "./testbed_reports"),
            report_formats=data.get("report_formats", ["console", "json", "html"]),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "TestbedConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            TestbedConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "agents": [
                {
                    "complexity": agent.complexity.value,
                    "domain": agent.domain.value,
                    "name": agent.name,
                }
                for agent in self.agents
            ],
            "protection_levels": [level.value for level in self.protection_levels],
            "attack_suites": self.attack_suites,
            "max_attacks_per_suite": self.max_attacks_per_suite,
            "llm_model": self.llm_model,
            "llm_api_base": self.llm_api_base,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "output_dir": self.output_dir,
            "report_formats": self.report_formats,
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML configuration file.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
