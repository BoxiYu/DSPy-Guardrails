"""
Promptfoo-style Configuration - Declarative YAML Configuration

Provides a promptfoo-inspired configuration system for security testing.

Example YAML:
    description: "Airline agent security test"
    targets:
      - type: http
        url: http://localhost:9000/agents/{id}/chat

    plugins:
      - prompt-injection
      - jailbreak
      - id: mcp
        numTests: 15

    strategies:
      - base64
      - jailbreak

    frameworks:
      - owasp:llm

    parallel: true
    maxWorkers: 5

    cache:
      enabled: true
      ttl: 3600

    output:
      formats: [console, json, html]
      dir: ./reports
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Strategy aliases for promptfoo-style names.
_STRATEGY_ALIAS_MAP = {
    "unicode-confusables": "unicode_confusables",
    "unicode_confusables": "unicode_confusables",
    "zero-width": "zero_width",
    "zero_width": "zero_width",
    "piglatin": "pig_latin",
    "pig-latin": "pig_latin",
    "multi-layer-obfuscation": "multi_layer",
    "multi_layer_obfuscation": "multi_layer",
    "unicode-escape": "unicode_escape",
    "unicode_escape": "unicode_escape",
    "word-splitting": "word_splitting",
    "word_splitting": "word_splitting",
    "jailbreak": "jailbreak_transform",
    "prompt-injection": "jailbreak_transform",
}
_NOOP_STRATEGIES = {"basic", "default"}
_SUPPORTED_SEVERITIES = {"low", "medium", "high", "critical"}
_SUPPORTED_OUTPUT_FORMATS = {"console", "json", "html", "junit", "sarif"}


def normalize_strategy_id(strategy_id: str) -> str:
    """Normalize promptfoo-style strategy ID to local strategy registry name."""
    normalized = strategy_id.strip().lower()
    return _STRATEGY_ALIAS_MAP.get(normalized, normalized)


def get_supported_strategy_ids() -> set[str]:
    """Get all accepted strategy IDs (including aliases)."""
    from ..redteam.strategies import list_strategies

    supported = set(list_strategies())
    supported.update(_STRATEGY_ALIAS_MAP.keys())
    supported.update(_NOOP_STRATEGIES)
    return supported


@dataclass
class TargetConfig:
    """Target configuration for security testing.

    Attributes:
        type: Target type (http, guardrail, mock, dspy)
        url: URL for HTTP targets
        name: Optional name for the target
        headers: HTTP headers for requests
        method: HTTP method (default: POST)
        timeout: Request timeout in seconds
        options: Additional target-specific options
    """
    type: str = "http"
    url: str = ""
    name: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    method: str = "POST"
    timeout: int = 30
    options: dict[str, Any] = field(default_factory=dict)

    def to_target_string(self) -> str:
        """Convert to target string format."""
        if self.type == "guardrail":
            return f"guardrail:{self.url}"
        elif self.type == "http":
            return self.url
        elif self.type == "mock":
            return "mock"
        else:
            return f"{self.type}:{self.url}"

    @classmethod
    def from_dict(cls, data: str | dict[str, Any]) -> "TargetConfig":
        """Create from dict or string."""
        if isinstance(data, str):
            # Parse string format: "http://url" or "guardrail:name"
            if data.startswith("guardrail:"):
                return cls(type="guardrail", url=data[10:])
            elif data.startswith("http://") or data.startswith("https://"):
                return cls(type="http", url=data)
            elif data == "mock":
                return cls(type="mock")
            else:
                return cls(type="http", url=data)
        else:
            return cls(**data)


@dataclass
class PluginConfig:
    """Plugin configuration.

    Attributes:
        id: Plugin identifier (e.g., "prompt-injection", "jailbreak", "mcp")
        numTests: Number of tests to run for this plugin
        severity: Minimum severity filter
        options: Plugin-specific options
    """
    id: str
    numTests: int = 10
    severity: str = "medium"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> "PluginConfig":
        """Create from string or dict."""
        if isinstance(value, str):
            return cls(id=value)
        else:
            return cls(**value)


@dataclass
class StrategyConfig:
    """Strategy configuration.

    Attributes:
        id: Strategy identifier (e.g., "base64", "rot13", "leetspeak")
        options: Strategy-specific options
    """
    id: str
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> "StrategyConfig":
        """Create from string or dict."""
        if isinstance(value, str):
            return cls(id=value)
        else:
            return cls(**value)


@dataclass
class CacheConfig:
    """Cache configuration.

    Attributes:
        enabled: Whether caching is enabled
        ttl: Time-to-live in seconds
        dir: Cache directory
        max_size: Maximum cache entries
    """
    enabled: bool = True
    ttl: int = 3600
    dir: str = ".cache/promptfoo"
    max_size: int = 10000


@dataclass
class OutputConfig:
    """Output configuration.

    Attributes:
        formats: Output formats (console, json, html, junit, sarif)
        dir: Output directory for reports
        filename: Base filename for reports
    """
    formats: list[str] = field(default_factory=lambda: ["console"])
    dir: str = "./reports"
    filename: str = "security-report"


@dataclass
class PromptfooConfig:
    """Main promptfoo-style configuration.

    Attributes:
        description: Description of the security test
        targets: List of target configurations
        plugins: List of plugin configurations
        strategies: List of strategy configurations
        frameworks: List of preset frameworks (e.g., "owasp:llm", "mitre:atlas")
        parallel: Enable parallel execution
        maxWorkers: Maximum parallel workers
        cache: Cache configuration
        output: Output configuration
        thresholds: Pass/fail thresholds
        metadata: Additional metadata
    """
    description: str = ""
    targets: list[TargetConfig] = field(default_factory=list)
    plugins: list[PluginConfig] = field(default_factory=list)
    strategies: list[StrategyConfig] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    parallel: bool = True
    maxWorkers: int = 5
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration and raise ValueError on invalid fields."""
        errors: list[str] = []

        # Validate plugin IDs and options.
        if self.plugins:
            from .plugins import get_plugin

            for plugin in self.plugins:
                plugin_id = plugin.id
                if not isinstance(plugin_id, str) or not plugin_id.strip():
                    errors.append(f"Invalid plugin id: {plugin_id!r}")
                    continue

                plugin_id = plugin_id.strip()
                try:
                    get_plugin(plugin_id)
                except ValueError:
                    errors.append(f"Unknown plugin id: {plugin_id}")

                try:
                    num_tests = int(plugin.numTests)
                except (TypeError, ValueError):
                    errors.append(
                        f"Invalid numTests for plugin '{plugin_id}': {plugin.numTests!r}"
                    )
                else:
                    if num_tests <= 0:
                        errors.append(
                            f"Invalid numTests for plugin '{plugin_id}': {plugin.numTests}"
                        )

                severity = str(plugin.severity).strip().lower()
                if severity not in _SUPPORTED_SEVERITIES:
                    errors.append(
                        f"Invalid severity for plugin '{plugin_id}': {plugin.severity}"
                    )

        # Validate strategy IDs.
        from ..redteam.strategies import list_strategies

        runtime_supported = set(list_strategies())
        for strategy in self.strategies:
            strategy_id_raw = strategy.id
            if not isinstance(strategy_id_raw, str) or not strategy_id_raw.strip():
                errors.append(f"Invalid strategy id: {strategy_id_raw!r}")
                continue

            strategy_id = strategy_id_raw.strip().lower()
            normalized = normalize_strategy_id(strategy_id)
            if normalized not in _NOOP_STRATEGIES and normalized not in runtime_supported:
                errors.append(f"Unknown strategy id: {strategy.id}")

            if strategy.options is not None and not isinstance(strategy.options, dict):
                errors.append(
                    f"Invalid options for strategy '{strategy.id}': expected dict, "
                    f"got {type(strategy.options).__name__}"
                )

        # Validate thresholds.
        for key, value in self.thresholds.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                errors.append(f"Threshold '{key}' must be numeric, got: {value!r}")
                continue
            if not 0.0 <= numeric <= 1.0:
                errors.append(
                    f"Threshold '{key}' must be in [0.0, 1.0], got: {numeric}"
                )

        # Validate workers and output format.
        if self.maxWorkers <= 0:
            errors.append(f"maxWorkers must be > 0, got: {self.maxWorkers}")
        for output_format in self.output.formats:
            normalized_output_format = str(output_format).strip().lower()
            if normalized_output_format not in _SUPPORTED_OUTPUT_FORMATS:
                errors.append(
                    f"Unsupported output format '{output_format}'. "
                    f"Supported: {sorted(_SUPPORTED_OUTPUT_FORMATS)}"
                )

        if errors:
            raise ValueError("; ".join(errors))

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        validate: bool = True,
    ) -> "PromptfooConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            PromptfooConfig instance

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If configuration is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(
                "Configuration root must be a mapping/object."
            )

        return cls.from_dict(data, validate=validate)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        validate: bool = True,
    ) -> "PromptfooConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            PromptfooConfig instance
        """
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a dictionary/mapping.")

        # Parse targets
        targets = []
        for target_data in data.get("targets", []):
            targets.append(TargetConfig.from_dict(target_data))

        # Parse plugins
        plugins = []
        for plugin_data in data.get("plugins", []):
            plugins.append(PluginConfig.from_value(plugin_data))

        # Parse strategies
        strategies = []
        for strategy_data in data.get("strategies", []):
            strategies.append(StrategyConfig.from_value(strategy_data))

        # Parse cache config
        cache_data = data.get("cache", {})
        if isinstance(cache_data, dict):
            cache = CacheConfig(**cache_data)
        else:
            cache = CacheConfig(enabled=bool(cache_data))

        # Parse output config
        output_data = data.get("output", {})
        if isinstance(output_data, dict):
            output = OutputConfig(**output_data)
        else:
            output = OutputConfig()

        config = cls(
            description=data.get("description", ""),
            targets=targets,
            plugins=plugins,
            strategies=strategies,
            frameworks=data.get("frameworks", []),
            parallel=data.get("parallel", True),
            maxWorkers=data.get("maxWorkers", 5),
            cache=cache,
            output=output,
            thresholds=data.get("thresholds", {}),
            metadata=data.get("metadata", {}),
        )
        if validate:
            config.validate()
        return config

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        validate: bool = True,
    ) -> "PromptfooConfig":
        """Create configuration from a preset.

        Args:
            preset_name: Preset name (e.g., "owasp:llm", "mitre:atlas", "quick-scan")

        Returns:
            PromptfooConfig instance with preset plugins and settings
        """
        from .presets import get_preset

        preset = get_preset(preset_name)
        config = preset.to_config()
        if validate:
            config.validate()
        return config

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        Returns:
            YAML string representation
        """
        data = {
            "description": self.description,
            "targets": [
                {
                    "type": t.type,
                    "url": t.url,
                    **({"name": t.name} if t.name else {}),
                    **({"headers": t.headers} if t.headers else {}),
                    **({"method": t.method} if t.method != "POST" else {}),
                    **({"timeout": t.timeout} if t.timeout != 30 else {}),
                    **({"options": t.options} if t.options else {}),
                }
                for t in self.targets
            ],
            "plugins": [
                p.id if p.numTests == 10 and not p.options
                else {
                    "id": p.id,
                    **({"numTests": p.numTests} if p.numTests != 10 else {}),
                    **({"severity": p.severity} if p.severity != "medium" else {}),
                    **({"options": p.options} if p.options else {}),
                }
                for p in self.plugins
            ],
            "strategies": [
                s.id if not s.options
                else {"id": s.id, "options": s.options}
                for s in self.strategies
            ],
            "frameworks": self.frameworks,
            "parallel": self.parallel,
            "maxWorkers": self.maxWorkers,
            "cache": {
                "enabled": self.cache.enabled,
                "ttl": self.cache.ttl,
                "dir": self.cache.dir,
            },
            "output": {
                "formats": self.output.formats,
                "dir": self.output.dir,
            },
        }

        if self.thresholds:
            data["thresholds"] = self.thresholds

        if self.metadata:
            data["metadata"] = self.metadata

        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def merge(self, other: "PromptfooConfig") -> "PromptfooConfig":
        """Merge with another configuration.

        The other configuration takes precedence for scalar values.
        Lists are concatenated.

        Args:
            other: Configuration to merge with

        Returns:
            New merged configuration
        """
        return PromptfooConfig(
            description=other.description or self.description,
            targets=self.targets + other.targets,
            plugins=self.plugins + other.plugins,
            strategies=self.strategies + other.strategies,
            frameworks=list(set(self.frameworks + other.frameworks)),
            parallel=other.parallel,
            maxWorkers=other.maxWorkers,
            cache=other.cache,
            output=other.output,
            thresholds={**self.thresholds, **other.thresholds},
            metadata={**self.metadata, **other.metadata},
        )

    def get_plugin_ids(self) -> list[str]:
        """Get list of plugin IDs."""
        return [p.id for p in self.plugins]

    def get_strategy_ids(self) -> list[str]:
        """Get list of strategy IDs."""
        return [s.id for s in self.strategies]


def load_config(path: str | Path) -> PromptfooConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        PromptfooConfig instance
    """
    return PromptfooConfig.from_yaml(path)


def create_default_config(
    target: str = "http://localhost:9000/chat",
    preset: str | None = None,
) -> PromptfooConfig:
    """Create a default configuration.

    Args:
        target: Target URL or guardrail name
        preset: Optional preset to use

    Returns:
        PromptfooConfig with default settings
    """
    if preset:
        config = PromptfooConfig.from_preset(preset)
        if target:
            config.targets = [TargetConfig.from_dict(target)]
        return config

    return PromptfooConfig(
        description="Security test configuration",
        targets=[TargetConfig.from_dict(target)],
        plugins=[
            PluginConfig(id="prompt-injection"),
            PluginConfig(id="jailbreak"),
        ],
        strategies=[
            StrategyConfig(id="base64"),
        ],
        parallel=True,
        maxWorkers=5,
        cache=CacheConfig(enabled=True),
        output=OutputConfig(formats=["console", "json"]),
    )
