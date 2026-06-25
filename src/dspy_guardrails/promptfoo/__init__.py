"""
Promptfoo-style Security Testing Framework for dspyGuardrails

This module provides promptfoo-inspired features for declarative security testing:

Features:
    - Declarative YAML configuration
    - OWASP LLM Top 10 and MITRE ATLAS presets
    - LLM call caching with cost tracking
    - Concurrent test execution
    - CI/CD integration helpers
    - Enhanced CLI commands

Usage:
    # CLI usage
    $ dspy-guardrails redteam init --preset owasp:llm
    $ dspy-guardrails redteam run --config promptfoo.yaml
    $ dspy-guardrails redteam run --preset quick-scan

    # Programmatic usage
    from dspy_guardrails.promptfoo import (
        PromptfooConfig,
        ConcurrentTestRunner,
        LLMCallCache,
        get_preset,
        OWASP_LLM_TOP10,
        MITRE_ATLAS,
    )

    # Load config from YAML
    config = PromptfooConfig.from_yaml("promptfoo.yaml")

    # Or use preset
    config = PromptfooConfig.from_preset("owasp:llm")

    # Run with caching and concurrency
    cache = LLMCallCache()
    runner = ConcurrentTestRunner(config, cache=cache)
    results = runner.run(target)

    # Get cache statistics
    print(cache.get_cost_summary())
"""

from .cache import (
    CacheStats,
    LLMCallCache,
    clear_llm_cache,
    get_llm_cache,
)
from .config import (
    CacheConfig,
    OutputConfig,
    PluginConfig,
    PromptfooConfig,
    StrategyConfig,
    TargetConfig,
    create_default_config,
    load_config,
)
from .plugins import (
    Plugin,
    PluginRegistry,
    get_plugin,
    list_plugins,
    register_plugin,
)
from .presets import (
    MITRE_ATLAS,
    OWASP_LLM_TOP10,
    QUICK_SCAN,
    Preset,
    PresetRegistry,
    get_preset,
    list_presets,
)
from .runner import (
    ConcurrentTestRunner,
    RunProgress,
    RunSummary,
    TestResult,
)

__all__ = [
    # Config
    "PromptfooConfig",
    "TargetConfig",
    "PluginConfig",
    "StrategyConfig",
    "OutputConfig",
    "CacheConfig",
    "load_config",
    "create_default_config",
    # Presets
    "Preset",
    "PresetRegistry",
    "get_preset",
    "list_presets",
    "OWASP_LLM_TOP10",
    "MITRE_ATLAS",
    "QUICK_SCAN",
    # Plugins
    "Plugin",
    "PluginRegistry",
    "get_plugin",
    "list_plugins",
    "register_plugin",
    # Cache
    "LLMCallCache",
    "CacheStats",
    "get_llm_cache",
    "clear_llm_cache",
    # Runner
    "ConcurrentTestRunner",
    "TestResult",
    "RunProgress",
    "RunSummary",
]
