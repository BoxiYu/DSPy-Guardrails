"""
Tests for promptfoo-style features
"""

import tempfile

import pytest


class TestPromptfooConfig:
    """Tests for PromptfooConfig"""

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        from dspy_guardrails.promptfoo.config import PromptfooConfig

        data = {
            "description": "Test config",
            "targets": [
                {"type": "http", "url": "http://localhost:9000"}
            ],
            "plugins": [
                "prompt-injection",
                {"id": "jailbreak", "numTests": 20}
            ],
            "strategies": ["base64"],
            "parallel": True,
            "maxWorkers": 5,
        }

        config = PromptfooConfig.from_dict(data)

        assert config.description == "Test config"
        assert len(config.targets) == 1
        assert config.targets[0].type == "http"
        assert config.targets[0].url == "http://localhost:9000"
        assert len(config.plugins) == 2
        assert config.plugins[0].id == "prompt-injection"
        assert config.plugins[1].id == "jailbreak"
        assert config.plugins[1].numTests == 20
        assert len(config.strategies) == 1
        assert config.strategies[0].id == "base64"
        assert config.parallel is True
        assert config.maxWorkers == 5

    def test_config_to_yaml(self):
        """Test converting config to YAML"""
        from dspy_guardrails.promptfoo.config import PluginConfig, PromptfooConfig, TargetConfig

        config = PromptfooConfig(
            description="Test",
            targets=[TargetConfig(type="http", url="http://test.com")],
            plugins=[PluginConfig(id="prompt-injection")],
        )

        yaml_str = config.to_yaml()
        assert "description: Test" in yaml_str
        assert "prompt-injection" in yaml_str

    def test_config_save_load(self):
        """Test saving and loading config"""
        from dspy_guardrails.promptfoo.config import PluginConfig, PromptfooConfig, TargetConfig

        config = PromptfooConfig(
            description="Save/Load Test",
            targets=[TargetConfig(type="http", url="http://localhost:8000")],
            plugins=[PluginConfig(id="prompt-injection", numTests=15)],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.save(f.name)
            loaded = PromptfooConfig.from_yaml(f.name)

        assert loaded.description == "Save/Load Test"
        assert len(loaded.targets) == 1
        assert loaded.targets[0].url == "http://localhost:8000"
        assert len(loaded.plugins) == 1
        assert loaded.plugins[0].numTests == 15

    def test_config_from_preset(self):
        """Test creating config from preset"""
        from dspy_guardrails.promptfoo.config import PromptfooConfig

        config = PromptfooConfig.from_preset("quick-scan")

        assert "Quick" in config.description
        assert len(config.plugins) > 0
        assert config.parallel is True

    def test_config_rejects_unknown_plugin(self):
        """Unknown plugin IDs should fail validation."""
        from dspy_guardrails.promptfoo.config import PromptfooConfig

        with pytest.raises(ValueError, match="Unknown plugin id"):
            PromptfooConfig.from_dict(
                {
                    "plugins": [{"id": "non-existent-plugin"}],
                }
            )

    def test_config_rejects_unknown_strategy(self):
        """Unknown strategy IDs should fail validation."""
        from dspy_guardrails.promptfoo.config import PromptfooConfig

        with pytest.raises(ValueError, match="Unknown strategy id"):
            PromptfooConfig.from_dict(
                {
                    "plugins": [{"id": "prompt-injection"}],
                    "strategies": [{"id": "unknown-strategy"}],
                }
            )

    def test_config_accepts_strategy_alias(self):
        """Promptfoo-style strategy aliases should validate."""
        from dspy_guardrails.promptfoo.config import PromptfooConfig

        config = PromptfooConfig.from_dict(
            {
                "plugins": [{"id": "prompt-injection"}],
                "strategies": [{"id": "unicode-confusables"}],
            }
        )

        assert len(config.strategies) == 1
        assert config.strategies[0].id == "unicode-confusables"

    def test_config_rejects_invalid_strategy_options_type(self):
        """Strategy options must be mapping-like in strict mode."""
        from dspy_guardrails.promptfoo.config import PromptfooConfig

        with pytest.raises(ValueError, match="Invalid options for strategy"):
            PromptfooConfig.from_dict(
                {
                    "plugins": [{"id": "prompt-injection"}],
                    "strategies": [{"id": "base64", "options": ["bad"]}],
                }
            )


class TestPresets:
    """Tests for presets"""

    def test_get_preset(self):
        """Test getting presets"""
        from dspy_guardrails.promptfoo.presets import get_preset

        # OWASP preset
        owasp = get_preset("owasp:llm")
        assert owasp.name == "OWASP LLM Top 10"
        assert len(owasp.plugins) == 10

        # MITRE preset
        mitre = get_preset("mitre:atlas")
        assert mitre.name == "MITRE ATLAS"

        # Quick scan preset
        quick = get_preset("quick-scan")
        assert "Quick" in quick.name

    def test_list_presets(self):
        """Test listing presets"""
        from dspy_guardrails.promptfoo.presets import list_presets

        presets = list_presets()
        assert "owasp:llm" in presets
        assert "mitre:atlas" in presets
        assert "quick-scan" in presets

    def test_preset_to_config(self):
        """Test converting preset to config"""
        from dspy_guardrails.promptfoo.presets import get_preset

        preset = get_preset("quick-scan")
        config = preset.to_config()

        assert config.frameworks == ["quick-scan"]
        assert len(config.plugins) > 0


class TestPlugins:
    """Tests for plugins"""

    def test_get_plugin(self):
        """Test getting plugins"""
        from dspy_guardrails.promptfoo.plugins import get_plugin

        plugin = get_plugin("prompt-injection")
        assert plugin.id == "prompt-injection"
        assert plugin.owasp_mapping == "LLM01"

    def test_list_plugins(self):
        """Test listing plugins"""
        from dspy_guardrails.promptfoo.plugins import list_plugins

        plugins = list_plugins()
        assert "prompt-injection" in plugins
        assert "jailbreak" in plugins

    def test_plugin_get_payloads(self):
        """Test getting payloads from plugin"""
        from dspy_guardrails.promptfoo.plugins import get_plugin

        plugin = get_plugin("prompt-injection")
        payloads = plugin.get_payloads(num_tests=5, severity="high")

        assert len(payloads) <= 5
        # All should be high or critical
        for p in payloads:
            severity = p.severity.value if hasattr(p.severity, 'value') else p.severity
            assert severity in ["high", "critical"]


class TestLLMCache:
    """Tests for LLM cache"""

    def test_cache_basic(self):
        """Test basic cache operations"""
        from dspy_guardrails.promptfoo.cache import LLMCallCache

        cache = LLMCallCache(persist=False)

        # Set and get
        cache.set_llm_response("test prompt", {"response": "test"}, model="gpt-4")
        result = cache.get_llm_response("test prompt", model="gpt-4")

        assert result is not None
        assert result["response"] == "test"

    def test_cache_normalization(self):
        """Test prompt normalization"""
        from dspy_guardrails.promptfoo.cache import LLMCallCache

        cache = LLMCallCache(persist=False)

        # Set with extra whitespace
        cache.set_llm_response("  test   prompt  ", {"response": "test"}, model="gpt-4")

        # Get with normalized prompt should still work
        result = cache.get_llm_response("test prompt", model="gpt-4")
        assert result is not None

    def test_cache_stats(self):
        """Test cache statistics"""
        from dspy_guardrails.promptfoo.cache import LLMCallCache

        cache = LLMCallCache(persist=False)

        # Miss
        cache.get_llm_response("missing", model="gpt-4")

        # Set and hit
        cache.set_llm_response("test", {"r": "ok"}, model="gpt-4")
        cache.get_llm_response("test", model="gpt-4")

        stats = cache.get_cost_summary()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1


class TestRunner:
    """Tests for concurrent test runner"""

    def test_runner_with_mock_target(self):
        """Test runner with mock target"""
        from dspy_guardrails.promptfoo.config import PluginConfig, PromptfooConfig
        from dspy_guardrails.promptfoo.runner import ConcurrentTestRunner
        from dspy_guardrails.testing.targets import MockTarget

        config = PromptfooConfig(
            plugins=[PluginConfig(id="prompt-injection", numTests=3)],
            parallel=True,
            maxWorkers=2,
        )

        target = MockTarget(
            block_fn=lambda p: "ignore" in p.lower()
        )

        runner = ConcurrentTestRunner(config)
        summary = runner.run(target)

        assert summary.total_tests > 0
        assert summary.blocked_attacks + summary.successful_attacks <= summary.total_tests

    def test_runner_progress(self):
        """Test runner progress tracking"""
        from dspy_guardrails.promptfoo.config import PluginConfig, PromptfooConfig
        from dspy_guardrails.promptfoo.runner import ConcurrentTestRunner, RunProgress
        from dspy_guardrails.testing.targets import MockTarget

        config = PromptfooConfig(
            plugins=[PluginConfig(id="prompt-injection", numTests=5)],
            parallel=False,
        )

        target = MockTarget()
        runner = ConcurrentTestRunner(config)

        progress_updates = []
        def on_progress(p: RunProgress):
            progress_updates.append(p.percent_complete)

        runner.run(target, on_progress=on_progress)

        # Should have progress updates
        assert len(progress_updates) > 0

    def test_runner_applies_strategy_transformations(self):
        """Configured strategies should expand generated test cases."""
        from dspy_guardrails.promptfoo.config import (
            PluginConfig,
            PromptfooConfig,
            StrategyConfig,
        )
        from dspy_guardrails.promptfoo.runner import ConcurrentTestRunner
        from dspy_guardrails.testing.targets import MockTarget

        config = PromptfooConfig(
            plugins=[PluginConfig(id="prompt-injection", numTests=2)],
            strategies=[StrategyConfig(id="base64")],
            parallel=False,
        )

        target = MockTarget(block_fn=lambda p: "ignore" in p.lower())
        runner = ConcurrentTestRunner(config)
        summary = runner.run(target)

        # Base tests + transformed tests
        assert summary.total_tests >= 4


class TestCIIntegration:
    """Tests for CI/CD integration"""

    def test_generate_github_workflow(self):
        """Test generating GitHub workflow"""
        from dspy_guardrails.testing.ci import generate_workflow

        workflow = generate_workflow(
            provider="github",
            config_path="security.yaml",
            threshold=0.85,
        )

        assert "name: AI Security Scan" in workflow
        assert "security.yaml" in workflow
        assert "dspy-guardrails" in workflow

    def test_generate_gitlab_workflow(self):
        """Test generating GitLab workflow"""
        from dspy_guardrails.testing.ci import generate_workflow

        workflow = generate_workflow(
            provider="gitlab",
            config_path="test.yaml",
        )

        assert "stages:" in workflow
        assert "security" in workflow

    def test_generate_jenkins_workflow(self):
        """Test generating Jenkins workflow"""
        from dspy_guardrails.testing.ci import generate_workflow

        workflow = generate_workflow(
            provider="jenkins",
        )

        assert "pipeline" in workflow
        assert "stages" in workflow


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
