"""Tests for SecurityPlatform core class."""

import pytest
from dspy_guardrails.platform import SecurityPlatform, PlatformConfig
from dspy_guardrails.platform.targets import (
    UnifiedTarget,
    TargetType,
    TargetCapability,
    TargetResponse,
)


class SimpleTarget(UnifiedTarget):
    """Simple test target for unit tests."""

    target_type = TargetType.GUARDRAIL
    capabilities = [TargetCapability.SINGLE_TURN]

    def invoke(self, prompt: str) -> TargetResponse:
        blocked = "ignore" in prompt.lower()
        return TargetResponse(response="OK", was_blocked=blocked)

    def invoke_multi_turn(self, messages):
        return self.invoke(messages[-1]["content"])

    def reset_session(self):
        pass


class TestPlatformInit:
    """Test SecurityPlatform initialization."""

    def test_platform_init_with_target(self):
        """Test platform initialization with a UnifiedTarget instance."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        assert platform.target is not None
        assert platform.config is not None

    def test_platform_init_with_config(self):
        """Test platform initialization with custom config."""
        target = SimpleTarget()
        config = PlatformConfig(attack_budget=200)
        platform = SecurityPlatform(target, config=config)
        assert platform.config.attack_budget == 200

    def test_platform_init_with_default_config(self):
        """Test platform uses default config when none provided."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        assert platform.config.attack_budget == 100  # Default value
        assert platform.config.parallel is True


class TestFluentAPI:
    """Test SecurityPlatform fluent API methods."""

    def test_platform_fluent_api(self):
        """Test chaining fluent API methods."""
        target = SimpleTarget()
        platform = (
            SecurityPlatform(target)
            .with_attacks("injection", "jailbreak")
            .with_scanners("quick_scan")
            .with_reports("html", "json")
        )
        assert "injection" in platform.config.attacks
        assert "jailbreak" in platform.config.attacks
        assert "quick_scan" in platform.config.scanners
        assert "html" in platform.config.report.formats

    def test_with_training(self):
        """Test with_training fluent method."""
        target = SimpleTarget()
        platform = SecurityPlatform(target).with_training(
            enabled=True,
            mode="redblue",
            epochs=20,
        )
        assert platform.config.training.enabled is True
        assert platform.config.training.mode == "redblue"
        assert platform.config.training.epochs == 20

    def test_with_output_dir(self):
        """Test with_output_dir fluent method."""
        target = SimpleTarget()
        platform = SecurityPlatform(target).with_output_dir("/tmp/reports")
        assert str(platform.config.report.output_dir) == "/tmp/reports"

    def test_fluent_api_returns_self(self):
        """Test that fluent methods return self for chaining."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)

        result = platform.with_attacks("injection")
        assert result is platform

        result = platform.with_scanners("quick")
        assert result is platform

        result = platform.with_reports("json")
        assert result is platform

        result = platform.with_training(enabled=True)
        assert result is platform


class TestTargetResolution:
    """Test target resolution logic."""

    def test_resolve_unified_target(self):
        """Test resolving UnifiedTarget instance."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        assert isinstance(platform.target, UnifiedTarget)

    def test_resolve_string_target(self):
        """Test resolving string target to HTTPTarget."""
        platform = SecurityPlatform("http://localhost:8000")
        assert isinstance(platform.target, UnifiedTarget)

    def test_resolve_dict_target(self):
        """Test resolving dict target to HTTPTarget."""
        platform = SecurityPlatform({"type": "http", "url": "http://localhost:8000"})
        assert isinstance(platform.target, UnifiedTarget)

    def test_resolve_invalid_target_type(self):
        """Test that invalid target type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported target type"):
            SecurityPlatform(12345)


class TestExecutionMethods:
    """Test execution methods."""

    def test_scan_returns_result(self):
        """Test that scan returns a result dictionary."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        result = platform.scan()
        assert "success" in result
        assert "vulnerabilities" in result
        assert "metrics" in result

    def test_attack_returns_result(self):
        """Test that attack returns a result dictionary."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        result = platform.attack(budget=5)
        assert "success" in result
        assert "attack_results" in result
        assert "metrics" in result

    def test_train_disabled_by_default(self):
        """Test that train returns not enabled message by default."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        result = platform.train()
        assert result["success"] is False
        assert "not enabled" in result["message"]

    def test_report_returns_paths(self):
        """Test that report returns list of paths."""
        import tempfile
        target = SimpleTarget()
        with tempfile.TemporaryDirectory() as tmpdir:
            platform = SecurityPlatform(target).with_reports("json").with_output_dir(tmpdir)
            platform.scan()
            files = platform.report()
            assert isinstance(files, list)

    def test_run_all_returns_complete_result(self):
        """Test that run_all returns complete results."""
        import tempfile
        target = SimpleTarget()
        with tempfile.TemporaryDirectory() as tmpdir:
            platform = SecurityPlatform(target).with_reports("json").with_output_dir(tmpdir)
            result = platform.run_all()
            assert "scan" in result
            assert "attack" in result


class TestClassMethods:
    """Test class methods."""

    def test_from_yaml_missing_file_raises_error(self):
        """Test that from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            SecurityPlatform.from_yaml("nonexistent_config.yaml")

    def test_from_yaml_loads_config(self):
        """Test that from_yaml loads configuration correctly."""
        import tempfile
        yaml_content = '''target:
  type: guardrail
  value: no_injection

scan:
  enabled: true
  categories:
    - injection

attack:
  enabled: true
  budget: 20

report:
  formats:
    - json
  output_dir: ./reports
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            platform = SecurityPlatform.from_yaml(f.name)
            assert platform.target is not None
            assert "injection" in platform.config.attacks
        # Cleanup
        import os
        os.unlink(f.name)


class TestPlatformAttributes:
    """Test platform attributes and state."""

    def test_registry_initialized(self):
        """Test that plugin registry is initialized."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        assert platform.registry is not None

    def test_results_initialized_empty(self):
        """Test that results list is initialized empty."""
        target = SimpleTarget()
        platform = SecurityPlatform(target)
        assert platform._results == []
