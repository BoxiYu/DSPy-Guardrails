"""Tests for YAML configuration parsing."""

import pytest
import tempfile
import os
from pathlib import Path

from dspy_guardrails.platform.cli.yaml_config import (
    SecurityConfig,
    TargetConfig,
    ScanConfig,
    AttackConfig,
    ReportConfig,
    create_sample_config,
    validate_config_file,
)


class TestTargetConfig:
    """TargetConfig tests."""

    def test_target_config_guardrail(self):
        """Test guardrail target configuration."""
        config = TargetConfig(type="guardrail", value="no_injection")
        assert config.type == "guardrail"
        assert config.value == "no_injection"
        assert config.name is None

    def test_target_config_with_name(self):
        """Test target configuration with optional name."""
        config = TargetConfig(type="guardrail", value="safe", name="Safety Check")
        assert config.name == "Safety Check"

    def test_target_config_http(self):
        """Test HTTP target configuration."""
        config = TargetConfig(type="http", value="http://localhost:8000")
        assert config.type == "http"
        assert config.value == "http://localhost:8000"

    def test_to_target_string_guardrail(self):
        """Test to_target_string for guardrail target."""
        config = TargetConfig(type="guardrail", value="no_injection")
        assert config.to_target_string() == "guardrail:no_injection"

    def test_to_target_string_http(self):
        """Test to_target_string for HTTP target."""
        config = TargetConfig(type="http", value="http://localhost:8000")
        assert config.to_target_string() == "http://localhost:8000"


class TestScanConfig:
    """ScanConfig tests."""

    def test_scan_config_defaults(self):
        """Test scan configuration defaults."""
        config = ScanConfig()
        assert config.enabled is True
        assert config.max_payloads == 20
        assert config.categories == ["injection", "jailbreak"]
        assert config.severity == "medium"

    def test_scan_config_custom(self):
        """Test custom scan configuration."""
        config = ScanConfig(
            enabled=False,
            max_payloads=100,
            categories=["injection", "bypass"],
            severity="high",
        )
        assert config.enabled is False
        assert config.max_payloads == 100
        assert config.categories == ["injection", "bypass"]
        assert config.severity == "high"


class TestAttackConfig:
    """AttackConfig tests."""

    def test_attack_config_defaults(self):
        """Test attack configuration defaults."""
        config = AttackConfig()
        assert config.enabled is True
        assert config.budget == 100
        assert config.types == ["injection", "jailbreak", "bypass"]
        assert config.use_llm is False
        assert config.stop_on_success is False

    def test_attack_config_custom(self):
        """Test custom attack configuration."""
        config = AttackConfig(
            enabled=True,
            budget=200,
            types=["mcp", "injection"],
            use_llm=True,
            stop_on_success=True,
        )
        assert config.budget == 200
        assert config.use_llm is True
        assert config.stop_on_success is True


class TestReportConfig:
    """ReportConfig tests."""

    def test_report_config_defaults(self):
        """Test report configuration defaults."""
        config = ReportConfig()
        assert config.formats == ["console"]
        assert config.output_dir == "./reports"

    def test_report_config_custom(self):
        """Test custom report configuration."""
        config = ReportConfig(
            formats=["console", "json", "html"],
            output_dir="/tmp/security_reports",
        )
        assert "json" in config.formats
        assert "html" in config.formats
        assert config.output_dir == "/tmp/security_reports"


class TestSecurityConfig:
    """SecurityConfig tests."""

    def test_from_dict_minimal(self):
        """Test loading minimal config from dict."""
        data = {
            "target": {"type": "guardrail", "value": "no_injection"}
        }
        config = SecurityConfig.from_dict(data)
        assert config.target.type == "guardrail"
        assert config.target.value == "no_injection"
        # Defaults should be applied
        assert config.scan.enabled is True
        assert config.attack.enabled is True

    def test_from_dict_full(self):
        """Test loading full config from dict."""
        data = {
            "target": {"type": "guardrail", "value": "safe", "name": "Full Test"},
            "scan": {"enabled": True, "max_payloads": 50, "categories": ["injection"]},
            "attack": {"enabled": True, "budget": 200, "use_llm": True},
            "report": {"formats": ["json", "html"], "output_dir": "/tmp/reports"},
        }
        config = SecurityConfig.from_dict(data)
        assert config.target.name == "Full Test"
        assert config.scan.max_payloads == 50
        assert config.attack.use_llm is True
        assert "json" in config.report.formats

    def test_from_yaml_file(self):
        """Test loading from YAML file."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection
  name: "YAML Test"

scan:
  enabled: true
  max_payloads: 30
  categories:
    - injection
    - jailbreak

attack:
  enabled: false
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = SecurityConfig.from_yaml(f.name)
                assert config.target.value == "no_injection"
                assert config.target.name == "YAML Test"
                assert config.scan.max_payloads == 30
                assert config.attack.enabled is False
            finally:
                os.unlink(f.name)

    def test_from_yaml_minimal(self):
        """Test loading minimal YAML file."""
        yaml_content = '''
target:
  type: guardrail
  value: safe
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = SecurityConfig.from_yaml(f.name)
                assert config.target.value == "safe"
                # Defaults applied
                assert config.scan.enabled is True
                assert config.attack.budget == 100
            finally:
                os.unlink(f.name)

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            SecurityConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            try:
                with pytest.raises(ValueError) as exc_info:
                    SecurityConfig.from_yaml(f.name)
                assert "Empty configuration" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_to_yaml(self):
        """Test saving to YAML file."""
        config = SecurityConfig(
            target=TargetConfig(type="guardrail", value="no_injection"),
            scan=ScanConfig(max_payloads=25),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            config.to_yaml(path)

            # Verify file was created
            assert os.path.exists(path)

            # Load back and verify
            loaded = SecurityConfig.from_yaml(path)
            assert loaded.target.value == "no_injection"
            assert loaded.scan.max_payloads == 25

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = SecurityConfig(
            target=TargetConfig(type="http", value="http://localhost:8000"),
            scan=ScanConfig(enabled=False),
        )
        data = config.to_dict()
        assert data["target"]["type"] == "http"
        assert data["scan"]["enabled"] is False

    def test_defaults(self):
        """Test all default values are applied."""
        data = {"target": {"type": "guardrail", "value": "safe"}}
        config = SecurityConfig.from_dict(data)

        # Scan defaults
        assert config.scan.enabled is True
        assert config.scan.max_payloads == 20
        assert config.scan.severity == "medium"

        # Attack defaults
        assert config.attack.enabled is True
        assert config.attack.budget == 100
        assert config.attack.use_llm is False
        assert config.attack.stop_on_success is False

        # Report defaults
        assert config.report.formats == ["console"]
        assert config.report.output_dir == "./reports"


class TestCreateSampleConfig:
    """Sample config generation tests."""

    def test_sample_config_is_valid_yaml(self):
        """Test that sample config is valid YAML."""
        import yaml
        sample = create_sample_config()
        data = yaml.safe_load(sample)
        assert "target" in data
        assert "scan" in data
        assert "attack" in data
        assert "report" in data

    def test_sample_config_is_loadable(self):
        """Test that sample config can be loaded as SecurityConfig."""
        import yaml
        sample = create_sample_config()
        data = yaml.safe_load(sample)
        config = SecurityConfig.from_dict(data)
        assert config.target.type == "guardrail"
        assert config.target.value == "no_injection"
        assert config.scan.max_payloads == 50

    def test_sample_config_has_comments(self):
        """Test that sample config includes helpful comments."""
        sample = create_sample_config()
        assert "#" in sample  # Has comments
        assert "dspyGuardrails" in sample


class TestValidateConfigFile:
    """Config file validation tests."""

    def test_validate_valid_config(self):
        """Test validating a valid config file."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                is_valid, error = validate_config_file(f.name)
                assert is_valid is True
                assert error is None
            finally:
                os.unlink(f.name)

    def test_validate_missing_file(self):
        """Test validating a non-existent file."""
        is_valid, error = validate_config_file("/nonexistent/config.yaml")
        assert is_valid is False
        assert "not found" in error.lower()

    def test_validate_invalid_yaml(self):
        """Test validating invalid YAML."""
        yaml_content = '''
target:
  type: guardrail
  value: [invalid yaml structure
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                is_valid, error = validate_config_file(f.name)
                assert is_valid is False
                # Should mention YAML error or configuration error
                assert error is not None
            finally:
                os.unlink(f.name)

    def test_validate_missing_required_field(self):
        """Test validating config with missing required field."""
        yaml_content = '''
scan:
  enabled: true
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                is_valid, error = validate_config_file(f.name)
                assert is_valid is False
                # Should mention missing target
                assert error is not None
            finally:
                os.unlink(f.name)


class TestConfigRoundTrip:
    """Tests for config save/load round-trip."""

    def test_full_roundtrip(self):
        """Test full configuration round-trip."""
        original = SecurityConfig(
            target=TargetConfig(
                type="guardrail",
                value="safe",
                name="Round Trip Test"
            ),
            scan=ScanConfig(
                enabled=True,
                max_payloads=75,
                categories=["injection", "jailbreak", "bypass"],
                severity="high"
            ),
            attack=AttackConfig(
                enabled=True,
                budget=150,
                types=["injection", "mcp"],
                use_llm=True,
                stop_on_success=True
            ),
            report=ReportConfig(
                formats=["console", "json", "html"],
                output_dir="/custom/reports"
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.yaml")
            original.to_yaml(path)
            loaded = SecurityConfig.from_yaml(path)

            assert loaded.target.type == original.target.type
            assert loaded.target.value == original.target.value
            assert loaded.target.name == original.target.name
            assert loaded.scan.max_payloads == original.scan.max_payloads
            assert loaded.scan.categories == original.scan.categories
            assert loaded.attack.budget == original.attack.budget
            assert loaded.attack.use_llm == original.attack.use_llm
            assert loaded.report.formats == original.report.formats


class TestEdgeCases:
    """Edge case tests."""

    def test_http_target_with_path(self):
        """Test HTTP target with path."""
        data = {
            "target": {"type": "http", "value": "http://localhost:8000/api/chat"}
        }
        config = SecurityConfig.from_dict(data)
        assert config.target.to_target_string() == "http://localhost:8000/api/chat"

    def test_empty_categories(self):
        """Test empty categories list."""
        data = {
            "target": {"type": "guardrail", "value": "safe"},
            "scan": {"categories": []}
        }
        config = SecurityConfig.from_dict(data)
        assert config.scan.categories == []

    def test_all_phases_disabled(self):
        """Test config with all phases disabled."""
        data = {
            "target": {"type": "guardrail", "value": "safe"},
            "scan": {"enabled": False},
            "attack": {"enabled": False}
        }
        config = SecurityConfig.from_dict(data)
        assert config.scan.enabled is False
        assert config.attack.enabled is False

    def test_large_budget(self):
        """Test config with large attack budget."""
        data = {
            "target": {"type": "guardrail", "value": "safe"},
            "attack": {"budget": 10000}
        }
        config = SecurityConfig.from_dict(data)
        assert config.attack.budget == 10000

    def test_unicode_in_name(self):
        """Test unicode characters in target name."""
        data = {
            "target": {
                "type": "guardrail",
                "value": "no_injection",
                "name": "注入检测测试"
            }
        }
        config = SecurityConfig.from_dict(data)
        assert config.target.name == "注入检测测试"
