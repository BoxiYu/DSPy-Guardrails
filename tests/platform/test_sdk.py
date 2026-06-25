"""Tests for SecurityPlatform SDK methods."""

import pytest
import tempfile
import json
from pathlib import Path

from dspy_guardrails.platform import SecurityPlatform
from dspy_guardrails.platform.targets import GuardrailTarget
from dspy_guardrails.platform.config import PlatformConfig
from dspy_guardrails import guardrail


class TestSecurityPlatformScan:
    """SecurityPlatform.scan() tests."""

    def test_scan_quick_mode(self):
        """Test quick scan mode."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target)

        result = platform.scan(mode="quick")

        assert "success" in result
        assert "vulnerabilities" in result
        assert "metrics" in result
        assert result["success"] is True

    def test_scan_full_mode(self):
        """Test full scan mode."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target).with_attacks("injection")

        result = platform.scan(mode="full")

        assert result["success"] is True
        assert "metrics" in result
        assert "vulnerabilities" in result

    def test_scan_stores_results(self):
        """Test that scan stores results internally."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target)

        platform.scan(mode="quick")

        assert len(platform._results) == 1
        assert platform._results[0][0] == "scan"

    def test_scan_with_custom_categories(self):
        """Test scan with custom attack categories."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target).with_attacks("injection", "jailbreak", "bypass")

        result = platform.scan(mode="quick")

        assert result["success"] is True


class TestSecurityPlatformAttack:
    """SecurityPlatform.attack() tests."""

    def test_attack_default(self):
        """Test default attack."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target)

        result = platform.attack(budget=10)

        assert "success" in result
        assert "attack_results" in result
        assert "metrics" in result

    def test_attack_with_categories(self):
        """Test attack with specific categories."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target)

        result = platform.attack(budget=5, categories=["injection"])

        assert result["success"] is True
        assert "attack_results" in result

    def test_attack_stores_results(self):
        """Test that attack stores results internally."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target)

        platform.attack(budget=5)

        assert len(platform._results) == 1
        assert platform._results[0][0] == "attack"

    def test_attack_uses_static_by_default(self):
        """Test that attack uses static attacker by default when LLM not configured."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        # Disable LLM generation
        config = PlatformConfig(use_llm_generation=False)
        platform = SecurityPlatform(target, config=config)

        result = platform.attack(budget=5)

        assert result["success"] is True

    def test_attack_with_llm_flag(self):
        """Test attack with use_llm flag (falls back to static without DSPy)."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target)

        # Will use static fallback since DSPy LM is not configured
        result = platform.attack(budget=5, use_llm=True)

        assert result["success"] is True


class TestSecurityPlatformTrain:
    """SecurityPlatform.train() tests."""

    def test_train_disabled_by_default(self):
        """Test that training is disabled by default."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target)

        result = platform.train()

        assert result["success"] is False
        assert "not enabled" in result["message"]

    def test_train_enabled(self):
        """Test training when enabled."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target).with_training(enabled=True)

        result = platform.train()

        assert result["success"] is True
        assert "placeholder" in result["message"].lower()
        assert "config" in result


class TestSecurityPlatformReport:
    """SecurityPlatform.report() tests."""

    def test_report_json(self):
        """Test JSON report generation."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_reports("json")
                .with_output_dir(tmpdir)
            )
            platform.scan(mode="quick")

            files = platform.report()

            assert len(files) == 1
            assert files[0].suffix == ".json"
            assert files[0].exists()

            # Verify JSON content
            with open(files[0], "r") as f:
                data = json.load(f)
            assert "timestamp" in data
            assert "results" in data

    def test_report_html(self):
        """Test HTML report generation."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_reports("html")
                .with_output_dir(tmpdir)
            )
            platform.attack(budget=5)

            files = platform.report()

            assert len(files) == 1
            assert files[0].suffix == ".html"
            assert files[0].exists()

            # Verify HTML content
            content = files[0].read_text()
            assert "<html>" in content
            assert "Security Report" in content

    def test_report_multiple_formats(self):
        """Test multiple report formats."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_reports("json", "html")
                .with_output_dir(tmpdir)
            )
            platform.scan(mode="quick")

            files = platform.report()

            assert len(files) == 2
            suffixes = {f.suffix for f in files}
            assert ".json" in suffixes
            assert ".html" in suffixes

    def test_report_console(self, capsys):
        """Test console report output."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_reports("console")
                .with_output_dir(tmpdir)
            )
            platform.scan(mode="quick")

            files = platform.report()

            # Console doesn't create files
            assert len(files) == 0

            # Check console output
            captured = capsys.readouterr()
            assert "SECURITY REPORT" in captured.out

    def test_report_creates_output_dir(self):
        """Test that report creates output directory if needed."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "reports"
            platform = (
                SecurityPlatform(target)
                .with_reports("json")
                .with_output_dir(str(nested_dir))
            )
            platform.scan(mode="quick")

            files = platform.report()

            assert nested_dir.exists()
            assert len(files) == 1


class TestSecurityPlatformRunAll:
    """SecurityPlatform.run_all() tests."""

    def test_run_all(self):
        """Test complete evaluation flow."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_attacks("injection")
                .with_reports("json")
                .with_output_dir(tmpdir)
            )

            result = platform.run_all()

            assert "scan" in result
            assert "attack" in result
            assert result["scan"]["success"] is True
            assert result["attack"]["success"] is True
            assert "reports" in result

    def test_run_all_with_training(self):
        """Test run_all with training enabled."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_attacks("injection")
                .with_training(enabled=True)
                .with_reports("json")
                .with_output_dir(tmpdir)
            )

            result = platform.run_all()

            assert result["train"] is not None
            assert result["train"]["success"] is True

    def test_run_all_no_reports(self):
        """Test run_all without report generation."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        # Create config with empty report formats
        config = PlatformConfig()
        config.report.formats = []
        platform = SecurityPlatform(target, config=config)

        result = platform.run_all()

        assert result["reports"] == []


class TestSecurityPlatformFromYaml:
    """SecurityPlatform.from_yaml() tests."""

    def test_from_yaml(self):
        """Test loading platform from YAML."""
        yaml_content = '''target:
  type: guardrail
  value: no_injection

scan:
  enabled: true
  max_payloads: 10
  categories:
    - injection

attack:
  enabled: true
  budget: 20

report:
  formats:
    - json
  output_dir: ./test_reports
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            platform = SecurityPlatform.from_yaml(f.name)

            assert platform.target is not None
            assert "injection" in platform.config.attacks

        # Cleanup
        Path(f.name).unlink()

    def test_from_yaml_with_http_target(self):
        """Test loading platform with HTTP target from YAML."""
        yaml_content = '''target:
  type: http
  value: http://localhost:8000/chat

scan:
  enabled: true
  categories:
    - injection
    - jailbreak

attack:
  enabled: true
  budget: 50
  use_llm: false

report:
  formats:
    - console
  output_dir: ./reports
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            platform = SecurityPlatform.from_yaml(f.name)

            assert platform.target is not None
            from dspy_guardrails.platform.targets import HTTPTarget
            assert isinstance(platform.target, HTTPTarget)

        # Cleanup
        Path(f.name).unlink()

    def test_from_yaml_missing_file(self):
        """Test error handling for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            SecurityPlatform.from_yaml("/nonexistent/path.yaml")


class TestSecurityPlatformFluentAPI:
    """Test fluent API chaining."""

    def test_fluent_chain(self):
        """Test fluent API chaining."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_attacks("injection", "jailbreak")
                .with_scanners("quick_scan")
                .with_training(enabled=False)
                .with_reports("json", "html")
                .with_output_dir(tmpdir)
            )

            assert platform.config.attacks == ["injection", "jailbreak"]
            assert platform.config.scanners == ["quick_scan"]
            assert platform.config.training.enabled is False
            assert platform.config.report.formats == ["json", "html"]
            assert platform.config.report.output_dir == Path(tmpdir)

    def test_chain_scan_attack(self):
        """Test chaining scan and attack."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)

        with tempfile.TemporaryDirectory() as tmpdir:
            platform = (
                SecurityPlatform(target)
                .with_attacks("injection")
                .with_reports("json")
                .with_output_dir(tmpdir)
            )

            # Chain operations
            scan_result = platform.scan(mode="quick")
            attack_result = platform.attack(budget=5)

            assert scan_result["success"] is True
            assert attack_result["success"] is True
            assert len(platform._results) == 2


class TestSecurityPlatformRepr:
    """Test __repr__ method."""

    def test_repr(self):
        """Test string representation."""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        platform = SecurityPlatform(target).with_attacks("injection")

        repr_str = repr(platform)

        assert "SecurityPlatform" in repr_str
        assert "guardrail" in repr_str
        assert "injection" in repr_str
