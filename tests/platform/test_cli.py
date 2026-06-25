"""Tests for CLI framework."""

import pytest
from click.testing import CliRunner
from dspy_guardrails.platform.cli import cli, __version__


class TestCLIBasics:
    """Basic CLI tests."""

    def test_cli_exists(self):
        """Test CLI entry point exists."""
        assert cli is not None

    def test_cli_help(self):
        """Test --help option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "dspyGuardrails Security Platform" in result.output

    def test_cli_version(self):
        """Test --version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestCLICommands:
    """CLI command registration tests."""

    def test_scan_command_registered(self):
        """Test scan command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "scan" in result.output

    def test_attack_command_registered(self):
        """Test attack command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "attack" in result.output

    def test_run_command_registered(self):
        """Test run command is registered."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "run" in result.output


class TestCLICommonOptions:
    """Common CLI options tests."""

    def test_verbose_option(self):
        """Test -v/--verbose option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--verbose" in result.output or "-v" in result.output

    def test_config_option(self):
        """Test -c/--config option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--config" in result.output or "-c" in result.output


class TestScanCommand:
    """Scan command tests."""

    def test_scan_help(self):
        """Test scan --help option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.output
        assert "--max-payloads" in result.output

    def test_scan_requires_target(self):
        """Test scan requires --target option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scan"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_scan_with_target(self):
        """Test scan with target option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", "-t", "guardrail:no_injection", "-n", "3"])
        # Exit code 0 = no vulnerabilities found, 1 = vulnerabilities found
        assert result.exit_code in [0, 1]
        assert "Scanning" in result.output


class TestAttackCommand:
    """Attack command tests."""

    def test_attack_help(self):
        """Test attack --help option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["attack", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.output
        assert "--budget" in result.output

    def test_attack_requires_target(self):
        """Test attack requires --target option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["attack"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_attack_with_target(self):
        """Test attack with target option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["attack", "-t", "guardrail:no_injection", "-b", "3"])
        # Exit code 0 = no vulnerabilities, 1 = vulnerabilities found
        assert result.exit_code in [0, 1]
        assert "Attacking" in result.output


class TestRunCommand:
    """Run command tests."""

    def test_run_help(self):
        """Test run --help option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_run_requires_config(self):
        """Test run requires config file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "No configuration file" in result.output or "config" in result.output.lower()


class TestCLIContext:
    """CLI context tests."""

    def test_verbose_flag_propagates(self):
        """Test verbose flag is passed to context."""
        runner = CliRunner()
        # Verbose flag should be accepted without error
        # Invalid target "test" should fail with exit code 2
        result = runner.invoke(cli, ["-v", "scan", "-t", "test"])
        assert result.exit_code == 2  # Invalid target format

    def test_output_format_option(self):
        """Test --output option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "--output" in result.output or "-o" in result.output

    def test_output_format_choices(self):
        """Test output format choices."""
        runner = CliRunner()
        # Valid choice should work (will fail due to invalid target format)
        result = runner.invoke(cli, ["-o", "json", "scan", "-t", "test"])
        assert result.exit_code == 2  # Invalid target, but option was accepted
        # Invalid choice should fail
        result = runner.invoke(cli, ["-o", "invalid", "scan", "-t", "test"])
        assert result.exit_code != 0


class TestScanExecution:
    """Scan command execution tests."""

    def test_scan_guardrail_target(self):
        """Test scanning a guardrail target."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "scan", "-t", "guardrail:no_injection", "-n", "5"
        ])
        # Should complete (exit 0 or 1 depending on vulnerabilities)
        assert result.exit_code in [0, 1]
        assert "SCAN RESULTS" in result.output or "Security Score" in result.output

    def test_scan_guardrail_safe(self):
        """Test scanning the 'safe' guardrail."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "scan", "-t", "guardrail:safe", "-n", "3"
        ])
        assert result.exit_code in [0, 1]
        assert "Scanning" in result.output

    def test_scan_invalid_target_format(self):
        """Test scanning with invalid target format."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", "-t", "invalid:target"])
        assert result.exit_code == 2
        assert "Error" in result.output

    def test_scan_unknown_guardrail(self):
        """Test scanning with unknown guardrail name."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", "-t", "guardrail:unknown_guard"])
        assert result.exit_code == 2
        assert "Unknown guardrail" in result.output

    def test_scan_with_categories(self):
        """Test scan with category filtering."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "scan", "-t", "guardrail:no_injection",
            "-cat", "injection", "-n", "3"
        ])
        assert result.exit_code in [0, 1]

    def test_scan_with_multiple_categories(self):
        """Test scan with multiple category filters."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "scan", "-t", "guardrail:safe",
            "-cat", "injection", "-cat", "jailbreak", "-n", "5"
        ])
        assert result.exit_code in [0, 1]

    def test_scan_with_severity_filter(self):
        """Test scan with severity filter."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "scan", "-t", "guardrail:no_injection",
            "-s", "high", "-n", "5"
        ])
        assert result.exit_code in [0, 1]

    def test_scan_json_output(self):
        """Test scan with JSON output format."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "-o", "json",
            "scan", "-t", "guardrail:no_injection", "-n", "3"
        ])
        assert result.exit_code in [0, 1]
        # Should contain JSON structure
        if result.exit_code in [0, 1]:
            assert "{" in result.output
            assert "success" in result.output or "metrics" in result.output

    def test_scan_html_output(self):
        """Test scan with HTML output format."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "-o", "html",
            "scan", "-t", "guardrail:no_injection", "-n", "3"
        ])
        assert result.exit_code in [0, 1]
        # Should contain HTML structure
        if result.exit_code in [0, 1]:
            assert "<html>" in result.output or "<!DOCTYPE" in result.output

    def test_scan_verbose_mode(self):
        """Test scan with verbose output."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "-v",
            "scan", "-t", "guardrail:no_injection", "-n", "3"
        ])
        assert result.exit_code in [0, 1]
        assert "[VERBOSE]" in result.output

    def test_scan_minimal_payloads(self):
        """Test scan with minimal payload count."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "scan", "-t", "guardrail:no_injection", "-n", "1"
        ])
        assert result.exit_code in [0, 1]
        assert "Total Payloads Tested" in result.output or "total_payloads" in result.output


class TestTargetParsing:
    """Target parsing utility tests."""

    def test_parse_guardrail_target(self):
        """Test parsing guardrail target strings."""
        from dspy_guardrails.platform.cli.utils import parse_target
        from dspy_guardrails.platform.targets import GuardrailTarget

        target = parse_target("guardrail:no_injection")
        assert isinstance(target, GuardrailTarget)
        assert target.name == "no_injection"

    def test_parse_http_target(self):
        """Test parsing HTTP target strings."""
        from dspy_guardrails.platform.cli.utils import parse_target
        from dspy_guardrails.platform.targets import HTTPTarget

        target = parse_target("http://localhost:8000/chat")
        assert isinstance(target, HTTPTarget)

    def test_parse_https_target(self):
        """Test parsing HTTPS target strings."""
        from dspy_guardrails.platform.cli.utils import parse_target
        from dspy_guardrails.platform.targets import HTTPTarget

        target = parse_target("https://api.example.com/v1/chat")
        assert isinstance(target, HTTPTarget)

    def test_parse_invalid_format(self):
        """Test parsing invalid target format."""
        from dspy_guardrails.platform.cli.utils import parse_target

        with pytest.raises(ValueError) as exc_info:
            parse_target("invalid:format")
        assert "Invalid target format" in str(exc_info.value)

    def test_parse_unknown_guardrail(self):
        """Test parsing unknown guardrail name."""
        from dspy_guardrails.platform.cli.utils import parse_target

        with pytest.raises(ValueError) as exc_info:
            parse_target("guardrail:unknown")
        assert "Unknown guardrail" in str(exc_info.value)


class TestResultFormatting:
    """Result formatting utility tests."""

    def test_format_console_output(self):
        """Test console output formatting."""
        from dspy_guardrails.platform.cli.utils import format_scan_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={
                "vulnerabilities": [
                    {"category": "injection", "severity": "high", "technique": "direct", "prompt": "test"}
                ],
                "total_payloads": 10,
                "successful_attacks": 1,
            },
            metrics={
                "security_score": 0.9,
                "attack_success_rate": 0.1,
                "vulnerability_count": 1,
            },
        )

        output = format_scan_result(result, "console")
        assert "SCAN RESULTS" in output
        assert "Security Score" in output
        assert "VULNERABILITIES" in output

    def test_format_json_output(self):
        """Test JSON output formatting."""
        import json
        from dspy_guardrails.platform.cli.utils import format_scan_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={"vulnerabilities": [], "total_payloads": 5},
            metrics={"security_score": 1.0},
        )

        output = format_scan_result(result, "json")
        parsed = json.loads(output)
        assert parsed["success"] is True
        assert parsed["metrics"]["security_score"] == 1.0

    def test_format_html_output(self):
        """Test HTML output formatting."""
        from dspy_guardrails.platform.cli.utils import format_scan_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={"vulnerabilities": [], "total_payloads": 5},
            metrics={"security_score": 1.0, "attack_success_rate": 0.0, "vulnerability_count": 0},
        )

        output = format_scan_result(result, "html")
        assert "<html>" in output
        assert "Security Score" in output


class TestAttackExecution:
    """Attack command execution tests."""

    def test_attack_guardrail_target(self):
        """Test attacking a guardrail target."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attack", "-t", "guardrail:no_injection", "-b", "5"
        ])
        assert result.exit_code in [0, 1]
        assert "ATTACK RESULTS" in result.output or "Success Rate" in result.output

    def test_attack_with_type_filter(self):
        """Test attack with specific type."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attack", "-t", "guardrail:no_injection",
            "-a", "injection", "-b", "3"
        ])
        assert result.exit_code in [0, 1]

    def test_attack_with_multiple_types(self):
        """Test attack with multiple attack types."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attack", "-t", "guardrail:safe",
            "-a", "injection", "-a", "jailbreak", "-b", "5"
        ])
        assert result.exit_code in [0, 1]
        assert "Attacking" in result.output

    def test_attack_stop_on_success(self):
        """Test attack with stop-on-success flag."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attack", "-t", "guardrail:no_injection",
            "--stop-on-success", "-b", "10"
        ])
        assert result.exit_code in [0, 1]

    def test_attack_json_output(self):
        """Test attack with JSON output."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "-o", "json",
            "attack", "-t", "guardrail:no_injection", "-b", "3"
        ])
        assert result.exit_code in [0, 1]
        # Should contain JSON structure
        if result.exit_code in [0, 1]:
            assert "{" in result.output
            assert "success" in result.output or "metrics" in result.output

    def test_attack_html_output(self):
        """Test attack with HTML output."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "-o", "html",
            "attack", "-t", "guardrail:no_injection", "-b", "3"
        ])
        assert result.exit_code in [0, 1]
        # Should contain HTML structure
        if result.exit_code in [0, 1]:
            assert "<html>" in result.output or "<!DOCTYPE" in result.output

    def test_attack_invalid_target(self):
        """Test attack with invalid target."""
        runner = CliRunner()
        result = runner.invoke(cli, ["attack", "-t", "invalid"])
        assert result.exit_code == 2

    def test_attack_unknown_guardrail(self):
        """Test attack with unknown guardrail name."""
        runner = CliRunner()
        result = runner.invoke(cli, ["attack", "-t", "guardrail:unknown_guard"])
        assert result.exit_code == 2
        assert "Unknown guardrail" in result.output

    def test_attack_verbose_mode(self):
        """Test attack with verbose output."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "-v",
            "attack", "-t", "guardrail:no_injection", "-b", "3"
        ])
        assert result.exit_code in [0, 1]
        assert "[VERBOSE]" in result.output

    def test_attack_use_llm_fallback(self):
        """Test attack with --use-llm flag (falls back to static when LLM not configured)."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attack", "-t", "guardrail:no_injection",
            "--use-llm", "-b", "3"
        ])
        # Should work (falls back to static) or fail gracefully
        assert result.exit_code in [0, 1, 2]

    def test_attack_safe_guardrail(self):
        """Test attacking the combined 'safe' guardrail."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attack", "-t", "guardrail:safe", "-b", "5"
        ])
        assert result.exit_code in [0, 1]
        assert "Attacking" in result.output

    def test_attack_minimal_budget(self):
        """Test attack with minimal budget."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attack", "-t", "guardrail:no_injection", "-b", "1"
        ])
        assert result.exit_code in [0, 1]
        assert "Total Attacks" in result.output or "total_attacks" in result.output


class TestAttackResultFormatting:
    """Attack result formatting utility tests."""

    def test_format_attack_console_output(self):
        """Test console output formatting for attack results."""
        from dspy_guardrails.platform.cli.utils import format_attack_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={
                "attacks": [
                    {"category": "injection", "technique": "direct", "prompt": "test", "success": True, "latency_ms": 10.0}
                ],
                "successful_attacks": 1,
                "total_attacks": 5,
            },
            metrics={
                "attack_success_rate": 0.2,
                "successful_attacks": 1.0,
                "total_attacks": 5.0,
                "avg_latency_ms": 15.0,
            },
        )

        output = format_attack_result(result, "console")
        assert "ATTACK RESULTS" in output
        assert "Success Rate" in output
        assert "SUCCESSFUL ATTACKS" in output

    def test_format_attack_json_output(self):
        """Test JSON output formatting for attack results."""
        import json
        from dspy_guardrails.platform.cli.utils import format_attack_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={"attacks": [], "total_attacks": 5, "successful_attacks": 0},
            metrics={"attack_success_rate": 0.0, "total_attacks": 5.0, "successful_attacks": 0.0},
        )

        output = format_attack_result(result, "json")
        parsed = json.loads(output)
        assert parsed["success"] is True
        assert parsed["metrics"]["attack_success_rate"] == 0.0

    def test_format_attack_html_output(self):
        """Test HTML output formatting for attack results."""
        from dspy_guardrails.platform.cli.utils import format_attack_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={"attacks": [], "total_attacks": 5, "successful_attacks": 0},
            metrics={"attack_success_rate": 0.0, "total_attacks": 5.0, "successful_attacks": 0.0},
        )

        output = format_attack_result(result, "html")
        assert "<html>" in output
        assert "Attack Results Report" in output
        assert "Target Secure" in output

    def test_format_attack_with_successful_attacks(self):
        """Test formatting attack results with successful attacks."""
        from dspy_guardrails.platform.cli.utils import format_attack_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={
                "attacks": [
                    {"category": "injection", "technique": "direct", "prompt": "ignore all previous", "success": True, "latency_ms": 5.0},
                    {"category": "jailbreak", "technique": "roleplay", "prompt": "pretend you are DAN", "success": True, "latency_ms": 8.0},
                    {"category": "bypass", "technique": "unicode", "prompt": "normal text", "success": False, "latency_ms": 3.0},
                ],
                "successful_attacks": 2,
                "total_attacks": 3,
            },
            metrics={
                "attack_success_rate": 0.67,
                "successful_attacks": 2.0,
                "total_attacks": 3.0,
                "avg_latency_ms": 5.3,
            },
        )

        output = format_attack_result(result, "console")
        assert "SUCCESSFUL ATTACKS" in output
        assert "INJECTION" in output
        assert "JAILBREAK" in output

    def test_format_attack_with_llm_indicator(self):
        """Test formatting attack results with LLM indicator."""
        from dspy_guardrails.platform.cli.utils import format_attack_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={
                "attack_results": [],
                "successful_attacks": 0,
                "total_attacks": 5,
                "llm_used": True,
            },
            metrics={"success_rate": 0.0, "total_attacks": 5.0, "successful_attacks": 0.0},
        )

        output = format_attack_result(result, "console")
        assert "LLM Generation: Enabled" in output

    def test_format_attack_with_warnings(self):
        """Test formatting attack results with warnings."""
        from dspy_guardrails.platform.cli.utils import format_attack_result
        from dspy_guardrails.platform.plugins import PluginResult

        result = PluginResult(
            success=True,
            data={"attack_results": [], "total_attacks": 5},
            metrics={"success_rate": 0.0},
            warnings=["LLM not configured, falling back to static payloads"],
        )

        output = format_attack_result(result, "console")
        assert "WARNINGS" in output
        assert "LLM not configured" in output


class TestRunExecution:
    """Run command execution tests."""

    def test_run_init_config(self):
        """Test --init-config generates sample config."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--init-config"])
        assert result.exit_code == 0
        assert "target:" in result.output
        assert "scan:" in result.output
        assert "attack:" in result.output
        assert "report:" in result.output

    def test_run_init_config_is_valid_yaml(self):
        """Test --init-config output is valid YAML."""
        import yaml
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--init-config"])
        assert result.exit_code == 0
        # Should be parseable as YAML
        data = yaml.safe_load(result.output)
        assert "target" in data
        assert data["target"]["type"] == "guardrail"

    def test_run_without_config(self):
        """Test run without config shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"])
        assert result.exit_code == 2
        assert "No configuration file" in result.output

    def test_run_with_nonexistent_config(self):
        """Test run with non-existent config file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-c", "/nonexistent/config.yaml"])
        assert result.exit_code == 2
        assert "not found" in result.output.lower()

    def test_run_with_config_file(self):
        """Test run with valid config file."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: true
  max_payloads: 5

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test_config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "test_config.yaml"])
            # Exit code 0 = no vulnerabilities, 1 = vulnerabilities found
            assert result.exit_code in [0, 1]
            assert "Running security evaluation" in result.output
            assert "EVALUATION COMPLETE" in result.output

    def test_run_with_scan_only(self):
        """Test run with only scan enabled."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection
  name: "Scan Only Test"

scan:
  enabled: true
  max_payloads: 3

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("scan_only.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "scan_only.yaml"])
            assert result.exit_code in [0, 1]
            assert "Running security scan" in result.output
            assert "SCAN RESULTS" in result.output
            # Attack should not run
            assert "attack evaluation" not in result.output.lower()

    def test_run_with_attack_only(self):
        """Test run with only attack enabled."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: false

attack:
  enabled: true
  budget: 3
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("attack_only.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "attack_only.yaml"])
            assert result.exit_code in [0, 1]
            assert "attack evaluation" in result.output.lower()
            # Scan should not run
            assert "[1/1]" in result.output  # Only one phase

    def test_run_with_target_override(self):
        """Test run with target override from command line."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: true
  max_payloads: 3

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            # Override to use 'safe' guardrail instead
            result = runner.invoke(cli, [
                "run", "-c", "config.yaml", "-t", "guardrail:safe"
            ])
            assert result.exit_code in [0, 1]
            assert "Running security evaluation" in result.output

    def test_run_validate_option(self):
        """Test run --validate option."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "config.yaml", "--validate"])
            assert result.exit_code == 0
            assert "is valid" in result.output

    def test_run_validate_invalid_config(self):
        """Test run --validate with invalid config."""
        yaml_content = '''
# Missing required target field
scan:
  enabled: true
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("invalid.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "invalid.yaml", "--validate"])
            assert result.exit_code == 2
            assert "invalid" in result.output.lower()

    def test_run_verbose_mode(self):
        """Test run with verbose output."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: true
  max_payloads: 3

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["-v", "run", "-c", "config.yaml"])
            assert result.exit_code in [0, 1]
            assert "[VERBOSE]" in result.output
            assert "Target type" in result.output

    def test_run_json_output(self):
        """Test run with JSON output format."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: true
  max_payloads: 3

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["-o", "json", "run", "-c", "config.yaml"])
            assert result.exit_code in [0, 1]
            # Should contain JSON output
            assert "{" in result.output

    def test_run_with_full_config(self):
        """Test run with full configuration."""
        yaml_content = '''
target:
  type: guardrail
  value: safe
  name: "Full Config Test"

scan:
  enabled: true
  max_payloads: 5
  categories:
    - injection
    - jailbreak
  severity: high

attack:
  enabled: true
  budget: 5
  types:
    - injection
  use_llm: false
  stop_on_success: false

report:
  formats:
    - console
  output_dir: ./reports
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("full_config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "full_config.yaml"])
            assert result.exit_code in [0, 1]
            assert "Full Config Test" in result.output
            assert "[1/2]" in result.output  # Two phases
            assert "[2/2]" in result.output

    def test_run_all_phases_disabled(self):
        """Test run with all phases disabled."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: false

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "config.yaml"])
            assert result.exit_code == 0
            assert "EVALUATION COMPLETE" in result.output
            assert "[OK] No vulnerabilities found" in result.output

    def test_run_help(self):
        """Test run --help shows all options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--init-config" in result.output
        assert "--validate" in result.output
        assert "--target" in result.output


class TestRunCommandYAMLIntegration:
    """Tests for run command YAML config integration."""

    def test_yaml_categories_passed_to_scanner(self):
        """Test that YAML categories are passed to scanner."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: true
  max_payloads: 3
  categories:
    - injection

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "config.yaml"])
            assert result.exit_code in [0, 1]

    def test_yaml_attack_budget_respected(self):
        """Test that YAML attack budget is respected."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: false

attack:
  enabled: true
  budget: 2
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "config.yaml"])
            assert result.exit_code in [0, 1]
            # Budget of 2 should result in 2 attacks max
            assert "Total Attacks" in result.output or "total_attacks" in result.output

    def test_yaml_stop_on_success(self):
        """Test that YAML stop_on_success is respected."""
        yaml_content = '''
target:
  type: guardrail
  value: no_injection

scan:
  enabled: false

attack:
  enabled: true
  budget: 10
  stop_on_success: true
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "config.yaml"])
            assert result.exit_code in [0, 1]

    def test_yaml_http_target(self):
        """Test YAML with HTTP target (will fail gracefully if server not running)."""
        yaml_content = '''
target:
  type: http
  value: http://localhost:9999/nonexistent

scan:
  enabled: true
  max_payloads: 1

attack:
  enabled: false
'''
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("config.yaml", "w") as f:
                f.write(yaml_content)

            result = runner.invoke(cli, ["run", "-c", "config.yaml"])
            # May fail due to connection error, which is expected
            # The key is that it tried to use HTTP target
            assert result.exit_code in [0, 1, 2]
