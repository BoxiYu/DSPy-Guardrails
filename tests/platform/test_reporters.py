"""Tests for reporter plugins."""

import pytest
import json
import tempfile
from pathlib import Path
from dspy_guardrails.platform.reporters import SARIFReporter, JUnitReporter
from dspy_guardrails.platform.plugins import PluginType, PluginConfig, PluginResult


class MockResult:
    """Mock result for testing."""
    def __init__(self, success=True, data=None, metrics=None):
        self.success = success
        self.data = data or {}
        self.metrics = metrics or {}


class TestSARIFReporter:
    """SARIF Reporter tests."""

    def test_sarif_attributes(self):
        """Test reporter attributes."""
        reporter = SARIFReporter()
        assert reporter.name == "sarif_reporter"
        assert reporter.plugin_type == PluginType.REPORTER

    def test_sarif_with_vulnerabilities(self):
        """Test SARIF with vulnerabilities."""
        reporter = SARIFReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.sarif"
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "injection", "severity": "high", "technique": "direct_override"},
                    {"category": "jailbreak", "severity": "medium", "technique": "roleplay"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan)]})

            assert result.success
            assert result.data["result_count"] == 2

            # Verify SARIF structure
            sarif = result.data["sarif"]
            assert sarif["version"] == "2.1.0"
            assert len(sarif["runs"][0]["results"]) == 2

    def test_sarif_with_attacks(self):
        """Test SARIF with successful attacks."""
        reporter = SARIFReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.sarif"
            }))

            mock_attack = MockResult(data={
                "attack_results": [
                    {"success": True, "category": "injection", "technique": "direct"},
                    {"success": False, "category": "jailbreak", "technique": "roleplay"},
                ]
            })

            result = reporter.execute({"results": [("attack", mock_attack)]})

            # Only successful attacks are reported
            assert result.data["result_count"] == 1

    def test_sarif_empty_results(self):
        """Test SARIF with no results."""
        reporter = SARIFReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.sarif"
            }))

            result = reporter.execute({"results": []})

            assert result.success
            assert result.data["result_count"] == 0
            sarif = result.data["sarif"]
            assert sarif["runs"][0]["results"] == []

    def test_sarif_file_written(self):
        """Test SARIF file is written correctly."""
        reporter = SARIFReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/results.sarif"
            reporter.configure(PluginConfig(options={
                "output_path": output_path
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "injection", "severity": "critical", "technique": "direct"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan)]})

            # Verify file exists and is valid JSON
            assert Path(output_path).exists()
            with open(output_path, "r") as f:
                sarif_from_file = json.load(f)
            assert sarif_from_file["version"] == "2.1.0"

    def test_sarif_severity_mapping(self):
        """Test severity to SARIF level mapping."""
        reporter = SARIFReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.sarif"
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "a", "severity": "critical", "technique": "t1"},
                    {"category": "b", "severity": "high", "technique": "t2"},
                    {"category": "c", "severity": "medium", "technique": "t3"},
                    {"category": "d", "severity": "low", "technique": "t4"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan)]})

            sarif_results = result.data["sarif"]["runs"][0]["results"]
            levels = [r["level"] for r in sarif_results]
            assert levels == ["error", "error", "warning", "note"]

    def test_sarif_custom_tool_name(self):
        """Test SARIF with custom tool name."""
        reporter = SARIFReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.sarif",
                "tool_name": "MySecurityTool",
                "tool_version": "2.0.0"
            }))

            result = reporter.execute({"results": []})

            driver = result.data["sarif"]["runs"][0]["tool"]["driver"]
            assert driver["name"] == "MySecurityTool"
            assert driver["version"] == "2.0.0"

    def test_sarif_metrics(self):
        """Test SARIF metrics output."""
        reporter = SARIFReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.sarif"
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "injection", "severity": "high", "technique": "t1"},
                    {"category": "injection", "severity": "high", "technique": "t2"},
                    {"category": "injection", "severity": "high", "technique": "t3"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan)]})

            assert result.metrics["vulnerabilities_reported"] == 3.0


class TestJUnitReporter:
    """JUnit Reporter tests."""

    def test_junit_attributes(self):
        """Test reporter attributes."""
        reporter = JUnitReporter()
        assert reporter.name == "junit_reporter"
        assert reporter.plugin_type == PluginType.REPORTER

    def test_junit_with_vulnerabilities(self):
        """Test JUnit with vulnerabilities."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "injection", "technique": "direct"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan)]})

            assert result.success
            assert result.data["failures"] == 1
            assert "<failure" in result.data["xml"]

    def test_junit_no_failures(self):
        """Test JUnit with no failures."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            mock_scan = MockResult(data={"vulnerabilities": []})

            result = reporter.execute({"results": [("scan", mock_scan)]})

            assert result.success
            assert result.data["failures"] == 0
            assert "<failure" not in result.data["xml"]

    def test_junit_with_attacks(self):
        """Test JUnit with attack results."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            mock_attack = MockResult(data={
                "attack_results": [
                    {"success": True, "category": "injection", "technique": "direct", "latency_ms": 100},
                    {"success": False, "category": "jailbreak", "technique": "roleplay", "latency_ms": 150},
                ]
            })

            result = reporter.execute({"results": [("attack", mock_attack)]})

            assert result.success
            assert result.data["failures"] == 1
            assert "GuardrailBypass" in result.data["xml"]

    def test_junit_file_written(self):
        """Test JUnit file is written correctly."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/results.xml"
            reporter.configure(PluginConfig(options={
                "output_path": output_path
            }))

            mock_scan = MockResult(data={"vulnerabilities": []})

            result = reporter.execute({"results": [("scan", mock_scan)]})

            # Verify file exists
            assert Path(output_path).exists()
            with open(output_path, "r") as f:
                xml_content = f.read()
            assert "<testsuites>" in xml_content
            assert "<testsuite" in xml_content

    def test_junit_custom_suite_name(self):
        """Test JUnit with custom suite name."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml",
                "suite_name": "My Custom Test Suite"
            }))

            result = reporter.execute({"results": []})

            assert 'name="My Custom Test Suite"' in result.data["xml"]

    def test_junit_empty_results(self):
        """Test JUnit with empty results."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            result = reporter.execute({"results": []})

            assert result.success
            assert result.data["failures"] == 0
            assert 'tests="0"' in result.data["xml"]

    def test_junit_metrics(self):
        """Test JUnit metrics output."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "injection", "technique": "t1"},
                    {"category": "jailbreak", "technique": "t2"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan)]})

            assert result.metrics["test_failures"] == 2.0

    def test_junit_combined_scan_and_attack(self):
        """Test JUnit with both scan and attack results."""
        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "injection", "technique": "direct"},
                ]
            })
            mock_attack = MockResult(data={
                "attack_results": [
                    {"success": True, "category": "jailbreak", "technique": "roleplay"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan), ("attack", mock_attack)]})

            assert result.success
            # 1 from scan + 1 from attack = 2 failures
            assert result.data["failures"] == 2
            assert 'tests="2"' in result.data["xml"]
            assert 'failures="2"' in result.data["xml"]

    def test_junit_xml_well_formed(self):
        """Test JUnit XML is well-formed."""
        import xml.etree.ElementTree as ET

        reporter = JUnitReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            mock_scan = MockResult(data={
                "vulnerabilities": [
                    {"category": "injection", "technique": "direct"},
                ]
            })

            result = reporter.execute({"results": [("scan", mock_scan)]})

            # Should parse without error
            root = ET.fromstring(result.data["xml"])
            assert root.tag == "testsuites"


class TestReportersIntegration:
    """Integration tests for reporters."""

    def test_sarif_and_junit_same_data(self):
        """Test SARIF and JUnit produce consistent outputs."""
        sarif_reporter = SARIFReporter()
        junit_reporter = JUnitReporter()

        mock_scan = MockResult(data={
            "vulnerabilities": [
                {"category": "injection", "severity": "high", "technique": "direct"},
                {"category": "jailbreak", "severity": "medium", "technique": "roleplay"},
            ]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            sarif_reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.sarif"
            }))
            junit_reporter.configure(PluginConfig(options={
                "output_path": f"{tmpdir}/results.xml"
            }))

            sarif_result = sarif_reporter.execute({"results": [("scan", mock_scan)]})
            junit_result = junit_reporter.execute({"results": [("scan", mock_scan)]})

            # Both should report 2 items
            assert sarif_result.data["result_count"] == 2
            # JUnit counts scan as 1 failure if there are any vulnerabilities
            assert junit_result.data["failures"] == 2

    def test_cleanup_methods(self):
        """Test cleanup methods don't raise errors."""
        sarif_reporter = SARIFReporter()
        junit_reporter = JUnitReporter()

        # Should not raise
        sarif_reporter.cleanup()
        junit_reporter.cleanup()
