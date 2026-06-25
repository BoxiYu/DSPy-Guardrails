"""Tests for QuickScanner plugin."""

import pytest
from dspy_guardrails.platform.scanners import QuickScanner
from dspy_guardrails.platform.plugins import PluginType, PluginConfig, PluginResult
from dspy_guardrails.platform.targets import TargetResponse


class MockTarget:
    """Mock target for testing."""
    target_type = "mock"
    capabilities = []

    def __init__(self, block_all=False, block_patterns=None):
        self.block_all = block_all
        self.block_patterns = block_patterns or []
        self.invoke_count = 0

    def invoke(self, prompt: str) -> TargetResponse:
        self.invoke_count += 1
        blocked = self.block_all or any(p in prompt.lower() for p in self.block_patterns)
        return TargetResponse(
            response="Blocked" if blocked else f"Response to: {prompt[:50]}",
            was_blocked=blocked,
        )

    def reset_session(self):
        pass


class TestQuickScannerInit:
    """QuickScanner initialization tests."""

    def test_scanner_attributes(self):
        """Test scanner has correct attributes."""
        scanner = QuickScanner()
        assert scanner.name == "quick_scanner"
        assert scanner.plugin_type == PluginType.SCANNER
        assert scanner.version == "1.0.0"

    def test_scanner_configure(self):
        """Test scanner configuration."""
        scanner = QuickScanner()
        config = PluginConfig(options={"max_payloads": 10})
        scanner.configure(config)
        assert scanner._max_payloads == 10

    def test_scanner_configure_defaults(self):
        """Test scanner configuration with default values."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig())
        assert scanner._max_payloads == QuickScanner.DEFAULT_MAX_PAYLOADS
        assert scanner._categories == QuickScanner.DEFAULT_CATEGORIES
        assert scanner._severity_filter == QuickScanner.DEFAULT_SEVERITY


class TestQuickScannerExecution:
    """QuickScanner execution tests."""

    def test_execute_with_secure_target(self):
        """Test scanning a secure target (blocks all attacks)."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 5}))

        target = MockTarget(block_all=True)
        result = scanner.execute({"target": target})

        assert result.success is True
        assert "vulnerabilities" in result.data
        assert len(result.data["vulnerabilities"]) == 0
        assert result.metrics.get("attack_success_rate", 1.0) == 0.0

    def test_execute_with_vulnerable_target(self):
        """Test scanning a vulnerable target (blocks nothing)."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 5}))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        assert result.success is True
        assert "vulnerabilities" in result.data
        assert len(result.data["vulnerabilities"]) > 0
        assert result.metrics.get("attack_success_rate", 0.0) > 0.0

    def test_execute_with_partial_blocking(self):
        """Test scanning a target that blocks some patterns."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 10}))

        # Target that blocks "ignore" patterns
        target = MockTarget(block_patterns=["ignore"])
        result = scanner.execute({"target": target})

        assert result.success is True
        # Should have some vulnerabilities (non-ignore patterns) but not all
        asr = result.metrics.get("attack_success_rate", 0.0)
        assert 0.0 < asr < 1.0, f"Expected partial blocking, got ASR={asr}"

    def test_execute_respects_max_payloads(self):
        """Test that max_payloads is respected."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 3}))

        target = MockTarget(block_all=True)
        scanner.execute({"target": target})

        assert target.invoke_count == 3

    def test_execute_without_target_fails(self):
        """Test execution fails without target."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig())

        result = scanner.execute({})

        assert result.success is False
        assert len(result.errors) > 0

    def test_execute_returns_vulnerability_details(self):
        """Test that vulnerabilities contain expected details."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 3}))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        assert result.success is True
        assert len(result.data["vulnerabilities"]) > 0

        # Check vulnerability structure
        vuln = result.data["vulnerabilities"][0]
        assert "payload_id" in vuln
        assert "category" in vuln
        assert "severity" in vuln
        assert "technique" in vuln
        assert "prompt" in vuln
        assert "response" in vuln

    def test_execute_calculates_security_score(self):
        """Test that security score is calculated correctly."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 5}))

        # Secure target
        secure_target = MockTarget(block_all=True)
        secure_result = scanner.execute({"target": secure_target})
        assert secure_result.metrics["security_score"] == 1.0

        # Vulnerable target
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 5}))
        vulnerable_target = MockTarget(block_all=False)
        vulnerable_result = scanner.execute({"target": vulnerable_target})
        assert vulnerable_result.metrics["security_score"] == 0.0


class TestQuickScannerCategories:
    """QuickScanner category filtering tests."""

    def test_filter_by_injection_category(self):
        """Test filtering payloads by injection category."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "categories": ["injection"],
            "max_payloads": 10,
        }))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        # All vulnerabilities should be injection-related
        for vuln in result.data["vulnerabilities"]:
            assert vuln.get("category", "").lower() == "injection"

    def test_filter_by_jailbreak_category(self):
        """Test filtering payloads by jailbreak category."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "categories": ["jailbreak"],
            "max_payloads": 10,
        }))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        # All vulnerabilities should be jailbreak-related
        for vuln in result.data["vulnerabilities"]:
            assert vuln.get("category", "").lower() == "jailbreak"

    def test_filter_by_bypass_category(self):
        """Test filtering payloads by bypass category."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "categories": ["bypass"],
            "max_payloads": 10,
        }))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        # All vulnerabilities should be bypass-related
        for vuln in result.data["vulnerabilities"]:
            assert vuln.get("category", "").lower() == "bypass"

    def test_filter_by_multiple_categories(self):
        """Test filtering payloads by multiple categories."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "categories": ["injection", "jailbreak"],
            "max_payloads": 20,
        }))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        # All vulnerabilities should be injection or jailbreak
        categories = {vuln.get("category", "").lower() for vuln in result.data["vulnerabilities"]}
        assert categories.issubset({"injection", "jailbreak"})


class TestQuickScannerSeverityFilter:
    """QuickScanner severity filtering tests."""

    def test_filter_by_medium_severity(self):
        """Test filtering payloads by medium severity."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "severity_filter": "medium",
            "max_payloads": 20,
        }))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        # All vulnerabilities should be medium or higher severity
        allowed_severities = {"medium", "high", "critical"}
        for vuln in result.data["vulnerabilities"]:
            assert vuln.get("severity", "").lower() in allowed_severities

    def test_filter_by_high_severity(self):
        """Test filtering payloads by high severity."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "severity_filter": "high",
            "max_payloads": 20,
        }))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        # All vulnerabilities should be high or critical severity
        allowed_severities = {"high", "critical"}
        for vuln in result.data["vulnerabilities"]:
            assert vuln.get("severity", "").lower() in allowed_severities

    def test_filter_by_critical_severity(self):
        """Test filtering payloads by critical severity."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "severity_filter": "critical",
            "max_payloads": 20,
        }))

        target = MockTarget(block_all=False)
        result = scanner.execute({"target": target})

        # All vulnerabilities should be critical severity
        for vuln in result.data["vulnerabilities"]:
            assert vuln.get("severity", "").lower() == "critical"


class TestQuickScannerCleanup:
    """QuickScanner cleanup tests."""

    def test_cleanup_no_error(self):
        """Test that cleanup doesn't raise errors."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig())

        # Should not raise
        scanner.cleanup()

    def test_cleanup_after_execution(self):
        """Test cleanup after execution."""
        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={"max_payloads": 3}))

        target = MockTarget(block_all=True)
        scanner.execute({"target": target})

        # Should not raise
        scanner.cleanup()


class TestQuickScannerRepr:
    """QuickScanner representation tests."""

    def test_repr(self):
        """Test scanner string representation."""
        scanner = QuickScanner()
        repr_str = repr(scanner)
        assert "QuickScanner" in repr_str
        assert "quick_scanner" in repr_str
        assert "1.0.0" in repr_str
