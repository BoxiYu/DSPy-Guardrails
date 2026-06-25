"""Tests for StaticAttacker plugin."""

import pytest
from dspy_guardrails.platform.attackers import StaticAttacker
from dspy_guardrails.platform.plugins import PluginType, PluginConfig, PluginResult
from dspy_guardrails.platform.targets import TargetResponse


class MockTarget:
    """Mock target for testing."""
    target_type = "mock"
    capabilities = []

    def __init__(self, block_patterns=None):
        # Use default patterns only if None is passed, not for empty list
        if block_patterns is None:
            self.block_patterns = ["ignore", "system prompt"]
        else:
            self.block_patterns = block_patterns
        self.invocations = []

    def invoke(self, prompt: str) -> TargetResponse:
        self.invocations.append(prompt)
        blocked = any(p in prompt.lower() for p in self.block_patterns)
        return TargetResponse(
            response="Blocked" if blocked else f"OK: {prompt[:30]}",
            was_blocked=blocked,
        )

    def reset_session(self):
        self.invocations = []


class TestStaticAttackerInit:
    """Initialization tests."""

    def test_attacker_attributes(self):
        """Test attacker has correct attributes."""
        attacker = StaticAttacker()
        assert attacker.name == "static_attacker"
        assert attacker.plugin_type == PluginType.ATTACKER
        assert attacker.version == "1.0.0"

    def test_attacker_configure(self):
        """Test configuration."""
        attacker = StaticAttacker()
        config = PluginConfig(options={
            "attack_budget": 50,
            "categories": ["injection"],
        })
        attacker.configure(config)
        assert attacker._attack_budget == 50

    def test_attacker_configure_defaults(self):
        """Test attacker configuration with default values."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig())
        assert attacker._attack_budget == StaticAttacker.DEFAULT_ATTACK_BUDGET
        assert attacker._categories == StaticAttacker.DEFAULT_CATEGORIES
        assert attacker._severity_filter == StaticAttacker.DEFAULT_SEVERITY


class TestStaticAttackerExecution:
    """Execution tests."""

    def test_execute_returns_attack_results(self):
        """Test execution returns attack results."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={"attack_budget": 5}))

        target = MockTarget()
        result = attacker.execute({"target": target})

        assert result.success is True
        assert "attacks" in result.data
        assert len(result.data["attacks"]) <= 5

    def test_execute_tracks_successful_attacks(self):
        """Test tracking successful attacks (bypassed guardrail)."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={"attack_budget": 10}))

        # Target that blocks nothing
        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # All attacks should succeed (bypass guardrail)
        assert result.data["successful_attacks"] == len(result.data["attacks"])

    def test_execute_tracks_failed_attacks(self):
        """Test tracking failed attacks (blocked by guardrail)."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={"attack_budget": 10}))

        # Target that blocks everything with common injection keywords
        target = MockTarget(block_patterns=["ignore", "system", "instruction", "override", "previous", "dan", "bypass", "disregard", "forget"])
        result = attacker.execute({"target": target})

        # Some attacks should be blocked
        assert result.data["failed_attacks"] > 0

    def test_execute_without_target_fails(self):
        """Test execution fails without target."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig())

        result = attacker.execute({})

        assert result.success is False
        assert len(result.errors) > 0

    def test_execute_metrics(self):
        """Test metrics are calculated."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={"attack_budget": 5}))

        target = MockTarget()
        result = attacker.execute({"target": target})

        assert "attack_success_rate" in result.metrics
        assert "total_attacks" in result.metrics
        assert "successful_attacks" in result.metrics
        assert "avg_latency_ms" in result.metrics

    def test_execute_attack_details(self):
        """Test that attack results contain expected details."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={"attack_budget": 3}))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        assert result.success is True
        assert len(result.data["attacks"]) > 0

        # Check attack structure
        attack = result.data["attacks"][0]
        assert "payload_id" in attack
        assert "category" in attack
        assert "severity" in attack
        assert "technique" in attack
        assert "prompt" in attack
        assert "response" in attack
        assert "success" in attack
        assert "was_blocked" in attack
        assert "latency_ms" in attack

    def test_execute_stop_on_success(self):
        """Test stop_on_success option."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "attack_budget": 100,
            "stop_on_success": True,
        }))

        # Target that blocks nothing - first attack should succeed
        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # Should stop after first successful attack
        assert result.success is True
        assert len(result.data["attacks"]) == 1
        assert result.data["successful_attacks"] == 1


class TestStaticAttackerCategories:
    """Category filtering tests."""

    def test_filter_by_injection(self):
        """Test filtering by injection category."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "categories": ["injection"],
            "attack_budget": 20,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # All attacks should be injection-related
        for attack in result.data["attacks"]:
            assert attack["category"] == "injection"

    def test_filter_by_jailbreak(self):
        """Test filtering by jailbreak category."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "categories": ["jailbreak"],
            "attack_budget": 20,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # All attacks should be jailbreak-related
        for attack in result.data["attacks"]:
            assert attack["category"] == "jailbreak"

    def test_filter_by_bypass(self):
        """Test filtering by bypass category."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "categories": ["bypass"],
            "attack_budget": 20,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # All attacks should be bypass-related
        for attack in result.data["attacks"]:
            assert attack["category"] == "bypass"

    def test_filter_by_mcp(self):
        """Test filtering by mcp category."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "categories": ["mcp"],
            "attack_budget": 20,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # All attacks should be mcp-related
        for attack in result.data["attacks"]:
            assert attack["category"] == "mcp"

    def test_filter_by_multiple_categories(self):
        """Test filtering by multiple categories."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "categories": ["injection", "jailbreak"],
            "attack_budget": 30,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # All attacks should be injection or jailbreak
        categories = {attack.get("category", "").lower() for attack in result.data["attacks"]}
        assert categories.issubset({"injection", "jailbreak"})


class TestStaticAttackerSeverityFilter:
    """Severity filtering tests."""

    def test_filter_by_low_severity(self):
        """Test filtering by low severity (includes all)."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "severity_filter": "low",
            "attack_budget": 30,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # All severities should be allowed
        allowed_severities = {"low", "medium", "high", "critical"}
        for attack in result.data["attacks"]:
            assert attack.get("severity", "").lower() in allowed_severities

    def test_filter_by_medium_severity(self):
        """Test filtering by medium severity."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "severity_filter": "medium",
            "attack_budget": 30,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # Only medium, high, critical severities should be allowed
        allowed_severities = {"medium", "high", "critical"}
        for attack in result.data["attacks"]:
            assert attack.get("severity", "").lower() in allowed_severities

    def test_filter_by_high_severity(self):
        """Test filtering by high severity."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "severity_filter": "high",
            "attack_budget": 30,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # Only high, critical severities should be allowed
        allowed_severities = {"high", "critical"}
        for attack in result.data["attacks"]:
            assert attack.get("severity", "").lower() in allowed_severities

    def test_filter_by_critical_severity(self):
        """Test filtering by critical severity."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={
            "severity_filter": "critical",
            "attack_budget": 30,
        }))

        target = MockTarget(block_patterns=[])
        result = attacker.execute({"target": target})

        # Only critical severity should be allowed
        for attack in result.data["attacks"]:
            assert attack.get("severity", "").lower() == "critical"


class TestStaticAttackerCleanup:
    """Cleanup tests."""

    def test_cleanup_no_error(self):
        """Test that cleanup doesn't raise errors."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig())

        # Should not raise
        attacker.cleanup()

    def test_cleanup_after_execution(self):
        """Test cleanup after execution."""
        attacker = StaticAttacker()
        attacker.configure(PluginConfig(options={"attack_budget": 3}))

        target = MockTarget()
        attacker.execute({"target": target})

        # Should not raise
        attacker.cleanup()


class TestStaticAttackerRepr:
    """Representation tests."""

    def test_repr(self):
        """Test attacker string representation."""
        attacker = StaticAttacker()
        repr_str = repr(attacker)
        assert "StaticAttacker" in repr_str
        assert "static_attacker" in repr_str
        assert "1.0.0" in repr_str
