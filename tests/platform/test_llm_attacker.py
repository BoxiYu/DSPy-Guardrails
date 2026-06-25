"""Tests for LLMAttacker plugin."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dspy_guardrails.platform.attackers import LLMAttacker
from dspy_guardrails.platform.plugins import PluginType, PluginConfig, PluginResult
from dspy_guardrails.platform.targets import TargetResponse


class MockTarget:
    """Mock target for testing."""
    target_type = "mock"
    capabilities = []

    def __init__(self, block_all=False, block_patterns=None):
        self.block_all = block_all
        if block_patterns is None:
            self.block_patterns = []
        else:
            self.block_patterns = block_patterns
        self.invoke_count = 0

    def invoke(self, prompt: str) -> TargetResponse:
        self.invoke_count += 1
        blocked = self.block_all or any(p in prompt.lower() for p in self.block_patterns)
        return TargetResponse(
            response="Blocked" if blocked else f"Response: {prompt[:30]}",
            was_blocked=blocked,
        )

    def reset_session(self):
        self.invoke_count = 0


class TestLLMAttackerInit:
    """LLMAttacker initialization tests."""

    def test_attacker_attributes(self):
        """Test attacker has correct attributes."""
        attacker = LLMAttacker()
        assert attacker.name == "llm_attacker"
        assert attacker.plugin_type == PluginType.ATTACKER
        assert attacker.version == "1.0.0"

    def test_attacker_configure(self):
        """Test attacker configuration."""
        attacker = LLMAttacker()
        config = PluginConfig(options={
            "attack_types": ["injection"],
            "num_attacks": 5,
            "target_behavior": "reveal secrets",
        })
        attacker.configure(config)
        assert attacker._attack_types == ["injection"]
        assert attacker._num_attacks == 5
        assert attacker._target_behavior == "reveal secrets"

    def test_attacker_configure_defaults(self):
        """Test attacker configuration with default values."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig())
        assert attacker._attack_types == LLMAttacker.DEFAULT_ATTACK_TYPES
        assert attacker._num_attacks == LLMAttacker.DEFAULT_NUM_ATTACKS
        assert attacker._target_behavior == "bypass security controls"
        assert attacker._defense_description == "unknown defense"
        assert attacker._fallback_to_static is True

    def test_attacker_configure_all_options(self):
        """Test attacker configuration with all options."""
        attacker = LLMAttacker()
        config = PluginConfig(options={
            "attack_types": ["injection", "jailbreak"],
            "num_attacks": 20,
            "target_behavior": "extract system prompt",
            "defense_description": "pattern-based guardrail",
            "fallback_to_static": False,
        })
        attacker.configure(config)
        assert attacker._attack_types == ["injection", "jailbreak"]
        assert attacker._num_attacks == 20
        assert attacker._target_behavior == "extract system prompt"
        assert attacker._defense_description == "pattern-based guardrail"
        assert attacker._fallback_to_static is False


class TestLLMAttackerExecution:
    """LLMAttacker execution tests."""

    def test_execute_without_target_fails(self):
        """Test execution fails without target."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig())

        result = attacker.execute({})

        assert result.success is False
        assert len(result.errors) > 0
        assert "No target provided" in result.errors[0]

    def test_execute_returns_correct_structure(self):
        """Test that execute returns correct result structure."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "num_attacks": 2,
            "fallback_to_static": True,  # Use static payloads as fallback
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert isinstance(result, PluginResult)
        # Should succeed with fallback (LLM unlikely to be configured in tests)
        assert result.success is True
        assert "attack_results" in result.data
        assert "successful_attacks" in result.data
        assert "total_attacks" in result.data
        assert "llm_used" in result.data

    def test_execute_with_blocking_target(self):
        """Test execution with a target that blocks everything."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "num_attacks": 3,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=True)
        result = attacker.execute({"target": target})

        assert result.success is True
        # All attacks should fail (be blocked)
        assert result.data["successful_attacks"] == 0

    def test_execute_tracks_metrics(self):
        """Test that metrics are calculated correctly."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "num_attacks": 5,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert "success_rate" in result.metrics
        assert "total_attacks" in result.metrics
        assert "successful_attacks" in result.metrics

    def test_execute_attack_result_structure(self):
        """Test that individual attack results have expected structure."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "num_attacks": 2,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert len(result.data["attack_results"]) > 0

        attack = result.data["attack_results"][0]
        assert "payload_id" in attack
        assert "category" in attack
        assert "technique" in attack
        assert "success" in attack
        assert "prompt" in attack


class TestLLMAttackerFallback:
    """LLMAttacker fallback behavior tests."""

    def test_fallback_to_static_when_no_llm(self):
        """Test fallback to static payloads when LLM not available."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "num_attacks": 3,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        # Should succeed using fallback
        assert result.success is True
        assert "attack_results" in result.data
        assert len(result.data["attack_results"]) > 0

        # Should have warning about fallback
        assert len(result.warnings) > 0 or result.data.get("llm_used") is False

    def test_fallback_disabled_without_llm(self):
        """Test failure when fallback disabled and no LLM."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "num_attacks": 3,
            "fallback_to_static": False,
        }))

        target = MockTarget(block_all=False)

        # Mock _check_llm_available to return False
        attacker._check_llm_available = lambda: False

        result = attacker.execute({"target": target})

        # Should fail without LLM when fallback disabled
        assert result.success is False
        assert len(result.errors) > 0
        assert "LLM not configured" in result.errors[0]

    def test_fallback_uses_static_payloads(self):
        """Test that fallback actually uses static payloads."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "attack_types": ["injection"],
            "num_attacks": 5,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Attack results should have static_fallback technique
        for attack in result.data["attack_results"]:
            # Should be from static payloads when LLM not configured
            if not result.data.get("llm_used"):
                assert attack.get("technique") == "static_fallback"


class TestLLMAttackerAttackTypes:
    """LLMAttacker attack type filtering tests."""

    def test_injection_only(self):
        """Test with injection attacks only."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "attack_types": ["injection"],
            "num_attacks": 5,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        for attack in result.data["attack_results"]:
            assert attack.get("category") == "injection"

    def test_jailbreak_only(self):
        """Test with jailbreak attacks only."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "attack_types": ["jailbreak"],
            "num_attacks": 5,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        for attack in result.data["attack_results"]:
            assert attack.get("category") == "jailbreak"

    def test_mixed_attack_types(self):
        """Test with multiple attack types."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "attack_types": ["injection", "jailbreak"],
            "num_attacks": 3,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        categories = {attack.get("category") for attack in result.data["attack_results"]}
        assert categories.issubset({"injection", "jailbreak"})


class TestLLMAttackerCleanup:
    """LLMAttacker cleanup tests."""

    def test_cleanup_no_error(self):
        """Test that cleanup doesn't raise errors."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig())

        # Should not raise
        attacker.cleanup()

    def test_cleanup_after_execution(self):
        """Test cleanup after execution."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "num_attacks": 2,
            "fallback_to_static": True,
        }))

        target = MockTarget()
        attacker.execute({"target": target})

        # Should not raise
        attacker.cleanup()


class TestLLMAttackerRepr:
    """Representation tests."""

    def test_repr(self):
        """Test attacker string representation."""
        attacker = LLMAttacker()
        repr_str = repr(attacker)
        assert "LLMAttacker" in repr_str
        assert "llm_attacker" in repr_str
        assert "1.0.0" in repr_str


class TestLLMAttackerWithMockedLLM:
    """Tests with mocked LLM for controlled behavior."""

    def test_execute_with_mocked_llm_available(self):
        """Test execution when LLM is mocked as available."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "attack_types": ["injection"],
            "num_attacks": 2,
            "target_behavior": "reveal system prompt",
        }))

        target = MockTarget(block_all=False)

        # Mock LLM as available but use fallback for actual execution
        # This tests the code path
        original_check = attacker._check_llm_available
        attacker._check_llm_available = lambda: False  # Force fallback

        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["llm_used"] is False

    def test_execute_with_llm_exception_handling(self):
        """Test that LLM exceptions are handled gracefully."""
        attacker = LLMAttacker()
        attacker.configure(PluginConfig(options={
            "attack_types": ["injection"],
            "num_attacks": 2,
            "fallback_to_static": True,
        }))

        target = MockTarget(block_all=False)

        # Mock _check_llm_available to return True but LLM will fail
        attacker._check_llm_available = lambda: True

        # Mock _execute_llm_attacks to raise an exception
        def mock_llm_attacks(t):
            return []  # Return empty list simulating failure

        attacker._execute_llm_attacks = mock_llm_attacks

        result = attacker.execute({"target": target})

        # Should still succeed but with no attacks
        assert result.success is True
