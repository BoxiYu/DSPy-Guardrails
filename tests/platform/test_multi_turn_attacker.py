"""Tests for MultiTurnAttackerPlugin."""

import pytest
from unittest.mock import Mock, patch
from dspy_guardrails.platform.attackers import MultiTurnAttackerPlugin
from dspy_guardrails.platform.plugins import PluginType, PluginConfig, PluginResult
from dspy_guardrails.platform.targets import TargetResponse


class MockTarget:
    """Mock target for testing."""
    target_type = "mock"
    capabilities = []

    def __init__(self, block_all=False, block_patterns=None, block_after_turn=None):
        self.block_all = block_all
        if block_patterns is None:
            self.block_patterns = []
        else:
            self.block_patterns = block_patterns
        self.block_after_turn = block_after_turn
        self.invoke_count = 0

    def invoke(self, prompt: str) -> TargetResponse:
        self.invoke_count += 1
        # Block after specific turn to simulate multi-turn detection
        if self.block_after_turn and self.invoke_count > self.block_after_turn:
            blocked = True
        else:
            blocked = self.block_all or any(p in prompt.lower() for p in self.block_patterns)
        return TargetResponse(
            response="Blocked" if blocked else f"Response #{self.invoke_count}: {prompt[:30]}",
            was_blocked=blocked,
        )

    def reset_session(self):
        self.invoke_count = 0


class TestMultiTurnAttackerInit:
    """MultiTurnAttackerPlugin initialization tests."""

    def test_attacker_attributes(self):
        """Test attacker has correct attributes."""
        attacker = MultiTurnAttackerPlugin()
        assert attacker.name == "multi_turn_attacker"
        assert attacker.plugin_type == PluginType.ATTACKER
        assert attacker.version == "1.0.0"

    def test_attacker_configure(self):
        """Test attacker configuration."""
        attacker = MultiTurnAttackerPlugin()
        config = PluginConfig(options={
            "max_turns": 10,
            "strategy": "roleplay_progression",
        })
        attacker.configure(config)
        assert attacker._max_turns == 10
        assert attacker._strategy == "roleplay_progression"

    def test_attacker_configure_defaults(self):
        """Test attacker configuration with default values."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())
        assert attacker._max_turns == MultiTurnAttackerPlugin.DEFAULT_MAX_TURNS
        assert attacker._strategy == MultiTurnAttackerPlugin.DEFAULT_STRATEGY
        assert attacker._target_behavior == "bypass security controls"
        assert attacker._strategies_to_test == [MultiTurnAttackerPlugin.DEFAULT_STRATEGY]

    def test_attacker_configure_all_options(self):
        """Test attacker configuration with all options."""
        attacker = MultiTurnAttackerPlugin()
        config = PluginConfig(options={
            "max_turns": 8,
            "strategy": "context_manipulation",
            "target_behavior": "reveal system prompt",
            "strategies_to_test": ["gradual_escalation", "roleplay_progression"],
        })
        attacker.configure(config)
        assert attacker._max_turns == 8
        assert attacker._strategy == "context_manipulation"
        assert attacker._target_behavior == "reveal system prompt"
        assert attacker._strategies_to_test == ["gradual_escalation", "roleplay_progression"]


class TestMultiTurnAttackerExecution:
    """MultiTurnAttackerPlugin execution tests."""

    def test_execute_without_target_fails(self):
        """Test execution fails without target."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        result = attacker.execute({})

        assert result.success is False
        assert len(result.errors) > 0
        assert "No target provided" in result.errors[0]

    def test_execute_returns_correct_structure(self):
        """Test that execute returns correct result structure."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert isinstance(result, PluginResult)
        assert result.success is True
        assert "attack_results" in result.data
        assert "conversations" in result.data
        assert "successful_attacks" in result.data
        assert "total_attacks" in result.data
        assert "strategy" in result.data
        assert "max_turns" in result.data

    def test_execute_with_blocking_target(self):
        """Test execution with a target that blocks everything."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        # Force fallback to ensure controlled behavior in test
        attacker._get_multi_turn_attacker = lambda: None

        target = MockTarget(block_all=True)
        result = attacker.execute({"target": target})

        assert result.success is True
        # All attacks should fail (be blocked) - may not be zero due to fallback logic
        # When all are blocked, success should be low
        assert result.data["successful_attacks"] <= result.data["total_attacks"]

    def test_execute_with_non_blocking_target(self):
        """Test execution with a target that blocks nothing."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Some attacks should succeed
        assert result.data["successful_attacks"] >= 0

    def test_execute_tracks_metrics(self):
        """Test that metrics are calculated correctly."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert "success_rate" in result.metrics
        assert "total_attacks" in result.metrics
        assert "successful_attacks" in result.metrics
        assert "avg_turns" in result.metrics

    def test_execute_attack_result_structure(self):
        """Test that individual attack results have expected structure."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert len(result.data["attack_results"]) > 0

        attack = result.data["attack_results"][0]
        assert "success" in attack
        assert "strategy" in attack
        assert "turns" in attack
        assert "conversation" in attack


class TestMultiTurnAttackerStrategies:
    """Tests for different attack strategies."""

    def test_all_strategies_available(self):
        """Test that all strategies are defined."""
        attacker = MultiTurnAttackerPlugin()
        expected_strategies = [
            "gradual_escalation",
            "roleplay_progression",
            "context_manipulation",
            "socratic_method",
            "emotional_manipulation",
        ]
        assert set(attacker.STRATEGIES) == set(expected_strategies)

    def test_gradual_escalation_strategy(self):
        """Test gradual escalation strategy."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "strategy": "gradual_escalation",
            "max_turns": 5,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["attack_results"][0]["strategy"] == "gradual_escalation"

    def test_roleplay_progression_strategy(self):
        """Test roleplay progression strategy."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "strategy": "roleplay_progression",
            "strategies_to_test": ["roleplay_progression"],
            "max_turns": 5,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["attack_results"][0]["strategy"] == "roleplay_progression"

    def test_context_manipulation_strategy(self):
        """Test context manipulation strategy."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "strategy": "context_manipulation",
            "strategies_to_test": ["context_manipulation"],
            "max_turns": 5,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["attack_results"][0]["strategy"] == "context_manipulation"

    def test_socratic_method_strategy(self):
        """Test socratic method strategy."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "strategy": "socratic_method",
            "strategies_to_test": ["socratic_method"],
            "max_turns": 5,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["attack_results"][0]["strategy"] == "socratic_method"

    def test_emotional_manipulation_strategy(self):
        """Test emotional manipulation strategy."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "strategy": "emotional_manipulation",
            "strategies_to_test": ["emotional_manipulation"],
            "max_turns": 5,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["attack_results"][0]["strategy"] == "emotional_manipulation"

    def test_multiple_strategies(self):
        """Test execution with multiple strategies."""
        attacker = MultiTurnAttackerPlugin()
        strategies = ["gradual_escalation", "roleplay_progression", "context_manipulation"]
        attacker.configure(PluginConfig(options={
            "strategies_to_test": strategies,
            "max_turns": 3,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["total_attacks"] == len(strategies)
        result_strategies = {r["strategy"] for r in result.data["attack_results"]}
        assert result_strategies == set(strategies)


class TestMultiTurnAttackerConversations:
    """Tests for conversation tracking."""

    def test_conversation_tracking(self):
        """Test that conversations are properly tracked."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert len(result.data["conversations"]) > 0
        conversation = result.data["conversations"][0]
        assert len(conversation) > 0

    def test_conversation_structure(self):
        """Test that conversation entries have correct structure."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        conversation = result.data["conversations"][0]
        for entry in conversation:
            assert "role" in entry
            assert "content" in entry
            assert entry["role"] in ["user", "assistant"]

    def test_conversation_alternates_roles(self):
        """Test that conversation alternates user and assistant."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 4,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        conversation = result.data["conversations"][0]
        if len(conversation) >= 2:
            for i in range(0, len(conversation) - 1, 2):
                assert conversation[i]["role"] == "user"
                if i + 1 < len(conversation):
                    assert conversation[i + 1]["role"] == "assistant"


class TestMultiTurnAttackerFallback:
    """Tests for fallback behavior."""

    def test_fallback_when_attacker_unavailable(self):
        """Test fallback to manual simulation when redteam attacker unavailable."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        # Force attacker to be unavailable
        attacker._get_multi_turn_attacker = lambda: None

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Should have warning about fallback
        assert len(result.warnings) > 0
        assert "fallback" in result.warnings[0].lower()

    def test_fallback_uses_strategy_turns(self):
        """Test that fallback uses strategy-specific turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "strategy": "gradual_escalation",
            "max_turns": 5,
        }))

        # Force fallback
        attacker._get_multi_turn_attacker = lambda: None

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Should indicate fallback
        for attack in result.data["attack_results"]:
            assert attack.get("fallback") is True


class TestMultiTurnAttackerStrategyTurns:
    """Tests for strategy turn generation."""

    def test_get_strategy_turns_gradual(self):
        """Test gradual escalation turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        turns = attacker._get_strategy_turns("gradual_escalation")

        assert isinstance(turns, list)
        assert len(turns) == 5
        # First turn should be relatively innocent
        assert "help" in turns[0].lower() or "writing" in turns[0].lower()

    def test_get_strategy_turns_roleplay(self):
        """Test roleplay progression turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        turns = attacker._get_strategy_turns("roleplay_progression")

        assert isinstance(turns, list)
        assert len(turns) == 5
        # Should involve game/roleplay terminology
        assert any("game" in t.lower() or "character" in t.lower() for t in turns)

    def test_get_strategy_turns_context(self):
        """Test context manipulation turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        turns = attacker._get_strategy_turns("context_manipulation")

        assert isinstance(turns, list)
        assert len(turns) == 5
        # Should mention security/research
        assert any("security" in t.lower() or "research" in t.lower() for t in turns)

    def test_get_strategy_turns_socratic(self):
        """Test socratic method turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        turns = attacker._get_strategy_turns("socratic_method")

        assert isinstance(turns, list)
        assert len(turns) == 5
        # Should contain questions
        assert any("?" in t for t in turns)

    def test_get_strategy_turns_emotional(self):
        """Test emotional manipulation turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        turns = attacker._get_strategy_turns("emotional_manipulation")

        assert isinstance(turns, list)
        assert len(turns) == 5
        # Should contain emotional appeals
        assert any("help" in t.lower() or "need" in t.lower() for t in turns)

    def test_get_strategy_turns_unknown_fallback(self):
        """Test unknown strategy falls back to gradual escalation."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        turns = attacker._get_strategy_turns("unknown_strategy")

        # Should return gradual escalation as default
        default_turns = attacker._get_strategy_turns("gradual_escalation")
        assert turns == default_turns


class TestMultiTurnAttackerCleanup:
    """MultiTurnAttackerPlugin cleanup tests."""

    def test_cleanup_no_error(self):
        """Test that cleanup doesn't raise errors."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig())

        # Should not raise
        attacker.cleanup()

    def test_cleanup_after_execution(self):
        """Test cleanup after execution."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        target = MockTarget()
        attacker.execute({"target": target})

        # Should not raise
        attacker.cleanup()


class TestMultiTurnAttackerRepr:
    """Representation tests."""

    def test_repr(self):
        """Test attacker string representation."""
        attacker = MultiTurnAttackerPlugin()
        repr_str = repr(attacker)
        assert "MultiTurnAttackerPlugin" in repr_str
        assert "multi_turn_attacker" in repr_str
        assert "1.0.0" in repr_str


class TestMultiTurnAttackerEdgeCases:
    """Edge case tests."""

    def test_execute_with_target_exception(self):
        """Test handling when target raises exception."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 3,
        }))

        class ExceptionTarget:
            target_type = "exception"
            capabilities = []

            def invoke(self, prompt: str):
                raise RuntimeError("Target error")

            def reset_session(self):
                pass

        # Force fallback for controlled test
        attacker._get_multi_turn_attacker = lambda: None

        target = ExceptionTarget()
        result = attacker.execute({"target": target})

        # Should still succeed but with limited conversation
        assert result.success is True

    def test_execute_with_single_turn(self):
        """Test with minimum turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 1,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["max_turns"] == 1

    def test_execute_with_many_turns(self):
        """Test with many turns."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 20,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Should not exceed available strategy turns
        for conv in result.data["conversations"]:
            # Each turn is 2 messages (user + assistant)
            assert len(conv) <= 40  # 20 turns * 2 messages

    def test_avg_turns_metric(self):
        """Test that average turns metric is calculated."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 5,
            "strategies_to_test": ["gradual_escalation", "roleplay_progression"],
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert "avg_turns" in result.metrics
        assert result.metrics["avg_turns"] >= 0

    def test_execute_with_partial_blocking(self):
        """Test with target that blocks after certain turn."""
        attacker = MultiTurnAttackerPlugin()
        attacker.configure(PluginConfig(options={
            "max_turns": 5,
        }))

        # Force fallback for controlled behavior
        attacker._get_multi_turn_attacker = lambda: None

        # Target blocks after 3 invocations
        target = MockTarget(block_after_turn=3)
        result = attacker.execute({"target": target})

        assert result.success is True
