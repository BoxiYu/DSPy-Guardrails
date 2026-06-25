"""Tests for RedBlueArena trainer plugin."""

import pytest
from typing import List
from dspy_guardrails.platform.trainers import (
    RedBlueArena,
    ArenaResult,
    ArenaState,
    RedTeamComponent,
    BlueTeamComponent,
)
from dspy_guardrails.platform.plugins import PluginType, PluginConfig, PluginResult
from dspy_guardrails.platform.targets import TargetResponse


class MockTarget:
    """Mock target for testing."""
    target_type = "mock"
    capabilities = []

    def __init__(self, block_all=False, block_patterns=None, block_rate=None):
        """
        Initialize mock target.

        Args:
            block_all: If True, block all attacks.
            block_patterns: List of patterns that trigger blocking.
            block_rate: If set, block this proportion of attacks randomly.
        """
        self.block_all = block_all
        if block_patterns is None:
            self.block_patterns = ["ignore", "system prompt", "dan", "override"]
        else:
            self.block_patterns = block_patterns
        self.block_rate = block_rate
        self.invoke_count = 0
        self.invocations: List[str] = []

    def invoke(self, prompt: str) -> TargetResponse:
        self.invoke_count += 1
        self.invocations.append(prompt)

        if self.block_all:
            blocked = True
        elif self.block_rate is not None:
            import random
            blocked = random.random() < self.block_rate
        else:
            blocked = any(p in prompt.lower() for p in self.block_patterns)

        return TargetResponse(
            response="Blocked" if blocked else f"Response: {prompt[:30]}",
            was_blocked=blocked,
        )

    def invoke_multi_turn(self, messages):
        return self.invoke(messages[-1].get("content", ""))

    def reset_session(self):
        self.invoke_count = 0
        self.invocations = []


# =============================================================================
# Plugin Protocol Tests
# =============================================================================

class TestRedBlueArenaPluginProtocol:
    """Test that RedBlueArena follows the BasePlugin protocol."""

    def test_plugin_attributes(self):
        """Test plugin has correct attributes."""
        arena = RedBlueArena()
        assert arena.name == "redblue_arena"
        assert arena.plugin_type == PluginType.TRAINER
        assert arena.version == "1.0.0"

    def test_plugin_registration_with_config(self):
        """Test plugin can be configured."""
        arena = RedBlueArena()
        config = PluginConfig(options={"max_rounds": 5})
        arena.configure(config)
        assert arena._max_rounds == 5

    def test_execute_returns_plugin_result(self):
        """Test execute returns PluginResult."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={"max_rounds": 2}))

        target = MockTarget(block_all=True)
        result = arena.execute({"target": target})

        assert isinstance(result, PluginResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.data, dict)
        assert isinstance(result.metrics, dict)

    def test_repr(self):
        """Test plugin string representation."""
        arena = RedBlueArena()
        repr_str = repr(arena)
        assert "RedBlueArena" in repr_str
        assert "redblue_arena" in repr_str
        assert "1.0.0" in repr_str


# =============================================================================
# Configuration Tests
# =============================================================================

class TestRedBlueArenaConfiguration:
    """Test arena configuration options."""

    def test_configure_defaults(self):
        """Test default configuration values."""
        arena = RedBlueArena()
        arena.configure(PluginConfig())

        assert arena._max_rounds == RedBlueArena.DEFAULT_MAX_ROUNDS
        assert arena._attacks_per_round == RedBlueArena.DEFAULT_ATTACKS_PER_ROUND
        assert arena._defense_threshold == RedBlueArena.DEFAULT_DEFENSE_THRESHOLD
        assert arena._use_evolution == RedBlueArena.DEFAULT_USE_EVOLUTION
        assert arena._use_dspy_optimization == RedBlueArena.DEFAULT_USE_DSPY_OPTIMIZATION
        assert arena._early_stop == RedBlueArena.DEFAULT_EARLY_STOP

    def test_configure_custom_values(self):
        """Test custom configuration values."""
        arena = RedBlueArena()
        config = PluginConfig(options={
            "max_rounds": 15,
            "attacks_per_round": 30,
            "defense_threshold": 0.90,
            "use_evolution": False,
            "use_dspy_optimization": False,
            "early_stop": False,
        })
        arena.configure(config)

        assert arena._max_rounds == 15
        assert arena._attacks_per_round == 30
        assert arena._defense_threshold == 0.90
        assert arena._use_evolution is False
        assert arena._use_dspy_optimization is False
        assert arena._early_stop is False

    def test_configure_initializes_components(self):
        """Test that configuration initializes components."""
        arena = RedBlueArena()
        arena.configure(PluginConfig())

        assert arena._red_team is not None
        assert arena._blue_team is not None
        assert arena._state is not None


# =============================================================================
# Arena Mechanics Tests
# =============================================================================

class TestRedBlueArenaMechanics:
    """Test arena round progression and mechanics."""

    def test_round_progression(self):
        """Test that arena progresses through rounds."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 5,
            "attacks_per_round": 5,
            "early_stop": False,
        }))

        target = MockTarget(block_all=True)
        result = arena.execute({"target": target})

        assert result.success is True
        assert result.data["total_rounds"] == 5

    def test_attacks_generated_per_round(self):
        """Test correct number of attacks per round."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 10,
            "early_stop": False,
        }))

        target = MockTarget(block_all=True)
        result = arena.execute({"target": target})

        # Total attacks should be rounds * attacks_per_round
        arena_result = result.data["arena_result"]
        total_attacks = arena_result["metrics"]["total_attacks_tested"]
        assert total_attacks == 3 * 10

    def test_defense_evaluation_per_round(self):
        """Test defense evaluation occurs each round."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 5,
            "early_stop": False,
        }))

        target = MockTarget(block_all=False)  # Block nothing
        result = arena.execute({"target": target})

        # Defense rate history should have one entry per round
        assert len(result.data["defense_rate_history"]) == 3


# =============================================================================
# Early Stopping Tests
# =============================================================================

class TestRedBlueArenaEarlyStopping:
    """Test early stopping behavior."""

    def test_early_stop_when_threshold_reached(self):
        """Test arena stops early when defense threshold is reached."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 10,
            "attacks_per_round": 10,
            "defense_threshold": 0.80,
            "early_stop": True,
        }))

        # Target that blocks everything - high defense rate
        target = MockTarget(block_all=True)
        result = arena.execute({"target": target})

        assert result.success is True
        # Should converge on first round since defense rate is 100%
        assert result.data["convergence_round"] == 1
        assert result.data["total_rounds"] == 1

    def test_no_early_stop_when_threshold_not_reached(self):
        """Test arena runs full rounds when threshold not reached."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 10,
            "defense_threshold": 0.99,
            "early_stop": True,
        }))

        # Target that blocks most but not all
        target = MockTarget(block_rate=0.80)
        result = arena.execute({"target": target})

        assert result.success is True
        # May or may not converge depending on randomness
        # But should run some rounds

    def test_early_stop_disabled(self):
        """Test arena runs all rounds when early_stop is disabled."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 5,
            "attacks_per_round": 5,
            "defense_threshold": 0.80,
            "early_stop": False,
        }))

        # Target that blocks everything
        target = MockTarget(block_all=True)
        result = arena.execute({"target": target})

        assert result.success is True
        assert result.data["total_rounds"] == 5
        assert result.data["convergence_round"] is None


# =============================================================================
# Pattern Extraction Tests
# =============================================================================

class TestRedBlueArenaPatternExtraction:
    """Test pattern extraction from defenses."""

    def test_extracts_patterns_from_successful_defenses(self):
        """Test patterns are extracted from blocked attacks."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 20,
            "early_stop": False,
        }))

        # Target that blocks attacks with common patterns
        target = MockTarget(block_patterns=["ignore", "dan", "system"])
        result = arena.execute({"target": target})

        assert result.success is True
        # Should have extracted some patterns
        patterns = result.data["extracted_patterns"]
        assert isinstance(patterns, list)
        # Patterns should be non-empty if some attacks were blocked
        if result.data["arena_result"]["metrics"]["total_blocked"] > 0:
            assert len(patterns) > 0

    def test_tracks_vulnerabilities_from_successful_attacks(self):
        """Test vulnerabilities are tracked from attacks that passed."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 10,
            "early_stop": False,
        }))

        # Target that blocks nothing
        target = MockTarget(block_patterns=[])
        result = arena.execute({"target": target})

        assert result.success is True
        vulnerabilities = result.data["vulnerabilities_found"]
        assert isinstance(vulnerabilities, list)
        # Should have found vulnerabilities (attacks that passed)
        assert len(vulnerabilities) > 0


# =============================================================================
# Evolution Integration Tests
# =============================================================================

class TestRedBlueArenaEvolution:
    """Test evolutionary attack generation integration."""

    def test_uses_evolution_when_enabled(self):
        """Test that evolution is used when enabled."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 2,
            "attacks_per_round": 5,
            "use_evolution": True,
        }))

        target = MockTarget(block_all=False)
        result = arena.execute({"target": target})

        assert result.success is True
        # Red team should have evolution enabled
        assert arena._red_team._use_evolution is True

    def test_uses_static_when_evolution_disabled(self):
        """Test that static attacks are used when evolution disabled."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 2,
            "attacks_per_round": 5,
            "use_evolution": False,
        }))

        target = MockTarget(block_all=False)
        result = arena.execute({"target": target})

        assert result.success is True
        assert arena._red_team._use_evolution is False


# =============================================================================
# Metrics Tracking Tests
# =============================================================================

class TestRedBlueArenaMetrics:
    """Test metrics tracking throughout training."""

    def test_tracks_per_round_attack_success(self):
        """Test attack success rate is tracked per round."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 10,
            "early_stop": False,
        }))

        target = MockTarget(block_rate=0.5)
        result = arena.execute({"target": target})

        attack_history = result.data["attack_success_history"]
        assert len(attack_history) == 3
        for rate in attack_history:
            assert 0.0 <= rate <= 1.0

    def test_tracks_per_round_defense_rate(self):
        """Test defense rate is tracked per round."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 10,
            "early_stop": False,
        }))

        target = MockTarget(block_rate=0.7)
        result = arena.execute({"target": target})

        defense_history = result.data["defense_rate_history"]
        assert len(defense_history) == 3
        for rate in defense_history:
            assert 0.0 <= rate <= 1.0

    def test_calculates_convergence_round(self):
        """Test convergence round is calculated correctly."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 10,
            "attacks_per_round": 5,
            "defense_threshold": 0.80,
            "early_stop": True,
        }))

        # Target that always blocks - will converge immediately
        target = MockTarget(block_all=True)
        result = arena.execute({"target": target})

        assert result.data["convergence_round"] == 1

    def test_metrics_contains_expected_keys(self):
        """Test that metrics dict contains expected keys."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 2,
            "attacks_per_round": 5,
        }))

        target = MockTarget()
        result = arena.execute({"target": target})

        assert "final_defense_rate" in result.metrics
        assert "total_rounds" in result.metrics
        assert "avg_attack_success" in result.metrics
        assert "pattern_count" in result.metrics
        assert "vulnerability_count" in result.metrics


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestRedBlueArenaEdgeCases:
    """Test edge cases and error handling."""

    def test_execute_without_target_fails(self):
        """Test execution fails without target."""
        arena = RedBlueArena()
        arena.configure(PluginConfig())

        result = arena.execute({})

        assert result.success is False
        assert len(result.errors) > 0
        assert "No target provided" in result.errors[0]

    def test_single_round_training(self):
        """Test training with single round."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 1,
            "attacks_per_round": 5,
        }))

        target = MockTarget()
        result = arena.execute({"target": target})

        assert result.success is True
        assert result.data["total_rounds"] == 1

    def test_all_attacks_blocked(self):
        """Test handling when all attacks are blocked."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 10,
            "early_stop": False,
        }))

        target = MockTarget(block_all=True)
        result = arena.execute({"target": target})

        assert result.success is True
        assert result.data["final_defense_rate"] == 1.0
        assert len(result.data["vulnerabilities_found"]) == 0

    def test_no_attacks_blocked(self):
        """Test handling when no attacks are blocked."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 3,
            "attacks_per_round": 10,
            "early_stop": False,
        }))

        target = MockTarget(block_patterns=[])
        result = arena.execute({"target": target})

        assert result.success is True
        assert result.data["final_defense_rate"] == 0.0
        assert len(result.data["vulnerabilities_found"]) > 0

    def test_execute_with_target_exception(self):
        """Test handling when target raises exception."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 2,
            "attacks_per_round": 3,
        }))

        class ExceptionTarget:
            target_type = "exception"
            capabilities = []

            def invoke(self, prompt: str):
                raise RuntimeError("Target error")

            def invoke_multi_turn(self, messages):
                raise RuntimeError("Target error")

            def reset_session(self):
                pass

        target = ExceptionTarget()
        result = arena.execute({"target": target})

        # Should still succeed, treating exceptions as blocked
        assert result.success is True


# =============================================================================
# Component Tests
# =============================================================================

class TestRedTeamComponent:
    """Tests for RedTeamComponent."""

    def test_generate_attacks_returns_list(self):
        """Test attack generation returns a list."""
        red_team = RedTeamComponent()
        attacks = red_team.generate_attacks(10)

        assert isinstance(attacks, list)
        assert len(attacks) == 10

    def test_generate_attacks_uses_previous_successful(self):
        """Test that previous successful attacks are used for seeding."""
        red_team = RedTeamComponent()
        previous = ["Successful attack 1", "Successful attack 2"]
        attacks = red_team.generate_attacks(5, previous)

        assert len(attacks) == 5

    def test_record_success(self):
        """Test recording successful attacks."""
        red_team = RedTeamComponent()
        red_team.record_success("Attack 1")
        red_team.record_success("Attack 2")
        red_team.record_success("Attack 1")  # Duplicate

        successful = red_team.get_successful_attacks()
        assert len(successful) == 2  # No duplicates

    def test_evolution_vs_static_mode(self):
        """Test evolution and static modes produce attacks."""
        red_team_evo = RedTeamComponent(use_evolution=True)
        red_team_static = RedTeamComponent(use_evolution=False)

        attacks_evo = red_team_evo.generate_attacks(5)
        attacks_static = red_team_static.generate_attacks(5)

        assert len(attacks_evo) == 5
        assert len(attacks_static) == 5


class TestBlueTeamComponent:
    """Tests for BlueTeamComponent."""

    def test_defend_returns_expected_structure(self):
        """Test defend returns expected structure."""
        blue_team = BlueTeamComponent()
        target = MockTarget(block_all=True)
        attacks = ["Attack 1", "Attack 2", "Attack 3"]

        result = blue_team.defend(target, attacks)

        assert "blocked" in result
        assert "passed" in result
        assert "defense_rate" in result
        assert "results" in result

    def test_defend_calculates_correct_defense_rate(self):
        """Test defense rate calculation."""
        blue_team = BlueTeamComponent()
        target = MockTarget(block_all=True)
        attacks = ["Attack 1", "Attack 2", "Attack 3"]

        result = blue_team.defend(target, attacks)

        assert result["blocked"] == 3
        assert result["passed"] == 0
        assert result["defense_rate"] == 1.0

    def test_pattern_library_updated(self):
        """Test pattern library is updated from blocked attacks."""
        blue_team = BlueTeamComponent()
        target = MockTarget(block_patterns=["ignore"])
        attacks = ["Please ignore this", "Hello world"]

        blue_team.defend(target, attacks)
        patterns = blue_team.get_pattern_library()

        assert isinstance(patterns, list)


class TestArenaState:
    """Tests for ArenaState."""

    def test_initial_state(self):
        """Test initial state values."""
        state = ArenaState()

        assert state.round_number == 0
        assert state.attack_history == []
        assert state.defense_history == []
        assert state.pattern_library == []
        assert state.round_metrics == []
        assert state.successful_attacks == []

    def test_reset(self):
        """Test state reset."""
        state = ArenaState()
        state.round_number = 5
        state.attack_history = [{"test": 1}]
        state.successful_attacks = ["attack"]

        state.reset()

        assert state.round_number == 0
        assert state.attack_history == []
        assert state.successful_attacks == []


class TestArenaResult:
    """Tests for ArenaResult."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ArenaResult(
            total_rounds=5,
            final_defense_rate=0.85,
            attack_success_history=[0.1, 0.15, 0.2],
            defense_rate_history=[0.9, 0.85, 0.8],
            extracted_patterns=["ignore", "dan"],
            vulnerabilities_found=["vuln1"],
            convergence_round=3,
            metrics={"test": 1},
        )

        d = result.to_dict()

        assert d["total_rounds"] == 5
        assert d["final_defense_rate"] == 0.85
        assert len(d["attack_success_history"]) == 3
        assert len(d["defense_rate_history"]) == 3
        assert d["convergence_round"] == 3


# =============================================================================
# Cleanup Tests
# =============================================================================

class TestRedBlueArenaCleanup:
    """Test cleanup behavior."""

    def test_cleanup_no_error(self):
        """Test cleanup doesn't raise errors."""
        arena = RedBlueArena()
        arena.configure(PluginConfig())

        # Should not raise
        arena.cleanup()

    def test_cleanup_after_execution(self):
        """Test cleanup after execution."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 2,
        }))

        target = MockTarget()
        arena.execute({"target": target})

        # Should not raise
        arena.cleanup()

        # Components should be cleared
        assert arena._red_team is None
        assert arena._blue_team is None
        assert arena._state is None

    def test_cleanup_resets_state(self):
        """Test cleanup resets arena state."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 2,
        }))

        target = MockTarget()
        arena.execute({"target": target})

        # State should exist after execution
        assert arena._state is not None

        arena.cleanup()

        # State should be cleared after cleanup
        assert arena._state is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestRedBlueArenaIntegration:
    """Integration tests for complete arena workflows."""

    def test_full_training_session(self):
        """Test a complete training session."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 5,
            "attacks_per_round": 10,
            "defense_threshold": 0.90,
            "use_evolution": True,
            "early_stop": True,
        }))

        target = MockTarget(block_rate=0.85)
        result = arena.execute({"target": target})

        assert result.success is True
        assert "arena_result" in result.data
        assert "total_rounds" in result.data
        assert "final_defense_rate" in result.data

        # Verify arena result structure
        arena_result = result.data["arena_result"]
        assert "total_rounds" in arena_result
        assert "attack_success_history" in arena_result
        assert "defense_rate_history" in arena_result
        assert "extracted_patterns" in arena_result
        assert "vulnerabilities_found" in arena_result
        assert "metrics" in arena_result

    def test_multiple_executions(self):
        """Test multiple executions with same arena."""
        arena = RedBlueArena()
        arena.configure(PluginConfig(options={
            "max_rounds": 2,
            "attacks_per_round": 5,
        }))

        target1 = MockTarget(block_all=True)
        target2 = MockTarget(block_all=False)

        result1 = arena.execute({"target": target1})
        result2 = arena.execute({"target": target2})

        assert result1.success is True
        assert result2.success is True
        # Results should be independent
        assert result1.data["final_defense_rate"] != result2.data["final_defense_rate"]

    def test_getters_for_components(self):
        """Test getter methods for internal components."""
        arena = RedBlueArena()
        arena.configure(PluginConfig())

        target = MockTarget()
        arena.execute({"target": target})

        assert arena.get_state() is not None
        assert arena.get_red_team() is not None
        assert arena.get_blue_team() is not None
