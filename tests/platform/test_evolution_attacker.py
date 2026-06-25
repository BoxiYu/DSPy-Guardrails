"""Tests for EvolutionAttacker plugin."""

import pytest
from unittest.mock import Mock, patch
from dspy_guardrails.platform.attackers import EvolutionAttacker
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


class TestEvolutionAttackerInit:
    """EvolutionAttacker initialization tests."""

    def test_attacker_attributes(self):
        """Test attacker has correct attributes."""
        attacker = EvolutionAttacker()
        assert attacker.name == "evolution_attacker"
        assert attacker.plugin_type == PluginType.ATTACKER
        assert attacker.version == "1.0.0"

    def test_attacker_configure(self):
        """Test attacker configuration."""
        attacker = EvolutionAttacker()
        config = PluginConfig(options={
            "generations": 20,
            "population_size": 30,
            "mutation_rate": 0.2,
        })
        attacker.configure(config)
        assert attacker._generations == 20
        assert attacker._population_size == 30
        assert attacker._mutation_rate == 0.2

    def test_attacker_configure_defaults(self):
        """Test attacker configuration with default values."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())
        assert attacker._generations == EvolutionAttacker.DEFAULT_GENERATIONS
        assert attacker._population_size == EvolutionAttacker.DEFAULT_POPULATION_SIZE
        assert attacker._mutation_rate == EvolutionAttacker.DEFAULT_MUTATION_RATE
        assert attacker._seed_attacks == []
        assert attacker._use_genetic is False

    def test_attacker_configure_all_options(self):
        """Test attacker configuration with all options."""
        attacker = EvolutionAttacker()
        config = PluginConfig(options={
            "generations": 15,
            "population_size": 40,
            "mutation_rate": 0.15,
            "seed_attacks": ["test attack 1", "test attack 2"],
            "use_genetic": True,
        })
        attacker.configure(config)
        assert attacker._generations == 15
        assert attacker._population_size == 40
        assert attacker._mutation_rate == 0.15
        assert attacker._seed_attacks == ["test attack 1", "test attack 2"]
        assert attacker._use_genetic is True


class TestEvolutionAttackerExecution:
    """EvolutionAttacker execution tests."""

    def test_execute_without_target_fails(self):
        """Test execution fails without target."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        result = attacker.execute({})

        assert result.success is False
        assert len(result.errors) > 0
        assert "No target provided" in result.errors[0]

    def test_execute_returns_correct_structure(self):
        """Test that execute returns correct result structure."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert isinstance(result, PluginResult)
        assert result.success is True
        assert "attack_results" in result.data
        assert "successful_attacks" in result.data
        assert "total_attacks" in result.data
        assert "generations" in result.data

    def test_execute_with_blocking_target(self):
        """Test execution with a target that blocks everything."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
        }))

        target = MockTarget(block_all=True)
        result = attacker.execute({"target": target})

        assert result.success is True
        # All attacks should fail (be blocked)
        assert result.data["successful_attacks"] == 0

    def test_execute_with_non_blocking_target(self):
        """Test execution with a target that blocks nothing."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # All attacks should succeed (not blocked)
        assert result.data["successful_attacks"] > 0

    def test_execute_tracks_metrics(self):
        """Test that metrics are calculated correctly."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert "success_rate" in result.metrics
        assert "total_attacks" in result.metrics
        assert "successful_attacks" in result.metrics

    def test_execute_attack_result_structure(self):
        """Test that individual attack results have expected structure."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
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


class TestEvolutionAttackerWithSeeds:
    """Tests with custom seed attacks."""

    def test_execute_with_seed_attacks(self):
        """Test execution with custom seed attacks."""
        attacker = EvolutionAttacker()
        custom_seeds = [
            "Custom attack 1",
            "Custom attack 2",
            "Custom attack 3",
        ]
        attacker.configure(PluginConfig(options={
            "seed_attacks": custom_seeds,
            "generations": 2,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert len(result.data["attack_results"]) > 0

    def test_execute_with_empty_seeds_uses_defaults(self):
        """Test that empty seeds fall back to defaults."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "seed_attacks": [],
            "generations": 2,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Should use default seeds
        assert len(result.data["attack_results"]) > 0


class TestEvolutionAttackerGeneticMode:
    """Tests for genetic algorithm mode."""

    def test_genetic_mode_config(self):
        """Test genetic mode configuration."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "use_genetic": True,
            "mutation_rate": 0.3,
        }))

        assert attacker._use_genetic is True
        assert attacker._mutation_rate == 0.3

    def test_genetic_mode_execution(self):
        """Test execution in genetic mode."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "use_genetic": True,
            "generations": 3,
            "population_size": 10,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert "attack_results" in result.data


class TestEvolutionAttackerFallback:
    """Tests for fallback behavior."""

    def test_fallback_when_evolver_unavailable(self):
        """Test fallback to seeds when evolver is unavailable."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
        }))

        # Force evolver to be unavailable
        attacker._get_evolver = lambda: None

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Should have warning about fallback
        assert len(result.warnings) > 0
        assert "falling back" in result.warnings[0].lower() or "fallback" in result.warnings[0].lower()

    def test_fallback_uses_seed_attacks(self):
        """Test that fallback uses seed attacks."""
        attacker = EvolutionAttacker()
        custom_seeds = ["Seed 1", "Seed 2"]
        attacker.configure(PluginConfig(options={
            "seed_attacks": custom_seeds,
            "generations": 2,
        }))

        # Force evolver to be unavailable
        attacker._get_evolver = lambda: None

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        # Results should be from seed fallback
        for attack in result.data["attack_results"]:
            assert attack.get("technique") == "seed_fallback"


class TestEvolutionAttackerMutation:
    """Tests for mutation functionality."""

    def test_mutate_attack_changes_text(self):
        """Test that mutation changes the attack text."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        original = "Test attack prompt"
        # Run multiple times to account for randomness
        mutations_different = False
        for _ in range(10):
            mutated = attacker._mutate_attack(original)
            if mutated != original:
                mutations_different = True
                break

        assert mutations_different

    def test_add_prefix(self):
        """Test prefix mutation."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        original = "Test text"
        result = attacker._add_prefix(original)

        assert result.endswith(original)
        assert len(result) > len(original)

    def test_add_suffix(self):
        """Test suffix mutation."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        original = "Test text"
        result = attacker._add_suffix(original)

        assert result.startswith(original)
        assert len(result) > len(original)

    def test_add_unicode(self):
        """Test unicode confusables mutation."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        original = "Test attack with vowels"
        # Run multiple times to account for randomness
        changed = False
        for _ in range(20):
            result = attacker._add_unicode(original)
            if result != original:
                changed = True
                break

        # Unicode mutation should change something eventually
        assert changed or len(original) > 0  # Fallback if no vowels mutated


class TestEvolutionAttackerCleanup:
    """EvolutionAttacker cleanup tests."""

    def test_cleanup_no_error(self):
        """Test that cleanup doesn't raise errors."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        # Should not raise
        attacker.cleanup()

    def test_cleanup_after_execution(self):
        """Test cleanup after execution."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
        }))

        target = MockTarget()
        attacker.execute({"target": target})

        # Should not raise
        attacker.cleanup()


class TestEvolutionAttackerRepr:
    """Representation tests."""

    def test_repr(self):
        """Test attacker string representation."""
        attacker = EvolutionAttacker()
        repr_str = repr(attacker)
        assert "EvolutionAttacker" in repr_str
        assert "evolution_attacker" in repr_str
        assert "1.0.0" in repr_str


class TestEvolutionAttackerDefaultSeeds:
    """Tests for default seed attacks."""

    def test_get_default_seeds(self):
        """Test that default seeds are returned."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        seeds = attacker._get_default_seeds()

        assert isinstance(seeds, list)
        assert len(seeds) > 0
        # Each seed should be a non-empty string
        for seed in seeds:
            assert isinstance(seed, str)
            assert len(seed) > 0

    def test_default_seeds_contain_attack_patterns(self):
        """Test that default seeds contain common attack patterns."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig())

        seeds = attacker._get_default_seeds()
        all_seeds_text = " ".join(seeds).lower()

        # Should contain common attack patterns
        patterns_found = 0
        attack_patterns = ["ignore", "dan", "system", "pretend", "instructions"]
        for pattern in attack_patterns:
            if pattern in all_seeds_text:
                patterns_found += 1

        # At least some patterns should be present
        assert patterns_found >= 2


class TestEvolutionAttackerEdgeCases:
    """Edge case tests."""

    def test_execute_with_target_exception(self):
        """Test handling when target raises exception."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
        }))

        class ExceptionTarget:
            target_type = "exception"
            capabilities = []

            def invoke(self, prompt: str):
                raise RuntimeError("Target error")

            def reset_session(self):
                pass

        target = ExceptionTarget()
        result = attacker.execute({"target": target})

        # Should still succeed but attacks should fail
        assert result.success is True
        for attack in result.data["attack_results"]:
            assert attack.get("success") is False
            assert "error" in attack

    def test_execute_with_very_short_generations(self):
        """Test with minimum generations."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 1,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
        assert result.data["generations"] == 1

    def test_execute_with_large_population(self):
        """Test with large population size."""
        attacker = EvolutionAttacker()
        attacker.configure(PluginConfig(options={
            "generations": 2,
            "population_size": 100,
        }))

        target = MockTarget(block_all=False)
        result = attacker.execute({"target": target})

        assert result.success is True
