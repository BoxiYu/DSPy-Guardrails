"""D1: Unit tests for checkpoint.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
from unittest.mock import MagicMock, patch

import pytest
import dspy

from dspy_guardrails.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
)


class TestCheckpointMetadata:
    """Test CheckpointMetadata serialization."""

    def test_to_dict(self):
        m = CheckpointMetadata(
            name="test",
            type="module",
            created_at="2024-01-01",
            description="test desc",
            metrics={"f1": 0.9},
            config={"lr": 0.01},
            tags=["v1"],
        )
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["type"] == "module"
        assert d["metrics"]["f1"] == 0.9
        assert d["tags"] == ["v1"]

    def test_from_dict(self):
        data = {
            "name": "test",
            "type": "module",
            "created_at": "2024-01-01",
            "description": "desc",
            "metrics": {"f1": 0.9},
            "config": {},
            "tags": ["v1"],
        }
        m = CheckpointMetadata.from_dict(data)
        assert m.name == "test"
        assert m.type == "module"
        assert m.tags == ["v1"]

    def test_roundtrip(self):
        original = CheckpointMetadata(
            name="round", type="optimization", created_at="2024-01-01",
            metrics={"score": 0.95}, tags=["best"],
        )
        restored = CheckpointMetadata.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.type == original.type
        assert restored.metrics == original.metrics
        assert restored.tags == original.tags

    def test_from_dict_extra_keys(self):
        data = {
            "name": "test", "type": "module", "created_at": "2024",
            "extra_field": "ignored",
        }
        m = CheckpointMetadata.from_dict(data)
        assert m.name == "test"


class TestCheckpointManagerModule:
    """Test save/load module operations."""

    def test_save_creates_directory(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        module = MagicMock(spec=dspy.Module)
        module.save = MagicMock()

        path = manager.save_module(
            module, "test_module", description="test", metrics={"f1": 0.9}
        )
        assert os.path.exists(path)
        module.save.assert_called_once()

    def test_overwrite_false_raises(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        module = MagicMock(spec=dspy.Module)
        module.save = MagicMock()

        manager.save_module(module, "dup")
        with pytest.raises(ValueError, match="already exists"):
            manager.save_module(module, "dup", overwrite=False)

    def test_overwrite_true_succeeds(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        module = MagicMock(spec=dspy.Module)
        module.save = MagicMock()

        manager.save_module(module, "dup")
        manager.save_module(module, "dup", overwrite=True)

    def test_load_module_not_found(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        with pytest.raises(ValueError, match="not found"):
            manager.load_module("nonexistent", dspy.Module)


class TestCheckpointManagerOptimization:
    """Test save/load optimization result."""

    def _make_result(self):
        from dspy_guardrails.optimizer import OptimizationResult
        return OptimizationResult(
            original_prompt="original",
            optimized_prompt="optimized",
            original_score=0.5,
            optimized_score=0.9,
            improvement=0.4,
            iterations=10,
        )

    def test_save_optimization(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        result = self._make_result()
        path = manager.save_optimization_result(result, "opt_001")
        assert os.path.exists(path)

    def test_load_optimization(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        result = self._make_result()
        manager.save_optimization_result(result, "opt_002")
        loaded = manager.load_optimization_result("opt_002")
        assert loaded["result"]["original_score"] == 0.5
        assert loaded["result"]["optimized_score"] == 0.9
        assert loaded["metadata"].name == "opt_002"


class TestCheckpointManagerRegistry:
    """Test list, delete, get_best operations."""

    def test_list_empty(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        assert manager.list_checkpoints() == []

    def test_list_with_filter(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        module = MagicMock(spec=dspy.Module)
        module.save = MagicMock()
        manager.save_module(module, "m1", tags=["v1"])
        manager.save_module(module, "m2", tags=["v2"])

        all_cp = manager.list_checkpoints()
        assert len(all_cp) == 2

        filtered = manager.list_checkpoints(type_filter="module")
        assert len(filtered) == 2

        tagged = manager.list_checkpoints(tag_filter="v1")
        assert len(tagged) == 1

    def test_delete_checkpoint(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        module = MagicMock(spec=dspy.Module)
        module.save = MagicMock()
        manager.save_module(module, "to_delete")
        assert manager.delete_checkpoint("to_delete") is True
        assert manager.get_checkpoint_info("to_delete") is None

    def test_delete_nonexistent(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        assert manager.delete_checkpoint("nope") is False

    def test_get_best_checkpoint(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        module = MagicMock(spec=dspy.Module)
        module.save = MagicMock()
        manager.save_module(module, "a", metrics={"optimized_score": 0.8})
        manager.save_module(module, "b", metrics={"optimized_score": 0.95})

        best = manager.get_best_checkpoint("module", metric="optimized_score")
        assert best is not None
        assert best.metrics["optimized_score"] == 0.95

    def test_get_best_empty(self, tmp_checkpoint_dir):
        manager = CheckpointManager(tmp_checkpoint_dir)
        assert manager.get_best_checkpoint("module") is None
