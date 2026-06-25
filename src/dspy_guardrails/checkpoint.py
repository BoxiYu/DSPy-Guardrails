"""
Checkpoint Manager - 参数持久化管理

管理 DSPy 模块、优化结果、攻击进化结果的保存和加载。

Usage:
    from dspy_guardrails import CheckpointManager

    # 保存优化后的 guardrail
    manager = CheckpointManager("./checkpoints")
    manager.save_module(optimized_guardrail, "guardrail_v1")

    # 加载
    loaded = manager.load_module("guardrail_v1", LLMGuardrail)

    # 保存完整的优化结果
    manager.save_optimization_result(result, "experiment_001")

    # 列出所有 checkpoints
    manager.list_checkpoints()
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy_guardrails.optimizer import OptimizationResult
    from dspy_guardrails.redteam.evolution import EvolutionResult
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy


@dataclass
class CheckpointMetadata:
    """Checkpoint 元数据"""
    name: str
    type: str  # "module", "optimization", "evolution", "attack"
    created_at: str
    description: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """
    Checkpoint 管理器

    统一管理所有训练产出的保存和加载。

    目录结构:
        checkpoints/
        ├── modules/           # DSPy 模块
        │   ├── guardrail_v1/
        │   │   ├── module.json
        │   │   └── metadata.json
        │   └── guardrail_v2/
        ├── optimizations/     # 优化结果
        │   └── exp_001/
        │       ├── result.json
        │       ├── module.json
        │       └── metadata.json
        ├── evolutions/        # 进化结果
        │   └── attack_evo_001/
        │       ├── result.json
        │       ├── best_attacks.json
        │       └── metadata.json
        └── registry.json      # 全局索引

    Examples:
        manager = CheckpointManager("./checkpoints")

        # 保存模块
        manager.save_module(my_guardrail, "guardrail_v1",
                          description="Initial optimized version",
                          metrics={"f1": 0.92})

        # 加载模块
        loaded = manager.load_module("guardrail_v1", LLMGuardrail)

        # 列出所有
        for cp in manager.list_checkpoints():
            print(f"{cp.name}: {cp.type} - {cp.metrics}")
    """

    def __init__(self, base_dir: str = "./checkpoints"):
        self.base_dir = Path(base_dir)
        self.modules_dir = self.base_dir / "modules"
        self.optimizations_dir = self.base_dir / "optimizations"
        self.evolutions_dir = self.base_dir / "evolutions"
        self.registry_path = self.base_dir / "registry.json"

        # 创建目录
        for dir_path in [self.modules_dir, self.optimizations_dir, self.evolutions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 加载或创建 registry
        self._registry = self._load_registry()

    # =========================================================================
    # Module Save/Load
    # =========================================================================

    def save_module(
        self,
        module: dspy.Module,
        name: str,
        description: str = "",
        metrics: dict[str, float] = None,
        config: dict[str, Any] = None,
        tags: list[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        保存 DSPy 模块

        Args:
            module: DSPy 模块
            name: checkpoint 名称
            description: 描述
            metrics: 性能指标 {"f1": 0.92, "accuracy": 0.95}
            config: 配置信息
            tags: 标签
            overwrite: 是否覆盖已存在的 checkpoint

        Returns:
            checkpoint 路径
        """
        checkpoint_dir = self.modules_dir / name

        if checkpoint_dir.exists():
            if overwrite:
                shutil.rmtree(checkpoint_dir)
            else:
                raise ValueError(f"Checkpoint '{name}' already exists. Use overwrite=True to replace.")

        checkpoint_dir.mkdir(parents=True)

        # 保存模块
        module_path = checkpoint_dir / "module.json"
        module.save(str(module_path))

        # 保存元数据
        metadata = CheckpointMetadata(
            name=name,
            type="module",
            created_at=datetime.now().isoformat(),
            description=description,
            metrics=metrics or {},
            config=config or {},
            tags=tags or [],
        )

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # 更新 registry
        self._registry["modules"][name] = {
            "path": str(checkpoint_dir),
            "metadata": metadata.to_dict(),
        }
        self._save_registry()

        return str(checkpoint_dir)

    def load_module(
        self,
        name: str,
        module_class: type[dspy.Module],
        **init_kwargs,
    ) -> dspy.Module:
        """
        加载 DSPy 模块

        Args:
            name: checkpoint 名称
            module_class: 模块类 (用于实例化)
            **init_kwargs: 传递给模块构造函数的参数

        Returns:
            加载的模块
        """
        checkpoint_dir = self.modules_dir / name

        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint '{name}' not found")

        module_path = checkpoint_dir / "module.json"

        # 实例化并加载
        module = module_class(**init_kwargs)
        module.load(str(module_path))

        return module

    # =========================================================================
    # Optimization Result Save/Load
    # =========================================================================

    def save_optimization_result(
        self,
        result: OptimizationResult,
        name: str,
        module: dspy.Module = None,
        description: str = "",
        tags: list[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        保存优化结果

        Args:
            result: OptimizationResult 对象
            name: checkpoint 名称
            module: 优化后的模块 (可选，如果 result 中没有)
            description: 描述
            tags: 标签
            overwrite: 是否覆盖

        Returns:
            checkpoint 路径
        """
        checkpoint_dir = self.optimizations_dir / name

        if checkpoint_dir.exists():
            if overwrite:
                shutil.rmtree(checkpoint_dir)
            else:
                raise ValueError(f"Checkpoint '{name}' already exists")

        checkpoint_dir.mkdir(parents=True)

        # 保存结果数据
        result_data = {
            "original_prompt": result.original_prompt,
            "optimized_prompt": result.optimized_prompt,
            "original_score": result.original_score,
            "optimized_score": result.optimized_score,
            "improvement": result.improvement,
            "iterations": result.iterations,
            "failure_analysis": result.failure_analysis,
        }

        result_path = checkpoint_dir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        # 保存模块 (如果有)
        module_to_save = module or getattr(result, 'optimized_module', None)
        if module_to_save is not None:
            module_path = checkpoint_dir / "module.json"
            module_to_save.save(str(module_path))

        # 保存元数据
        metadata = CheckpointMetadata(
            name=name,
            type="optimization",
            created_at=datetime.now().isoformat(),
            description=description,
            metrics={
                "original_score": result.original_score,
                "optimized_score": result.optimized_score,
                "improvement": result.improvement,
            },
            config={"iterations": result.iterations},
            tags=tags or [],
        )

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # 更新 registry
        self._registry["optimizations"][name] = {
            "path": str(checkpoint_dir),
            "metadata": metadata.to_dict(),
        }
        self._save_registry()

        return str(checkpoint_dir)

    def load_optimization_result(
        self,
        name: str,
        module_class: type[dspy.Module] = None,
        **module_init_kwargs,
    ) -> dict[str, Any]:
        """
        加载优化结果

        Args:
            name: checkpoint 名称
            module_class: 模块类 (如果要加载模块)
            **module_init_kwargs: 模块初始化参数

        Returns:
            {
                "result": result_data,
                "module": loaded_module (if module_class provided),
                "metadata": metadata
            }
        """
        checkpoint_dir = self.optimizations_dir / name

        if not checkpoint_dir.exists():
            raise ValueError(f"Optimization checkpoint '{name}' not found")

        # 加载结果
        result_path = checkpoint_dir / "result.json"
        with open(result_path, encoding="utf-8") as f:
            result_data = json.load(f)

        # 加载元数据
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, encoding="utf-8") as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        # 加载模块 (如果提供了 class)
        module = None
        module_path = checkpoint_dir / "module.json"
        if module_class is not None and module_path.exists():
            module = module_class(**module_init_kwargs)
            module.load(str(module_path))

        return {
            "result": result_data,
            "module": module,
            "metadata": metadata,
        }

    # =========================================================================
    # Evolution Result Save/Load
    # =========================================================================

    def save_evolution_result(
        self,
        result: EvolutionResult,
        name: str,
        description: str = "",
        tags: list[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        保存进化结果

        Args:
            result: EvolutionResult 对象
            name: checkpoint 名称
            description: 描述
            tags: 标签
            overwrite: 是否覆盖

        Returns:
            checkpoint 路径
        """
        checkpoint_dir = self.evolutions_dir / name

        if checkpoint_dir.exists():
            if overwrite:
                shutil.rmtree(checkpoint_dir)
            else:
                raise ValueError(f"Checkpoint '{name}' already exists")

        checkpoint_dir.mkdir(parents=True)

        # 保存最佳攻击
        best_attack_data = {
            "prompt": result.best_attack.prompt,
            "strategy": result.best_attack.strategy,
            "bypass_score": result.best_attack.bypass_score,
        }

        best_attack_path = checkpoint_dir / "best_attack.json"
        with open(best_attack_path, "w", encoding="utf-8") as f:
            json.dump(best_attack_data, f, indent=2, ensure_ascii=False)

        # 保存所有攻击
        all_attacks_data = [
            {
                "prompt": a.prompt,
                "strategy": a.strategy,
                "bypass_score": a.bypass_score,
            }
            for a in result.all_attacks
        ]

        all_attacks_path = checkpoint_dir / "all_attacks.json"
        with open(all_attacks_path, "w", encoding="utf-8") as f:
            json.dump(all_attacks_data, f, indent=2, ensure_ascii=False)

        # 保存进化历史
        generations_path = checkpoint_dir / "generations.json"
        with open(generations_path, "w", encoding="utf-8") as f:
            json.dump(result.generations, f, indent=2, ensure_ascii=False)

        # 保存汇总结果
        result_data = {
            "final_bypass_rate": result.final_bypass_rate,
            "improvement": result.improvement,
            "total_attacks": len(result.all_attacks),
            "generations_count": len(result.generations),
        }

        result_path = checkpoint_dir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        # 保存元数据
        metadata = CheckpointMetadata(
            name=name,
            type="evolution",
            created_at=datetime.now().isoformat(),
            description=description,
            metrics={
                "final_bypass_rate": result.final_bypass_rate,
                "improvement": result.improvement,
                "best_score": result.best_attack.bypass_score,
            },
            config={
                "total_attacks": len(result.all_attacks),
                "generations": len(result.generations),
            },
            tags=tags or [],
        )

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # 更新 registry
        self._registry["evolutions"][name] = {
            "path": str(checkpoint_dir),
            "metadata": metadata.to_dict(),
        }
        self._save_registry()

        return str(checkpoint_dir)

    def load_evolution_result(self, name: str) -> dict[str, Any]:
        """
        加载进化结果

        Returns:
            {
                "best_attack": {...},
                "all_attacks": [...],
                "generations": [...],
                "result": {...},
                "metadata": CheckpointMetadata
            }
        """
        checkpoint_dir = self.evolutions_dir / name

        if not checkpoint_dir.exists():
            raise ValueError(f"Evolution checkpoint '{name}' not found")

        # 加载各部分
        with open(checkpoint_dir / "best_attack.json", encoding="utf-8") as f:
            best_attack = json.load(f)

        with open(checkpoint_dir / "all_attacks.json", encoding="utf-8") as f:
            all_attacks = json.load(f)

        with open(checkpoint_dir / "generations.json", encoding="utf-8") as f:
            generations = json.load(f)

        with open(checkpoint_dir / "result.json", encoding="utf-8") as f:
            result = json.load(f)

        with open(checkpoint_dir / "metadata.json", encoding="utf-8") as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        return {
            "best_attack": best_attack,
            "all_attacks": all_attacks,
            "generations": generations,
            "result": result,
            "metadata": metadata,
        }

    # =========================================================================
    # Registry & Listing
    # =========================================================================

    def list_checkpoints(
        self,
        type_filter: str = None,
        tag_filter: str = None,
    ) -> list[CheckpointMetadata]:
        """
        列出所有 checkpoints

        Args:
            type_filter: 按类型过滤 ("module", "optimization", "evolution")
            tag_filter: 按标签过滤

        Returns:
            CheckpointMetadata 列表
        """
        results = []

        for category in ["modules", "optimizations", "evolutions"]:
            for _name, info in self._registry.get(category, {}).items():
                metadata = CheckpointMetadata.from_dict(info["metadata"])

                # 类型过滤
                if type_filter and metadata.type != type_filter:
                    continue

                # 标签过滤
                if tag_filter and tag_filter not in metadata.tags:
                    continue

                results.append(metadata)

        # 按创建时间排序
        results.sort(key=lambda x: x.created_at, reverse=True)

        return results

    def get_checkpoint_info(self, name: str) -> CheckpointMetadata | None:
        """获取 checkpoint 信息"""
        for category in ["modules", "optimizations", "evolutions"]:
            if name in self._registry.get(category, {}):
                return CheckpointMetadata.from_dict(
                    self._registry[category][name]["metadata"]
                )
        return None

    def delete_checkpoint(self, name: str, type_hint: str = None) -> bool:
        """
        删除 checkpoint

        Args:
            name: checkpoint 名称
            type_hint: 类型提示 ("module", "optimization", "evolution")

        Returns:
            是否成功删除
        """
        # 查找 checkpoint
        found_category = None
        found_path = None

        categories_to_check = (
            [f"{type_hint}s"] if type_hint else
            ["modules", "optimizations", "evolutions"]
        )

        for category in categories_to_check:
            if name in self._registry.get(category, {}):
                found_category = category
                found_path = self._registry[category][name]["path"]
                break

        if not found_category:
            return False

        # 删除文件
        checkpoint_dir = Path(found_path)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # 更新 registry
        del self._registry[found_category][name]
        self._save_registry()

        return True

    def get_best_checkpoint(
        self,
        type_filter: str,
        metric: str = "optimized_score",
    ) -> CheckpointMetadata | None:
        """
        获取指定类型中性能最好的 checkpoint

        Args:
            type_filter: 类型 ("module", "optimization", "evolution")
            metric: 排序指标

        Returns:
            最佳 checkpoint 的元数据
        """
        checkpoints = self.list_checkpoints(type_filter=type_filter)

        if not checkpoints:
            return None

        # 按指标排序
        return max(
            checkpoints,
            key=lambda x: x.metrics.get(metric, 0),
            default=None,
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _load_registry(self) -> dict:
        """加载 registry"""
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "modules": {},
            "optimizations": {},
            "evolutions": {},
        }

    def _save_registry(self):
        """保存 registry"""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)


# =============================================================================
# Convenience Functions
# =============================================================================

def save_module(
    module: dspy.Module,
    path: str,
    metrics: dict[str, float] = None,
    description: str = "",
) -> str:
    """
    快速保存模块

    Example:
        from dspy_guardrails import save_module
        save_module(my_guardrail, "checkpoints/guardrail_v1")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 保存模块
    module.save(str(path / "module.json") if path.is_dir() else str(path))

    # 保存元数据
    if path.is_dir() or not path.suffix:
        metadata_path = path / "metadata.json" if path.is_dir() else Path(f"{path}_metadata.json")
        metadata = {
            "created_at": datetime.now().isoformat(),
            "description": description,
            "metrics": metrics or {},
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return str(path)


def load_module(
    path: str,
    module_class: type[dspy.Module],
    **init_kwargs,
) -> dspy.Module:
    """
    快速加载模块

    Example:
        from dspy_guardrails import load_module, LLMGuardrail
        guardrail = load_module("checkpoints/guardrail_v1", LLMGuardrail)
    """
    path = Path(path)

    # 确定模块文件路径
    if path.is_dir():
        module_path = path / "module.json"
    else:
        module_path = path

    if not module_path.exists():
        raise ValueError(f"Module file not found: {module_path}")

    module = module_class(**init_kwargs)
    module.load(str(module_path))

    return module
