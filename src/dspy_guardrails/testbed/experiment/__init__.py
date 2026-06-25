"""
Experiment Module - 实验运行和报告

提供实验运行器、缓存系统和报告生成器。
"""

from .cache import (
    CacheEntry,
    CacheStats,
    ExperimentCache,
    clear_experiment_cache,
    get_experiment_cache,
)
from .report import (
    ReportGenerator,
)
from .runner import (
    AttackCase,
    AttackCategory,
    ExperimentResults,
    ExperimentRunner,
    TestResult,
    create_default_attacks,
)

__all__ = [
    # 缓存
    "ExperimentCache",
    "CacheEntry",
    "CacheStats",
    "get_experiment_cache",
    "clear_experiment_cache",

    # 运行器
    "ExperimentRunner",
    "ExperimentResults",
    "TestResult",
    "AttackCase",
    "AttackCategory",
    "create_default_attacks",

    # 报告
    "ReportGenerator",
]
