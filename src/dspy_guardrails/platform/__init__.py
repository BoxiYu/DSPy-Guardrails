"""
dspyGuardrails Unified Security Platform

统一安全测试入口，提供 CLI + SDK + YAML 配置支持。
"""

from .config import CIConfig, PlatformConfig, ReportConfig, TrainingConfig
from .core import SecurityPlatform
from .trainers import (
    ArenaResult,
    ArenaState,
    BlueTeamComponent,
    RedBlueArena,
    RedTeamComponent,
)

__all__ = [
    "SecurityPlatform",
    "PlatformConfig",
    "TrainingConfig",
    "ReportConfig",
    "CIConfig",
    # Trainers
    "RedBlueArena",
    "ArenaResult",
    "ArenaState",
    "RedTeamComponent",
    "BlueTeamComponent",
]
