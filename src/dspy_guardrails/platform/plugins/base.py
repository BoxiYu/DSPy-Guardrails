"""Plugin system base classes and protocols."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PluginType(Enum):
    """插件类型"""
    SCANNER = "scanner"
    ATTACKER = "attacker"
    TRAINER = "trainer"
    REPORTER = "reporter"
    TARGET = "target"


class PluginConfig(BaseModel):
    """插件配置基类"""
    enabled: bool = True
    priority: int = 0
    options: dict[str, Any] = Field(default_factory=dict)


class PluginResult(BaseModel):
    """插件执行结果"""
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    def merge(self, other: "PluginResult") -> "PluginResult":
        """合并两个结果"""
        return PluginResult(
            success=self.success and other.success,
            data={**self.data, **other.data},
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            metrics={**self.metrics, **other.metrics},
        )


class BasePlugin(ABC):
    """所有插件的抽象基类"""

    name: str
    version: str
    plugin_type: PluginType

    @abstractmethod
    def configure(self, config: PluginConfig) -> None:
        """配置插件"""
        pass

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> PluginResult:
        """执行插件逻辑"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} v{self.version}>"
