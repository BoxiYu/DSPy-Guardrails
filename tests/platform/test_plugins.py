"""Tests for plugin system."""

import pytest
from dspy_guardrails.platform.plugins.base import (
    BasePlugin,
    PluginConfig,
    PluginResult,
    PluginType,
)


class MockPlugin(BasePlugin):
    """测试用 Mock 插件"""
    name = "mock_plugin"
    version = "1.0.0"
    plugin_type = PluginType.SCANNER

    def configure(self, config: PluginConfig) -> None:
        self._config = config

    def execute(self, context: dict) -> PluginResult:
        return PluginResult(
            success=True,
            data={"mock": "result"},
            metrics={"test_metric": 1.0},
        )

    def cleanup(self) -> None:
        pass


def test_plugin_config_defaults():
    """测试插件配置默认值"""
    config = PluginConfig()
    assert config.enabled is True
    assert config.priority == 0


def test_plugin_result_success():
    """测试成功的插件结果"""
    result = PluginResult(success=True, data={"key": "value"})
    assert result.success is True
    assert result.errors == []


def test_plugin_result_failure():
    """测试失败的插件结果"""
    result = PluginResult(
        success=False,
        data={},
        errors=["Error 1", "Error 2"],
    )
    assert result.success is False
    assert len(result.errors) == 2


def test_mock_plugin_execute():
    """测试 Mock 插件执行"""
    plugin = MockPlugin()
    plugin.configure(PluginConfig())
    result = plugin.execute({})

    assert result.success is True
    assert result.data["mock"] == "result"
    assert result.metrics["test_metric"] == 1.0


def test_plugin_type_enum():
    """测试插件类型枚举"""
    assert PluginType.SCANNER.value == "scanner"
    assert PluginType.ATTACKER.value == "attacker"
    assert PluginType.TRAINER.value == "trainer"
    assert PluginType.REPORTER.value == "reporter"
    assert PluginType.TARGET.value == "target"
