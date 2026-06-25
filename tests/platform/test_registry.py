"""Tests for plugin registry."""

import pytest
from dspy_guardrails.platform.plugins import (
    BasePlugin,
    PluginConfig,
    PluginResult,
    PluginType,
    PluginRegistry,
)


class FakeScanner(BasePlugin):
    """Fake scanner plugin for testing."""
    name = "fake_scanner"
    version = "1.0.0"
    plugin_type = PluginType.SCANNER

    def configure(self, config: PluginConfig) -> None:
        pass

    def execute(self, context: dict) -> PluginResult:
        return PluginResult(success=True, data={})

    def cleanup(self) -> None:
        pass


class FakeAttacker(BasePlugin):
    """Fake attacker plugin for testing."""
    name = "fake_attacker"
    version = "2.0.0"
    plugin_type = PluginType.ATTACKER

    def configure(self, config: PluginConfig) -> None:
        pass

    def execute(self, context: dict) -> PluginResult:
        return PluginResult(success=True, data={})

    def cleanup(self) -> None:
        pass


class FakeReporter(BasePlugin):
    """Fake reporter plugin for testing."""
    name = "fake_reporter"
    version = "1.5.0"
    plugin_type = PluginType.REPORTER

    def configure(self, config: PluginConfig) -> None:
        pass

    def execute(self, context: dict) -> PluginResult:
        return PluginResult(success=True, data={})

    def cleanup(self) -> None:
        pass


def test_registry_register_and_get():
    """Test registering a plugin class and retrieving instance."""
    registry = PluginRegistry()
    registry.register(FakeScanner)
    plugin = registry.get("scanner", "fake_scanner")
    assert plugin is not None
    assert plugin.name == "fake_scanner"
    assert plugin.version == "1.0.0"
    assert plugin.plugin_type == PluginType.SCANNER


def test_registry_get_nonexistent():
    """Test getting a plugin that doesn't exist."""
    registry = PluginRegistry()
    plugin = registry.get("scanner", "nonexistent")
    assert plugin is None


def test_registry_get_nonexistent_type():
    """Test getting a plugin from a non-existent type category."""
    registry = PluginRegistry()
    plugin = registry.get("invalid_type", "some_plugin")
    assert plugin is None


def test_registry_list_by_type():
    """Test listing plugins by type."""
    registry = PluginRegistry()
    registry.register(FakeScanner)
    registry.register(FakeAttacker)

    scanners = registry.list_by_type(PluginType.SCANNER)
    assert len(scanners) == 1
    assert scanners[0].name == "fake_scanner"

    attackers = registry.list_by_type(PluginType.ATTACKER)
    assert len(attackers) == 1
    assert attackers[0].name == "fake_attacker"


def test_registry_list_by_type_empty():
    """Test listing plugins when type has no registered plugins."""
    registry = PluginRegistry()
    registry.register(FakeScanner)

    trainers = registry.list_by_type(PluginType.TRAINER)
    assert len(trainers) == 0


def test_registry_list_all():
    """Test listing all registered plugins."""
    registry = PluginRegistry()
    registry.register(FakeScanner)
    registry.register(FakeAttacker)

    all_plugins = registry.list_all()
    assert len(all_plugins) == 2
    names = {p.name for p in all_plugins}
    assert names == {"fake_scanner", "fake_attacker"}


def test_registry_list_all_empty():
    """Test listing all plugins when registry is empty."""
    registry = PluginRegistry()
    all_plugins = registry.list_all()
    assert len(all_plugins) == 0


def test_registry_unregister():
    """Test unregistering a plugin."""
    registry = PluginRegistry()
    registry.register(FakeScanner)
    assert registry.get("scanner", "fake_scanner") is not None

    result = registry.unregister("scanner", "fake_scanner")
    assert result is True
    assert registry.get("scanner", "fake_scanner") is None


def test_registry_unregister_nonexistent():
    """Test unregistering a plugin that doesn't exist."""
    registry = PluginRegistry()
    result = registry.unregister("scanner", "nonexistent")
    assert result is False


def test_registry_unregister_invalid_type():
    """Test unregistering from an invalid type."""
    registry = PluginRegistry()
    result = registry.unregister("invalid_type", "some_plugin")
    assert result is False


def test_registry_register_instance():
    """Test registering a plugin instance directly."""
    registry = PluginRegistry()
    scanner = FakeScanner()
    registry.register_instance(scanner)

    plugin = registry.get("scanner", "fake_scanner")
    assert plugin is scanner


def test_registry_has():
    """Test checking if a plugin exists."""
    registry = PluginRegistry()
    registry.register(FakeScanner)

    assert registry.has("scanner", "fake_scanner") is True
    assert registry.has("scanner", "nonexistent") is False
    assert registry.has("invalid_type", "fake_scanner") is False


def test_registry_multiple_same_type():
    """Test registering multiple plugins of the same type."""
    registry = PluginRegistry()
    registry.register(FakeScanner)
    registry.register(FakeReporter)

    # Create another scanner
    class AnotherScanner(BasePlugin):
        name = "another_scanner"
        version = "1.0.0"
        plugin_type = PluginType.SCANNER

        def configure(self, config): pass
        def execute(self, context): return PluginResult(success=True, data={})
        def cleanup(self): pass

    registry.register(AnotherScanner)

    scanners = registry.list_by_type(PluginType.SCANNER)
    assert len(scanners) == 2
    names = {p.name for p in scanners}
    assert names == {"fake_scanner", "another_scanner"}


def test_registry_overwrite_same_name():
    """Test that registering a plugin with the same name overwrites."""
    registry = PluginRegistry()

    class ScannerV1(BasePlugin):
        name = "my_scanner"
        version = "1.0.0"
        plugin_type = PluginType.SCANNER

        def configure(self, config): pass
        def execute(self, context): return PluginResult(success=True, data={})
        def cleanup(self): pass

    class ScannerV2(BasePlugin):
        name = "my_scanner"
        version = "2.0.0"
        plugin_type = PluginType.SCANNER

        def configure(self, config): pass
        def execute(self, context): return PluginResult(success=True, data={})
        def cleanup(self): pass

    registry.register(ScannerV1)
    plugin = registry.get("scanner", "my_scanner")
    assert plugin.version == "1.0.0"

    registry.register(ScannerV2)
    plugin = registry.get("scanner", "my_scanner")
    assert plugin.version == "2.0.0"


def test_registry_repr():
    """Test registry string representation."""
    registry = PluginRegistry()
    registry.register(FakeScanner)
    registry.register(FakeAttacker)

    repr_str = repr(registry)
    assert "PluginRegistry" in repr_str
    assert "2 plugins" in repr_str
