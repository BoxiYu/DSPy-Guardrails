"""Plugin registry for discovering and managing plugins."""


from .base import BasePlugin, PluginType


class PluginRegistry:
    """
    插件注册表

    管理插件的注册、查找、列举和注销。
    插件按类型分组存储，支持通过类型和名称快速查找。

    Usage:
        registry = PluginRegistry()
        registry.register(MyScanner)
        plugin = registry.get("scanner", "my_scanner")
    """

    def __init__(self):
        """初始化注册表，为每种插件类型创建空字典。"""
        self._plugins: dict[str, dict[str, BasePlugin]] = {
            t.value: {} for t in PluginType
        }

    def register(self, plugin_cls: type[BasePlugin]) -> None:
        """
        注册插件类（自动实例化）。

        Args:
            plugin_cls: 插件类（必须继承 BasePlugin）
        """
        plugin = plugin_cls()
        self.register_instance(plugin)

    def register_instance(self, plugin: BasePlugin) -> None:
        """
        注册插件实例。

        Args:
            plugin: 已实例化的插件对象
        """
        type_key = plugin.plugin_type.value
        self._plugins[type_key][plugin.name] = plugin

    def unregister(self, plugin_type: str, name: str) -> bool:
        """
        注销插件。

        Args:
            plugin_type: 插件类型字符串 (e.g., "scanner", "attacker")
            name: 插件名称

        Returns:
            True 如果成功注销，False 如果插件不存在
        """
        if plugin_type in self._plugins and name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][name]
            return True
        return False

    def get(self, plugin_type: str, name: str) -> BasePlugin | None:
        """
        获取指定插件。

        Args:
            plugin_type: 插件类型字符串
            name: 插件名称

        Returns:
            插件实例，不存在则返回 None
        """
        return self._plugins.get(plugin_type, {}).get(name)

    def list_by_type(self, plugin_type: PluginType) -> list[BasePlugin]:
        """
        列出指定类型的所有插件。

        Args:
            plugin_type: 插件类型枚举

        Returns:
            该类型下所有已注册的插件列表
        """
        return list(self._plugins.get(plugin_type.value, {}).values())

    def list_all(self) -> list[BasePlugin]:
        """
        列出所有已注册的插件。

        Returns:
            所有插件的列表
        """
        all_plugins = []
        for type_plugins in self._plugins.values():
            all_plugins.extend(type_plugins.values())
        return all_plugins

    def has(self, plugin_type: str, name: str) -> bool:
        """
        检查插件是否存在。

        Args:
            plugin_type: 插件类型字符串
            name: 插件名称

        Returns:
            True 如果插件存在，False 否则
        """
        return name in self._plugins.get(plugin_type, {})

    def __repr__(self) -> str:
        """返回注册表的字符串表示。"""
        total = len(self.list_all())
        return f"<PluginRegistry {total} plugins>"
