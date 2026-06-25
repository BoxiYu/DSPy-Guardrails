"""
工具基类 - Tool Base Classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolCategory(Enum):
    """工具类别"""
    QUERY = "query"           # 查询类 - 只读
    MODIFY = "modify"         # 修改类 - 写操作
    EXTERNAL = "external"     # 外部API
    KNOWLEDGE = "knowledge"   # 知识库


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return f"Success: {self.data}"
        return f"Error: {self.error}"


class BaseTool(ABC):
    """工具基类"""

    def __init__(self):
        self._call_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """工具类别"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """参数定义（JSON Schema格式）"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass

    def __call__(self, **kwargs) -> ToolResult:
        """调用工具"""
        self._call_count += 1
        return self.execute(**kwargs)

    @property
    def call_count(self) -> int:
        """调用次数"""
        return self._call_count

    def reset_count(self):
        """重置调用计数"""
        self._call_count = 0

    def to_openai_function(self) -> dict:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_dspy_tool(self) -> dict:
        """转换为 DSPy Tool 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "execute": self.execute,
        }


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有工具"""
        return list(self._tools.keys())

    def list_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """按类别列出工具"""
        return [t for t in self._tools.values() if t.category == category]

    def get_all(self) -> list[BaseTool]:
        """获取所有工具"""
        return list(self._tools.values())

    def reset_all_counts(self):
        """重置所有工具调用计数"""
        for tool in self._tools.values():
            tool.reset_count()


# 全局工具注册表
tool_registry = ToolRegistry()
