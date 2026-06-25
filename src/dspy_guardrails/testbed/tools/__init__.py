"""
工具模块 - Tools Module

提供航空客服场景的各类工具，分为四大类：
- 查询类 (Query): 只读操作，查询航班、订单、客户信息
- 修改类 (Modify): 写操作，取消订单、改签座位、更新信息
- 外部API类 (External): 调用外部服务，天气、酒店、通知
- 知识库类 (Knowledge): 搜索FAQ、政策文档
"""

from .base import (
    BaseTool,
    ToolCategory,
    ToolRegistry,
    ToolResult,
    tool_registry,
)
from .external_tools import (
    CheckWeatherTool,
    SearchHotelsTool,
    SendNotificationTool,
)
from .knowledge_tools import (
    GetPolicyTool,
    SearchFAQTool,
)
from .modify_tools import (
    CancelBookingTool,
    ChangeSeatTool,
    CreateRefundRequestTool,
    UpdateCustomerInfoTool,
)
from .query_tools import (
    GetAvailableSeatsTool,
    GetBookingTool,
    GetCustomerProfileTool,
    QueryFlightStatusTool,
    SearchFlightsTool,
)

# 注册所有工具到全局注册表
_all_tools = [
    # 查询类工具 (5)
    QueryFlightStatusTool(),
    GetBookingTool(),
    SearchFlightsTool(),
    GetCustomerProfileTool(),
    GetAvailableSeatsTool(),

    # 修改类工具 (4)
    CancelBookingTool(),
    ChangeSeatTool(),
    UpdateCustomerInfoTool(),
    CreateRefundRequestTool(),

    # 外部API类工具 (3)
    SearchHotelsTool(),
    SendNotificationTool(),
    CheckWeatherTool(),

    # 知识库类工具 (2)
    SearchFAQTool(),
    GetPolicyTool(),
]

for tool in _all_tools:
    tool_registry.register(tool)


def get_all_tools() -> list[BaseTool]:
    """获取所有工具"""
    return tool_registry.get_all()


def get_tool(name: str) -> BaseTool:
    """获取指定工具"""
    return tool_registry.get(name)


def get_tools_by_category(category: ToolCategory) -> list[BaseTool]:
    """按类别获取工具"""
    return tool_registry.list_by_category(category)


def get_tools_as_openai_functions() -> list[dict]:
    """获取所有工具的 OpenAI Function 格式"""
    return [tool.to_openai_function() for tool in get_all_tools()]


__all__ = [
    # 基类
    "BaseTool",
    "ToolCategory",
    "ToolResult",
    "ToolRegistry",
    "tool_registry",

    # 查询类工具
    "QueryFlightStatusTool",
    "GetBookingTool",
    "SearchFlightsTool",
    "GetCustomerProfileTool",
    "GetAvailableSeatsTool",

    # 修改类工具
    "CancelBookingTool",
    "ChangeSeatTool",
    "UpdateCustomerInfoTool",
    "CreateRefundRequestTool",

    # 外部API类工具
    "SearchHotelsTool",
    "SendNotificationTool",
    "CheckWeatherTool",

    # 知识库类工具
    "SearchFAQTool",
    "GetPolicyTool",

    # 辅助函数
    "get_all_tools",
    "get_tool",
    "get_tools_by_category",
    "get_tools_as_openai_functions",
]
