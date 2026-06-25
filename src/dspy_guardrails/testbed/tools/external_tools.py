"""
外部API工具 - External API Tools
调用外部服务，如天气、酒店、通知等
"""

import random
from datetime import datetime

from ..data.hotels import calculate_arrangement_cost, get_available_hotels
from .base import BaseTool, ToolCategory, ToolResult


class SearchHotelsTool(BaseTool):
    """搜索酒店工具"""

    @property
    def name(self) -> str:
        return "search_hotels"

    @property
    def description(self) -> str:
        return "搜索机场附近的酒店，用于航班延误或取消时的住宿安排"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXTERNAL

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "airport_code": {
                    "type": "string",
                    "description": "机场代码，如 PEK (北京首都)"
                },
                "room_type": {
                    "type": "string",
                    "enum": ["Standard", "Deluxe", "Suite"],
                    "description": "房间类型"
                },
                "nights": {
                    "type": "integer",
                    "description": "入住晚数",
                    "default": 1
                }
            },
            "required": ["airport_code"]
        }

    def execute(self, airport_code: str, room_type: str = "Standard", nights: int = 1) -> ToolResult:
        hotels = get_available_hotels(airport_code, room_type)

        if not hotels:
            return ToolResult(
                success=True,
                data=[],
                metadata={
                    "airport_code": airport_code,
                    "message": f"未找到 {airport_code} 机场附近的可用酒店"
                }
            )

        # 计算费用
        for hotel in hotels:
            cost_info = calculate_arrangement_cost(
                hotel["hotel"]["hotel_id"],
                room_type,
                nights
            )
            hotel["estimated_cost"] = cost_info

        return ToolResult(
            success=True,
            data=hotels,
            metadata={
                "airport_code": airport_code,
                "room_type": room_type,
                "nights": nights,
                "count": len(hotels)
            }
        )


class SendNotificationTool(BaseTool):
    """发送通知工具"""

    @property
    def name(self) -> str:
        return "send_notification"

    @property
    def description(self) -> str:
        return "向客户发送短信或邮件通知"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXTERNAL

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "客户ID"
                },
                "channel": {
                    "type": "string",
                    "enum": ["sms", "email", "both"],
                    "description": "通知渠道"
                },
                "message": {
                    "type": "string",
                    "description": "通知内容"
                },
                "template": {
                    "type": "string",
                    "enum": ["flight_change", "booking_confirm", "refund_status", "general"],
                    "description": "消息模板类型"
                }
            },
            "required": ["customer_id", "channel", "message"]
        }

    def execute(self, customer_id: str, channel: str, message: str, template: str = "general") -> ToolResult:
        # 模拟发送通知
        from ..data.customers import get_customer

        customer = get_customer(customer_id)
        if not customer:
            return ToolResult(
                success=False,
                error=f"未找到客户 {customer_id}"
            )

        notifications_sent = []

        if channel in ["sms", "both"]:
            phone = customer.get("phone")
            if phone:
                notifications_sent.append({
                    "channel": "sms",
                    "recipient": phone[:3] + "****" + phone[-4:] if len(phone) >= 11 else "****",
                    "status": "Sent",
                    "message_id": f"SMS{random.randint(10000, 99999)}"
                })
            else:
                notifications_sent.append({
                    "channel": "sms",
                    "status": "Failed",
                    "error": "客户未提供电话号码"
                })

        if channel in ["email", "both"]:
            email = customer.get("email")
            if email:
                notifications_sent.append({
                    "channel": "email",
                    "recipient": email[:3] + "***@" + email.split("@")[-1] if "@" in email else "***",
                    "status": "Sent",
                    "message_id": f"EMAIL{random.randint(10000, 99999)}"
                })
            else:
                notifications_sent.append({
                    "channel": "email",
                    "status": "Failed",
                    "error": "客户未提供邮箱地址"
                })

        success = any(n["status"] == "Sent" for n in notifications_sent)

        return ToolResult(
            success=success,
            data={
                "customer_id": customer_id,
                "notifications": notifications_sent,
                "template": template,
                "sent_at": datetime.now().isoformat()
            },
            error=None if success else "所有通知渠道发送失败",
            metadata={
                "customer_id": customer_id,
                "action": "send_notification"
            }
        )


class CheckWeatherTool(BaseTool):
    """查询天气工具"""

    @property
    def name(self) -> str:
        return "check_weather"

    @property
    def description(self) -> str:
        return "查询机场所在城市的天气情况，用于判断航班是否可能受影响"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXTERNAL

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "airport_code": {
                    "type": "string",
                    "description": "机场代码"
                },
                "date": {
                    "type": "string",
                    "description": "查询日期，格式 YYYY-MM-DD"
                }
            },
            "required": ["airport_code"]
        }

    def execute(self, airport_code: str, date: str = None) -> ToolResult:
        # 模拟天气数据
        weather_data = {
            "PEK": {"city": "北京", "temp": 2, "condition": "晴", "wind": "北风3级", "visibility": "良好"},
            "SHA": {"city": "上海", "temp": 8, "condition": "多云", "wind": "东风2级", "visibility": "良好"},
            "CAN": {"city": "广州", "temp": 18, "condition": "阴", "wind": "南风2级", "visibility": "一般"},
            "SZX": {"city": "深圳", "temp": 20, "condition": "晴", "wind": "东南风2级", "visibility": "良好"},
            "CTU": {"city": "成都", "temp": 6, "condition": "雾", "wind": "静风", "visibility": "差"},
            "HGH": {"city": "杭州", "temp": 10, "condition": "小雨", "wind": "东北风3级", "visibility": "一般"},
            "XIY": {"city": "西安", "temp": 0, "condition": "雪", "wind": "北风4级", "visibility": "差"},
            "KMG": {"city": "昆明", "temp": 15, "condition": "晴", "wind": "西南风2级", "visibility": "良好"},
        }

        airport = airport_code.upper()
        if airport not in weather_data:
            # 生成随机天气
            weather = {
                "city": airport,
                "temp": random.randint(-5, 30),
                "condition": random.choice(["晴", "多云", "阴", "小雨", "雾"]),
                "wind": f"{random.choice(['东', '南', '西', '北'])}风{random.randint(1, 4)}级",
                "visibility": random.choice(["良好", "一般", "差"])
            }
        else:
            weather = weather_data[airport]

        # 评估对航班的影响
        flight_impact = "无影响"
        if weather["condition"] in ["雾", "雪", "大雨", "暴雨"]:
            flight_impact = "可能延误"
        if weather["visibility"] == "差":
            flight_impact = "可能延误或取消"

        weather["flight_impact"] = flight_impact
        weather["airport_code"] = airport
        weather["query_date"] = date or datetime.now().strftime("%Y-%m-%d")

        return ToolResult(
            success=True,
            data=weather,
            metadata={
                "airport_code": airport,
                "date": date
            }
        )
