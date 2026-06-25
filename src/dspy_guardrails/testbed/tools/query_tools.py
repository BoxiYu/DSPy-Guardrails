"""
查询类工具 - Query Tools
只读操作，查询航班、订单、客户等信息
"""

from ..data.bookings import get_booking as get_booking_db
from ..data.customers import get_customer, get_customer_by_phone
from ..data.flights import get_flight
from ..data.flights import search_flights as search_flights_db
from ..data.seat_maps import get_available_seats_by_preference, get_flight_availability
from .base import BaseTool, ToolCategory, ToolResult


class QueryFlightStatusTool(BaseTool):
    """查询航班状态工具"""

    @property
    def name(self) -> str:
        return "query_flight_status"

    @property
    def description(self) -> str:
        return "查询航班实时状态，包括出发时间、到达时间、登机口、延误信息等"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.QUERY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "flight_number": {
                    "type": "string",
                    "description": "航班号，如 CA1234"
                }
            },
            "required": ["flight_number"]
        }

    def execute(self, flight_number: str) -> ToolResult:
        flight = get_flight(flight_number)
        if not flight:
            return ToolResult(
                success=False,
                error=f"未找到航班 {flight_number}"
            )
        return ToolResult(
            success=True,
            data=flight,
            metadata={"flight_number": flight_number}
        )


class GetBookingTool(BaseTool):
    """获取订单信息工具"""

    @property
    def name(self) -> str:
        return "get_booking"

    @property
    def description(self) -> str:
        return "根据确认码获取订单详细信息，包括乘客、航班、座位、状态等"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.QUERY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "confirmation_code": {
                    "type": "string",
                    "description": "订单确认码，如 ABC123"
                }
            },
            "required": ["confirmation_code"]
        }

    def execute(self, confirmation_code: str) -> ToolResult:
        booking = get_booking_db(confirmation_code)
        if not booking:
            return ToolResult(
                success=False,
                error=f"未找到订单 {confirmation_code}"
            )
        return ToolResult(
            success=True,
            data=booking,
            metadata={"confirmation_code": confirmation_code}
        )


class SearchFlightsTool(BaseTool):
    """搜索航班工具"""

    @property
    def name(self) -> str:
        return "search_flights"

    @property
    def description(self) -> str:
        return "根据出发地和目的地搜索可用航班"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.QUERY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "出发机场代码，如 PEK (北京)"
                },
                "destination": {
                    "type": "string",
                    "description": "到达机场代码，如 SHA (上海)"
                },
                "date": {
                    "type": "string",
                    "description": "出发日期，格式 YYYY-MM-DD（可选）"
                }
            },
            "required": ["origin", "destination"]
        }

    def execute(self, origin: str, destination: str, date: str = None) -> ToolResult:
        flights = search_flights_db(origin, destination, date)
        if not flights:
            return ToolResult(
                success=True,
                data=[],
                metadata={
                    "origin": origin,
                    "destination": destination,
                    "message": "未找到符合条件的航班"
                }
            )
        return ToolResult(
            success=True,
            data=flights,
            metadata={
                "origin": origin,
                "destination": destination,
                "count": len(flights)
            }
        )


class GetCustomerProfileTool(BaseTool):
    """获取客户信息工具"""

    @property
    def name(self) -> str:
        return "get_customer_profile"

    @property
    def description(self) -> str:
        return "获取客户详细信息，包括会员等级、里程余额、偏好设置等"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.QUERY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "客户ID，如 C001"
                },
                "phone": {
                    "type": "string",
                    "description": "客户电话号码（可选，用于查找客户）"
                }
            },
            "required": []
        }

    def execute(self, customer_id: str = None, phone: str = None) -> ToolResult:
        customer = None
        if customer_id:
            customer = get_customer(customer_id)
        elif phone:
            customer = get_customer_by_phone(phone)

        if not customer:
            return ToolResult(
                success=False,
                error="未找到客户信息，请提供客户ID或电话号码"
            )

        # 脱敏处理敏感信息
        safe_customer = customer.copy()
        if "id_number" in safe_customer:
            id_num = safe_customer["id_number"]
            safe_customer["id_number"] = id_num[:6] + "****" + id_num[-4:] if len(id_num) > 10 else "****"
        if "phone" in safe_customer and safe_customer["phone"]:
            phone_num = safe_customer["phone"]
            if len(phone_num) >= 11:
                safe_customer["phone"] = phone_num[:3] + "****" + phone_num[-4:]

        return ToolResult(
            success=True,
            data=safe_customer,
            metadata={"customer_id": customer.get("customer_id")}
        )


class GetAvailableSeatsTool(BaseTool):
    """获取可用座位工具"""

    @property
    def name(self) -> str:
        return "get_available_seats"

    @property
    def description(self) -> str:
        return "查询航班可用座位，可按偏好筛选（靠窗/靠走道）"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.QUERY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "flight_number": {
                    "type": "string",
                    "description": "航班号"
                },
                "preference": {
                    "type": "string",
                    "enum": ["window", "aisle", "middle", "any"],
                    "description": "座位偏好：window(靠窗), aisle(靠走道), middle(中间), any(任意)"
                }
            },
            "required": ["flight_number"]
        }

    def execute(self, flight_number: str, preference: str = "any") -> ToolResult:
        availability = get_flight_availability(flight_number)
        if not availability:
            return ToolResult(
                success=False,
                error=f"未找到航班 {flight_number} 的座位信息"
            )

        seats = get_available_seats_by_preference(flight_number, preference)
        return ToolResult(
            success=True,
            data={
                "flight": flight_number,
                "preference": preference,
                "available_seats": seats,
                "count": len(seats)
            },
            metadata={"flight_number": flight_number}
        )
