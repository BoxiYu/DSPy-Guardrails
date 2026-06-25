"""
修改类工具 - Modify Tools
写操作，修改订单、座位、客户信息等
"""

from datetime import datetime

from ..data.bookings import get_booking
from ..data.customers import get_customer
from ..data.seat_maps import is_seat_available
from .base import BaseTool, ToolCategory, ToolResult


class CancelBookingTool(BaseTool):
    """取消订单工具"""

    @property
    def name(self) -> str:
        return "cancel_booking"

    @property
    def description(self) -> str:
        return "取消航班订单，计算退款金额并处理退款"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.MODIFY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "confirmation_code": {
                    "type": "string",
                    "description": "订单确认码"
                },
                "reason": {
                    "type": "string",
                    "description": "取消原因"
                }
            },
            "required": ["confirmation_code", "reason"]
        }

    def execute(self, confirmation_code: str, reason: str) -> ToolResult:
        booking = get_booking(confirmation_code)
        if not booking:
            return ToolResult(
                success=False,
                error=f"未找到订单 {confirmation_code}"
            )

        # 检查订单状态
        status = booking.get("status")
        if status == "Cancelled":
            return ToolResult(
                success=False,
                error="订单已取消，无需重复操作"
            )
        if status == "Completed":
            return ToolResult(
                success=False,
                error="订单已完成，无法取消"
            )

        # 计算退款金额（模拟）
        original_price = booking.get("price", 0)
        cancellation_fee = original_price * 0.1  # 10% 手续费
        refund_amount = original_price - cancellation_fee

        # 更新订单状态（模拟）
        # 实际实现中这里会更新数据库
        result_data = {
            "confirmation_code": confirmation_code,
            "previous_status": status,
            "new_status": "Cancelled",
            "original_price": original_price,
            "cancellation_fee": cancellation_fee,
            "refund_amount": refund_amount,
            "reason": reason,
            "cancelled_at": datetime.now().isoformat()
        }

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "confirmation_code": confirmation_code,
                "action": "cancel"
            }
        )


class ChangeSeatTool(BaseTool):
    """改签座位工具"""

    @property
    def name(self) -> str:
        return "change_seat"

    @property
    def description(self) -> str:
        return "更换航班座位，可能产生选座费用"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.MODIFY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "confirmation_code": {
                    "type": "string",
                    "description": "订单确认码"
                },
                "new_seat": {
                    "type": "string",
                    "description": "新座位号，如 12A"
                }
            },
            "required": ["confirmation_code", "new_seat"]
        }

    def execute(self, confirmation_code: str, new_seat: str) -> ToolResult:
        booking = get_booking(confirmation_code)
        if not booking:
            return ToolResult(
                success=False,
                error=f"未找到订单 {confirmation_code}"
            )

        # 检查订单状态
        status = booking.get("status")
        if status in ["Cancelled", "Completed"]:
            return ToolResult(
                success=False,
                error=f"订单状态为 {status}，无法更换座位"
            )

        flight = booking.get("flight")
        old_seat = booking.get("seat")

        # 检查座位是否可用
        if not is_seat_available(flight, new_seat):
            return ToolResult(
                success=False,
                error=f"座位 {new_seat} 不可用，请选择其他座位"
            )

        # 计算选座费用（模拟）
        seat_fee = 0
        if new_seat.endswith("A") or new_seat.endswith("K") or new_seat.endswith("H"):
            seat_fee = 50  # 靠窗
        row = int("".join(filter(str.isdigit, new_seat)))
        if row <= 10:
            seat_fee += 80  # 前排

        result_data = {
            "confirmation_code": confirmation_code,
            "flight": flight,
            "old_seat": old_seat,
            "new_seat": new_seat,
            "seat_fee": seat_fee,
            "changed_at": datetime.now().isoformat()
        }

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "confirmation_code": confirmation_code,
                "action": "change_seat"
            }
        )


class UpdateCustomerInfoTool(BaseTool):
    """更新客户信息工具"""

    @property
    def name(self) -> str:
        return "update_customer_info"

    @property
    def description(self) -> str:
        return "更新客户联系方式、偏好设置等信息"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.MODIFY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "客户ID"
                },
                "updates": {
                    "type": "object",
                    "description": "要更新的字段",
                    "properties": {
                        "phone": {"type": "string", "description": "新电话号码"},
                        "email": {"type": "string", "description": "新邮箱地址"},
                        "seat_preference": {
                            "type": "string",
                            "enum": ["window", "aisle", "no_preference"],
                            "description": "座位偏好"
                        },
                        "meal_preference": {
                            "type": "string",
                            "description": "餐食偏好"
                        }
                    }
                }
            },
            "required": ["customer_id", "updates"]
        }

    def execute(self, customer_id: str, updates: dict) -> ToolResult:
        customer = get_customer(customer_id)
        if not customer:
            return ToolResult(
                success=False,
                error=f"未找到客户 {customer_id}"
            )

        # 验证更新字段
        allowed_fields = ["phone", "email", "seat_preference", "meal_preference"]
        invalid_fields = [k for k in updates.keys() if k not in allowed_fields]
        if invalid_fields:
            return ToolResult(
                success=False,
                error=f"不允许更新的字段: {invalid_fields}"
            )

        # 模拟更新
        old_values = {}
        for field in updates:
            if field in customer:
                old_values[field] = customer[field]
            elif field == "seat_preference" and "preferences" in customer:
                old_values[field] = customer["preferences"].get("seat")
            elif field == "meal_preference" and "preferences" in customer:
                old_values[field] = customer["preferences"].get("meal")

        result_data = {
            "customer_id": customer_id,
            "old_values": old_values,
            "new_values": updates,
            "updated_at": datetime.now().isoformat()
        }

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "customer_id": customer_id,
                "action": "update_info"
            }
        )


class CreateRefundRequestTool(BaseTool):
    """创建退款请求工具"""

    @property
    def name(self) -> str:
        return "create_refund_request"

    @property
    def description(self) -> str:
        return "创建退款申请，支持部分退款或全额退款"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.MODIFY

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "booking_code": {
                    "type": "string",
                    "description": "订单确认码"
                },
                "reason": {
                    "type": "string",
                    "description": "退款原因"
                },
                "amount": {
                    "type": "number",
                    "description": "退款金额（可选，不填则全额退款）"
                }
            },
            "required": ["booking_code", "reason"]
        }

    def execute(self, booking_code: str, reason: str, amount: float = None) -> ToolResult:
        booking = get_booking(booking_code)
        if not booking:
            return ToolResult(
                success=False,
                error=f"未找到订单 {booking_code}"
            )

        original_price = booking.get("price", 0)
        refund_amount = amount if amount else original_price

        # 验证退款金额
        if refund_amount > original_price:
            return ToolResult(
                success=False,
                error=f"退款金额 {refund_amount} 超过订单金额 {original_price}"
            )

        # 生成退款单号
        import random
        refund_id = f"REF{random.randint(100, 999)}"

        result_data = {
            "refund_id": refund_id,
            "booking_code": booking_code,
            "original_amount": original_price,
            "refund_amount": refund_amount,
            "reason": reason,
            "status": "Pending",
            "estimated_completion": "3-5个工作日",
            "created_at": datetime.now().isoformat()
        }

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "booking_code": booking_code,
                "action": "create_refund"
            }
        )
