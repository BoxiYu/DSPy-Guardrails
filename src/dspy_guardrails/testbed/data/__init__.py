"""
Mock Data Module - 实验用模拟数据

提供航空客服场景的完整模拟数据，用于测试 Agent 和 Guardrail。
"""

from .agent_notes import AGENT_NOTES_DB
from .audit_logs import AUDIT_LOGS_DB
from .baggage import BAGGAGE_CLAIMS_DB, BAGGAGE_DB
from .bookings import BOOKINGS_DB
from .customers import CUSTOMERS_DB
from .flights import FLIGHTS_DB
from .hotels import HOTELS_DB
from .internal import INTERNAL_DB
from .knowledge_base import KNOWLEDGE_BASE
from .payments import PAYMENTS_DB
from .promotions import PROMOTIONS_DB
from .refunds import REFUNDS_DB
from .seat_maps import SEAT_MAPS_DB
from .special_services import SPECIAL_SERVICES_DB
from .watchlist import WATCHLIST_DB

__all__ = [
    "FLIGHTS_DB",
    "BOOKINGS_DB",
    "CUSTOMERS_DB",
    "PAYMENTS_DB",
    "REFUNDS_DB",
    "BAGGAGE_DB",
    "BAGGAGE_CLAIMS_DB",
    "SPECIAL_SERVICES_DB",
    "PROMOTIONS_DB",
    "WATCHLIST_DB",
    "AGENT_NOTES_DB",
    "SEAT_MAPS_DB",
    "HOTELS_DB",
    "KNOWLEDGE_BASE",
    "INTERNAL_DB",
    "AUDIT_LOGS_DB",
]
