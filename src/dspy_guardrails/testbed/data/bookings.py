"""
订单数据 - Mock Bookings Database
"""

BOOKINGS_DB = {
    # 正常订单
    "ABC123": {
        "confirmation_code": "ABC123",
        "passenger_name": "张三",
        "passenger_id": "C001",
        "flight": "CA1234",
        "seat": "12A",
        "cabin_class": "Economy",
        "price": 1280,
        "currency": "CNY",
        "status": "Confirmed",
        "created_at": "2025-01-10 09:30:00",
        "meal_preference": "Standard",
        "baggage_allowance": "23kg",
        "frequent_flyer": "CA88888888",
        "contact_phone": "13812345678",
        "contact_email": "zhangsan@example.com",
    },
    "DEF456": {
        "confirmation_code": "DEF456",
        "passenger_name": "李四",
        "passenger_id": "C002",
        "flight": "MU5678",
        "seat": "3C",
        "cabin_class": "Business",
        "price": 4580,
        "currency": "CNY",
        "status": "Confirmed",
        "created_at": "2025-01-08 14:22:00",
        "meal_preference": "Vegetarian",
        "baggage_allowance": "32kg x 2",
        "frequent_flyer": "MU12345678",
        "contact_phone": "13987654321",
        "contact_email": "lisi@company.com",
        "lounge_access": True,
        "priority_boarding": True,
    },
    "GHI789": {
        "confirmation_code": "GHI789",
        "passenger_name": "王五",
        "passenger_id": "C003",
        "flight": "HU7890",
        "seat": "22F",
        "cabin_class": "Economy",
        "price": 980,
        "currency": "CNY",
        "status": "Affected",
        "affected_reason": "航班取消",
        "rebooking_options": ["HU7892", "CA4102"],
        "created_at": "2025-01-05 11:15:00",
        "meal_preference": "Standard",
        "baggage_allowance": "23kg",
        "contact_phone": "13611112222",
        "contact_email": "wangwu@test.com",
    },
    "JKL012": {
        "confirmation_code": "JKL012",
        "passenger_name": "赵六",
        "passenger_id": "C004",
        "flight": "CA985",
        "seat": "1A",
        "cabin_class": "First",
        "price": 42000,
        "currency": "CNY",
        "status": "Confirmed",
        "created_at": "2025-01-02 08:00:00",
        "meal_preference": "Chef Special",
        "baggage_allowance": "40kg x 3",
        "frequent_flyer": "CA00000001",
        "contact_phone": "13900001111",
        "contact_email": "zhao.liu@corp.com",
        "lounge_access": True,
        "priority_boarding": True,
        "chauffeur_service": True,
        "vip_services": ["专属休息室", "专车接送", "快速通道"],
    },
    # 已取消订单
    "MNO345": {
        "confirmation_code": "MNO345",
        "passenger_name": "钱七",
        "passenger_id": "C005",
        "flight": "CZ3456",
        "seat": "18B",
        "cabin_class": "Economy",
        "price": 1180,
        "currency": "CNY",
        "status": "Cancelled",
        "cancelled_at": "2025-01-12 16:30:00",
        "cancel_reason": "行程变更",
        "refund_status": "Completed",
        "refund_amount": 1062,  # 扣除手续费
        "refund_id": "REF001",
        "created_at": "2025-01-03 10:00:00",
    },
    # 待支付订单
    "PQR678": {
        "confirmation_code": "PQR678",
        "passenger_name": "孙八",
        "passenger_id": "C006",
        "flight": "FM9012",
        "seat": "25D",
        "cabin_class": "Economy",
        "price": 1560,
        "currency": "CNY",
        "status": "Pending Payment",
        "payment_deadline": "2025-01-16 18:00:00",
        "created_at": "2025-01-15 12:00:00",
        "hold_until": "2025-01-16 18:00:00",
        "contact_phone": "13722223333",
    },
    # 已完成行程
    "STU901": {
        "confirmation_code": "STU901",
        "passenger_name": "周九",
        "passenger_id": "C007",
        "flight": "CA1888",
        "seat": "8A",
        "cabin_class": "Economy",
        "price": 1380,
        "currency": "CNY",
        "status": "Completed",
        "created_at": "2025-01-12 20:00:00",
        "checked_in_at": "2025-01-15 04:30:00",
        "boarded_at": "2025-01-15 05:45:00",
        "meal_preference": "Standard",
        "baggage_allowance": "23kg",
        "actual_baggage": "18kg",
    },
    # 改签订单
    "VWX234": {
        "confirmation_code": "VWX234",
        "passenger_name": "吴十",
        "passenger_id": "C008",
        "flight": "MU2345",
        "seat": "15C",
        "cabin_class": "Economy",
        "price": 890,
        "currency": "CNY",
        "status": "Changed",
        "original_flight": "MU2343",
        "change_fee": 100,
        "change_reason": "时间冲突",
        "changed_at": "2025-01-14 09:00:00",
        "created_at": "2025-01-10 15:30:00",
    },
    # 国际航班订单
    "YZA567": {
        "confirmation_code": "YZA567",
        "passenger_name": "郑十一",
        "passenger_id": "C009",
        "flight": "MU583",
        "seat": "35K",
        "cabin_class": "Economy",
        "price": 8900,
        "currency": "CNY",
        "status": "Confirmed",
        "created_at": "2025-01-05 18:00:00",
        "passport_number": "E12345678",
        "passport_expiry": "2028-05-20",
        "visa_status": "ESTA Approved",
        "meal_preference": "Asian Vegetarian",
        "special_request": "靠窗座位",
    },
    # 团队订单
    "BCD890": {
        "confirmation_code": "BCD890",
        "passenger_name": "华为出差团",
        "passenger_id": "C010",
        "flight": "CA173",
        "seats": ["20A", "20B", "20C", "21A", "21B", "21C"],
        "passenger_count": 6,
        "cabin_class": "Economy",
        "price": 15600,
        "unit_price": 2600,
        "currency": "CNY",
        "status": "Confirmed",
        "created_at": "2025-01-08 10:00:00",
        "group_leader": "王经理",
        "corporate_account": "CORP-HW-2024",
        "invoice_required": True,
        "invoice_title": "华为技术有限公司",
    },
    # 婴儿票
    "EFG123": {
        "confirmation_code": "EFG123",
        "passenger_name": "李小明",
        "passenger_id": "C011",
        "flight": "CZ6789",
        "seat": "30A",  # 与成人同坐
        "cabin_class": "Economy",
        "price": 98,  # 婴儿票价
        "currency": "CNY",
        "status": "Confirmed",
        "ticket_type": "Infant",
        "date_of_birth": "2024-06-15",
        "accompanying_adult": "HIJ456",
        "bassinet_requested": True,
        "created_at": "2025-01-11 14:00:00",
    },
    "HIJ456": {
        "confirmation_code": "HIJ456",
        "passenger_name": "李妈妈",
        "passenger_id": "C012",
        "flight": "CZ6789",
        "seat": "30A",
        "cabin_class": "Economy",
        "price": 1280,
        "currency": "CNY",
        "status": "Confirmed",
        "infant_booking": "EFG123",
        "created_at": "2025-01-11 14:00:00",
        "special_request": "靠近洗手间座位",
    },
}


def get_booking(confirmation_code: str) -> dict:
    """获取订单信息"""
    return BOOKINGS_DB.get(confirmation_code.upper())


def get_bookings_by_passenger(passenger_id: str) -> list:
    """按乘客获取订单"""
    return [b for b in BOOKINGS_DB.values() if b.get("passenger_id") == passenger_id]


def get_bookings_by_flight(flight_number: str) -> list:
    """按航班获取订单"""
    return [b for b in BOOKINGS_DB.values() if b.get("flight") == flight_number.upper()]


def get_bookings_by_status(status: str) -> list:
    """按状态获取订单"""
    return [b for b in BOOKINGS_DB.values() if b.get("status") == status]
