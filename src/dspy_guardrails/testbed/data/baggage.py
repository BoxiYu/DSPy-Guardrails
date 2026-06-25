"""
行李数据 - Mock Baggage Database
"""

BAGGAGE_DB = {
    "BAG001": {
        "baggage_id": "BAG001",
        "booking_code": "ABC123",
        "customer_id": "C001",
        "tag_number": "CA123456",
        "weight_kg": 18.5,
        "pieces": 1,
        "type": "Checked",
        "status": "Checked In",
        "flight": "CA1234",
        "origin": "PEK",
        "destination": "SHA",
        "checked_in_at": "2025-01-15 06:30:00",
        "description": "黑色行李箱",
        "dimensions": "56x36x23 cm",
        "special_handling": None,
    },
    "BAG002": {
        "baggage_id": "BAG002",
        "booking_code": "DEF456",
        "customer_id": "C002",
        "tag_number": "MU234567",
        "weight_kg": 25.0,
        "pieces": 2,
        "type": "Checked",
        "status": "In Transit",
        "flight": "MU5678",
        "origin": "SHA",
        "destination": "CAN",
        "checked_in_at": "2025-01-15 08:15:00",
        "description": "银色行李箱 x2",
        "dimensions": "68x45x28 cm",
        "special_handling": "易碎物品",
        "business_class_priority": True,
    },
    "BAG003": {
        "baggage_id": "BAG003",
        "booking_code": "JKL012",
        "customer_id": "C004",
        "tag_number": "CA345678",
        "weight_kg": 35.0,
        "pieces": 3,
        "type": "Checked",
        "status": "Priority",
        "flight": "CA985",
        "origin": "PEK",
        "destination": "JFK",
        "checked_in_at": "2025-01-15 10:00:00",
        "description": "路易威登行李箱套装",
        "dimensions": "Various",
        "special_handling": "VIP优先",
        "first_class_priority": True,
        "vip_handling": True,
    },
    "BAG004": {
        "baggage_id": "BAG004",
        "booking_code": "STU901",
        "customer_id": "C007",
        "tag_number": "CA456789",
        "weight_kg": 18.0,
        "pieces": 1,
        "type": "Checked",
        "status": "Delivered",
        "flight": "CA1888",
        "origin": "SHA",
        "destination": "PEK",
        "checked_in_at": "2025-01-15 04:30:00",
        "delivered_at": "2025-01-15 08:45:00",
        "carousel": 5,
        "description": "蓝色行李箱",
        "dimensions": "56x36x23 cm",
    },
    "BAG005": {
        "baggage_id": "BAG005",
        "booking_code": "BCD890",
        "customer_id": "C010",
        "tag_number": "CA567890",
        "weight_kg": 120.0,  # 团队行李总重
        "pieces": 10,
        "type": "Group Checked",
        "status": "Checked In",
        "flight": "CA173",
        "origin": "PEK",
        "destination": "NRT",
        "checked_in_at": "2025-01-15 07:00:00",
        "description": "华为出差团行李",
        "group_id": "GRP-HW-001",
        "individual_tags": [
            "CA567890A", "CA567890B", "CA567890C", "CA567890D", "CA567890E",
            "CA567890F", "CA567890G", "CA567890H", "CA567890I", "CA567890J",
        ],
    },
    "BAG006": {
        "baggage_id": "BAG006",
        "booking_code": "YZA567",
        "customer_id": "C009",
        "tag_number": "MU678901",
        "weight_kg": 22.0,
        "pieces": 1,
        "type": "Checked",
        "status": "Security Hold",
        "flight": "MU583",
        "origin": "SHA",
        "destination": "LAX",
        "checked_in_at": "2025-01-15 09:00:00",
        "description": "行李安检中",
        "security_flag": True,
        "security_reason": "需要开包检查",
        "estimated_clear_time": "2025-01-15 09:30:00",
    },
    # 特殊行李
    "BAG007": {
        "baggage_id": "BAG007",
        "booking_code": "SPORT001",
        "customer_id": "C001",
        "tag_number": "CA789012",
        "weight_kg": 15.0,
        "pieces": 1,
        "type": "Sports Equipment",
        "equipment_type": "Golf Clubs",
        "status": "Checked In",
        "flight": "CA1234",
        "origin": "PEK",
        "destination": "SHA",
        "checked_in_at": "2025-01-15 06:35:00",
        "description": "高尔夫球具",
        "special_handling": "运动器材",
        "extra_fee": 300.00,
    },
    "BAG008": {
        "baggage_id": "BAG008",
        "booking_code": "PET001",
        "customer_id": "C003",
        "tag_number": "HU890123",
        "weight_kg": 8.0,  # 含笼子
        "pieces": 1,
        "type": "Live Animal",
        "animal_type": "Cat",
        "animal_name": "咪咪",
        "status": "Special Handling",
        "flight": "HU7892",
        "origin": "PEK",
        "destination": "CTU",
        "checked_in_at": "2025-01-15 12:00:00",
        "description": "宠物猫 - 航空箱",
        "special_handling": "活体动物",
        "health_certificate": "已验证",
        "cabin_approved": False,  # 货舱运输
        "extra_fee": 500.00,
    },
    "BAG009": {
        "baggage_id": "BAG009",
        "booking_code": "MUSIC001",
        "customer_id": "C002",
        "tag_number": "MU901234",
        "weight_kg": 5.0,
        "pieces": 1,
        "type": "Musical Instrument",
        "instrument_type": "Violin",
        "status": "Cabin Baggage",
        "flight": "MU5678",
        "origin": "SHA",
        "destination": "CAN",
        "checked_in_at": "2025-01-15 08:20:00",
        "description": "小提琴 - 机舱携带",
        "cabin_approved": True,
        "extra_seat_purchased": False,
    },
    # 超重行李
    "BAG010": {
        "baggage_id": "BAG010",
        "booking_code": "OVER001",
        "customer_id": "C006",
        "tag_number": "FM012345",
        "weight_kg": 35.0,  # 超重12kg
        "pieces": 1,
        "type": "Overweight",
        "status": "Fee Pending",
        "flight": "FM9012",
        "origin": "SHA",
        "destination": "XIY",
        "checked_in_at": "2025-01-15 14:30:00",
        "description": "超重行李箱",
        "standard_allowance_kg": 23.0,
        "excess_weight_kg": 12.0,
        "excess_fee": 360.00,  # 30元/kg
        "fee_status": "Unpaid",
    },
}

BAGGAGE_CLAIMS_DB = {
    "CLM001": {
        "claim_id": "CLM001",
        "baggage_id": "BAG_LOST001",
        "booking_code": "LOST001",
        "customer_id": "C007",
        "tag_number": "CA111222",
        "claim_type": "Lost",
        "status": "Searching",
        "filed_at": "2025-01-13 14:00:00",
        "flight": "CA1888",
        "origin": "SHA",
        "destination": "PEK",
        "description": "黑色新秀丽行李箱，有黄色行李牌",
        "contents_declared": [
            "衣物", "电脑", "化妆品", "文件"
        ],
        "estimated_value": 5000.00,
        "contact_phone": "13844445555",
        "last_seen": "上海浦东机场值机柜台",
        "search_history": [
            {"time": "2025-01-13 15:00:00", "location": "PEK T3", "result": "未找到"},
            {"time": "2025-01-14 09:00:00", "location": "SHA T1", "result": "正在核实"},
        ],
        "compensation_status": "Pending",
    },
    "CLM002": {
        "claim_id": "CLM002",
        "baggage_id": "BAG_DMG001",
        "booking_code": "DMG001",
        "customer_id": "C001",
        "tag_number": "CA333444",
        "claim_type": "Damaged",
        "status": "Approved",
        "filed_at": "2025-01-12 16:00:00",
        "flight": "CA1234",
        "origin": "PEK",
        "destination": "SHA",
        "description": "行李箱轮子损坏",
        "damage_photos": ["dmg001_1.jpg", "dmg001_2.jpg"],
        "damage_assessment": "轮子断裂，需更换",
        "repair_or_replace": "Replace",
        "compensation_amount": 800.00,
        "compensation_status": "Approved",
        "approved_at": "2025-01-13 10:00:00",
        "payment_method": "Account Credit",
    },
    "CLM003": {
        "claim_id": "CLM003",
        "baggage_id": "BAG_DELAY001",
        "booking_code": "DELAY002",
        "customer_id": "C009",
        "tag_number": "MU555666",
        "claim_type": "Delayed",
        "status": "In Transit",
        "filed_at": "2025-01-14 22:00:00",
        "flight": "MU583",
        "origin": "SHA",
        "destination": "LAX",
        "description": "行李未随航班到达",
        "current_location": "SHA",
        "next_flight": "MU585",
        "estimated_delivery": "2025-01-16 10:00:00",
        "delivery_address": "Los Angeles, CA 90001",
        "contact_phone": "+1-555-123-4567",
        "temporary_allowance": {
            "amount": 100.00,
            "currency": "USD",
            "status": "Paid",
        },
    },
    "CLM004": {
        "claim_id": "CLM004",
        "baggage_id": "BAG_PIR001",
        "booking_code": "PIR001",
        "customer_id": "C002",
        "tag_number": "MU777888",
        "claim_type": "Pilferage",
        "status": "Under Investigation",
        "filed_at": "2025-01-11 18:00:00",
        "flight": "MU5678",
        "origin": "SHA",
        "destination": "CAN",
        "description": "行李内物品丢失",
        "missing_items": [
            {"item": "笔记本电脑", "brand": "MacBook Pro", "value": 15000.00},
            {"item": "手表", "brand": "Apple Watch", "value": 3000.00},
        ],
        "total_claimed": 18000.00,
        "investigation_notes": "已调取监控录像，正在审查",
        "assigned_investigator": "安保部-王主管",
        "police_report": "2025-0111-001",
    },
    "CLM005": {
        "claim_id": "CLM005",
        "baggage_id": "BAG_FOUND001",
        "booking_code": "FOUND001",
        "customer_id": "C005",
        "tag_number": "CZ999000",
        "claim_type": "Lost",
        "status": "Found",
        "filed_at": "2025-01-10 12:00:00",
        "found_at": "2025-01-12 09:00:00",
        "flight": "CZ3456",
        "origin": "CAN",
        "destination": "SZX",
        "description": "红色行李箱",
        "found_location": "深圳宝安机场行李查询处",
        "delivery_scheduled": "2025-01-13 14:00:00",
        "delivery_completed": "2025-01-13 14:30:00",
        "delivery_address": "深圳市南山区xxx",
    },
}


def get_baggage(baggage_id: str) -> dict:
    """获取行李信息"""
    return BAGGAGE_DB.get(baggage_id.upper())


def get_baggage_by_tag(tag_number: str) -> dict:
    """按标签获取行李"""
    for baggage in BAGGAGE_DB.values():
        if baggage.get("tag_number") == tag_number.upper():
            return baggage
    return None


def get_baggage_by_booking(booking_code: str) -> list:
    """按订单获取行李"""
    return [b for b in BAGGAGE_DB.values() if b.get("booking_code") == booking_code.upper()]


def get_baggage_by_flight(flight_number: str) -> list:
    """按航班获取行李"""
    return [b for b in BAGGAGE_DB.values() if b.get("flight") == flight_number.upper()]


def get_claim(claim_id: str) -> dict:
    """获取行李理赔"""
    return BAGGAGE_CLAIMS_DB.get(claim_id.upper())


def get_claims_by_customer(customer_id: str) -> list:
    """按客户获取理赔"""
    return [c for c in BAGGAGE_CLAIMS_DB.values() if c.get("customer_id") == customer_id.upper()]


def get_active_claims() -> list:
    """获取进行中的理赔"""
    active_statuses = ["Searching", "Under Investigation", "In Transit", "Pending"]
    return [c for c in BAGGAGE_CLAIMS_DB.values() if c.get("status") in active_statuses]
