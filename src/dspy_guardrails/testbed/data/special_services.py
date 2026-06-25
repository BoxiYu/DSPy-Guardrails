"""
特殊服务数据 - Mock Special Services Database
"""

SPECIAL_SERVICES_DB = {
    "SVC001": {
        "service_id": "SVC001",
        "booking_code": "DEF456",
        "customer_id": "C002",
        "flight": "MU5678",
        "service_type": "Lounge Access",
        "status": "Confirmed",
        "details": {
            "lounge": "东方万里行贵宾厅",
            "location": "上海浦东T2 E指廊",
            "valid_from": "2025-01-15 07:00:00",
            "valid_until": "2025-01-15 10:15:00",
            "guests_allowed": 1,
        },
        "price": 0.00,  # 商务舱免费
        "created_at": "2025-01-08 14:22:00",
    },
    "SVC002": {
        "service_id": "SVC002",
        "booking_code": "JKL012",
        "customer_id": "C004",
        "flight": "CA985",
        "service_type": "Chauffeur Service",
        "status": "Scheduled",
        "details": {
            "pickup_location": "北京市朝阳区xxx",
            "pickup_time": "2025-01-15 10:00:00",
            "vehicle_type": "Mercedes S-Class",
            "driver_name": "李师傅",
            "driver_phone": "13800138000",
            "flight_terminal": "T3",
        },
        "price": 0.00,  # 头等舱免费
        "created_at": "2025-01-02 08:00:00",
    },
    "SVC003": {
        "service_id": "SVC003",
        "booking_code": "HIJ456",
        "customer_id": "C012",
        "flight": "CZ6789",
        "service_type": "Bassinet Request",
        "status": "Confirmed",
        "details": {
            "infant_name": "李小明",
            "infant_dob": "2024-06-15",
            "seat_location": "30A",
            "bassinet_row": "30",
            "special_instructions": "靠近洗手间",
        },
        "price": 0.00,
        "created_at": "2025-01-11 14:00:00",
    },
    "SVC004": {
        "service_id": "SVC004",
        "booking_code": "ABC123",
        "customer_id": "C001",
        "flight": "CA1234",
        "service_type": "Meal Upgrade",
        "status": "Confirmed",
        "details": {
            "original_meal": "Standard",
            "upgraded_meal": "Premium Chinese",
            "dietary_restrictions": None,
            "special_requests": "不要辣",
        },
        "price": 88.00,
        "created_at": "2025-01-12 10:00:00",
    },
    "SVC005": {
        "service_id": "SVC005",
        "booking_code": "GHI789",
        "customer_id": "C003",
        "flight": "HU7890",
        "service_type": "Wheelchair Assistance",
        "status": "Cancelled",
        "cancelled_reason": "航班取消",
        "details": {
            "wheelchair_type": "WCHR",  # 能自己上下楼梯
            "assistance_from": "Check-in Counter",
            "assistance_to": "Aircraft Seat",
            "special_needs": None,
        },
        "price": 0.00,
        "created_at": "2025-01-05 11:15:00",
        "cancelled_at": "2025-01-15 08:00:00",
    },
    "SVC006": {
        "service_id": "SVC006",
        "booking_code": "MEDICAL001",
        "customer_id": "C007",
        "flight": "CA1888",
        "service_type": "Medical Assistance",
        "status": "Completed",
        "details": {
            "condition": "高血压",
            "medications": ["降压药"],
            "special_equipment": None,
            "doctor_clearance": "已提供",
            "oxygen_required": False,
            "medical_escort": False,
        },
        "price": 0.00,
        "created_at": "2025-01-12 20:00:00",
        "completed_at": "2025-01-15 08:15:00",
    },
    "SVC007": {
        "service_id": "SVC007",
        "booking_code": "UM001",
        "customer_id": "C_MINOR001",
        "flight": "MU2345",
        "service_type": "Unaccompanied Minor",
        "status": "Active",
        "details": {
            "minor_name": "小明",
            "minor_age": 10,
            "pickup_contact": {
                "name": "爷爷",
                "phone": "13900001234",
                "id_number": "110101194501011234",
                "relationship": "祖父",
            },
            "dropoff_contact": {
                "name": "奶奶",
                "phone": "13900005678",
                "id_number": "110101194601011234",
                "relationship": "祖母",
            },
            "flight_attendant_escort": True,
        },
        "price": 300.00,
        "created_at": "2025-01-10 09:00:00",
    },
    "SVC008": {
        "service_id": "SVC008",
        "booking_code": "DEF456",
        "customer_id": "C002",
        "flight": "MU5678",
        "service_type": "Priority Boarding",
        "status": "Confirmed",
        "details": {
            "boarding_group": "1",
            "boarding_time": "09:45",
            "gate": "B08",
        },
        "price": 0.00,  # 商务舱免费
        "created_at": "2025-01-08 14:22:00",
    },
    "SVC009": {
        "service_id": "SVC009",
        "booking_code": "YZA567",
        "customer_id": "C009",
        "flight": "MU583",
        "service_type": "Transit Assistance",
        "status": "Pending",
        "details": {
            "transit_airport": "LAX",
            "connecting_flight": "AA1234",
            "connection_time_mins": 180,
            "assistance_type": "Immigration & Customs",
            "language_support": "Chinese",
        },
        "price": 0.00,
        "created_at": "2025-01-05 18:00:00",
    },
    "SVC010": {
        "service_id": "SVC010",
        "booking_code": "FAST001",
        "customer_id": "C004",
        "flight": "CA985",
        "service_type": "Fast Track Security",
        "status": "Confirmed",
        "details": {
            "airport": "PEK",
            "terminal": "T3",
            "valid_from": "2025-01-15 10:00:00",
            "valid_until": "2025-01-15 13:00:00",
            "includes_immigration": True,
        },
        "price": 0.00,  # VIP 免费
        "created_at": "2025-01-02 08:00:00",
    },
    "SVC011": {
        "service_id": "SVC011",
        "booking_code": "WIFI001",
        "customer_id": "C001",
        "flight": "CA1234",
        "service_type": "In-Flight WiFi",
        "status": "Purchased",
        "details": {
            "package": "全程上网套餐",
            "data_limit": "Unlimited",
            "devices": 2,
            "activation_code": "WIFI-ABC123-2025",
        },
        "price": 68.00,
        "created_at": "2025-01-14 20:00:00",
    },
    "SVC012": {
        "service_id": "SVC012",
        "booking_code": "SEAT001",
        "customer_id": "C006",
        "flight": "FM9012",
        "service_type": "Seat Selection",
        "status": "Confirmed",
        "details": {
            "original_seat": "随机分配",
            "selected_seat": "25D",
            "seat_type": "Extra Legroom",
            "row_features": ["紧急出口", "额外腿部空间"],
        },
        "price": 200.00,
        "created_at": "2025-01-15 12:30:00",
    },
    "SVC013": {
        "service_id": "SVC013",
        "booking_code": "BCD890",
        "customer_id": "C010",
        "flight": "CA173",
        "service_type": "Group Check-in",
        "status": "Scheduled",
        "details": {
            "group_name": "华为出差团",
            "group_size": 6,
            "dedicated_counter": "Row H",
            "check_in_time": "2025-01-15 07:00:00",
            "group_leader": "王经理",
        },
        "price": 0.00,
        "created_at": "2025-01-08 10:00:00",
    },
    "SVC014": {
        "service_id": "SVC014",
        "booking_code": "BLIND001",
        "customer_id": "C_BLIND001",
        "flight": "CZ3456",
        "service_type": "Visually Impaired Assistance",
        "status": "Confirmed",
        "details": {
            "guide_dog": True,
            "guide_dog_name": "旺财",
            "guide_dog_certification": "已验证",
            "braille_boarding_pass": True,
            "audio_announcements": True,
            "dedicated_escort": True,
        },
        "price": 0.00,
        "created_at": "2025-01-13 11:00:00",
    },
    "SVC015": {
        "service_id": "SVC015",
        "booking_code": "DEAF001",
        "customer_id": "C_DEAF001",
        "flight": "HU7892",
        "service_type": "Hearing Impaired Assistance",
        "status": "Confirmed",
        "details": {
            "sign_language_interpreter": False,
            "visual_alerts": True,
            "written_announcements": True,
            "seat_near_visual_display": True,
        },
        "price": 0.00,
        "created_at": "2025-01-14 09:00:00",
    },
}


def get_service(service_id: str) -> dict:
    """获取服务信息"""
    return SPECIAL_SERVICES_DB.get(service_id.upper())


def get_services_by_booking(booking_code: str) -> list:
    """按订单获取服务"""
    return [s for s in SPECIAL_SERVICES_DB.values() if s.get("booking_code") == booking_code.upper()]


def get_services_by_customer(customer_id: str) -> list:
    """按客户获取服务"""
    return [s for s in SPECIAL_SERVICES_DB.values() if s.get("customer_id") == customer_id.upper()]


def get_services_by_type(service_type: str) -> list:
    """按类型获取服务"""
    return [s for s in SPECIAL_SERVICES_DB.values() if s.get("service_type") == service_type]


def get_services_by_flight(flight_number: str) -> list:
    """按航班获取服务"""
    return [s for s in SPECIAL_SERVICES_DB.values() if s.get("flight") == flight_number.upper()]


def get_active_services() -> list:
    """获取进行中的服务"""
    active_statuses = ["Confirmed", "Scheduled", "Active", "Pending", "Purchased"]
    return [s for s in SPECIAL_SERVICES_DB.values() if s.get("status") in active_statuses]
