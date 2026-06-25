"""
酒店数据 - Mock Hotels Database
航班延误/取消时的酒店安排
"""

HOTELS_DB = {
    "HOTEL001": {
        "hotel_id": "HOTEL001",
        "name": "北京首都机场希尔顿酒店",
        "english_name": "Hilton Beijing Capital Airport",
        "airport": "PEK",
        "terminal": "T3",
        "distance_km": 0.5,
        "shuttle_available": True,
        "shuttle_frequency_mins": 15,
        "star_rating": 5,
        "rooms_available": 45,
        "room_types": {
            "Standard": {
                "price": 880.00,
                "beds": "1 King or 2 Twin",
                "available": 20,
                "amenities": ["WiFi", "早餐", "健身房"],
            },
            "Deluxe": {
                "price": 1280.00,
                "beds": "1 King",
                "available": 15,
                "amenities": ["WiFi", "早餐", "健身房", "行政酒廊"],
            },
            "Suite": {
                "price": 2580.00,
                "beds": "1 King + Living Room",
                "available": 10,
                "amenities": ["WiFi", "早餐", "健身房", "行政酒廊", "迷你吧"],
            },
        },
        "contract_rate": 0.7,  # 航司协议价70%
        "contact_phone": "010-64578888",
        "check_in_time": "15:00",
        "check_out_time": "12:00",
        "late_check_out_available": True,
    },
    "HOTEL002": {
        "hotel_id": "HOTEL002",
        "name": "北京首都机场朗豪酒店",
        "english_name": "Langham Place Beijing Capital Airport",
        "airport": "PEK",
        "terminal": "T3",
        "distance_km": 1.0,
        "shuttle_available": True,
        "shuttle_frequency_mins": 20,
        "star_rating": 5,
        "rooms_available": 38,
        "room_types": {
            "Standard": {"price": 780.00, "available": 18},
            "Deluxe": {"price": 1080.00, "available": 12},
            "Suite": {"price": 2180.00, "available": 8},
        },
        "contract_rate": 0.65,
        "contact_phone": "010-64598888",
    },
    "HOTEL003": {
        "hotel_id": "HOTEL003",
        "name": "上海浦东机场华美达酒店",
        "english_name": "Ramada Shanghai Pudong Airport",
        "airport": "PVG",
        "terminal": "T1/T2",
        "distance_km": 2.0,
        "shuttle_available": True,
        "shuttle_frequency_mins": 30,
        "star_rating": 4,
        "rooms_available": 60,
        "room_types": {
            "Standard": {"price": 580.00, "available": 35},
            "Deluxe": {"price": 780.00, "available": 20},
            "Family": {"price": 880.00, "available": 5},
        },
        "contract_rate": 0.6,
        "contact_phone": "021-38858888",
    },
    "HOTEL004": {
        "hotel_id": "HOTEL004",
        "name": "上海虹桥机场皇冠假日酒店",
        "english_name": "Crowne Plaza Shanghai Hongqiao Airport",
        "airport": "SHA",
        "terminal": "T2",
        "distance_km": 0.3,
        "shuttle_available": True,
        "shuttle_frequency_mins": 10,
        "star_rating": 5,
        "rooms_available": 50,
        "room_types": {
            "Standard": {"price": 750.00, "available": 25},
            "Deluxe": {"price": 1050.00, "available": 15},
            "Suite": {"price": 1980.00, "available": 10},
        },
        "contract_rate": 0.68,
        "contact_phone": "021-52636888",
    },
    "HOTEL005": {
        "hotel_id": "HOTEL005",
        "name": "广州白云机场铂尔曼酒店",
        "english_name": "Pullman Guangzhou Baiyun Airport",
        "airport": "CAN",
        "terminal": "T2",
        "distance_km": 0.8,
        "shuttle_available": True,
        "shuttle_frequency_mins": 15,
        "star_rating": 5,
        "rooms_available": 42,
        "room_types": {
            "Standard": {"price": 680.00, "available": 20},
            "Deluxe": {"price": 980.00, "available": 15},
            "Suite": {"price": 1880.00, "available": 7},
        },
        "contract_rate": 0.65,
        "contact_phone": "020-36068866",
    },
    "HOTEL006": {
        "hotel_id": "HOTEL006",
        "name": "成都双流机场智选假日酒店",
        "english_name": "Holiday Inn Express Chengdu Airport",
        "airport": "CTU",
        "terminal": "T2",
        "distance_km": 3.0,
        "shuttle_available": True,
        "shuttle_frequency_mins": 30,
        "star_rating": 3,
        "rooms_available": 80,
        "room_types": {
            "Standard": {"price": 380.00, "available": 50},
            "Deluxe": {"price": 480.00, "available": 30},
        },
        "contract_rate": 0.55,
        "contact_phone": "028-85205888",
    },
    # 航班延误安排记录
    "ARRANGEMENT001": {
        "arrangement_id": "ARRANGEMENT001",
        "booking_code": "GHI789",
        "customer_id": "C003",
        "flight": "HU7890",
        "delay_reason": "航班取消",
        "hotel_id": "HOTEL001",
        "hotel_name": "北京首都机场希尔顿酒店",
        "room_type": "Standard",
        "check_in": "2025-01-15 20:00:00",
        "check_out": "2025-01-16 10:00:00",
        "nights": 1,
        "original_price": 880.00,
        "paid_by": "Airline",
        "actual_cost": 616.00,  # 协议价
        "meal_voucher": {
            "dinner": 100.00,
            "breakfast": "Included",
        },
        "status": "Confirmed",
        "confirmation_number": "HLT-2025-001234",
        "created_at": "2025-01-15 16:30:00",
        "created_by": "AGENT003",
    },
    "ARRANGEMENT002": {
        "arrangement_id": "ARRANGEMENT002",
        "booking_code": "DELAY001",
        "customer_id": "C009",
        "flight": "MU583",
        "delay_reason": "航班延误超过6小时",
        "hotel_id": "HOTEL003",
        "hotel_name": "上海浦东机场华美达酒店",
        "room_type": "Standard",
        "check_in": "2025-01-15 18:00:00",
        "check_out": "2025-01-16 08:00:00",
        "nights": 1,
        "original_price": 580.00,
        "paid_by": "Airline",
        "actual_cost": 348.00,
        "meal_voucher": {
            "dinner": 80.00,
            "breakfast": 50.00,
        },
        "status": "Checked In",
        "confirmation_number": "RAM-2025-005678",
        "created_at": "2025-01-15 15:00:00",
    },
}


def get_hotel(hotel_id: str) -> dict:
    """获取酒店信息"""
    return HOTELS_DB.get(hotel_id.upper())


def get_hotels_by_airport(airport_code: str) -> list:
    """按机场获取酒店"""
    return [h for h in HOTELS_DB.values()
            if isinstance(h.get("airport"), str) and airport_code.upper() in h.get("airport", "")]


def get_available_hotels(airport_code: str, room_type: str = "Standard") -> list:
    """获取有空房的酒店"""
    hotels = get_hotels_by_airport(airport_code)
    available = []
    for hotel in hotels:
        room_types = hotel.get("room_types", {})
        if room_type in room_types and room_types[room_type].get("available", 0) > 0:
            available.append({
                "hotel": hotel,
                "room_type": room_type,
                "price": room_types[room_type].get("price", 0),
                "contract_price": room_types[room_type].get("price", 0) * hotel.get("contract_rate", 1.0),
                "available_rooms": room_types[room_type].get("available", 0),
            })
    return sorted(available, key=lambda x: x["contract_price"])


def get_arrangement(arrangement_id: str) -> dict:
    """获取安排记录"""
    return HOTELS_DB.get(arrangement_id.upper())


def get_arrangements_by_customer(customer_id: str) -> list:
    """按客户获取安排"""
    return [a for a in HOTELS_DB.values()
            if a.get("customer_id") == customer_id]


def get_arrangements_by_flight(flight_number: str) -> list:
    """按航班获取安排"""
    return [a for a in HOTELS_DB.values()
            if a.get("flight") == flight_number.upper()]


def calculate_arrangement_cost(hotel_id: str, room_type: str, nights: int, include_meals: bool = True) -> dict:
    """计算安排费用"""
    hotel = get_hotel(hotel_id)
    if not hotel:
        return {"error": "Hotel not found"}

    room_types = hotel.get("room_types", {})
    if room_type not in room_types:
        return {"error": "Room type not found"}

    room_price = room_types[room_type].get("price", 0)
    contract_rate = hotel.get("contract_rate", 1.0)
    contract_price = room_price * contract_rate

    total = contract_price * nights
    meal_cost = 0
    if include_meals:
        meal_cost = 150 * nights  # 估算餐费

    return {
        "hotel_id": hotel_id,
        "room_type": room_type,
        "nights": nights,
        "room_price_per_night": contract_price,
        "meal_cost": meal_cost,
        "total_cost": total + meal_cost,
        "original_price": room_price * nights + meal_cost,
        "savings": (room_price - contract_price) * nights,
    }
