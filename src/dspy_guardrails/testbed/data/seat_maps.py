"""
座位图数据 - Mock Seat Maps Database
各机型座位布局和可用性
"""

SEAT_MAPS_DB = {
    # A330-300 国内配置
    "A330-300-DOM": {
        "aircraft_type": "A330-300",
        "configuration": "Domestic",
        "total_seats": 292,
        "classes": {
            "Business": {
                "rows": list(range(1, 7)),  # 1-6排
                "layout": "2-2-2",
                "seats_per_row": 6,
                "total_seats": 36,
                "features": ["平躺座椅", "个人娱乐系统", "充电插座", "WiFi"],
            },
            "Premium Economy": {
                "rows": list(range(10, 15)),  # 10-14排
                "layout": "2-4-2",
                "seats_per_row": 8,
                "total_seats": 40,
                "features": ["加宽座椅", "额外腿部空间", "优先登机"],
            },
            "Economy": {
                "rows": list(range(20, 47)),  # 20-46排
                "layout": "2-4-2",
                "seats_per_row": 8,
                "total_seats": 216,
                "features": ["标准座椅", "个人屏幕"],
            },
        },
        "special_seats": {
            "emergency_exit": ["20A", "20B", "20C", "20D", "20E", "20F", "20G", "20H",
                              "35A", "35B", "35C", "35D", "35E", "35F", "35G", "35H"],
            "bulkhead": ["20A", "20B", "20C", "20D", "20E", "20F", "20G", "20H"],
            "bassinet": ["20A", "20H"],
            "extra_legroom": list(range(20, 22)) + list(range(35, 37)),
        },
    },
    # B737-800
    "B737-800": {
        "aircraft_type": "B737-800",
        "configuration": "Standard",
        "total_seats": 162,
        "classes": {
            "Business": {
                "rows": list(range(1, 4)),  # 1-3排
                "layout": "2-2",
                "seats_per_row": 4,
                "total_seats": 12,
                "features": ["加宽座椅", "优先服务"],
            },
            "Economy": {
                "rows": list(range(5, 30)),  # 5-29排
                "layout": "3-3",
                "seats_per_row": 6,
                "total_seats": 150,
                "features": ["标准座椅"],
            },
        },
        "special_seats": {
            "emergency_exit": ["12A", "12B", "12C", "12D", "12E", "12F"],
            "bulkhead": ["5A", "5B", "5C", "5D", "5E", "5F"],
            "extra_legroom": list(range(12, 14)),
        },
    },
    # B777-300ER 国际配置
    "B777-300ER-INTL": {
        "aircraft_type": "B777-300ER",
        "configuration": "International",
        "total_seats": 304,
        "classes": {
            "First": {
                "rows": list(range(1, 3)),  # 1-2排
                "layout": "1-2-1",
                "seats_per_row": 4,
                "total_seats": 8,
                "features": ["私人套房", "180度平躺", "专属洗手间", "机上酒吧"],
            },
            "Business": {
                "rows": list(range(5, 14)),  # 5-13排
                "layout": "1-2-1",
                "seats_per_row": 4,
                "total_seats": 36,
                "features": ["平躺座椅", "直通走道", "个人娱乐"],
            },
            "Premium Economy": {
                "rows": list(range(20, 25)),  # 20-24排
                "layout": "2-4-2",
                "seats_per_row": 8,
                "total_seats": 40,
                "features": ["加宽座椅", "额外腿部空间"],
            },
            "Economy": {
                "rows": list(range(30, 58)),  # 30-57排
                "layout": "3-4-3",
                "seats_per_row": 10,
                "total_seats": 220,
                "features": ["标准座椅", "个人屏幕"],
            },
        },
    },
    # A320neo
    "A320neo": {
        "aircraft_type": "A320neo",
        "configuration": "Standard",
        "total_seats": 156,
        "classes": {
            "Business": {
                "rows": list(range(1, 4)),
                "layout": "2-2",
                "seats_per_row": 4,
                "total_seats": 12,
            },
            "Economy": {
                "rows": list(range(5, 29)),
                "layout": "3-3",
                "seats_per_row": 6,
                "total_seats": 144,
            },
        },
    },
}

# 特定航班座位可用性
FLIGHT_SEAT_AVAILABILITY = {
    "CA1234": {
        "flight": "CA1234",
        "aircraft": "A330-300-DOM",
        "date": "2025-01-15",
        "available": {
            "Business": ["3A", "3B", "4C", "4D", "5A", "5B", "6C", "6D"],
            "Premium Economy": ["10A", "10B", "11G", "11H", "12A", "12B"],
            "Economy": [
                "22A", "22B", "22H", "23C", "23D", "23E", "23F",
                "25A", "25B", "25G", "25H", "26A", "26B", "26G", "26H",
                "30C", "30D", "30E", "30F", "31A", "31B", "31G", "31H",
                "35A", "35B", "35G", "35H",  # 紧急出口排
                "40A", "40B", "40C", "40D", "40E", "40F", "40G", "40H",
            ],
        },
        "blocked": {
            "12A": "Occupied by ABC123",
            "20A": "Reserved for bassinet",
            "20H": "Reserved for bassinet",
        },
        "pricing": {
            "emergency_exit": 200,
            "extra_legroom": 150,
            "window": 50,
            "aisle": 50,
            "front_economy": 80,
        },
    },
    "MU5678": {
        "flight": "MU5678",
        "aircraft": "B737-800",
        "date": "2025-01-15",
        "available": {
            "Business": ["2A", "2B", "3C", "3D"],
            "Economy": [
                "8A", "8B", "8E", "8F",
                "10A", "10B", "10C", "10D", "10E", "10F",
                "15C", "15D",
                "20A", "20B", "20C", "20D", "20E", "20F",
                "25A", "25B", "25C", "25D", "25E", "25F",
            ],
        },
        "blocked": {
            "3C": "Occupied by DEF456",
        },
        "pricing": {
            "emergency_exit": 180,
            "window": 40,
            "aisle": 40,
        },
    },
    "CA985": {
        "flight": "CA985",
        "aircraft": "B777-300ER-INTL",
        "date": "2025-01-15",
        "available": {
            "First": ["1K", "2A"],  # 大部分已售
            "Business": ["7A", "7K", "9D", "9G", "11A", "11K"],
            "Premium Economy": ["21A", "21B", "22G", "22H", "23A", "23B"],
            "Economy": [
                "35A", "35B", "35J", "35K",
                "40A", "40B", "40C", "40H", "40J", "40K",
                "45D", "45E", "45F", "45G",
                "50A", "50B", "50J", "50K",
            ],
        },
        "blocked": {
            "1A": "Occupied by JKL012",
        },
        "pricing": {
            "window": 100,
            "aisle": 100,
            "extra_legroom": 300,
            "bulkhead": 250,
        },
    },
}


def get_seat_map(aircraft_type: str) -> dict:
    """获取机型座位图"""
    return SEAT_MAPS_DB.get(aircraft_type)


def get_flight_availability(flight_number: str) -> dict:
    """获取航班座位可用性"""
    return FLIGHT_SEAT_AVAILABILITY.get(flight_number.upper())


def is_seat_available(flight_number: str, seat: str) -> bool:
    """检查座位是否可用"""
    availability = get_flight_availability(flight_number)
    if not availability:
        return False

    seat = seat.upper()
    for cabin_seats in availability.get("available", {}).values():
        if seat in cabin_seats:
            return True
    return False


def get_seat_price(flight_number: str, seat: str) -> float:
    """获取座位选座费用"""
    availability = get_flight_availability(flight_number)
    if not availability:
        return 0.0

    pricing = availability.get("pricing", {})
    seat = seat.upper()
    row = int("".join(filter(str.isdigit, seat)))
    col = "".join(filter(str.isalpha, seat))

    total_price = 0.0

    # 检查特殊座位
    aircraft = availability.get("aircraft")
    seat_map = get_seat_map(aircraft)
    if seat_map:
        special = seat_map.get("special_seats", {})
        if seat in special.get("emergency_exit", []):
            total_price += pricing.get("emergency_exit", 0)
        if row in special.get("extra_legroom", []):
            total_price += pricing.get("extra_legroom", 0)

    # 检查位置
    if col in ["A", "K", "H"]:  # 靠窗
        total_price += pricing.get("window", 0)
    elif col in ["C", "D", "G", "J"]:  # 靠走道（取决于布局）
        total_price += pricing.get("aisle", 0)

    return total_price


def get_available_seats_by_preference(flight_number: str, preference: str) -> list:
    """按偏好获取可用座位"""
    availability = get_flight_availability(flight_number)
    if not availability:
        return []

    all_available = []
    for cabin_seats in availability.get("available", {}).values():
        all_available.extend(cabin_seats)

    if preference == "window":
        return [s for s in all_available if s[-1] in ["A", "K", "H"]]
    elif preference == "aisle":
        return [s for s in all_available if s[-1] in ["C", "D", "G", "J"]]
    elif preference == "middle":
        return [s for s in all_available if s[-1] in ["B", "E", "F"]]
    else:
        return all_available
