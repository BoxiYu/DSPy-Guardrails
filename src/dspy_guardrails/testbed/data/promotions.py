"""
促销活动数据 - Mock Promotions Database
"""

PROMOTIONS_DB = {
    "PROMO001": {
        "promotion_id": "PROMO001",
        "name": "春节特惠",
        "code": "CNY2025",
        "type": "Discount",
        "discount_type": "percentage",
        "discount_value": 15,  # 15% off
        "valid_from": "2025-01-15 00:00:00",
        "valid_until": "2025-02-15 23:59:59",
        "applicable_routes": ["ALL"],
        "applicable_classes": ["Economy", "Premium Economy"],
        "min_purchase": 500.00,
        "max_discount": 300.00,
        "usage_limit_per_user": 2,
        "total_quota": 10000,
        "used_count": 3567,
        "status": "Active",
        "terms": [
            "仅限经济舱和超级经济舱",
            "不可与其他优惠叠加",
            "退改签按原价计算",
        ],
    },
    "PROMO002": {
        "promotion_id": "PROMO002",
        "name": "商务舱升舱优惠",
        "code": "UPGRADE50",
        "type": "Upgrade",
        "discount_type": "fixed",
        "discount_value": 500.00,  # 500元优惠
        "valid_from": "2025-01-01 00:00:00",
        "valid_until": "2025-03-31 23:59:59",
        "applicable_routes": ["PEK-SHA", "SHA-PEK", "PEK-CAN", "CAN-PEK"],
        "upgrade_from": "Economy",
        "upgrade_to": "Business",
        "min_miles": 50000,  # 需要至少50000里程
        "usage_limit_per_user": 1,
        "total_quota": 500,
        "used_count": 123,
        "status": "Active",
        "terms": [
            "需使用里程+现金支付",
            "限金卡及以上会员",
            "出发前24小时可申请",
        ],
    },
    "PROMO003": {
        "promotion_id": "PROMO003",
        "name": "首次注册奖励",
        "code": "NEWUSER",
        "type": "Bonus",
        "bonus_type": "miles",
        "bonus_value": 1000,  # 1000里程
        "valid_from": "2025-01-01 00:00:00",
        "valid_until": "2025-12-31 23:59:59",
        "applicable_users": ["new"],
        "usage_limit_per_user": 1,
        "total_quota": None,  # 无限制
        "used_count": 15678,
        "status": "Active",
        "auto_apply": True,
        "terms": [
            "仅限新注册用户",
            "首次预订后自动发放",
            "里程90天内有效",
        ],
    },
    "PROMO004": {
        "promotion_id": "PROMO004",
        "name": "生日月双倍里程",
        "code": "BIRTHDAY",
        "type": "Multiplier",
        "multiplier": 2.0,  # 双倍里程
        "multiplier_target": "miles",
        "valid_from": "2025-01-01 00:00:00",
        "valid_until": "2025-12-31 23:59:59",
        "applicable_routes": ["ALL"],
        "applicable_classes": ["ALL"],
        "usage_limit_per_user": None,  # 生日月内无限制
        "status": "Active",
        "auto_apply": True,
        "terms": [
            "会员生日当月有效",
            "自动应用，无需输入代码",
            "与其他里程活动可叠加",
        ],
    },
    "PROMO005": {
        "promotion_id": "PROMO005",
        "name": "行李额外优惠",
        "code": "BAGFREE",
        "type": "Service",
        "service_type": "Free Baggage",
        "free_baggage_kg": 10,  # 额外10kg
        "valid_from": "2025-01-10 00:00:00",
        "valid_until": "2025-01-31 23:59:59",
        "applicable_routes": ["国内航线"],
        "applicable_classes": ["Economy"],
        "usage_limit_per_user": 1,
        "total_quota": 5000,
        "used_count": 2345,
        "status": "Active",
        "terms": [
            "仅限国内经济舱航班",
            "每位旅客限用一次",
            "需在购票时使用",
        ],
    },
    "PROMO006": {
        "promotion_id": "PROMO006",
        "name": "企业团购优惠",
        "code": "CORP2025",
        "type": "Corporate",
        "discount_type": "percentage",
        "discount_value": 20,  # 20% off
        "valid_from": "2025-01-01 00:00:00",
        "valid_until": "2025-12-31 23:59:59",
        "applicable_routes": ["ALL"],
        "applicable_classes": ["Economy", "Business"],
        "min_tickets": 5,  # 最少5张票
        "applicable_users": ["corporate"],
        "status": "Active",
        "terms": [
            "需企业账户预订",
            "最少5人同行",
            "需提前7天预订",
        ],
    },
    "PROMO007": {
        "promotion_id": "PROMO007",
        "name": "周末特价",
        "code": "WEEKEND",
        "type": "Flash Sale",
        "discount_type": "fixed",
        "discount_value": 100.00,
        "valid_from": "2025-01-18 00:00:00",  # 周六
        "valid_until": "2025-01-19 23:59:59",  # 周日
        "applicable_routes": ["SHA-HGH", "HGH-SHA", "PEK-TJN", "TJN-PEK"],
        "applicable_classes": ["Economy"],
        "usage_limit_per_user": 2,
        "total_quota": 1000,
        "used_count": 0,
        "status": "Scheduled",
        "terms": [
            "限周末出行",
            "先到先得",
            "不可退改",
        ],
    },
    "PROMO008": {
        "promotion_id": "PROMO008",
        "name": "学生特惠",
        "code": "STUDENT",
        "type": "Discount",
        "discount_type": "percentage",
        "discount_value": 25,  # 25% off
        "valid_from": "2025-01-01 00:00:00",
        "valid_until": "2025-12-31 23:59:59",
        "applicable_routes": ["ALL"],
        "applicable_classes": ["Economy"],
        "applicable_users": ["student"],
        "verification_required": True,
        "verification_type": "Student ID",
        "usage_limit_per_user": 4,  # 每年4次
        "status": "Active",
        "terms": [
            "需验证学生身份",
            "每年最多使用4次",
            "限国内航线",
        ],
    },
    "PROMO009": {
        "promotion_id": "PROMO009",
        "name": "里程兑换优惠",
        "code": "MILES50",
        "type": "Miles Redemption",
        "discount_type": "percentage",
        "discount_value": 50,  # 里程兑换减50%
        "valid_from": "2025-01-15 00:00:00",
        "valid_until": "2025-01-31 23:59:59",
        "applicable_routes": ["PEK-SHA", "SHA-CAN", "CAN-PEK"],
        "applicable_classes": ["Economy"],
        "min_miles_balance": 20000,
        "usage_limit_per_user": 1,
        "total_quota": 200,
        "used_count": 45,
        "status": "Active",
        "terms": [
            "限指定航线",
            "里程账户余额需≥20000",
            "兑换后不可退改",
        ],
    },
    "PROMO010": {
        "promotion_id": "PROMO010",
        "name": "老年优惠",
        "code": "SENIOR",
        "type": "Discount",
        "discount_type": "percentage",
        "discount_value": 10,  # 10% off
        "valid_from": "2025-01-01 00:00:00",
        "valid_until": "2025-12-31 23:59:59",
        "applicable_routes": ["国内航线"],
        "applicable_classes": ["Economy"],
        "applicable_users": ["senior"],
        "age_requirement": 65,  # 65岁及以上
        "usage_limit_per_user": None,
        "status": "Active",
        "auto_apply": True,
        "terms": [
            "65周岁及以上旅客",
            "自动识别，无需输入代码",
            "可与生日优惠叠加",
        ],
    },
    # 已过期活动
    "PROMO011": {
        "promotion_id": "PROMO011",
        "name": "元旦特惠",
        "code": "NEWYEAR",
        "type": "Discount",
        "discount_type": "percentage",
        "discount_value": 20,
        "valid_from": "2024-12-25 00:00:00",
        "valid_until": "2025-01-03 23:59:59",
        "applicable_routes": ["ALL"],
        "applicable_classes": ["ALL"],
        "total_quota": 5000,
        "used_count": 4892,
        "status": "Expired",
    },
    # 即将开始
    "PROMO012": {
        "promotion_id": "PROMO012",
        "name": "情人节特惠",
        "code": "LOVE214",
        "type": "Discount",
        "discount_type": "percentage",
        "discount_value": 14,  # 14% off (214)
        "valid_from": "2025-02-10 00:00:00",
        "valid_until": "2025-02-16 23:59:59",
        "applicable_routes": ["ALL"],
        "applicable_classes": ["Economy", "Business"],
        "min_tickets": 2,  # 需两人同行
        "total_quota": 2140,
        "used_count": 0,
        "status": "Scheduled",
        "terms": [
            "需两人同行",
            "同一订单预订",
            "出行日期需含2月14日",
        ],
    },
}


def get_promotion(promotion_id: str) -> dict:
    """获取活动信息"""
    return PROMOTIONS_DB.get(promotion_id.upper())


def get_promotion_by_code(code: str) -> dict:
    """按优惠码获取活动"""
    for promo in PROMOTIONS_DB.values():
        if promo.get("code") == code.upper():
            return promo
    return None


def get_active_promotions() -> list:
    """获取进行中的活动"""
    return [p for p in PROMOTIONS_DB.values() if p.get("status") == "Active"]


def get_promotions_by_type(promo_type: str) -> list:
    """按类型获取活动"""
    return [p for p in PROMOTIONS_DB.values() if p.get("type") == promo_type]


def validate_promotion(code: str, booking_info: dict) -> dict:
    """验证优惠码是否适用"""
    promo = get_promotion_by_code(code)
    if not promo:
        return {"valid": False, "reason": "优惠码不存在"}
    if promo.get("status") != "Active":
        return {"valid": False, "reason": f"优惠码状态: {promo.get('status')}"}
    if promo.get("total_quota") and promo.get("used_count", 0) >= promo.get("total_quota"):
        return {"valid": False, "reason": "优惠码已用完"}
    return {"valid": True, "promotion": promo}
