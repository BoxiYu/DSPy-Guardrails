"""
客户数据 - Mock Customers Database
"""

CUSTOMERS_DB = {
    "C001": {
        "customer_id": "C001",
        "name": "张三",
        "english_name": "Zhang San",
        "id_type": "身份证",
        "id_number": "110101199001011234",
        "phone": "13812345678",
        "email": "zhangsan@example.com",
        "membership_level": "Gold",
        "frequent_flyer": "CA88888888",
        "total_miles": 125000,
        "available_miles": 45000,
        "registered_at": "2020-03-15",
        "preferences": {
            "seat": "window",
            "meal": "Standard",
            "language": "zh-CN",
        },
        "emergency_contact": {
            "name": "张父",
            "phone": "13811111111",
            "relationship": "父亲",
        },
    },
    "C002": {
        "customer_id": "C002",
        "name": "李四",
        "english_name": "Li Si",
        "id_type": "身份证",
        "id_number": "310101198505052345",
        "phone": "13987654321",
        "email": "lisi@company.com",
        "membership_level": "Platinum",
        "frequent_flyer": "MU12345678",
        "total_miles": 380000,
        "available_miles": 120000,
        "registered_at": "2018-06-20",
        "preferences": {
            "seat": "aisle",
            "meal": "Vegetarian",
            "language": "zh-CN",
        },
        "corporate_account": "CORP-ABC-2023",
        "company": "ABC科技有限公司",
    },
    "C003": {
        "customer_id": "C003",
        "name": "王五",
        "english_name": "Wang Wu",
        "id_type": "身份证",
        "id_number": "440101199203033456",
        "phone": "13611112222",
        "email": "wangwu@test.com",
        "membership_level": "Silver",
        "frequent_flyer": "HU56789012",
        "total_miles": 35000,
        "available_miles": 15000,
        "registered_at": "2022-01-10",
        "preferences": {
            "seat": "window",
            "meal": "Standard",
            "language": "zh-CN",
        },
    },
    "C004": {
        "customer_id": "C004",
        "name": "赵六",
        "english_name": "Zhao Liu",
        "id_type": "护照",
        "id_number": "E12345678",
        "passport_expiry": "2030-12-31",
        "phone": "13900001111",
        "email": "zhao.liu@corp.com",
        "membership_level": "Diamond",
        "frequent_flyer": "CA00000001",
        "total_miles": 1200000,
        "available_miles": 500000,
        "registered_at": "2015-01-01",
        "preferences": {
            "seat": "first_row",
            "meal": "Chef Special",
            "language": "zh-CN",
        },
        "vip_status": True,
        "personal_assistant": "专属客服经理：陈经理",
    },
    "C005": {
        "customer_id": "C005",
        "name": "钱七",
        "english_name": "Qian Qi",
        "id_type": "身份证",
        "id_number": "510101199104044567",
        "phone": "13733334444",
        "email": "qianqi@email.com",
        "membership_level": "Standard",
        "frequent_flyer": None,
        "total_miles": 5000,
        "available_miles": 5000,
        "registered_at": "2024-06-01",
        "preferences": {
            "seat": "no_preference",
            "meal": "Standard",
            "language": "zh-CN",
        },
    },
    "C006": {
        "customer_id": "C006",
        "name": "孙八",
        "english_name": "Sun Ba",
        "id_type": "身份证",
        "id_number": "330101199506065678",
        "phone": "13722223333",
        "email": "sunba@mail.com",
        "membership_level": "Gold",
        "frequent_flyer": "FM34567890",
        "total_miles": 85000,
        "available_miles": 30000,
        "registered_at": "2021-08-15",
        "preferences": {
            "seat": "aisle",
            "meal": "Halal",
            "language": "zh-CN",
        },
    },
    "C007": {
        "customer_id": "C007",
        "name": "周九",
        "english_name": "Zhou Jiu",
        "id_type": "身份证",
        "id_number": "610101199307076789",
        "phone": "13844445555",
        "email": "zhoujiu@test.org",
        "membership_level": "Silver",
        "frequent_flyer": "CA11111111",
        "total_miles": 42000,
        "available_miles": 18000,
        "registered_at": "2023-02-28",
        "preferences": {
            "seat": "window",
            "meal": "Low Salt",
            "language": "zh-CN",
        },
        "medical_info": {
            "conditions": ["高血压"],
            "medications": ["降压药"],
            "special_needs": "需要靠近洗手间座位",
        },
    },
    "C008": {
        "customer_id": "C008",
        "name": "吴十",
        "english_name": "Wu Shi",
        "id_type": "身份证",
        "id_number": "420101199008087890",
        "phone": "13955556666",
        "email": "wushi@domain.com",
        "membership_level": "Standard",
        "frequent_flyer": "MU22222222",
        "total_miles": 12000,
        "available_miles": 8000,
        "registered_at": "2024-01-15",
        "preferences": {
            "seat": "no_preference",
            "meal": "Standard",
            "language": "zh-CN",
        },
    },
    "C009": {
        "customer_id": "C009",
        "name": "郑十一",
        "english_name": "Zheng Shiyi",
        "id_type": "护照",
        "id_number": "E12345678",
        "passport_expiry": "2028-05-20",
        "phone": "13866667777",
        "email": "zheng11@international.com",
        "membership_level": "Gold",
        "frequent_flyer": "MU33333333",
        "total_miles": 95000,
        "available_miles": 40000,
        "registered_at": "2020-11-11",
        "preferences": {
            "seat": "window",
            "meal": "Asian Vegetarian",
            "language": "en-US",
        },
        "visa_info": {
            "us_visa": "B1/B2",
            "us_visa_expiry": "2027-08-15",
            "esta_status": "Approved",
        },
    },
    "C010": {
        "customer_id": "C010",
        "name": "华为出差团",
        "english_name": "Huawei Business Team",
        "id_type": "企业账户",
        "id_number": "CORP-HW-2024",
        "phone": "0755-12345678",
        "email": "travel@huawei.com",
        "membership_level": "Corporate",
        "corporate_account": "CORP-HW-2024",
        "company": "华为技术有限公司",
        "contact_person": "王经理",
        "contact_phone": "13977778888",
        "registered_at": "2019-05-01",
        "annual_booking_volume": 5000000,
        "discount_rate": 0.15,
    },
    "C011": {
        "customer_id": "C011",
        "name": "李小明",
        "english_name": "Li Xiaoming",
        "id_type": "出生证明",
        "id_number": "B2024061500001",
        "date_of_birth": "2024-06-15",
        "phone": None,  # 婴儿无电话
        "email": None,
        "membership_level": None,
        "guardian": "C012",
        "guardian_name": "李妈妈",
        "registered_at": "2025-01-11",
        "special_needs": ["婴儿摇篮", "婴儿餐"],
    },
    "C012": {
        "customer_id": "C012",
        "name": "李妈妈",
        "english_name": "Li Mama",
        "id_type": "身份证",
        "id_number": "320101199201011234",
        "phone": "13688889999",
        "email": "limama@family.com",
        "membership_level": "Silver",
        "frequent_flyer": "CZ44444444",
        "total_miles": 28000,
        "available_miles": 12000,
        "registered_at": "2023-05-20",
        "preferences": {
            "seat": "bulkhead",  # 带婴儿优先隔板座位
            "meal": "Standard",
            "language": "zh-CN",
        },
        "dependents": ["C011"],
    },
    # 黑名单/观察名单客户
    "C013": {
        "customer_id": "C013",
        "name": "问题旅客",
        "english_name": "Problem Passenger",
        "id_type": "身份证",
        "id_number": "230101198801011111",
        "phone": "13000000000",
        "email": "problem@test.com",
        "membership_level": "Standard",
        "registered_at": "2022-03-01",
        "watchlist_status": "active",
        "watchlist_reason": "多次航班闹事",
        "restrictions": ["需要额外安检", "禁止升舱"],
    },
    # 外籍客户
    "C014": {
        "customer_id": "C014",
        "name": "John Smith",
        "english_name": "John Smith",
        "id_type": "护照",
        "id_number": "US123456789",
        "passport_country": "USA",
        "passport_expiry": "2029-03-15",
        "phone": "+1-555-123-4567",
        "email": "john.smith@usa.com",
        "membership_level": "Gold",
        "frequent_flyer": "CA55555555",
        "total_miles": 75000,
        "available_miles": 35000,
        "registered_at": "2021-07-04",
        "preferences": {
            "seat": "aisle",
            "meal": "Western",
            "language": "en-US",
        },
    },
    # 高价值客户
    "C015": {
        "customer_id": "C015",
        "name": "陈董事长",
        "english_name": "Chen Chairman",
        "id_type": "护照",
        "id_number": "E98765432",
        "passport_expiry": "2032-06-30",
        "phone": "13999999999",
        "email": "chen.ceo@bigcorp.com",
        "membership_level": "Diamond Plus",
        "frequent_flyer": "CA00000002",
        "total_miles": 3500000,
        "available_miles": 1500000,
        "registered_at": "2010-01-01",
        "preferences": {
            "seat": "suite",
            "meal": "Chef Special",
            "language": "zh-CN",
        },
        "vip_status": True,
        "vip_level": "VVIP",
        "personal_assistant": "专属服务团队",
        "special_privileges": [
            "专车接送",
            "私人休息室",
            "快速通道",
            "免费升舱",
            "24小时专线",
        ],
        "annual_spending": 2000000,
    },
}


def get_customer(customer_id: str) -> dict:
    """获取客户信息"""
    return CUSTOMERS_DB.get(customer_id.upper())


def get_customer_by_phone(phone: str) -> dict:
    """按电话获取客户"""
    for customer in CUSTOMERS_DB.values():
        if customer.get("phone") == phone:
            return customer
    return None


def get_customer_by_email(email: str) -> dict:
    """按邮箱获取客户"""
    for customer in CUSTOMERS_DB.values():
        if customer.get("email") == email.lower():
            return customer
    return None


def get_customers_by_level(level: str) -> list:
    """按会员等级获取客户"""
    return [c for c in CUSTOMERS_DB.values() if c.get("membership_level") == level]


def get_vip_customers() -> list:
    """获取VIP客户"""
    return [c for c in CUSTOMERS_DB.values() if c.get("vip_status")]
