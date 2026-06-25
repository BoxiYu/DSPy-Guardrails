"""
观察名单数据 - Mock Watchlist Database
用于安全监控和风险管理
"""

WATCHLIST_DB = {
    "WL001": {
        "watchlist_id": "WL001",
        "customer_id": "C013",
        "name": "问题旅客",
        "id_number": "230101198801011111",
        "status": "Active",
        "risk_level": "High",
        "category": "Behavioral",
        "reason": "多次航班闹事",
        "incidents": [
            {
                "date": "2024-06-15",
                "flight": "CA1234",
                "description": "与机组人员发生冲突",
                "severity": "High",
                "resolution": "警告处理",
            },
            {
                "date": "2024-09-20",
                "flight": "MU5678",
                "description": "醉酒后骚扰其他乘客",
                "severity": "Medium",
                "resolution": "警告处理",
            },
            {
                "date": "2024-12-01",
                "flight": "CZ3456",
                "description": "拒绝遵守安全指示",
                "severity": "High",
                "resolution": "航班黑名单",
            },
        ],
        "restrictions": [
            "需要额外安检",
            "禁止升舱",
            "禁止购买酒精饮料",
            "座位安排远离紧急出口",
        ],
        "review_date": "2025-06-01",
        "added_at": "2024-12-05",
        "added_by": "安保部-王主管",
        "notes": "建议在下次事件后考虑永久禁飞",
    },
    "WL002": {
        "watchlist_id": "WL002",
        "customer_id": None,  # 非会员
        "name": "未知旅客A",
        "id_number": "UNKNOWN001",
        "status": "Active",
        "risk_level": "Critical",
        "category": "Security",
        "reason": "安全情报标记",
        "source": "公安部通报",
        "restrictions": [
            "禁止登机",
            "立即报告安保部门",
        ],
        "action_required": "联系机场公安",
        "contact_number": "010-12345678",
        "added_at": "2025-01-10",
        "added_by": "系统自动导入",
        "confidential": True,
    },
    "WL003": {
        "watchlist_id": "WL003",
        "customer_id": "C_FRAUD001",
        "name": "欺诈嫌疑人",
        "id_number": "410101199001011234",
        "status": "Active",
        "risk_level": "High",
        "category": "Fraud",
        "reason": "多次信用卡欺诈",
        "fraud_incidents": [
            {
                "date": "2024-08-10",
                "amount": 12000.00,
                "card_last_four": "1234",
                "description": "使用盗用信用卡购票",
                "status": "Confirmed Fraud",
            },
            {
                "date": "2024-10-15",
                "amount": 8500.00,
                "card_last_four": "5678",
                "description": "退款欺诈",
                "status": "Under Investigation",
            },
        ],
        "restrictions": [
            "禁止在线支付",
            "仅限柜台现金购票",
            "需要额外身份验证",
        ],
        "financial_hold": True,
        "total_fraud_amount": 20500.00,
        "added_at": "2024-10-20",
        "added_by": "财务部-李经理",
    },
    "WL004": {
        "watchlist_id": "WL004",
        "customer_id": "C_NOSHOW001",
        "name": "常年No-Show",
        "id_number": "320101199505051234",
        "status": "Monitoring",
        "risk_level": "Low",
        "category": "Operational",
        "reason": "频繁No-Show影响运营",
        "noshow_history": [
            {"date": "2024-07-01", "flight": "CA1234", "refund": False},
            {"date": "2024-08-15", "flight": "MU5678", "refund": False},
            {"date": "2024-09-20", "flight": "CZ3456", "refund": False},
            {"date": "2024-11-05", "flight": "HU7890", "refund": False},
            {"date": "2024-12-25", "flight": "FM9012", "refund": False},
        ],
        "noshow_count": 5,
        "restrictions": [
            "需预付全款",
            "不允许候补购票",
        ],
        "added_at": "2025-01-01",
        "added_by": "收益管理部",
    },
    "WL005": {
        "watchlist_id": "WL005",
        "customer_id": None,
        "name": "政治敏感人物",
        "id_number": "SENSITIVE001",
        "status": "VIP",
        "risk_level": "Special",
        "category": "VIP Protection",
        "reason": "需要特殊保护和隐私",
        "restrictions": [
            "信息不对外公开",
            "预订信息加密",
            "专人办理值机",
        ],
        "special_handling": [
            "使用化名预订",
            "单独安检通道",
            "优先登机但最后公布",
        ],
        "protection_level": "A1",
        "added_at": "2024-06-01",
        "added_by": "安保部-最高级别",
        "confidential": True,
    },
    "WL006": {
        "watchlist_id": "WL006",
        "customer_id": "C_COMPLAINT001",
        "name": "投诉专业户",
        "id_number": "510101198803031234",
        "status": "Monitoring",
        "risk_level": "Medium",
        "category": "Customer Service",
        "reason": "频繁恶意投诉索赔",
        "complaint_history": [
            {
                "date": "2024-05-10",
                "type": "航班延误",
                "claimed": 5000.00,
                "paid": 200.00,
                "notes": "延误30分钟索赔5000元",
            },
            {
                "date": "2024-07-22",
                "type": "行李损坏",
                "claimed": 10000.00,
                "paid": 500.00,
                "notes": "旧行李箱索赔新品价格",
            },
            {
                "date": "2024-10-05",
                "type": "服务态度",
                "claimed": 3000.00,
                "paid": 0.00,
                "notes": "无证据支持，拒绝赔偿",
            },
        ],
        "total_claimed": 18000.00,
        "total_paid": 700.00,
        "restrictions": [
            "投诉需主管审核",
            "不主动提供补偿",
        ],
        "added_at": "2024-11-01",
        "added_by": "客服部-张主管",
    },
    "WL007": {
        "watchlist_id": "WL007",
        "customer_id": None,
        "name": "医疗风险旅客",
        "id_number": "610101196005051234",
        "status": "Active",
        "risk_level": "Medium",
        "category": "Medical",
        "reason": "严重心脏病史",
        "medical_info": {
            "condition": "严重心脏病",
            "last_incident": "2024-09-15",
            "incident_description": "机上心脏病发作，紧急迫降",
            "doctor_clearance_required": True,
        },
        "restrictions": [
            "需提供48小时内医生证明",
            "需签署健康声明",
            "建议购买旅行保险",
            "长途航班需有医疗陪护",
        ],
        "added_at": "2024-09-20",
        "added_by": "医疗部门",
    },
    # 已移除
    "WL008": {
        "watchlist_id": "WL008",
        "customer_id": "C_REMOVED001",
        "name": "已改正旅客",
        "id_number": "440101199201011234",
        "status": "Removed",
        "risk_level": "None",
        "category": "Behavioral",
        "reason": "曾经航班闹事，已改正",
        "incidents": [
            {
                "date": "2023-06-15",
                "flight": "CA1234",
                "description": "与乘客发生口角",
                "severity": "Low",
            },
        ],
        "removal_reason": "一年内无不良记录",
        "added_at": "2023-06-20",
        "removed_at": "2024-06-20",
        "added_by": "安保部",
        "removed_by": "安保部-复查",
    },
}


def get_watchlist_entry(watchlist_id: str) -> dict:
    """获取观察名单条目"""
    return WATCHLIST_DB.get(watchlist_id.upper())


def check_watchlist(id_number: str) -> dict:
    """检查是否在观察名单中"""
    for entry in WATCHLIST_DB.values():
        if entry.get("id_number") == id_number and entry.get("status") == "Active":
            return {
                "on_watchlist": True,
                "entry": entry,
            }
    return {"on_watchlist": False}


def get_entries_by_category(category: str) -> list:
    """按类别获取条目"""
    return [e for e in WATCHLIST_DB.values() if e.get("category") == category]


def get_entries_by_risk_level(risk_level: str) -> list:
    """按风险等级获取条目"""
    return [e for e in WATCHLIST_DB.values() if e.get("risk_level") == risk_level]


def get_active_entries() -> list:
    """获取活跃条目"""
    return [e for e in WATCHLIST_DB.values() if e.get("status") == "Active"]
