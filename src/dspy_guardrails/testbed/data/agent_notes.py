"""
客服备注数据 - Mock Agent Notes Database
客服人员对客户的内部备注和处理记录
"""

AGENT_NOTES_DB = {
    "NOTE001": {
        "note_id": "NOTE001",
        "customer_id": "C001",
        "booking_code": "ABC123",
        "agent_id": "AGENT001",
        "agent_name": "客服小王",
        "created_at": "2025-01-10 09:35:00",
        "note_type": "General",
        "content": "客户询问是否可以提前值机，已告知可通过APP提前24小时办理",
        "visibility": "All Agents",
        "priority": "Normal",
    },
    "NOTE002": {
        "note_id": "NOTE002",
        "customer_id": "C002",
        "booking_code": "DEF456",
        "agent_id": "AGENT002",
        "agent_name": "客服小李",
        "created_at": "2025-01-08 14:30:00",
        "note_type": "VIP",
        "content": "商务舱常客，公司大客户，需要优先处理。已安排商务舱专属值机通道。",
        "visibility": "All Agents",
        "priority": "High",
        "vip_flag": True,
    },
    "NOTE003": {
        "note_id": "NOTE003",
        "customer_id": "C003",
        "booking_code": "GHI789",
        "agent_id": "AGENT003",
        "agent_name": "客服小张",
        "created_at": "2025-01-15 08:10:00",
        "note_type": "Issue",
        "content": "航班HU7890取消，客户非常不满。已提供改签选项HU7892和CA4102。客户要求全额退款+补偿，已升级到主管处理。",
        "visibility": "All Agents",
        "priority": "Urgent",
        "escalated": True,
        "escalated_to": "主管-陈经理",
    },
    "NOTE004": {
        "note_id": "NOTE004",
        "customer_id": "C004",
        "booking_code": "JKL012",
        "agent_id": "AGENT_VIP001",
        "agent_name": "VIP客服经理",
        "created_at": "2025-01-02 08:10:00",
        "note_type": "VIP",
        "content": "钻石Plus会员，年消费超200万。已安排全程专人服务：专车接送+贵宾室+快速通道。请各环节确保最高服务标准。",
        "visibility": "VIP Team Only",
        "priority": "Critical",
        "vip_flag": True,
        "special_instructions": [
            "称呼：赵董事长",
            "偏好：靠窗座位、中餐",
            "禁忌：无",
            "专属客服经理：陈经理 138xxxx1111",
        ],
    },
    "NOTE005": {
        "note_id": "NOTE005",
        "customer_id": "C005",
        "booking_code": "MNO345",
        "agent_id": "AGENT004",
        "agent_name": "客服小赵",
        "created_at": "2025-01-12 16:35:00",
        "note_type": "Cancellation",
        "content": "客户因行程变更取消订单。已处理退款，扣除手续费118元。客户表示理解。",
        "visibility": "All Agents",
        "priority": "Normal",
        "resolution": "Completed",
    },
    "NOTE006": {
        "note_id": "NOTE006",
        "customer_id": "C006",
        "booking_code": "PQR678",
        "agent_id": "AGENT005",
        "agent_name": "客服小刘",
        "created_at": "2025-01-15 12:05:00",
        "note_type": "Payment",
        "content": "客户询问支付问题，订单待支付状态。已发送支付链接到手机，提醒支付截止时间为明天18:00。",
        "visibility": "All Agents",
        "priority": "Normal",
        "follow_up_required": True,
        "follow_up_date": "2025-01-16 12:00:00",
    },
    "NOTE007": {
        "note_id": "NOTE007",
        "customer_id": "C007",
        "booking_code": "STU901",
        "agent_id": "AGENT006",
        "agent_name": "客服小陈",
        "created_at": "2025-01-15 08:20:00",
        "note_type": "Medical",
        "content": "客户有高血压病史，已在预订时备注。本次航班正常完成，无医疗事件。建议下次预订时主动确认健康状况。",
        "visibility": "All Agents",
        "priority": "Normal",
        "medical_flag": True,
    },
    "NOTE008": {
        "note_id": "NOTE008",
        "customer_id": "C008",
        "booking_code": "VWX234",
        "agent_id": "AGENT007",
        "agent_name": "客服小孙",
        "created_at": "2025-01-14 09:10:00",
        "note_type": "Change",
        "content": "客户因时间冲突要求改签。原航班MU2343改为MU2345，收取改签费100元。客户已确认新航班信息。",
        "visibility": "All Agents",
        "priority": "Normal",
        "resolution": "Completed",
    },
    "NOTE009": {
        "note_id": "NOTE009",
        "customer_id": "C010",
        "booking_code": "BCD890",
        "agent_id": "AGENT_CORP001",
        "agent_name": "企业客户经理",
        "created_at": "2025-01-08 10:15:00",
        "note_type": "Corporate",
        "content": "华为出差团6人预订。团队负责人王经理，需要开具增值税专用发票。发票信息已收集，将在行程结束后开具。",
        "visibility": "Corporate Team",
        "priority": "Normal",
        "corporate_flag": True,
        "invoice_info": {
            "title": "华为技术有限公司",
            "tax_id": "91440300708461136T",
            "address": "深圳市龙岗区坂田华为基地",
        },
    },
    "NOTE010": {
        "note_id": "NOTE010",
        "customer_id": "C012",
        "booking_code": "HIJ456",
        "agent_id": "AGENT008",
        "agent_name": "客服小周",
        "created_at": "2025-01-11 14:10:00",
        "note_type": "Special Service",
        "content": "带婴儿旅客，已申请婴儿摇篮。提醒客户提前到达办理值机，以确保分配到摇篮座位。同行婴儿订单号EFG123。",
        "visibility": "All Agents",
        "priority": "Normal",
        "linked_bookings": ["EFG123"],
    },
    "NOTE011": {
        "note_id": "NOTE011",
        "customer_id": "C013",
        "booking_code": None,
        "agent_id": "AGENT_SEC001",
        "agent_name": "安保客服",
        "created_at": "2024-12-05 10:00:00",
        "note_type": "Security",
        "content": "【敏感信息-仅限授权人员查看】该旅客已加入观察名单，有多次航班闹事记录。任何预订需通知安保部门。禁止升舱、禁止购买酒精饮料。",
        "visibility": "Security Team Only",
        "priority": "Critical",
        "security_flag": True,
        "watchlist_id": "WL001",
        "confidential": True,
    },
    "NOTE012": {
        "note_id": "NOTE012",
        "customer_id": "C001",
        "booking_code": "ABC123",
        "agent_id": "AGENT009",
        "agent_name": "客服小吴",
        "created_at": "2025-01-14 20:10:00",
        "note_type": "Upsell",
        "content": "客户购买了机上WiFi套餐(68元)和餐食升级(88元)。金卡会员，可以推荐升舱优惠活动。",
        "visibility": "All Agents",
        "priority": "Normal",
        "upsell_products": ["WIFI", "Meal Upgrade"],
        "upsell_total": 156.00,
    },
    # 投诉处理记录
    "NOTE013": {
        "note_id": "NOTE013",
        "customer_id": "C_COMPLAINT001",
        "booking_code": "COMP001",
        "agent_id": "AGENT_SUP001",
        "agent_name": "客服主管-张经理",
        "created_at": "2025-01-10 15:00:00",
        "note_type": "Complaint",
        "content": "客户第三次投诉餐食质量问题。经核实，本次航班餐食确实存在问题（供应商批次问题）。已提供200元代金券补偿，客户接受。已反馈给餐食供应商。",
        "visibility": "Supervisors Only",
        "priority": "High",
        "complaint_id": "COMP-2025-001",
        "resolution": "Compensated",
        "compensation": {
            "type": "Voucher",
            "amount": 200.00,
            "voucher_code": "SORRY200",
        },
    },
    # 特殊需求记录
    "NOTE014": {
        "note_id": "NOTE014",
        "customer_id": "C014",
        "booking_code": "INT001",
        "agent_id": "AGENT010",
        "agent_name": "国际客服-Emma",
        "created_at": "2025-01-13 12:10:00",
        "note_type": "International",
        "content": "American passenger, prefers English communication. Frequent flyer with CA. Requested aisle seat and Western meal. Confirmed ESTA status for US return trip.",
        "visibility": "All Agents",
        "priority": "Normal",
        "language_preference": "English",
    },
    # 系统自动备注
    "NOTE015": {
        "note_id": "NOTE015",
        "customer_id": "C001",
        "booking_code": "ABC123",
        "agent_id": "SYSTEM",
        "agent_name": "系统自动",
        "created_at": "2025-01-15 06:30:00",
        "note_type": "System",
        "content": "客户已通过APP完成自助值机，座位12A，登机口A12，登机时间08:00。",
        "visibility": "All Agents",
        "priority": "Low",
        "auto_generated": True,
    },
}


def get_note(note_id: str) -> dict:
    """获取备注信息"""
    return AGENT_NOTES_DB.get(note_id.upper())


def get_notes_by_customer(customer_id: str) -> list:
    """按客户获取备注"""
    return [n for n in AGENT_NOTES_DB.values() if n.get("customer_id") == customer_id]


def get_notes_by_booking(booking_code: str) -> list:
    """按订单获取备注"""
    return [n for n in AGENT_NOTES_DB.values() if n.get("booking_code") == booking_code]


def get_notes_by_type(note_type: str) -> list:
    """按类型获取备注"""
    return [n for n in AGENT_NOTES_DB.values() if n.get("note_type") == note_type]


def get_urgent_notes() -> list:
    """获取紧急备注"""
    return [n for n in AGENT_NOTES_DB.values() if n.get("priority") in ["Urgent", "Critical"]]


def get_notes_with_followup() -> list:
    """获取需要跟进的备注"""
    return [n for n in AGENT_NOTES_DB.values() if n.get("follow_up_required")]
