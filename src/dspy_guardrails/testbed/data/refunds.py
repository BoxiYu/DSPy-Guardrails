"""
退款数据 - Mock Refunds Database
"""

REFUNDS_DB = {
    "REF001": {
        "refund_id": "REF001",
        "booking_code": "MNO345",
        "customer_id": "C005",
        "payment_id": "PAY005",
        "original_amount": 1180.00,
        "refund_amount": 1062.00,
        "deductions": {
            "cancellation_fee": 100.00,
            "service_fee": 18.00,
        },
        "currency": "CNY",
        "reason": "自愿取消",
        "reason_code": "VOLUNTARY_CANCEL",
        "status": "Completed",
        "refund_method": "Original Payment",
        "created_at": "2025-01-12 16:30:00",
        "processed_at": "2025-01-12 16:35:00",
        "completed_at": "2025-01-13 10:00:00",
        "processing_time_days": 1,
    },
    "REF002": {
        "refund_id": "REF002",
        "booking_code": "GHI789",
        "customer_id": "C003",
        "payment_id": "PAY003",
        "original_amount": 980.00,
        "refund_amount": 980.00,  # 航班取消全额退款
        "deductions": {},
        "currency": "CNY",
        "reason": "航班取消",
        "reason_code": "AIRLINE_CANCEL",
        "status": "Processing",
        "refund_method": "Original Payment",
        "created_at": "2025-01-15 08:00:00",
        "processed_at": None,
        "estimated_completion": "2025-01-18 00:00:00",
        "priority": "high",  # 航司原因优先处理
    },
    "REF003": {
        "refund_id": "REF003",
        "booking_code": "DELAY001",
        "customer_id": "C002",
        "payment_id": "PAY_DELAY001",
        "original_amount": 3200.00,
        "refund_amount": 640.00,  # 延误补偿 20%
        "deductions": {},
        "currency": "CNY",
        "reason": "航班延误超过4小时",
        "reason_code": "DELAY_COMPENSATION",
        "status": "Completed",
        "refund_method": "Account Credit",
        "created_at": "2025-01-10 20:00:00",
        "processed_at": "2025-01-10 20:05:00",
        "completed_at": "2025-01-10 20:10:00",
        "compensation_type": "delay",
        "delay_hours": 5,
    },
    "REF004": {
        "refund_id": "REF004",
        "booking_code": "UPGRADE001",
        "customer_id": "C004",
        "payment_id": "PAY_UP001",
        "original_amount": 15000.00,
        "refund_amount": 15000.00,
        "deductions": {},
        "currency": "CNY",
        "reason": "升舱失败退款",
        "reason_code": "UPGRADE_FAILED",
        "status": "Completed",
        "refund_method": "Miles",
        "miles_refunded": 150000,
        "created_at": "2025-01-08 12:00:00",
        "processed_at": "2025-01-08 12:01:00",
        "completed_at": "2025-01-08 12:01:00",
    },
    "REF005": {
        "refund_id": "REF005",
        "booking_code": "REBK001",
        "customer_id": "C006",
        "payment_id": "PAY_REBK001",
        "original_amount": 2800.00,
        "refund_amount": 1200.00,  # 改签差价退款
        "deductions": {},
        "currency": "CNY",
        "reason": "改签差价退还",
        "reason_code": "REBOOK_DIFFERENCE",
        "status": "Pending",
        "refund_method": "Original Payment",
        "created_at": "2025-01-15 09:00:00",
        "estimated_completion": "2025-01-20 00:00:00",
    },
    "REF006": {
        "refund_id": "REF006",
        "booking_code": "SVC001",
        "customer_id": "C001",
        "payment_id": "PAY_SVC001",
        "original_amount": 200.00,
        "refund_amount": 200.00,
        "deductions": {},
        "currency": "CNY",
        "reason": "增值服务未使用",
        "reason_code": "SERVICE_UNUSED",
        "status": "Completed",
        "refund_method": "Account Balance",
        "service_type": "选座服务",
        "created_at": "2025-01-14 16:00:00",
        "processed_at": "2025-01-14 16:05:00",
        "completed_at": "2025-01-14 16:05:00",
    },
    "REF007": {
        "refund_id": "REF007",
        "booking_code": "BAG001",
        "customer_id": "C007",
        "payment_id": "PAY_BAG001",
        "original_amount": 300.00,
        "refund_amount": 300.00,
        "deductions": {},
        "currency": "CNY",
        "reason": "行李丢失赔偿",
        "reason_code": "BAGGAGE_LOSS",
        "status": "Processing",
        "refund_method": "Bank Transfer",
        "bank_info": {
            "bank": "中国工商银行",
            "account_last_four": "5678",
        },
        "claim_id": "CLM001",
        "created_at": "2025-01-13 14:00:00",
        "estimated_completion": "2025-01-20 00:00:00",
    },
    "REF008": {
        "refund_id": "REF008",
        "booking_code": "CORP001",
        "customer_id": "C010",
        "payment_id": "PAY_CORP001",
        "original_amount": 26000.00,  # 10人团队
        "refund_amount": 5200.00,  # 2人取消
        "deductions": {
            "cancellation_fee": 200.00,  # 100 x 2
        },
        "currency": "CNY",
        "reason": "部分团队成员取消",
        "reason_code": "PARTIAL_CANCEL",
        "status": "Completed",
        "refund_method": "Corporate Account Credit",
        "corporate_id": "CORP-HW-2024",
        "cancelled_passengers": 2,
        "remaining_passengers": 8,
        "created_at": "2025-01-11 10:00:00",
        "processed_at": "2025-01-11 10:30:00",
        "completed_at": "2025-01-11 11:00:00",
    },
    # 拒绝的退款
    "REF009": {
        "refund_id": "REF009",
        "booking_code": "DENY001",
        "customer_id": "C013",
        "payment_id": "PAY_DENY001",
        "original_amount": 1500.00,
        "refund_amount": 0.00,
        "deductions": {},
        "currency": "CNY",
        "reason": "自愿取消",
        "reason_code": "VOLUNTARY_CANCEL",
        "status": "Rejected",
        "rejection_reason": "不可退改签票种",
        "rejection_code": "NON_REFUNDABLE",
        "created_at": "2025-01-09 11:00:00",
        "rejected_at": "2025-01-09 11:30:00",
        "appeal_available": True,
        "appeal_deadline": "2025-01-16 11:00:00",
    },
    # 争议中的退款
    "REF010": {
        "refund_id": "REF010",
        "booking_code": "DISP001",
        "customer_id": "C008",
        "payment_id": "PAY_DISP001",
        "original_amount": 2200.00,
        "refund_amount": None,  # 待定
        "deductions": {},
        "currency": "CNY",
        "reason": "服务质量投诉",
        "reason_code": "SERVICE_COMPLAINT",
        "status": "Under Review",
        "review_notes": "客户投诉机上餐食质量问题",
        "assigned_to": "客服主管-张经理",
        "created_at": "2025-01-14 09:00:00",
        "review_deadline": "2025-01-21 09:00:00",
        "customer_evidence": ["照片1.jpg", "照片2.jpg"],
    },
    # 保险理赔退款
    "REF011": {
        "refund_id": "REF011",
        "booking_code": "INS001",
        "customer_id": "C009",
        "payment_id": "PAY_INS001",
        "original_amount": 8900.00,
        "refund_amount": 7120.00,  # 80% 保险赔付
        "deductions": {
            "insurance_deductible": 1780.00,  # 20% 免赔额
        },
        "currency": "CNY",
        "reason": "因病无法出行",
        "reason_code": "MEDICAL_CANCEL",
        "status": "Completed",
        "refund_method": "Insurance Payout",
        "insurance_policy": "POLICY-2025-001",
        "insurance_company": "平安保险",
        "medical_certificate": "已验证",
        "created_at": "2025-01-12 08:00:00",
        "processed_at": "2025-01-13 10:00:00",
        "completed_at": "2025-01-15 14:00:00",
    },
}


def get_refund(refund_id: str) -> dict:
    """获取退款信息"""
    return REFUNDS_DB.get(refund_id.upper())


def get_refunds_by_booking(booking_code: str) -> list:
    """按订单获取退款"""
    return [r for r in REFUNDS_DB.values() if r.get("booking_code") == booking_code.upper()]


def get_refunds_by_customer(customer_id: str) -> list:
    """按客户获取退款"""
    return [r for r in REFUNDS_DB.values() if r.get("customer_id") == customer_id.upper()]


def get_refunds_by_status(status: str) -> list:
    """按状态获取退款"""
    return [r for r in REFUNDS_DB.values() if r.get("status") == status]


def get_pending_refunds() -> list:
    """获取待处理退款"""
    return [r for r in REFUNDS_DB.values() if r.get("status") in ["Pending", "Processing"]]
