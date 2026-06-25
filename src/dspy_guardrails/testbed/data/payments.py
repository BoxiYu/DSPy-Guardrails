"""
支付数据 - Mock Payments Database
"""

PAYMENTS_DB = {
    "PAY001": {
        "payment_id": "PAY001",
        "booking_code": "ABC123",
        "customer_id": "C001",
        "amount": 1280.00,
        "currency": "CNY",
        "method": "Alipay",
        "status": "Completed",
        "transaction_id": "ALI20250110093012345",
        "created_at": "2025-01-10 09:30:00",
        "completed_at": "2025-01-10 09:30:15",
        "description": "机票购买 - CA1234",
    },
    "PAY002": {
        "payment_id": "PAY002",
        "booking_code": "DEF456",
        "customer_id": "C002",
        "amount": 4580.00,
        "currency": "CNY",
        "method": "Corporate Card",
        "card_last_four": "8888",
        "status": "Completed",
        "transaction_id": "CORP20250108142212345",
        "created_at": "2025-01-08 14:22:00",
        "completed_at": "2025-01-08 14:22:30",
        "description": "商务舱机票 - MU5678",
        "invoice_required": True,
        "invoice_title": "ABC科技有限公司",
    },
    "PAY003": {
        "payment_id": "PAY003",
        "booking_code": "GHI789",
        "customer_id": "C003",
        "amount": 980.00,
        "currency": "CNY",
        "method": "WeChat Pay",
        "status": "Completed",
        "transaction_id": "WX20250105111512345",
        "created_at": "2025-01-05 11:15:00",
        "completed_at": "2025-01-05 11:15:20",
        "description": "机票购买 - HU7890",
    },
    "PAY004": {
        "payment_id": "PAY004",
        "booking_code": "JKL012",
        "customer_id": "C004",
        "amount": 42000.00,
        "currency": "CNY",
        "method": "Bank Transfer",
        "bank_name": "中国银行",
        "account_last_four": "1234",
        "status": "Completed",
        "transaction_id": "BOC20250102080012345",
        "created_at": "2025-01-02 08:00:00",
        "completed_at": "2025-01-02 08:05:00",
        "description": "头等舱机票 - CA985",
        "vip_discount": 0.1,
        "original_amount": 46666.67,
    },
    "PAY005": {
        "payment_id": "PAY005",
        "booking_code": "MNO345",
        "customer_id": "C005",
        "amount": 1180.00,
        "currency": "CNY",
        "method": "Credit Card",
        "card_type": "Visa",
        "card_last_four": "5678",
        "status": "Refunded",
        "transaction_id": "VISA20250103100012345",
        "created_at": "2025-01-03 10:00:00",
        "completed_at": "2025-01-03 10:00:45",
        "refunded_at": "2025-01-12 16:35:00",
        "refund_amount": 1062.00,
        "refund_id": "REF001",
        "description": "机票购买 - CZ3456 (已退款)",
    },
    "PAY006": {
        "payment_id": "PAY006",
        "booking_code": "PQR678",
        "customer_id": "C006",
        "amount": 1560.00,
        "currency": "CNY",
        "method": "Pending",
        "status": "Pending",
        "created_at": "2025-01-15 12:00:00",
        "expires_at": "2025-01-16 18:00:00",
        "description": "待支付 - FM9012",
        "payment_url": "https://pay.airline.com/order/PQR678",
    },
    "PAY007": {
        "payment_id": "PAY007",
        "booking_code": "STU901",
        "customer_id": "C007",
        "amount": 1380.00,
        "currency": "CNY",
        "method": "Miles + Cash",
        "miles_used": 10000,
        "miles_value": 100.00,
        "cash_amount": 1280.00,
        "status": "Completed",
        "transaction_id": "MIX20250112200012345",
        "created_at": "2025-01-12 20:00:00",
        "completed_at": "2025-01-12 20:01:00",
        "description": "里程+现金 - CA1888",
    },
    "PAY008": {
        "payment_id": "PAY008",
        "booking_code": "VWX234",
        "customer_id": "C008",
        "amount": 990.00,  # 原价890 + 改签费100
        "currency": "CNY",
        "method": "Alipay",
        "status": "Completed",
        "transaction_id": "ALI20250114090012345",
        "created_at": "2025-01-14 09:00:00",
        "completed_at": "2025-01-14 09:00:30",
        "description": "改签费 + 差价 - MU2345",
        "breakdown": {
            "change_fee": 100.00,
            "fare_difference": 0.00,
            "original_payment": "PAY_ORIG_008",
        },
    },
    "PAY009": {
        "payment_id": "PAY009",
        "booking_code": "YZA567",
        "customer_id": "C009",
        "amount": 8900.00,
        "currency": "CNY",
        "method": "Credit Card",
        "card_type": "MasterCard",
        "card_last_four": "9012",
        "status": "Completed",
        "transaction_id": "MC20250105180012345",
        "created_at": "2025-01-05 18:00:00",
        "completed_at": "2025-01-05 18:01:00",
        "description": "国际航班 - MU583",
        "exchange_rate": 7.25,  # USD to CNY
    },
    "PAY010": {
        "payment_id": "PAY010",
        "booking_code": "BCD890",
        "customer_id": "C010",
        "amount": 15600.00,
        "currency": "CNY",
        "method": "Corporate Account",
        "corporate_id": "CORP-HW-2024",
        "status": "Completed",
        "transaction_id": "CORP20250108100012345",
        "created_at": "2025-01-08 10:00:00",
        "completed_at": "2025-01-08 10:00:00",
        "description": "团队机票 - CA173 (6人)",
        "unit_price": 2600.00,
        "quantity": 6,
        "corporate_discount": 0.05,
        "invoice_required": True,
        "invoice_title": "华为技术有限公司",
        "invoice_tax_id": "91440300708461136T",
    },
    "PAY011": {
        "payment_id": "PAY011",
        "booking_code": "EFG123",
        "customer_id": "C012",  # 由母亲支付
        "amount": 98.00,
        "currency": "CNY",
        "method": "WeChat Pay",
        "status": "Completed",
        "transaction_id": "WX20250111140012345",
        "created_at": "2025-01-11 14:00:00",
        "completed_at": "2025-01-11 14:00:10",
        "description": "婴儿票 - CZ6789",
    },
    "PAY012": {
        "payment_id": "PAY012",
        "booking_code": "HIJ456",
        "customer_id": "C012",
        "amount": 1280.00,
        "currency": "CNY",
        "method": "WeChat Pay",
        "status": "Completed",
        "transaction_id": "WX20250111140112345",
        "created_at": "2025-01-11 14:01:00",
        "completed_at": "2025-01-11 14:01:15",
        "description": "成人票 - CZ6789",
    },
    # 失败的支付
    "PAY013": {
        "payment_id": "PAY013",
        "booking_code": "TEMP001",
        "customer_id": "C003",
        "amount": 2500.00,
        "currency": "CNY",
        "method": "Credit Card",
        "card_type": "Visa",
        "card_last_four": "1111",
        "status": "Failed",
        "failure_reason": "余额不足",
        "failure_code": "INSUFFICIENT_FUNDS",
        "created_at": "2025-01-14 15:30:00",
        "failed_at": "2025-01-14 15:30:05",
        "description": "支付失败 - 尝试预订",
    },
    # 外币支付
    "PAY014": {
        "payment_id": "PAY014",
        "booking_code": "INT001",
        "customer_id": "C014",
        "amount": 1200.00,
        "currency": "USD",
        "cny_amount": 8700.00,
        "exchange_rate": 7.25,
        "method": "Credit Card",
        "card_type": "American Express",
        "card_last_four": "3456",
        "status": "Completed",
        "transaction_id": "AMEX20250113120012345",
        "created_at": "2025-01-13 12:00:00",
        "completed_at": "2025-01-13 12:00:30",
        "description": "International booking - USD",
    },
    # 分期付款
    "PAY015": {
        "payment_id": "PAY015",
        "booking_code": "INST001",
        "customer_id": "C006",
        "amount": 12000.00,
        "currency": "CNY",
        "method": "Installment",
        "installment_plan": {
            "total_amount": 12000.00,
            "down_payment": 4000.00,
            "monthly_payment": 2000.00,
            "months": 4,
            "interest_rate": 0.0,  # 免息分期
            "paid_installments": 2,
            "remaining_amount": 4000.00,
        },
        "status": "In Progress",
        "transaction_id": "INST20250101100012345",
        "created_at": "2025-01-01 10:00:00",
        "description": "分期付款 - 家庭度假套餐",
        "next_payment_date": "2025-02-01",
    },
}


def get_payment(payment_id: str) -> dict:
    """获取支付信息"""
    return PAYMENTS_DB.get(payment_id.upper())


def get_payments_by_booking(booking_code: str) -> list:
    """按订单获取支付"""
    return [p for p in PAYMENTS_DB.values() if p.get("booking_code") == booking_code.upper()]


def get_payments_by_customer(customer_id: str) -> list:
    """按客户获取支付"""
    return [p for p in PAYMENTS_DB.values() if p.get("customer_id") == customer_id.upper()]


def get_payments_by_status(status: str) -> list:
    """按状态获取支付"""
    return [p for p in PAYMENTS_DB.values() if p.get("status") == status]


def get_pending_payments() -> list:
    """获取待支付订单"""
    return get_payments_by_status("Pending")
