"""
审计日志数据 - Mock Audit Logs Database
系统操作记录和安全审计
"""

AUDIT_LOGS_DB = {
    "LOG001": {
        "log_id": "LOG001",
        "timestamp": "2025-01-15 09:30:15",
        "event_type": "BOOKING_CREATE",
        "user_type": "Customer",
        "user_id": "C001",
        "session_id": "sess_abc123",
        "action": "创建订单",
        "resource": "booking",
        "resource_id": "ABC123",
        "details": {
            "flight": "CA1234",
            "passenger": "张三",
            "amount": 1280.00,
        },
        "ip_address": "223.104.xxx.xxx",
        "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
        "status": "Success",
    },
    "LOG002": {
        "log_id": "LOG002",
        "timestamp": "2025-01-15 09:30:45",
        "event_type": "PAYMENT_PROCESS",
        "user_type": "Customer",
        "user_id": "C001",
        "session_id": "sess_abc123",
        "action": "支付订单",
        "resource": "payment",
        "resource_id": "PAY001",
        "details": {
            "booking_code": "ABC123",
            "amount": 1280.00,
            "method": "Alipay",
            "transaction_id": "ALI20250110093012345",
        },
        "ip_address": "223.104.xxx.xxx",
        "status": "Success",
    },
    "LOG003": {
        "log_id": "LOG003",
        "timestamp": "2025-01-15 10:15:00",
        "event_type": "AGENT_LOGIN",
        "user_type": "Agent",
        "user_id": "EMP001",
        "session_id": "sess_agent001",
        "action": "客服登录",
        "resource": "session",
        "details": {
            "employee_name": "张经理",
            "department": "客服部",
            "login_method": "SSO",
        },
        "ip_address": "10.0.xxx.xxx",
        "user_agent": "Chrome/120.0.0.0",
        "status": "Success",
    },
    "LOG004": {
        "log_id": "LOG004",
        "timestamp": "2025-01-15 10:20:00",
        "event_type": "BOOKING_VIEW",
        "user_type": "Agent",
        "user_id": "EMP001",
        "session_id": "sess_agent001",
        "action": "查看订单",
        "resource": "booking",
        "resource_id": "ABC123",
        "details": {
            "customer_id": "C001",
            "reason": "客户咨询",
        },
        "ip_address": "10.0.xxx.xxx",
        "status": "Success",
    },
    "LOG005": {
        "log_id": "LOG005",
        "timestamp": "2025-01-15 10:25:00",
        "event_type": "SENSITIVE_DATA_ACCESS",
        "user_type": "Agent",
        "user_id": "EMP001",
        "session_id": "sess_agent001",
        "action": "访问敏感信息",
        "resource": "customer_pii",
        "resource_id": "C001",
        "details": {
            "data_type": "身份证号",
            "masked": True,
            "reason": "身份验证",
        },
        "ip_address": "10.0.xxx.xxx",
        "status": "Success",
        "sensitivity": "High",
    },
    "LOG006": {
        "log_id": "LOG006",
        "timestamp": "2025-01-15 11:00:00",
        "event_type": "REFUND_REQUEST",
        "user_type": "Customer",
        "user_id": "C005",
        "session_id": "sess_def456",
        "action": "申请退款",
        "resource": "refund",
        "resource_id": "REF001",
        "details": {
            "booking_code": "MNO345",
            "amount": 1180.00,
            "reason": "行程变更",
        },
        "ip_address": "116.228.xxx.xxx",
        "status": "Success",
    },
    "LOG007": {
        "log_id": "LOG007",
        "timestamp": "2025-01-15 11:30:00",
        "event_type": "GUARDRAIL_BLOCK",
        "user_type": "Customer",
        "user_id": "UNKNOWN",
        "session_id": "sess_suspicious001",
        "action": "请求被拦截",
        "resource": "chat",
        "details": {
            "input": "[REDACTED - Injection Attempt]",
            "guardrail": "injection_detection",
            "confidence": 0.95,
            "reason": "检测到注入攻击尝试",
        },
        "ip_address": "45.33.xxx.xxx",
        "status": "Blocked",
        "sensitivity": "Critical",
        "alert_sent": True,
    },
    "LOG008": {
        "log_id": "LOG008",
        "timestamp": "2025-01-15 11:31:00",
        "event_type": "SECURITY_ALERT",
        "user_type": "System",
        "user_id": "SYSTEM",
        "action": "安全告警",
        "resource": "security",
        "details": {
            "alert_type": "Injection Attack",
            "source_ip": "45.33.xxx.xxx",
            "related_log": "LOG007",
            "action_taken": "IP临时封禁",
            "notified": ["安保部", "IT部"],
        },
        "status": "Alert",
        "sensitivity": "Critical",
    },
    "LOG009": {
        "log_id": "LOG009",
        "timestamp": "2025-01-15 12:00:00",
        "event_type": "BOOKING_CANCEL",
        "user_type": "Customer",
        "user_id": "C005",
        "session_id": "sess_def456",
        "action": "取消订单",
        "resource": "booking",
        "resource_id": "MNO345",
        "details": {
            "original_status": "Confirmed",
            "new_status": "Cancelled",
            "refund_amount": 1062.00,
            "fee_deducted": 118.00,
        },
        "ip_address": "116.228.xxx.xxx",
        "status": "Success",
    },
    "LOG010": {
        "log_id": "LOG010",
        "timestamp": "2025-01-15 12:30:00",
        "event_type": "CHAT_SESSION",
        "user_type": "Customer",
        "user_id": "C003",
        "session_id": "sess_chat001",
        "action": "AI对话",
        "resource": "chat",
        "details": {
            "turns": 5,
            "topic": "航班取消咨询",
            "escalated": True,
            "escalated_to": "EMP001",
            "satisfaction_rating": 3,
        },
        "ip_address": "117.136.xxx.xxx",
        "status": "Completed",
    },
    "LOG011": {
        "log_id": "LOG011",
        "timestamp": "2025-01-15 13:00:00",
        "event_type": "PERMISSION_DENIED",
        "user_type": "Agent",
        "user_id": "EMP002",
        "session_id": "sess_agent002",
        "action": "权限不足",
        "resource": "refund_approval",
        "resource_id": "REF010",
        "details": {
            "requested_action": "批准大额退款",
            "refund_amount": 18000.00,
            "required_permission": "approve_high_value",
            "user_permission": ["view_basic", "edit_booking"],
        },
        "ip_address": "10.0.xxx.xxx",
        "status": "Denied",
    },
    "LOG012": {
        "log_id": "LOG012",
        "timestamp": "2025-01-15 13:30:00",
        "event_type": "WATCHLIST_CHECK",
        "user_type": "System",
        "user_id": "SYSTEM",
        "action": "观察名单检查",
        "resource": "watchlist",
        "details": {
            "checked_id": "230101198801011111",
            "result": "MATCH",
            "watchlist_id": "WL001",
            "risk_level": "High",
            "booking_blocked": True,
        },
        "status": "Alert",
        "sensitivity": "High",
    },
    "LOG013": {
        "log_id": "LOG013",
        "timestamp": "2025-01-15 14:00:00",
        "event_type": "DATA_EXPORT",
        "user_type": "Agent",
        "user_id": "EMP001",
        "session_id": "sess_agent001",
        "action": "导出数据",
        "resource": "report",
        "details": {
            "report_type": "Daily Booking Summary",
            "date_range": "2025-01-15",
            "record_count": 156,
            "contains_pii": False,
        },
        "ip_address": "10.0.xxx.xxx",
        "status": "Success",
        "approval_id": "APPR001",
    },
    "LOG014": {
        "log_id": "LOG014",
        "timestamp": "2025-01-15 14:30:00",
        "event_type": "FRAUD_DETECTION",
        "user_type": "System",
        "user_id": "SYSTEM",
        "action": "欺诈检测",
        "resource": "payment",
        "resource_id": "PAY_FRAUD001",
        "details": {
            "customer_id": "C_FRAUD001",
            "amount": 15000.00,
            "fraud_score": 0.85,
            "indicators": [
                "新设备",
                "异常地理位置",
                "高价值交易",
                "历史欺诈记录",
            ],
            "action_taken": "交易拦截",
        },
        "status": "Blocked",
        "sensitivity": "Critical",
        "alert_sent": True,
    },
    "LOG015": {
        "log_id": "LOG015",
        "timestamp": "2025-01-15 15:00:00",
        "event_type": "API_CALL",
        "user_type": "System",
        "user_id": "SYSTEM",
        "action": "外部API调用",
        "resource": "external_api",
        "details": {
            "api": "Weather API",
            "endpoint": "/forecast",
            "parameters": {"city": "Beijing"},
            "response_time_ms": 245,
            "status_code": 200,
        },
        "status": "Success",
    },
}


def get_log(log_id: str) -> dict:
    """获取日志条目"""
    return AUDIT_LOGS_DB.get(log_id.upper())


def get_logs_by_user(user_id: str) -> list:
    """按用户获取日志"""
    return [log for log in AUDIT_LOGS_DB.values() if log.get("user_id") == user_id]


def get_logs_by_event_type(event_type: str) -> list:
    """按事件类型获取日志"""
    return [log for log in AUDIT_LOGS_DB.values() if log.get("event_type") == event_type]


def get_logs_by_resource(resource: str, resource_id: str = None) -> list:
    """按资源获取日志"""
    logs = [log for log in AUDIT_LOGS_DB.values() if log.get("resource") == resource]
    if resource_id:
        logs = [log for log in logs if log.get("resource_id") == resource_id]
    return logs


def get_security_alerts() -> list:
    """获取安全告警"""
    alert_types = ["GUARDRAIL_BLOCK", "SECURITY_ALERT", "FRAUD_DETECTION", "WATCHLIST_CHECK"]
    return [log for log in AUDIT_LOGS_DB.values() if log.get("event_type") in alert_types]


def get_sensitive_access_logs() -> list:
    """获取敏感数据访问日志"""
    return [log for log in AUDIT_LOGS_DB.values() if log.get("sensitivity") in ["High", "Critical"]]


def get_logs_by_session(session_id: str) -> list:
    """按会话获取日志"""
    return sorted(
        [log for log in AUDIT_LOGS_DB.values() if log.get("session_id") == session_id],
        key=lambda x: x.get("timestamp", "")
    )


def get_blocked_requests() -> list:
    """获取被拦截的请求"""
    return [log for log in AUDIT_LOGS_DB.values() if log.get("status") in ["Blocked", "Denied"]]
