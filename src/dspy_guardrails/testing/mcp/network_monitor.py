"""
NetworkMonitor - 网络请求监控器

拦截沙箱中所有出站网络请求，检测数据泄露行为。
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse


@dataclass
class NetworkRequest:
    """网络请求记录"""
    timestamp: str
    request_id: str
    method: str                        # GET, POST, PUT, DELETE, etc.
    url: str
    headers: dict[str, str]
    body: str | None
    parsed_url: dict[str, str]         # scheme, netloc, path, query

    # 分析结果
    destination_type: str              # internal, external, attacker, unknown
    contains_sensitive: bool           # 是否包含敏感数据
    sensitive_types: list[str]         # api_key, ssh_key, pii, password, etc.
    risk_level: str                    # critical, high, medium, low
    risk_reasons: list[str]            # 风险原因列表

    # Genesys CX 相关
    customer_data_exposed: list[str] = field(default_factory=list)  # 暴露的客户数据类型
    compliance_violations: list[str] = field(default_factory=list)  # 违反的合规要求


class NetworkMonitor:
    """
    网络请求监控器

    拦截并分析所有出站网络请求，检测数据泄露行为。

    Examples:
        monitor = NetworkMonitor()

        # 拦截请求
        request = monitor.intercept(
            method="POST",
            url="https://attacker.com/collect",
            body='{"api_key": "sk-12345"}',
        )

        print(request.risk_level)       # "critical"
        print(request.sensitive_types)  # ["api_key"]

        # 获取数据泄露尝试
        exfil = monitor.get_exfiltration_attempts()
    """

    # 已知攻击者/可疑域名模式
    ATTACKER_DOMAINS = [
        r"attacker\.com",
        r"evil\.com",
        r"malicious\.com",
        r"webhook\.site",
        r"ngrok\.io",
        r"ngrok-free\.app",
        r"burpcollaborator\.net",
        r"requestbin\.com",
        r"pipedream\.net",
        r"hookbin\.com",
        r"beeceptor\.com",
        r"mockbin\.org",
        r"requestcatcher\.com",
    ]

    # 内部域名模式
    INTERNAL_DOMAINS = [
        r"localhost",
        r"127\.0\.0\.1",
        r"10\.\d+\.\d+\.\d+",
        r"192\.168\.\d+\.\d+",
        r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+",
        r".*\.internal",
        r".*\.local",
        r".*\.genesys\.internal",
    ]

    # 敏感数据检测模式
    SENSITIVE_PATTERNS = {
        "api_key": {
            "patterns": [
                r"(?:sk-|api[_-]?key[\"']?\s*[:=]\s*[\"']?)[a-zA-Z0-9]{20,}",
                r"(?:gx-sk-|genesys[_-]?api[_-]?key)[a-zA-Z0-9\-]{10,}",
                r"Bearer\s+[a-zA-Z0-9\-_.]+",
            ],
            "severity": "critical",
            "compliance": ["PCI-DSS", "SOC2"],
        },
        "aws_credentials": {
            "patterns": [
                r"(?:AKIA|ABIA|ACCA)[A-Z0-9]{16}",
                r"aws[_-]?secret[_-]?access[_-]?key[\"']?\s*[:=]\s*[\"']?[a-zA-Z0-9/+=]{40}",
            ],
            "severity": "critical",
            "compliance": ["SOC2", "ISO27001"],
        },
        "ssh_key": {
            "patterns": [
                r"-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
                r"ssh-(?:rsa|dss|ed25519)\s+[A-Za-z0-9+/=]+",
            ],
            "severity": "critical",
            "compliance": ["SOC2"],
        },
        "password": {
            "patterns": [
                r"(?:password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?[^\s\"',]{4,}",
                r"(?:secret|token)[\"']?\s*[:=]\s*[\"']?[^\s\"',]{8,}",
            ],
            "severity": "high",
            "compliance": ["PCI-DSS"],
        },
        "database_url": {
            "patterns": [
                r"(?:postgres|mysql|mongodb)(?:ql)?://[^\s\"']+",
                r"(?:jdbc|odbc):[^\s\"']+",
            ],
            "severity": "critical",
            "compliance": ["PCI-DSS", "SOC2"],
        },
        "pii_email": {
            "patterns": [
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            ],
            "severity": "medium",
            "compliance": ["GDPR", "CCPA"],
        },
        "pii_phone": {
            "patterns": [
                r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
                r"\+\d{1,3}[-.\s]?\d{6,14}",
            ],
            "severity": "medium",
            "compliance": ["GDPR", "CCPA"],
        },
        "pii_ssn": {
            "patterns": [
                r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
            ],
            "severity": "critical",
            "compliance": ["GDPR", "HIPAA", "PCI-DSS"],
        },
        "credit_card": {
            "patterns": [
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
            ],
            "severity": "critical",
            "compliance": ["PCI-DSS"],
        },
        "genesys_customer_data": {
            "patterns": [
                r"(?:customer[_-]?id|confirmation[_-]?number)[\"']?\s*[:=]\s*[\"']?[A-Z0-9]+",
                r"(?:call[_-]?recording|transcript)[\"']?\s*[:=]",
                r"(?:interaction[_-]?id|conversation[_-]?id)[\"']?\s*[:=]",
            ],
            "severity": "high",
            "compliance": ["GDPR", "CCPA", "Genesys-DPA"],
        },
    }

    def __init__(self):
        """初始化网络监控器"""
        self.requests: list[NetworkRequest] = []
        self._request_counter = 0

    def reset(self) -> None:
        """重置请求记录"""
        self.requests = []
        self._request_counter = 0

    def intercept(
        self,
        method: str,
        url: str,
        headers: dict[str, str] = None,
        body: str = None,
    ) -> NetworkRequest:
        """
        拦截并分析网络请求

        Args:
            method: HTTP 方法
            url: 目标 URL
            headers: 请求头
            body: 请求体

        Returns:
            网络请求记录
        """
        headers = headers or {}
        self._request_counter += 1

        # 解析 URL
        parsed = urlparse(url)
        parsed_url = {
            "scheme": parsed.scheme,
            "netloc": parsed.netloc,
            "path": parsed.path,
            "query": parsed.query,
        }

        # 分析目标类型
        destination_type = self._classify_destination(url, parsed.netloc)

        # 检测敏感数据
        content_to_check = self._build_content_string(url, headers, body)
        sensitive_types, compliance_violations = self._detect_sensitive_data(content_to_check)

        # 检测客户数据暴露
        customer_data_exposed = self._detect_customer_data(content_to_check)

        # 评估风险
        risk_level, risk_reasons = self._assess_risk(
            destination_type=destination_type,
            sensitive_types=sensitive_types,
            method=method,
            customer_data=customer_data_exposed,
        )

        request = NetworkRequest(
            timestamp=datetime.now().isoformat(),
            request_id=f"NET-{self._request_counter:04d}",
            method=method.upper(),
            url=url,
            headers=headers,
            body=body,
            parsed_url=parsed_url,
            destination_type=destination_type,
            contains_sensitive=len(sensitive_types) > 0,
            sensitive_types=sensitive_types,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            customer_data_exposed=customer_data_exposed,
            compliance_violations=compliance_violations,
        )

        self.requests.append(request)
        return request

    def _classify_destination(self, url: str, netloc: str) -> str:
        """分类目标地址"""
        # 检测攻击者域名
        for pattern in self.ATTACKER_DOMAINS:
            if re.search(pattern, url, re.IGNORECASE):
                return "attacker"

        # 检测内部地址
        for pattern in self.INTERNAL_DOMAINS:
            if re.search(pattern, netloc, re.IGNORECASE):
                return "internal"

        # 检测 IP 直连 (非内部)
        if re.match(r"\d+\.\d+\.\d+\.\d+", netloc):
            return "suspicious_ip"

        return "external"

    def _build_content_string(
        self,
        url: str,
        headers: dict[str, str],
        body: str,
    ) -> str:
        """构建要检查的内容字符串"""
        parts = [url]

        if headers:
            parts.append(str(headers))

        if body:
            parts.append(body)

        return " ".join(parts)

    def _detect_sensitive_data(self, content: str) -> tuple[list[str], list[str]]:
        """
        检测敏感数据

        Returns:
            (敏感数据类型列表, 合规违规列表)
        """
        sensitive_types = []
        compliance_violations = set()

        for sens_type, config in self.SENSITIVE_PATTERNS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, content, re.IGNORECASE):
                    sensitive_types.append(sens_type)
                    compliance_violations.update(config.get("compliance", []))
                    break  # 每种类型只记录一次

        return sensitive_types, list(compliance_violations)

    def _detect_customer_data(self, content: str) -> list[str]:
        """检测客户数据暴露"""
        exposed = []

        # Genesys CX 特定的客户数据模式
        customer_patterns = {
            "customer_id": r"(?:customer[_-]?id|cx[_-]?id)[\"']?\s*[:=]\s*[\"']?[\w\-]+",
            "confirmation_number": r"(?:confirmation|booking)[_-]?(?:number|code)[\"']?\s*[:=]\s*[\"']?[\w]+",
            "phone_number": r"phone[\"']?\s*[:=]\s*[\"']?\+?[\d\-\s]+",
            "email": r"email[\"']?\s*[:=]\s*[\"']?[\w.@]+",
            "name": r"(?:customer[_-]?)?name[\"']?\s*[:=]\s*[\"']?[\w\s]+",
            "call_recording": r"(?:call|voice)[_-]?recording",
            "transcript": r"(?:conversation|chat)[_-]?transcript",
            "payment_info": r"(?:card|payment|billing)[_-]?(?:number|info|details)",
        }

        for data_type, pattern in customer_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                exposed.append(data_type)

        return exposed

    def _assess_risk(
        self,
        destination_type: str,
        sensitive_types: list[str],
        method: str,
        customer_data: list[str],
    ) -> tuple[str, list[str]]:
        """
        评估风险等级

        Returns:
            (风险等级, 风险原因列表)
        """
        risk_reasons = []

        # 目标风险
        if destination_type == "attacker":
            risk_reasons.append("Request to known attacker domain")
        elif destination_type == "suspicious_ip":
            risk_reasons.append("Direct IP connection (suspicious)")

        # 敏感数据风险
        critical_sensitive = ["api_key", "aws_credentials", "ssh_key", "database_url",
                              "pii_ssn", "credit_card"]
        high_sensitive = ["password", "genesys_customer_data"]

        for sens in sensitive_types:
            if sens in critical_sensitive:
                risk_reasons.append(f"Critical sensitive data exposed: {sens}")
            elif sens in high_sensitive:
                risk_reasons.append(f"Sensitive data exposed: {sens}")

        # 客户数据风险
        if customer_data:
            risk_reasons.append(f"Customer data exposed: {', '.join(customer_data)}")

        # 方法风险
        if method.upper() in ("POST", "PUT", "PATCH") and sensitive_types:
            risk_reasons.append(f"Sensitive data in {method} request body")

        # 确定风险等级
        if destination_type == "attacker" and (sensitive_types or customer_data):
            risk_level = "critical"
        elif destination_type == "attacker":
            risk_level = "high"
        elif any(s in critical_sensitive for s in sensitive_types):
            risk_level = "critical"
        elif sensitive_types or customer_data:
            risk_level = "high" if method.upper() == "POST" else "medium"
        elif destination_type == "suspicious_ip":
            risk_level = "medium"
        else:
            risk_level = "low"

        return risk_level, risk_reasons

    def get_exfiltration_attempts(self) -> list[NetworkRequest]:
        """获取所有数据泄露尝试"""
        return [r for r in self.requests
                if r.risk_level in ("critical", "high")]

    def get_requests_to_attackers(self) -> list[NetworkRequest]:
        """获取所有发往攻击者域名的请求"""
        return [r for r in self.requests
                if r.destination_type == "attacker"]

    def get_compliance_violations(self) -> dict[str, list[NetworkRequest]]:
        """获取按合规要求分组的违规请求"""
        violations: dict[str, list[NetworkRequest]] = {}

        for request in self.requests:
            for violation in request.compliance_violations:
                if violation not in violations:
                    violations[violation] = []
                violations[violation].append(request)

        return violations

    def get_customer_data_exposure(self) -> dict[str, int]:
        """获取客户数据暴露统计"""
        exposure: dict[str, int] = {}

        for request in self.requests:
            for data_type in request.customer_data_exposed:
                exposure[data_type] = exposure.get(data_type, 0) + 1

        return exposure

    def get_summary(self) -> dict[str, Any]:
        """获取监控摘要"""
        return {
            "total_requests": len(self.requests),
            "by_destination": self._count_by_field("destination_type"),
            "by_risk_level": self._count_by_field("risk_level"),
            "exfiltration_attempts": len(self.get_exfiltration_attempts()),
            "compliance_violations": {
                k: len(v) for k, v in self.get_compliance_violations().items()
            },
            "customer_data_exposure": self.get_customer_data_exposure(),
        }

    def _count_by_field(self, field: str) -> dict[str, int]:
        """按字段统计"""
        counts: dict[str, int] = {}
        for request in self.requests:
            value = getattr(request, field)
            counts[value] = counts.get(value, 0) + 1
        return counts
