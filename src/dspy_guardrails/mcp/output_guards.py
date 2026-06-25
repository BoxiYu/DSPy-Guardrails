"""
Tool Output Guards - 工具输出防护

过滤和清理工具输出，防止间接注入和数据泄露。
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .core import GuardAction, GuardResult, ThreatCategory, ToolCallContext


@dataclass
class OutputSanitizationResult:
    """输出清理结果"""
    sanitized: bool
    original_length: int
    sanitized_length: int
    items_removed: int = 0
    items_redacted: int = 0
    threats_detected: list[str] = field(default_factory=list)


class ToolOutputGuard(ABC):
    """工具输出防护基类"""

    name: str = "base_output_guard"
    description: str = "Base output guard"

    @abstractmethod
    def filter(
        self,
        output: Any,
        context: ToolCallContext,
    ) -> GuardResult:
        """过滤工具输出"""
        pass

    @classmethod
    def indirect_injection_check(cls) -> "IndirectInjectionGuard":
        """创建间接注入防护"""
        return IndirectInjectionGuard()

    @classmethod
    def pii_filter(cls) -> "PIIOutputFilter":
        """创建 PII 过滤器"""
        return PIIOutputFilter()

    @classmethod
    def sensitive_data_filter(
        cls,
        patterns: list[str] = None,
    ) -> "SensitiveDataFilter":
        """创建敏感数据过滤器"""
        return SensitiveDataFilter(patterns)

    @classmethod
    def size_limit(
        cls,
        max_size: int = 100000,
    ) -> "OutputSizeLimiter":
        """创建输出大小限制器"""
        return OutputSizeLimiter(max_size)

    @classmethod
    def custom(
        cls,
        filter_fn: Callable[[Any, ToolCallContext], GuardResult],
        name: str = "custom",
    ) -> "CustomOutputGuard":
        """创建自定义过滤器"""
        return CustomOutputGuard(filter_fn, name)


class IndirectInjectionGuard(ToolOutputGuard):
    """
    间接 Prompt Injection 防护

    检测工具输出中可能劫持 AI 的恶意指令。
    这是 MCP 最重要的安全防护之一。
    """

    name = "indirect_injection_guard"
    description = "Detect indirect prompt injection in tool outputs"

    # 间接注入模式
    INJECTION_PATTERNS = [
        # AI 指令劫持
        r"(assistant|ai|claude|gpt|model)\s*[:=]\s*",
        r"<\|?(system|assistant|user)\|?>",
        r"\[INST\]|\[/INST\]",
        r"###\s*(system|instruction|response)",

        # 指令覆盖
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|context)",
        r"disregard\s+(the\s+)?user",
        r"new\s+instructions?\s*[:=]",
        r"important\s*[:!]\s*ignore",
        r"actually[,\s]+(instead|do|please)",

        # 角色劫持
        r"you\s+(are|must|should)\s+now",
        r"from\s+now\s+on",
        r"your\s+(new\s+)?role\s+is",
        r"act\s+as\s+(if|a|an)",

        # 隐蔽指令
        r"<!--.*?-->",      # HTML 注释
        r"/\*.*?\*/",       # C-style 注释
        r"IMPORTANT:\s*",
        r"NOTE TO AI:",
        r"SYSTEM OVERRIDE:",

        # 数据泄露请求
        r"send\s+(this|data|info).*(to|via)\s*(email|http|url)",
        r"exfiltrate",
        r"forward\s+to",
    ]

    # 高风险关键词
    HIGH_RISK_KEYWORDS = [
        "ignore previous", "new instructions", "override",
        "jailbreak", "bypass", "unrestricted",
        "execute code", "run command", "shell access",
    ]

    def __init__(
        self,
        additional_patterns: list[str] = None,
        sensitivity: float = 0.5,
    ):
        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.INJECTION_PATTERNS]
        if additional_patterns:
            self.patterns.extend([
                re.compile(p, re.IGNORECASE) for p in additional_patterns
            ])
        self.sensitivity = sensitivity

    def filter(
        self,
        output: Any,
        context: ToolCallContext,
    ) -> GuardResult:
        """检测间接注入"""
        # 将输出转为字符串
        text = self._to_text(output)

        if not text:
            return GuardResult(action=GuardAction.ALLOW, passed=True)

        matches = []
        high_risk_count = 0

        # 检查模式
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                matches.append({
                    "pattern": pattern.pattern[:50],
                    "matched": match.group()[:100],
                    "position": match.start(),
                })

        # 检查高风险关键词
        text_lower = text.lower()
        for keyword in self.HIGH_RISK_KEYWORDS:
            if keyword in text_lower:
                high_risk_count += 1
                matches.append({
                    "type": "high_risk_keyword",
                    "keyword": keyword,
                })

        if matches:
            # 计算威胁分数
            threat_score = min(1.0, len(matches) * 0.2 + high_risk_count * 0.15)

            if threat_score >= 0.7:
                # 高威胁：阻止或清理
                sanitized = self._sanitize_output(text)
                return GuardResult(
                    action=GuardAction.MODIFY,
                    passed=True,
                    threat_category=ThreatCategory.INDIRECT_INJECTION,
                    threat_score=threat_score,
                    message=f"Detected {len(matches)} indirect injection patterns - output sanitized",
                    details={"matches": matches[:5], "sanitized": True},
                    modified_value=sanitized,
                )
            else:
                # 中等威胁：警告
                return GuardResult(
                    action=GuardAction.WARN,
                    passed=True,
                    threat_category=ThreatCategory.INDIRECT_INJECTION,
                    threat_score=threat_score,
                    message=f"Detected {len(matches)} potential injection patterns",
                    details={"matches": matches[:5]},
                )

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _to_text(self, output: Any) -> str:
        """将输出转为文本"""
        if isinstance(output, str):
            return output
        elif isinstance(output, dict):
            return json.dumps(output, ensure_ascii=False)
        elif isinstance(output, list):
            return json.dumps(output, ensure_ascii=False)
        else:
            return str(output)

    def _sanitize_output(self, text: str) -> str:
        """清理输出中的恶意内容"""
        sanitized = text

        # 移除危险模式
        for pattern in self.patterns:
            sanitized = pattern.sub("[FILTERED]", sanitized)

        # 添加安全前缀
        warning = "[Note: This output has been sanitized for security]\n\n"

        return warning + sanitized


class PIIOutputFilter(ToolOutputGuard):
    """
    PII 输出过滤器

    检测并脱敏工具输出中的个人身份信息。
    """

    name = "pii_filter"
    description = "Filter PII from tool outputs"

    PII_PATTERNS = {
        "email": (
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "[EMAIL_REDACTED]"
        ),
        "phone": (
            r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "[PHONE_REDACTED]"
        ),
        "ssn": (
            r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
            "[SSN_REDACTED]"
        ),
        "credit_card": (
            r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
            "[CC_REDACTED]"
        ),
        "ip_address": (
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "[IP_REDACTED]"
        ),
        "api_key": (
            r'\b(?:sk|pk|api[_-]?key)[_-]?[A-Za-z0-9]{20,}\b',
            "[API_KEY_REDACTED]"
        ),
        "aws_key": (
            r'\b(?:AKIA|ABIA|ACCA)[A-Z0-9]{16}\b',
            "[AWS_KEY_REDACTED]"
        ),
        "private_key": (
            r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
            "[PRIVATE_KEY_REDACTED]"
        ),
    }

    def __init__(
        self,
        pii_types: list[str] = None,
        redact: bool = True,
    ):
        self.pii_types = pii_types or list(self.PII_PATTERNS.keys())
        self.redact = redact
        self.compiled_patterns = {
            name: (re.compile(pattern, re.IGNORECASE), replacement)
            for name, (pattern, replacement) in self.PII_PATTERNS.items()
            if name in self.pii_types
        }

    def filter(
        self,
        output: Any,
        context: ToolCallContext,
    ) -> GuardResult:
        """过滤 PII"""
        text = self._to_text(output)
        if not text:
            return GuardResult(action=GuardAction.ALLOW, passed=True)

        detected = []
        sanitized_text = text

        for pii_type, (pattern, replacement) in self.compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected.append({
                    "type": pii_type,
                    "count": len(matches),
                })
                if self.redact:
                    sanitized_text = pattern.sub(replacement, sanitized_text)

        if detected:
            return GuardResult(
                action=GuardAction.MODIFY if self.redact else GuardAction.WARN,
                passed=True,
                threat_category=ThreatCategory.PII_EXPOSURE,
                threat_score=min(1.0, len(detected) * 0.2),
                message=f"Detected {sum(d['count'] for d in detected)} PII items",
                details={"detected": detected, "redacted": self.redact},
                modified_value=sanitized_text if self.redact else None,
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _to_text(self, output: Any) -> str:
        if isinstance(output, str):
            return output
        elif isinstance(output, (dict, list)):
            return json.dumps(output, ensure_ascii=False)
        return str(output)


class SensitiveDataFilter(ToolOutputGuard):
    """
    敏感数据过滤器

    过滤特定的敏感数据模式。
    """

    name = "sensitive_data_filter"
    description = "Filter sensitive data patterns"

    DEFAULT_PATTERNS = [
        # 密码相关
        r"password\s*[:=]\s*['\"]?[\w!@#$%^&*]+",
        r"secret\s*[:=]\s*['\"]?[\w-]+",
        r"token\s*[:=]\s*['\"]?[\w-]+",

        # 数据库连接
        r"(?:mysql|postgres|mongodb)://[^\s]+",
        r"Data Source=[^\s;]+",

        # 云凭证
        r"AKIA[A-Z0-9]{16}",  # AWS
        r"ghp_[A-Za-z0-9]{36}",  # GitHub
        r"xox[baprs]-[0-9]+-[0-9A-Za-z]+",  # Slack
    ]

    def __init__(self, patterns: list[str] = None):
        all_patterns = self.DEFAULT_PATTERNS + (patterns or [])
        self.patterns = [re.compile(p, re.IGNORECASE) for p in all_patterns]

    def filter(
        self,
        output: Any,
        context: ToolCallContext,
    ) -> GuardResult:
        """过滤敏感数据"""
        text = str(output) if output else ""
        matches = []

        for pattern in self.patterns:
            for match in pattern.finditer(text):
                matches.append({
                    "pattern": pattern.pattern[:30],
                    "position": match.start(),
                })

        if matches:
            # 替换敏感内容
            sanitized = text
            for pattern in self.patterns:
                sanitized = pattern.sub("[SENSITIVE_DATA_REDACTED]", sanitized)

            return GuardResult(
                action=GuardAction.MODIFY,
                passed=True,
                threat_category=ThreatCategory.SENSITIVE_DATA,
                threat_score=0.7,
                message=f"Filtered {len(matches)} sensitive data patterns",
                details={"matches_count": len(matches)},
                modified_value=sanitized,
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)


class OutputSizeLimiter(ToolOutputGuard):
    """
    输出大小限制器

    防止过大的输出导致问题。
    """

    name = "output_size_limiter"
    description = "Limit output size"

    def __init__(
        self,
        max_size: int = 100000,
        truncate: bool = True,
    ):
        self.max_size = max_size
        self.truncate = truncate

    def filter(
        self,
        output: Any,
        context: ToolCallContext,
    ) -> GuardResult:
        """限制输出大小"""
        text = str(output) if output else ""
        size = len(text)

        if size > self.max_size:
            if self.truncate:
                truncated = text[:self.max_size] + f"\n\n[TRUNCATED: Output exceeded {self.max_size} chars]"
                return GuardResult(
                    action=GuardAction.MODIFY,
                    passed=True,
                    threat_category=ThreatCategory.RESOURCE_ABUSE,
                    threat_score=0.3,
                    message=f"Output truncated from {size} to {self.max_size} chars",
                    modified_value=truncated,
                )
            else:
                return GuardResult(
                    action=GuardAction.BLOCK,
                    passed=False,
                    threat_category=ThreatCategory.RESOURCE_ABUSE,
                    threat_score=0.5,
                    message=f"Output size {size} exceeds limit {self.max_size}",
                )

        return GuardResult(action=GuardAction.ALLOW, passed=True)


class CustomOutputGuard(ToolOutputGuard):
    """自定义输出防护"""

    def __init__(
        self,
        filter_fn: Callable[[Any, ToolCallContext], GuardResult],
        name: str = "custom",
    ):
        self.filter_fn = filter_fn
        self.name = name

    def filter(
        self,
        output: Any,
        context: ToolCallContext,
    ) -> GuardResult:
        return self.filter_fn(output, context)
