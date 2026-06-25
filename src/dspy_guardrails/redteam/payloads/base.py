"""
Payload Base Types - 攻击载荷基础类型
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PayloadCategory(Enum):
    """攻击载荷类别"""
    INJECTION = "injection"
    JAILBREAK = "jailbreak"
    MCP = "mcp"
    BYPASS = "bypass"
    HARMFUL = "harmful"
    RELEVANCE = "relevance"


class PayloadSeverity(Enum):
    """攻击严重性"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackPayload:
    """
    攻击载荷定义

    Attributes:
        id: 唯一标识符
        prompt: 攻击提示词
        category: 攻击类别
        technique: 攻击技术
        severity: 严重性等级
        source: 来源 (benchmark, custom, domain)
        expected_blocked: 期望被拦截
        metadata: 额外元数据
    """
    id: str
    prompt: str
    category: PayloadCategory
    technique: str = ""
    severity: PayloadSeverity = PayloadSeverity.MEDIUM
    source: str = "custom"
    expected_blocked: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            # 自动生成 ID
            import hashlib
            self.id = hashlib.md5(self.prompt[:50].encode()).hexdigest()[:8]


@dataclass
class PayloadTemplate:
    """
    攻击模板

    可以通过填充参数生成具体攻击载荷。
    """
    template: str
    category: PayloadCategory
    technique: str
    parameters: list[str] = field(default_factory=list)
    severity: PayloadSeverity = PayloadSeverity.MEDIUM

    def generate(self, **kwargs) -> AttackPayload:
        """生成攻击载荷"""
        prompt = self.template.format(**kwargs)
        return AttackPayload(
            id=f"{self.technique}_{hash(prompt) % 10000:04d}",
            prompt=prompt,
            category=self.category,
            technique=self.technique,
            severity=self.severity,
            metadata={"template": self.template, "params": kwargs},
        )


class BasePayloadProvider:
    """攻击载荷提供者基类"""

    @classmethod
    def get_all(cls) -> list[AttackPayload]:
        """获取所有载荷"""
        raise NotImplementedError

    @classmethod
    def get_templates(cls) -> list[PayloadTemplate]:
        """获取所有模板"""
        return []

    @classmethod
    def get_by_technique(cls, technique: str) -> list[AttackPayload]:
        """按技术获取载荷"""
        return [p for p in cls.get_all() if p.technique == technique]

    @classmethod
    def get_by_severity(cls, severity: PayloadSeverity) -> list[AttackPayload]:
        """按严重性获取载荷"""
        return [p for p in cls.get_all() if p.severity == severity]
