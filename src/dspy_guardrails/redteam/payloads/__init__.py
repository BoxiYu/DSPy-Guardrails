"""
Attack Payloads - 攻击载荷统一管理

所有攻击载荷的集中存放位置。

Usage:
    from dspy_guardrails.redteam.payloads import (
        InjectionPayloads,
        JailbreakPayloads,
        MCPPayloads,
        CLIPayloads,
        get_all_payloads,
    )

    # 获取所有注入攻击载荷
    payloads = InjectionPayloads.get_all()

    # 获取所有 CLI 攻击载荷
    cli_payloads = CLIPayloads().get_all()

    # 获取所有攻击载荷
    all_payloads = get_all_payloads()
"""

from .base import AttackPayload, PayloadCategory
from .bypass import BypassPayloads
from .cli import CLIAttackCategory, CLIPayload, CLIPayloads
from .injection import InjectionPayloads
from .jailbreak import JailbreakPayloads
from .mcp import MCPPayloads

__all__ = [
    "AttackPayload",
    "PayloadCategory",
    "InjectionPayloads",
    "JailbreakPayloads",
    "MCPPayloads",
    "BypassPayloads",
    "CLIPayloads",
    "CLIPayload",
    "CLIAttackCategory",
    "get_all_payloads",
    "get_payloads_by_category",
]


def get_all_payloads() -> list[AttackPayload]:
    """获取所有攻击载荷"""
    payloads = []
    payloads.extend(InjectionPayloads.get_all())
    payloads.extend(JailbreakPayloads.get_all())
    payloads.extend(MCPPayloads.get_all())
    payloads.extend(BypassPayloads.get_all())
    return payloads


def get_payloads_by_category(category: PayloadCategory) -> list[AttackPayload]:
    """按类别获取攻击载荷"""
    mapping = {
        PayloadCategory.INJECTION: InjectionPayloads,
        PayloadCategory.JAILBREAK: JailbreakPayloads,
        PayloadCategory.MCP: MCPPayloads,
        PayloadCategory.BYPASS: BypassPayloads,
    }
    provider = mapping.get(category)
    if provider:
        return provider.get_all()
    return []
