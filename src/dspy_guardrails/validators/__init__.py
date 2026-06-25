"""
Validators - 标准化 Validator 抽象层

提供类似 Guardrails AI 的 Validator 接口，支持 on_fail 策略。

Usage:
    from dspy_guardrails.validators import (
        Guard, Validator, OnFailAction,
        NoInjection, NoPII, NoToxicity, ValidLength, ValidChoices,
    )

    # 组合多个 validator
    guard = Guard(
        validators=[
            NoInjection(on_fail="exception"),
            NoPII(on_fail="fix"),
            NoToxicity(on_fail="reask"),
        ]
    )

    # 检查文本
    result = guard.validate(text)
    if result.is_valid:
        print(result.output)
    else:
        print(result.errors)

    # 带 LLM reask 的检查
    result = guard(text, llm=my_llm, max_reasks=3)
"""

from dspy_guardrails.validators.base import (
    FailResult,
    OnFailAction,
    PassResult,
    ValidationError,
    Validator,
    ValidatorResult,
)
from dspy_guardrails.validators.builtin import (
    NoInjection,
    NoMCPAttack,
    NoPII,
    NoToxicity,
    ValidChoices,
    ValidJSON,
    ValidLength,
    ValidRange,
    ValidRegex,
)
from dspy_guardrails.validators.guard import AsyncGuard, Guard, GuardHistoryEntry, GuardResult

__all__ = [
    # Base
    "Validator",
    "ValidatorResult",
    "PassResult",
    "FailResult",
    "OnFailAction",
    # Guard
    "ValidationError",
    "Guard",
    "GuardResult",
    "GuardHistoryEntry",
    "AsyncGuard",
    # Built-in validators
    "NoInjection",
    "NoPII",
    "NoToxicity",
    "NoMCPAttack",
    "ValidLength",
    "ValidChoices",
    "ValidRange",
    "ValidRegex",
    "ValidJSON",
]
