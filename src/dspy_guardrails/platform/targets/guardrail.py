"""GuardrailTarget - Adapter for guardrail functions."""

import time
from collections.abc import Callable

from .protocol import (
    TargetCapability,
    TargetResponse,
    TargetType,
    UnifiedTarget,
)


class GuardrailTarget(UnifiedTarget):
    """
    护栏函数适配器

    将 guardrail.no_injection() 等函数包装为 UnifiedTarget。

    Example:
        from dspy_guardrails import guardrail
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection)
        response = target.invoke("test input")
    """

    target_type = TargetType.GUARDRAIL
    capabilities = [TargetCapability.SINGLE_TURN]

    def __init__(
        self,
        guardrail_fn: Callable[[str], bool],
        name: str | None = None,
        threshold: float = 0.5,
        score_fn: Callable[[str], float] | None = None,
    ):
        """
        初始化护栏目标

        Args:
            guardrail_fn: 护栏检测函数 (返回 True 表示安全)
            name: 目标名称
            threshold: 拦截阈值 (用于评分函数)
            score_fn: 可选的评分函数 (返回 0-1 分数)
        """
        self._guardrail_fn = guardrail_fn
        self._name = name or guardrail_fn.__name__
        self._threshold = threshold
        self._score_fn = score_fn

    def invoke(self, prompt: str) -> TargetResponse:
        """执行护栏检测"""
        start_time = time.time()

        # 执行检测
        is_safe = self._guardrail_fn(prompt)

        # 计算分数 (如果有评分函数)
        score = 0.0
        if self._score_fn:
            score = self._score_fn(prompt)
        elif not is_safe:
            score = 1.0  # 不安全时默认分数为 1

        latency_ms = (time.time() - start_time) * 1000

        return TargetResponse(
            response="BLOCKED" if not is_safe else "PASSED",
            was_blocked=not is_safe,
            block_reason=f"{self._name} detection triggered" if not is_safe else None,
            guardrail_scores={self._name: score},
            latency_ms=latency_ms,
        )

    def invoke_multi_turn(
        self,
        messages: list[dict[str, str]],
    ) -> TargetResponse:
        """多轮调用 (护栏函数只检测最后一条消息)"""
        if not messages:
            return TargetResponse(response="No messages", was_blocked=False)

        last_message = messages[-1].get("content", "")
        return self.invoke(last_message)

    def reset_session(self) -> None:
        """护栏函数无状态，无需重置"""
        pass

    @property
    def name(self) -> str:
        return self._name
