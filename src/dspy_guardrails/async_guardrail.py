"""
Async Guardrail API - 异步版本的 guardrail 检查函数

为生产环境异步框架提供非阻塞的安全检测 API。

Usage:
    from dspy_guardrails.async_guardrail import (
        no_injection_async,
        safe_async,
        injection_score_async,
    )

    # 在 async 上下文中使用
    is_safe = await no_injection_async(text)
    score = await injection_score_async(text)

    # 并发检查多个文本
    import asyncio
    results = await asyncio.gather(
        no_injection_async(text1),
        no_injection_async(text2),
        no_injection_async(text3),
    )
"""

import asyncio
from collections.abc import Callable
from typing import Any

from dspy_guardrails.guardrail import guardrail

# =============================================================================
# Pattern-based async wrappers (use asyncio.to_thread for non-blocking)
# =============================================================================


async def no_injection_async(text: str) -> bool:
    """Async version of guardrail.no_injection()"""
    return await asyncio.to_thread(guardrail.no_injection, text)


async def no_pii_async(text: str) -> bool:
    """Async version of guardrail.no_pii()"""
    return await asyncio.to_thread(guardrail.no_pii, text)


async def no_toxicity_async(text: str) -> bool:
    """Async version of guardrail.no_toxicity()"""
    return await asyncio.to_thread(guardrail.no_toxicity, text)


async def safe_async(text: str) -> bool:
    """Async version of guardrail.safe()"""
    return await asyncio.to_thread(guardrail.safe, text)


async def safe_input_async(text: str) -> bool:
    """Async version of guardrail.safe_input()"""
    return await asyncio.to_thread(guardrail.safe_input, text)


async def safe_output_async(text: str) -> bool:
    """Async version of guardrail.safe_output()"""
    return await asyncio.to_thread(guardrail.safe_output, text)


async def injection_score_async(text: str) -> float:
    """Async version of guardrail.injection_score()"""
    return await asyncio.to_thread(guardrail.injection_score, text)


async def toxicity_async(text: str) -> float:
    """Async version of guardrail.toxicity()"""
    return await asyncio.to_thread(guardrail.toxicity, text)


async def pii_score_async(text: str) -> float:
    """Async version of guardrail.pii_score()"""
    return await asyncio.to_thread(guardrail.pii_score, text)


async def no_mcp_attack_async(text: str, context: str = "auto", threshold: float = 0.25) -> bool:
    """Async version of guardrail.no_mcp_attack()"""
    return await asyncio.to_thread(guardrail.no_mcp_attack, text, context, threshold)


async def mcp_security_score_async(text: str, context: str = "auto") -> float:
    """Async version of guardrail.mcp_security_score()"""
    return await asyncio.to_thread(guardrail.mcp_security_score, text, context)


async def safe_mcp_async(text: str, context: str = "auto") -> bool:
    """Async version of guardrail.safe_mcp()"""
    return await asyncio.to_thread(guardrail.safe_mcp, text, context)


# =============================================================================
# Concurrent check utilities
# =============================================================================


async def check_all_async(
    text: str,
    checks: list[str] | None = None,
) -> dict[str, bool | float]:
    """Run multiple guardrail checks concurrently.

    Args:
        text: Text to check.
        checks: List of check names. Defaults to ["no_injection", "no_toxicity", "no_pii"].

    Returns:
        Dict mapping check name to result.
    """
    if checks is None:
        checks = ["no_injection", "no_toxicity", "no_pii"]

    check_map: dict[str, Callable] = {
        "no_injection": no_injection_async,
        "no_pii": no_pii_async,
        "no_toxicity": no_toxicity_async,
        "safe": safe_async,
        "safe_input": safe_input_async,
        "safe_output": safe_output_async,
        "injection_score": injection_score_async,
        "toxicity": toxicity_async,
        "pii_score": pii_score_async,
        "no_mcp_attack": no_mcp_attack_async,
        "mcp_security_score": mcp_security_score_async,
        "safe_mcp": safe_mcp_async,
    }

    tasks = {}
    for name in checks:
        fn = check_map.get(name)
        if fn is not None:
            tasks[name] = fn(text)

    results_list = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results_list, strict=True))


async def batch_check_async(
    texts: list[str],
    check: str = "safe",
    max_concurrency: int = 50,
) -> list[bool | float]:
    """Check multiple texts concurrently with concurrency limit.

    Args:
        texts: List of texts to check.
        check: Check function name.
        max_concurrency: Maximum concurrent checks.

    Returns:
        List of results in same order as input texts.
    """
    check_map: dict[str, Callable] = {
        "no_injection": no_injection_async,
        "no_pii": no_pii_async,
        "no_toxicity": no_toxicity_async,
        "safe": safe_async,
        "injection_score": injection_score_async,
        "toxicity": toxicity_async,
        "pii_score": pii_score_async,
    }

    fn = check_map.get(check)
    if fn is None:
        raise ValueError(f"Unknown check: {check}. Available: {list(check_map.keys())}")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _limited(text: str) -> Any:
        async with semaphore:
            return await fn(text)

    return list(await asyncio.gather(*[_limited(t) for t in texts]))


# =============================================================================
# Async LLM Guardrail
# =============================================================================


class AsyncLLMGuardrail:
    """Async wrapper for LLMGuardrail.

    Usage:
        guard = AsyncLLMGuardrail()
        result = await guard.check("some text", "injection")
        is_safe = await guard.no_injection("some text")
    """

    def __init__(self, use_cot: bool = False):
        from dspy_guardrails.llm_guardrail import LLMGuardrail

        self._guard = LLMGuardrail(use_cot=use_cot)

    async def check(self, text: str, category: str) -> Any:
        """Async check using LLM."""
        return await asyncio.to_thread(self._guard.check, text, category)

    async def no_injection(self, text: str) -> bool:
        result = await self.check(text, "injection")
        return not result.is_unsafe

    async def no_toxicity(self, text: str) -> bool:
        result = await self.check(text, "toxicity")
        return not result.is_unsafe

    async def no_pii(self, text: str) -> bool:
        result = await self.check(text, "pii")
        return not result.is_unsafe

    async def safe(self, text: str) -> bool:
        results = await asyncio.gather(
            self.check(text, "injection"),
            self.check(text, "toxicity"),
        )
        return all(not r.is_unsafe for r in results)


class AsyncHybridGuardrail:
    """Async wrapper for HybridGuardrail.

    Usage:
        guard = AsyncHybridGuardrail()
        result = await guard.check("some text", "injection")
        # Also supports tuple unpacking:
        is_unsafe, confidence = await guard.check("some text", "injection")
    """

    def __init__(self, use_llm: bool = True, threshold: float = 0.2):
        from dspy_guardrails.llm_guardrail import HybridGuardrail

        self._guard = HybridGuardrail(use_llm=use_llm, threshold=threshold)

    async def check(self, text: str, category: str, use_llm_fallback: bool = True):
        from dspy_guardrails.llm_guardrail import HybridResult

        return await asyncio.to_thread(self._guard.check, text, category, use_llm_fallback)
