"""LlamaGuard-based guard targets for input and output checking.

Uses LlamaGuard 4 (12B) via OpenRouter API. Supports both:
- Input mode: checks user prompt only → safe/unsafe
- Output mode: checks user prompt + assistant response → safe/unsafe

This enables direct comparison of the SAME guardrail model in different placements.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

from dspy_guardrails.testing.targets import BaseTarget, TargetResponse
from .vllm_target import OpenRouterTarget


class LlamaGuardChecker:
    """Calls LlamaGuard 4 via OpenRouter to classify content as safe/unsafe."""

    def __init__(
        self,
        model: str = "meta-llama/llama-guard-4-12b",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def check_input(self, prompt: str) -> tuple[bool, str]:
        """Check if user prompt is safe. Returns (is_unsafe, category)."""
        messages = [{"role": "user", "content": prompt}]
        return self._call(messages)

    def check_output(self, prompt: str, response: str) -> tuple[bool, str]:
        """Check if assistant response is safe. Returns (is_unsafe, category)."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return self._call(messages)

    def _call(self, messages: list[dict]) -> tuple[bool, str]:
        """Call LlamaGuard API. Returns (is_unsafe: bool, category: str)."""
        try:
            resp = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 50,
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return False, f"api_error_{resp.status_code}"

            text = resp.json()["choices"][0]["message"]["content"].strip()
            is_unsafe = text.lower().startswith("unsafe")
            category = text.split("\n")[1].strip() if "\n" in text else ""
            return is_unsafe, category

        except Exception as exc:
            return False, f"error: {exc}"


class LlamaGuardInputTarget(BaseTarget):
    """Target with LlamaGuard 4 as INPUT guard.

    Flow: prompt → LlamaGuard checks prompt → if unsafe, block → else, call target LLM
    """

    def __init__(self, target_model: str = "qwen/qwen3-235b-a22b-2507"):
        super().__init__()
        self._target = OpenRouterTarget(model=target_model)
        self._guard = LlamaGuardChecker()

    def invoke(self, prompt: str) -> TargetResponse:
        is_unsafe, category = self._guard.check_input(prompt)
        if is_unsafe:
            return TargetResponse(
                response="",
                guardrail_status={"llamaguard_input": {"passed": False, "category": category}},
                metadata={"block_reason": f"LlamaGuard input: {category}", "guard": "llamaguard_input"},
            )
        return self._target.invoke(prompt)

    def reset_session(self):
        self._target.reset_session()
        self.clear_history()


class LlamaGuardOutputTarget(BaseTarget):
    """Target with LlamaGuard 4 as OUTPUT guard.

    Flow: prompt → target LLM generates response → LlamaGuard checks response → if unsafe, block
    """

    def __init__(self, target_model: str = "qwen/qwen3-235b-a22b-2507"):
        super().__init__()
        self._target = OpenRouterTarget(model=target_model)
        self._guard = LlamaGuardChecker()

    def invoke(self, prompt: str) -> TargetResponse:
        result = self._target.invoke(prompt)
        if result.metadata.get("error"):
            return result

        is_unsafe, category = self._guard.check_output(prompt, result.response)
        if is_unsafe:
            return TargetResponse(
                response="",
                guardrail_status={"llamaguard_output": {"passed": False, "category": category}},
                metadata={**result.metadata, "block_reason": f"LlamaGuard output: {category}", "guard": "llamaguard_output"},
                latency_ms=result.latency_ms,
            )
        result.metadata["guard"] = "llamaguard_output_passed"
        return result

    def reset_session(self):
        self._target.reset_session()
        self.clear_history()


class LlamaGuardBothTarget(BaseTarget):
    """Target with LlamaGuard 4 as both INPUT and OUTPUT guard.

    Flow: prompt → LlamaGuard checks prompt → target LLM → LlamaGuard checks response
    """

    def __init__(self, target_model: str = "qwen/qwen3-235b-a22b-2507"):
        super().__init__()
        self._target = OpenRouterTarget(model=target_model)
        self._guard = LlamaGuardChecker()

    def invoke(self, prompt: str) -> TargetResponse:
        # Input check
        is_unsafe, category = self._guard.check_input(prompt)
        if is_unsafe:
            return TargetResponse(
                response="",
                guardrail_status={"llamaguard_input": {"passed": False, "category": category}},
                metadata={"block_reason": f"LlamaGuard input: {category}", "guard": "llamaguard_input"},
            )

        # Generate response
        result = self._target.invoke(prompt)
        if result.metadata.get("error"):
            return result

        # Output check
        is_unsafe, category = self._guard.check_output(prompt, result.response)
        if is_unsafe:
            return TargetResponse(
                response="",
                guardrail_status={"llamaguard_output": {"passed": False, "category": category}},
                metadata={**result.metadata, "block_reason": f"LlamaGuard output: {category}", "guard": "llamaguard_output"},
                latency_ms=result.latency_ms,
            )

        result.metadata["guard"] = "llamaguard_both_passed"
        return result

    def reset_session(self):
        self._target.reset_session()
        self.clear_history()
