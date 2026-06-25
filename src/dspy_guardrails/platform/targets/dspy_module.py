"""DSPyModuleTarget - Adapter for DSPy modules."""

import time
from typing import Any

from .protocol import (
    TargetCapability,
    TargetResponse,
    TargetType,
    UnifiedTarget,
)


class DSPyModuleTarget(UnifiedTarget):
    """
    DSPy Module 适配器

    测试 DSPy 模块的安全性。支持单轮和多轮调用。

    Example:
        import dspy

        class MyQA(dspy.Module):
            def forward(self, question):
                return self.generate(question=question)

        target = DSPyModuleTarget(module=MyQA())
        response = target.invoke("What is 2+2?")
    """

    target_type = TargetType.DSPY_MODULE
    capabilities = [
        TargetCapability.SINGLE_TURN,
        TargetCapability.MULTI_TURN,
    ]

    def __init__(
        self,
        module: Any | None = None,
        input_field: str = "question",
        output_field: str = "answer",
        use_guardrails: bool = True,
    ):
        """
        初始化 DSPy 目标

        Args:
            module: DSPy 模块实例
            input_field: 输入字段名
            output_field: 输出字段名
            use_guardrails: 是否启用内置护栏检测
        """
        self._module = module
        self._input_field = input_field
        self._output_field = output_field
        self._use_guardrails = use_guardrails
        self._conversation: list[dict[str, str]] = []

    def invoke(self, prompt: str) -> TargetResponse:
        """执行 DSPy 模块调用"""
        start_time = time.time()

        # Check guardrails first if enabled
        guardrail_scores: dict[str, float] = {}

        if self._use_guardrails:
            guardrail_result = self._check_input_guardrails(prompt)
            guardrail_scores = guardrail_result.get("scores", {})
            if guardrail_result.get("blocked"):
                block_reason = guardrail_result.get("reason", "Blocked by guardrail")
                latency_ms = (time.time() - start_time) * 1000
                return TargetResponse(
                    response=block_reason,
                    was_blocked=True,
                    block_reason=block_reason,
                    guardrail_scores=guardrail_scores,
                    latency_ms=latency_ms,
                )

        try:
            # Call DSPy module
            if self._module:
                kwargs = {self._input_field: prompt}
                result = self._module(**kwargs)

                # Extract response
                if hasattr(result, self._output_field):
                    response = getattr(result, self._output_field)
                elif hasattr(result, "answer"):
                    response = result.answer
                elif hasattr(result, "response"):
                    response = result.response
                else:
                    response = str(result)
            else:
                # Mock response for testing
                response = f"Mock DSPy response to: {prompt[:50]}"

            latency_ms = (time.time() - start_time) * 1000

            # Check output guardrails
            if self._use_guardrails:
                output_check = self._check_output_guardrails(str(response))
                guardrail_scores.update(output_check.get("scores", {}))

            # Record conversation
            self._conversation.append({"role": "user", "content": prompt})
            self._conversation.append({"role": "assistant", "content": str(response)})

            return TargetResponse(
                response=str(response),
                was_blocked=False,
                guardrail_scores=guardrail_scores,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return TargetResponse(
                response=f"Error: {str(e)}",
                was_blocked=False,
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )

    def invoke_multi_turn(
        self,
        messages: list[dict[str, str]],
    ) -> TargetResponse:
        """多轮对话"""
        if not messages:
            return TargetResponse(response="No messages", was_blocked=False)

        # Add messages to conversation history
        for msg in messages[:-1]:
            self._conversation.append(msg)

        # Process last message
        last_message = messages[-1].get("content", "")
        return self.invoke(last_message)

    def reset_session(self) -> None:
        """重置会话"""
        self._conversation = []

    def _check_input_guardrails(self, text: str) -> dict[str, Any]:
        """Check input with guardrails."""
        try:
            from ... import guardrail

            scores = {
                "injection": guardrail.injection_score(text),
            }

            # Block if injection detected
            if not guardrail.no_injection(text):
                return {
                    "blocked": True,
                    "reason": "Injection detected in input",
                    "scores": scores,
                }

            return {"blocked": False, "scores": scores}
        except ImportError:
            # Guardrail module not available
            return {"blocked": False, "scores": {}}
        except Exception:
            return {"blocked": False, "scores": {}}

    def _check_output_guardrails(self, text: str) -> dict[str, Any]:
        """Check output with guardrails."""
        try:
            from ... import guardrail

            return {
                "scores": {
                    "toxicity": guardrail.toxicity(text),
                }
            }
        except ImportError:
            return {"scores": {}}
        except Exception:
            return {"scores": {}}

    @property
    def name(self) -> str:
        if self._module:
            return f"dspy:{self._module.__class__.__name__}"
        return "dspy:mock"

    @property
    def conversation(self) -> list[dict[str, str]]:
        return self._conversation

    @property
    def input_field(self) -> str:
        return self._input_field

    @property
    def output_field(self) -> str:
        return self._output_field
