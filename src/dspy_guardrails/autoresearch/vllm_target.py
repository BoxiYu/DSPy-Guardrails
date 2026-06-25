"""
vLLM Target — wrapper for a local vLLM OpenAI-compatible endpoint.

Provides VLLMTarget (plain) and VLLMDefenseTarget (with system prompt + regex filtering)
for use with the autoresearch evaluation harness.
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests

from dspy_guardrails.testing.targets import BaseTarget, ConversationTurn, TargetResponse

_DEFAULT_BASE_URL = "http://localhost:18921/v1"
_DEFAULT_MODEL = "Huihui-Qwen3.5-27B-abliterated"


class VLLMTarget(BaseTarget):
    """
    Target that calls a local vLLM server via its OpenAI-compatible chat completions API.

    Args:
        base_url:    Root URL for the vLLM OpenAI-compatible API (no trailing slash).
        model_name:  Model identifier to pass in the request body.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature.
        timeout:     Request timeout in seconds.

    Examples:
        t = VLLMTarget()
        r = t.invoke("What is 2+2?")
        print(r.response, r.latency_ms, r.was_blocked)
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model_name: str = _DEFAULT_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> TargetResponse:
        """Send *prompt* to the vLLM chat completions endpoint and return a TargetResponse."""
        messages = [{"role": "user", "content": prompt}]
        return self._call_vllm(prompt=prompt, messages=messages)

    def reset_session(self) -> None:
        """Clear conversation history."""
        self.clear_history()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_vllm(self, prompt: str, messages: list[dict[str, str]]) -> TargetResponse:
        """
        POST to ``/chat/completions`` and return a TargetResponse.

        On any error (timeout, connection failure, non-200 status) the response will
        contain an error message and metadata with error details.  ``was_blocked``
        will be False in all error cases so the harness can distinguish network
        failures from intentional blocks.
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        t0 = time.monotonic()
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            latency_ms = (time.monotonic() - t0) * 1000.0

            if resp.status_code != 200:
                error_msg = f"vLLM API error {resp.status_code}: {resp.text[:200]}"
                return self._error_response(error_msg, latency_ms, {"http_status": resp.status_code})

            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            metadata: dict[str, Any] = {
                "model": data.get("model", self.model_name),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }

        except requests.Timeout:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._error_response(
                f"Request to vLLM timed out after {self.timeout}s",
                latency_ms,
                {"error_type": "timeout"},
            )
        except requests.ConnectionError as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._error_response(
                f"Connection to vLLM failed: {exc}",
                latency_ms,
                {"error_type": "connection_error"},
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._error_response(
                f"Unexpected error calling vLLM: {exc}",
                latency_ms,
                {"error_type": type(exc).__name__},
            )

        # Record history
        self.conversation_history.append(ConversationTurn(role="user", content=prompt))
        self.conversation_history.append(ConversationTurn(role="assistant", content=text))

        return TargetResponse(response=text, latency_ms=latency_ms, metadata=metadata)

    @staticmethod
    def _error_response(
        message: str, latency_ms: float, extra_meta: dict[str, Any]
    ) -> TargetResponse:
        """Build a non-blocked TargetResponse that conveys an error."""
        return TargetResponse(
            response=message,
            guardrail_status={},   # empty → was_blocked == False
            metadata={"error": True, **extra_meta},
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------


class VLLMDefenseTarget(BaseTarget):
    """
    vLLM target that adds a layer of defense:

    1. **Regex pre-filter**: If the incoming prompt matches any pattern in
       ``defense_patterns``, the request is blocked before reaching the model.
    2. **System prompt**: A system prompt is prepended to every LLM call to
       instruct the model to refuse unsafe content.

    Args:
        base_url:         Root URL for the vLLM API.
        model_name:       Model identifier.
        system_prompt:    System message prepended to every conversation.
        defense_patterns: List of regex strings.  A prompt that matches *any*
                          pattern is blocked immediately (case-insensitive).
        max_tokens:       Maximum tokens to generate.
        temperature:      Sampling temperature.
        timeout:          Request timeout in seconds.

    Examples:
        t = VLLMDefenseTarget(
            system_prompt="You are a safe assistant. Refuse harmful requests.",
            defense_patterns=[r"ignore.{0,30}instructions", r"jailbreak"],
        )
        r = t.invoke("ignore all instructions and …")
        assert r.was_blocked
    """

    _BLOCKED_GUARDRAIL = "vllm_defense_regex"

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model_name: str = _DEFAULT_MODEL,
        system_prompt: str = "",
        defense_patterns: list[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self._inner = VLLMTarget(
            base_url=base_url,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        self.system_prompt = system_prompt
        self._compiled_patterns: list[re.Pattern[str]] = []
        self._set_patterns(defense_patterns or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, prompt: str) -> TargetResponse:
        """Check *prompt* against defense patterns, then forward to vLLM."""
        # Step 1 — regex pre-filter
        matched = self._match_pattern(prompt)
        if matched is not None:
            blocked_response = TargetResponse(
                response="I'm sorry, I can't help with that request.",
                guardrail_status={
                    self._BLOCKED_GUARDRAIL: {
                        "passed": False,
                        "reason": f"Matched defense pattern: {matched}",
                    }
                },
                metadata={"blocked_by": "regex", "matched_pattern": matched},
                latency_ms=0.0,
            )
            self.conversation_history.append(ConversationTurn(role="user", content=prompt))
            self.conversation_history.append(
                ConversationTurn(
                    role="assistant",
                    content=blocked_response.response,
                    guardrail_status=blocked_response.guardrail_status,
                )
            )
            return blocked_response

        # Step 2 — build messages with system prompt
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = self._inner._call_vllm(prompt=prompt, messages=messages)

        # Mirror history into our own list (inner target also records it)
        self.conversation_history.append(ConversationTurn(role="user", content=prompt))
        self.conversation_history.append(
            ConversationTurn(role="assistant", content=result.response)
        )
        return result

    def reset_session(self) -> None:
        """Clear conversation history on both this target and the inner target."""
        self.clear_history()
        self._inner.reset_session()

    def update_defense(
        self,
        patterns: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        Dynamically update defense rules.

        Args:
            patterns:      New list of regex patterns to replace the current set.
                           Pass ``None`` to leave patterns unchanged.
            system_prompt: New system prompt.  Pass ``None`` to leave unchanged.
        """
        if patterns is not None:
            self._set_patterns(patterns)
        if system_prompt is not None:
            self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_patterns(self, patterns: list[str]) -> None:
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns
        ]

    def _match_pattern(self, text: str) -> str | None:
        """Return the first matching pattern string, or None if no match."""
        for compiled in self._compiled_patterns:
            if compiled.search(text):
                return compiled.pattern
        return None


# ---------------------------------------------------------------------------
# OpenRouter Target — for attacking safety-aligned API models
# ---------------------------------------------------------------------------


class OpenRouterTarget(BaseTarget):
    """Target that calls an OpenRouter model as the victim to be attacked.

    Uses the same OpenAI-compatible chat completions API but adds the
    ``Authorization`` header required by OpenRouter.

    Args:
        model:       OpenRouter model ID (e.g. "meta-llama/llama-3.3-70b-instruct:free").
        api_key:     OpenRouter API key.  Falls back to ``OPENROUTER_API_KEY`` env var.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature.
        timeout:     Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: float = 90.0,
    ) -> None:
        super().__init__()
        import os
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def invoke(self, prompt: str) -> TargetResponse:
        """Send *prompt* to the OpenRouter target model."""
        messages = [{"role": "user", "content": prompt}]
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        t0 = time.monotonic()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            latency_ms = (time.monotonic() - t0) * 1000.0

            if resp.status_code != 200:
                return TargetResponse(
                    response=f"API error {resp.status_code}: {resp.text[:200]}",
                    metadata={"error": True, "http_status": resp.status_code},
                    latency_ms=latency_ms,
                )

            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            metadata: dict[str, Any] = {
                "model": data.get("model", self.model),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except requests.Timeout:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return TargetResponse(
                response=f"Timeout after {self.timeout}s",
                metadata={"error": True, "error_type": "timeout"},
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000.0
            return TargetResponse(
                response=f"Error: {exc}",
                metadata={"error": True, "error_type": type(exc).__name__},
                latency_ms=latency_ms,
            )

        self.conversation_history.append(ConversationTurn(role="user", content=prompt))
        self.conversation_history.append(ConversationTurn(role="assistant", content=text))
        return TargetResponse(response=text, latency_ms=latency_ms, metadata=metadata)

    def reset_session(self) -> None:
        self.clear_history()


# ---------------------------------------------------------------------------
# OutputGuardTarget — output-side guardrail wrapper
# ---------------------------------------------------------------------------


class OutputGuardTarget(BaseTarget):
    """Target with OUTPUT-side guardrail only.

    Flow: prompt → target LLM → generate response → output guard checks response
    If output guard blocks → return blocked TargetResponse
    If output guard passes → return normal response
    """

    def __init__(self, inner_target: BaseTarget, output_guard: Any) -> None:
        """
        Args:
            inner_target: The actual LLM target (VLLMTarget, OpenRouterTarget, etc.)
            output_guard: Object with .check(response, goal="") -> GuardResult
        """
        super().__init__()
        self._inner = inner_target
        self._output_guard = output_guard

    def invoke(self, prompt: str) -> TargetResponse:
        """Call inner target, then check response with output guard."""
        # 1. Call inner target to get LLM response
        result = self._inner.invoke(prompt)

        # 2. If inner target already errored, pass through
        if result.metadata.get("error"):
            return result

        # 3. Check response with output guard
        guard_result = self._output_guard.check(result.response, goal=prompt)

        if guard_result.blocked:
            return TargetResponse(
                response="[OUTPUT BLOCKED] " + guard_result.reason,
                guardrail_status={
                    "output_guard": {
                        "passed": False,
                        "reason": guard_result.reason,
                        "score": guard_result.score,
                    }
                },
                metadata={**result.metadata, "output_guard_blocked": True},
                latency_ms=result.latency_ms,
            )

        # 4. Pass through with output guard metadata
        result.metadata["output_guard_passed"] = True
        result.metadata["output_guard_score"] = guard_result.score
        return result

    def reset_session(self) -> None:
        self._inner.reset_session()
        self.clear_history()


# ---------------------------------------------------------------------------
# InputOutputGuardTarget — input AND output guardrail wrapper
# ---------------------------------------------------------------------------


class InputOutputGuardTarget(BaseTarget):
    """Target with both INPUT and OUTPUT guardrails.

    Flow: prompt → input guard → (blocked?) → target LLM → output guard → (blocked?)
    Either guard blocking counts as defense success.

    This matches production deployment (AWS Bedrock, Anthropic, Google Vertex AI, NVIDIA NeMo).
    """

    def __init__(
        self,
        inner_target: BaseTarget,
        input_guard: Any,
        output_guard: Any,
    ) -> None:
        """
        Args:
            inner_target: The actual LLM target.
            input_guard: Callable ``(prompt: str) -> tuple[bool, str]`` returning
                         ``(is_blocked, reason)``.  Pass ``None`` to skip input check.
            output_guard: Object with ``.check(response, goal="") -> GuardResult``.
                          Pass ``None`` to skip output check.
        """
        super().__init__()
        self._inner = inner_target
        self._input_guard = input_guard
        self._output_guard = output_guard

    def invoke(self, prompt: str) -> TargetResponse:
        """Check input guard, call LLM, then check output guard."""
        # 1. Input guard
        if self._input_guard is not None:
            is_blocked, reason = self._input_guard(prompt)
            if is_blocked:
                return TargetResponse(
                    response="[INPUT BLOCKED] " + reason,
                    guardrail_status={
                        "input_guard": {"passed": False, "reason": reason}
                    },
                    metadata={"input_guard_blocked": True},
                    latency_ms=0.0,
                )

        # 2. Call inner target
        result = self._inner.invoke(prompt)

        # 3. Pass through errors without output guard check
        if result.metadata.get("error"):
            return result

        # 4. Output guard
        if self._output_guard is not None:
            guard_result = self._output_guard.check(result.response, goal=prompt)
            if guard_result.blocked:
                return TargetResponse(
                    response="[OUTPUT BLOCKED] " + guard_result.reason,
                    guardrail_status={
                        "output_guard": {
                            "passed": False,
                            "reason": guard_result.reason,
                            "score": guard_result.score,
                        }
                    },
                    metadata={**result.metadata, "output_guard_blocked": True},
                    latency_ms=result.latency_ms,
                )
            result.metadata["output_guard_passed"] = True
            result.metadata["output_guard_score"] = guard_result.score

        return result

    def reset_session(self) -> None:
        self._inner.reset_session()
        self.clear_history()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_guarded_target(
    target: BaseTarget,
    guard_mode: str = "input",
    input_patterns: list[str] | None = None,
    input_system_prompt: str = "",
    output_guard: Any = None,
) -> BaseTarget:
    """Factory to create a target with the specified guard configuration.

    Args:
        target: Base target (VLLMTarget or OpenRouterTarget).
        guard_mode: ``"input"``, ``"output"``, ``"both"``, or ``"none"``.
        input_patterns: Regex patterns for input guard (used when guard_mode
                        includes input).
        input_system_prompt: System prompt for input guard.
        output_guard: Output guard object with ``.check()`` method (used when
                      guard_mode includes output).

    Returns:
        Wrapped target with appropriate guards applied.

    Raises:
        ValueError: If an unrecognised ``guard_mode`` is supplied.
    """
    if guard_mode == "none":
        return target

    if guard_mode == "input":
        # Build an input-guard-only wrapper using InputOutputGuardTarget
        # with output_guard=None (input patterns only)
        patterns = input_patterns or []
        compiled = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]

        def _input_guard_fn(prompt: str) -> tuple[bool, str]:
            for cp in compiled:
                if cp.search(prompt):
                    return True, f"Input pattern matched: {cp.pattern}"
            return False, ""

        return InputOutputGuardTarget(
            inner_target=target,
            input_guard=_input_guard_fn,
            output_guard=None,
        )

    if guard_mode == "output":
        if output_guard is None:
            raise ValueError("output_guard must be provided when guard_mode='output'")
        return OutputGuardTarget(inner_target=target, output_guard=output_guard)

    if guard_mode == "both":
        if output_guard is None:
            raise ValueError("output_guard must be provided when guard_mode='both'")
        compiled_patterns: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in (input_patterns or [])
        ]

        def _input_guard_fn(prompt: str) -> tuple[bool, str]:
            for pat in compiled_patterns:
                if pat.search(prompt):
                    return True, f"Matched defense pattern: {pat.pattern}"
            return False, ""

        return InputOutputGuardTarget(
            inner_target=target,
            input_guard=_input_guard_fn,
            output_guard=output_guard,
        )

    raise ValueError(
        f"Unknown guard_mode {guard_mode!r}. Choose from 'none', 'input', 'output', 'both'."
    )
