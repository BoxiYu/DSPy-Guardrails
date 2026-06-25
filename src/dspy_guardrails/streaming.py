"""
Streaming Guardrail - Incremental content detection for streams

Performs incremental safety checks on token streams, supporting AsyncGenerator I/O.

Usage:
    from dspy_guardrails.streaming import StreamGuardrail

    guard = StreamGuardrail(checks=["no_injection", "no_toxicity"])

    # Wrap an async token stream
    async for token in guard.filter(token_stream):
        print(token, end="", flush=True)

    # Check for violations
    if guard.violations:
        print(f"Blocked: {guard.violations}")
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from dspy_guardrails.guardrail import guardrail


@dataclass
class StreamViolation:
    """A violation detected during streaming."""

    check: str
    text: str
    position: int  # character offset in accumulated text
    on_fail: str = "block"  # action from validator's on_fail, or stream-level default


@dataclass
class StreamGuardrailConfig:
    """Configuration for stream guardrail behavior."""

    checks: list[str] = field(default_factory=lambda: ["no_injection", "no_toxicity"])
    buffer_size: int = 5
    sentence_delimiters: str = r"[.!?。！？\n]"
    on_violation: str = "block"  # "block", "warn", "pass"
    check_final: bool = True


class StreamGuardrail:
    """Incremental guardrail for token streams.

    Accumulates tokens into sentence-level chunks and runs guardrail checks
    on each chunk. Supports both blocking and warning modes.

    Args:
        checks: Guardrail check names to apply.
        buffer_size: Minimum tokens before attempting a check.
        on_violation: Action on violation - "block" stops the stream,
            "warn" emits but records, "pass" records only.
        check_final: Whether to check the final buffer on stream end.
    """

    def __init__(
        self,
        checks: list[str] | None = None,
        buffer_size: int = 5,
        on_violation: str = "block",
        check_final: bool = True,
    ):
        self.config = StreamGuardrailConfig(
            checks=checks or ["no_injection", "no_toxicity"],
            buffer_size=buffer_size,
            on_violation=on_violation,
            check_final=check_final,
        )
        self._buffer: list[str] = []
        self._accumulated: str = ""
        self._checked_up_to: int = 0
        self._violations: list[StreamViolation] = []
        self._sentence_re = re.compile(self.config.sentence_delimiters)
        self._check_fns = self._resolve_checks(self.config.checks)

    @property
    def violations(self) -> list[StreamViolation]:
        return list(self._violations)

    @property
    def is_clean(self) -> bool:
        return len(self._violations) == 0

    def reset(self) -> None:
        """Reset state for reuse."""
        self._buffer.clear()
        self._accumulated = ""
        self._checked_up_to = 0
        self._violations.clear()

    def _resolve_checks(self, names: list[str]) -> dict[str, Any]:
        check_map = {
            "no_injection": guardrail.no_injection,
            "no_pii": guardrail.no_pii,
            "no_toxicity": guardrail.no_toxicity,
            "safe": guardrail.safe,
            "safe_input": guardrail.safe_input,
            "safe_output": guardrail.safe_output,
            "no_mcp_attack": guardrail.no_mcp_attack,
        }
        resolved = {}
        for name in names:
            fn = check_map.get(name)
            if fn is not None:
                resolved[name] = fn
        return resolved

    def _check_text(self, text: str | None = None) -> StreamViolation | None:
        """Run all checks on accumulated text, return first violation or None.

        Checks the full accumulated text to avoid splitting patterns across
        sentence boundaries (e.g., email addresses containing dots).
        Dispatches to validator-based checking if from_validators() was used.
        """
        if getattr(self, "_use_validators", False):
            return self._check_text_with_validators(text)

        check_text = text if text is not None else self._accumulated
        if not check_text.strip():
            return None
        for name, fn in self._check_fns.items():
            if not fn(check_text):
                return StreamViolation(
                    check=name,
                    text=check_text,
                    position=self._checked_up_to,
                )
        return None

    async def _check_text_async(self, text: str | None = None) -> StreamViolation | None:
        return await asyncio.to_thread(self._check_text, text)

    def _try_extract_chunk(self) -> str | None:
        """Extract the unchecked portion if a sentence boundary has been reached.

        Returns the unchecked text up to the last sentence boundary,
        or None if no boundary found yet. Checks the full chunk rather
        than splitting into individual sentences to avoid breaking patterns
        that span a delimiter (e.g. emails containing dots).
        """
        unchecked = self._accumulated[self._checked_up_to:]
        if not unchecked:
            return None

        # Find the last sentence boundary
        last_match = None
        for m in self._sentence_re.finditer(unchecked):
            last_match = m

        if last_match is None:
            return None

        # Return everything up to and including the last boundary
        end = last_match.end()
        chunk = unchecked[:end]
        self._checked_up_to += end
        return chunk

    async def filter(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Filter an async token stream through guardrail checks.

        Yields tokens that pass checks. On violation with on_violation="block",
        stops yielding. With "warn" or "pass", continues yielding.

        Args:
            stream: Async iterator of string tokens.

        Yields:
            Tokens that pass guardrail checks.
        """
        self.reset()
        blocked = False

        async for token in stream:
            if blocked:
                break

            self._buffer.append(token)
            self._accumulated += token

            # Check when we have enough tokens or hit sentence boundary
            if len(self._buffer) >= self.config.buffer_size or self._sentence_re.search(token):
                violation = await self._check_text_async()
                if violation:
                    self._violations.append(violation)
                    if self.config.on_violation == "block":
                        blocked = True
                self._buffer.clear()

            if not blocked:
                yield token

        # Final check on full accumulated text
        if not blocked and self.config.check_final and self._accumulated.strip():
            if not self._violations:
                violation = await self._check_text_async()
                if violation:
                    self._violations.append(violation)

    async def check_stream(self, stream: AsyncIterator[str]) -> tuple[str, list[StreamViolation]]:
        """Consume an entire stream and return accumulated text + violations.

        Unlike filter(), this collects all text and checks it, returning
        the full text and any violations found.

        Returns:
            Tuple of (full_text, violations).
        """
        self.reset()
        tokens = []
        async for token in stream:
            tokens.append(token)
            self._accumulated += token

            if self._sentence_re.search(token):
                violation = await self._check_text_async()
                if violation:
                    self._violations.append(violation)

        # Final check
        if not self._violations and self._accumulated.strip():
            violation = await self._check_text_async()
            if violation:
                self._violations.append(violation)

        return self._accumulated, list(self._violations)

    # =========================================================================
    # Validator integration
    # =========================================================================

    @classmethod
    def from_validators(
        cls,
        validators: list[Any],
        buffer_size: int = 5,
        on_violation: str = "block",
        check_final: bool = True,
    ) -> StreamGuardrail:
        """Create a StreamGuardrail from Validator instances.

        Maps each validator's on_fail action to streaming behavior:
        - exception/refrain → block the stream
        - fix → apply fix_value and warn
        - noop → warn (record but continue)
        - reask → warn (cannot reask in streaming)

        Args:
            validators: List of Validator instances.
            buffer_size: Token buffer before checking.
            on_violation: Default violation action.
            check_final: Check final buffer.

        Returns:
            Configured StreamGuardrail.
        """
        instance = cls(
            checks=[],
            buffer_size=buffer_size,
            on_violation=on_violation,
            check_final=check_final,
        )
        # Replace check functions with validator-based checks
        instance._validators = list(validators)
        instance._check_fns = {}  # Clear default check fns
        instance._use_validators = True
        return instance

    def _check_text_with_validators(self, text: str | None = None) -> StreamViolation | None:
        """Run validator instances on text, return first violation."""
        from dspy_guardrails.validators.base import FailResult as VFailResult

        check_text = text if text is not None else self._accumulated
        if not check_text.strip():
            return None

        for validator in self._validators:
            result = validator.validate(check_text)
            if isinstance(result, VFailResult):
                # Map on_fail to streaming action
                action = _on_fail_to_stream_action(validator.on_fail)
                return StreamViolation(
                    check=validator.name,
                    text=check_text,
                    position=self._checked_up_to,
                    on_fail=action,
                )
        return None

    def _orig_check_text(self, text: str | None = None) -> StreamViolation | None:
        """Original check using guardrail functions."""
        check_text = text if text is not None else self._accumulated
        if not check_text.strip():
            return None
        for name, fn in self._check_fns.items():
            if not fn(check_text):
                return StreamViolation(
                    check=name,
                    text=check_text,
                    position=self._checked_up_to,
                )
        return None


def _on_fail_to_stream_action(on_fail: Any) -> str:
    """Map validator on_fail to streaming action string."""
    from dspy_guardrails.validators.base import OnFailAction

    if isinstance(on_fail, str):
        try:
            on_fail = OnFailAction(on_fail)
        except ValueError:
            return "warn"

    mapping = {
        OnFailAction.EXCEPTION: "block",
        OnFailAction.REFRAIN: "block",
        OnFailAction.FIX: "warn",
        OnFailAction.FIX_REASK: "warn",
        OnFailAction.REASK: "warn",
        OnFailAction.NOOP: "warn",
        OnFailAction.FILTER: "block",
    }
    return mapping.get(on_fail, "warn")
