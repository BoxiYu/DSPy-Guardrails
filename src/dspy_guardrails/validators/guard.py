"""
Guard - Orchestrator for running multiple validators with on_fail strategies.

The Guard class chains validators together and applies their on_fail
actions when validation fails, including reask loops with LLM.

Usage:
    guard = Guard(
        validators=[
            NoInjection(on_fail="exception"),
            NoPII(on_fail="fix"),
            NoToxicity(on_fail="reask"),
            ValidLength(min=10, on_fail="reask"),
        ]
    )

    # Simple validation
    result = guard.validate("some text")
    print(result.is_valid, result.output, result.errors)

    # Transparent LLM wrapper (like Guardrails AI)
    result = guard(
        llm=my_llm,
        messages=[{"role": "user", "content": "Write a response"}],
        max_reasks=3,
    )

    # Post-process existing output (parse mode)
    result = guard.parse("some LLM output")

    # Call history
    for entry in guard.history:
        print(entry)

    # Async with parallel validators
    async_guard = AsyncGuard(validators=[...])
    result = await async_guard.validate("text")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from dspy_guardrails.validators.base import (
    FailResult,
    OnFailAction,
    PassResult,
    ValidationError,
    Validator,
    ValidatorResult,
)

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Result of running a Guard."""

    is_valid: bool
    output: Any  # Final output (may be fixed)
    raw_output: Any  # Original LLM/input output before fixes
    errors: list[FailResult] = field(default_factory=list)
    reask_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def error_messages(self) -> list[str]:
        return [e.error_message for e in self.errors]


@dataclass
class GuardHistoryEntry:
    """A single call recorded in Guard.history."""

    input_value: Any
    output: GuardResult
    llm_calls: int = 0
    reask_count: int = 0
    duration_ms: float = 0.0
    timestamp: float = 0.0


class Guard:
    """Orchestrator that runs validators and applies on_fail strategies.

    Validators are run in order. Each validator's on_fail action determines
    what happens when it fails:

    - noop: Record error, continue with original value
    - exception: Raise ValidationError immediately
    - fix: Use validator.fix() to auto-correct, continue with fixed value
    - fix_reask: Fix, re-validate; if still fails, fall through to reask
    - reask: Re-prompt the LLM with the error message (requires llm param)
    - filter: Set field to None (for structured output)
    - refrain: Return empty result

    Supports transparent LLM wrapping:
        result = guard(llm=my_llm, messages=[...])

    And parse mode (post-process without LLM call):
        result = guard.parse("existing output")
    """

    def __init__(self, validators: list[Validator] | None = None, name: str = ""):
        self.validators = validators or []
        self.name = name
        self._history: list[GuardHistoryEntry] = []

    @property
    def history(self) -> list[GuardHistoryEntry]:
        """Call history for audit/debugging."""
        return list(self._history)

    def use(self, validator: Validator) -> Guard:
        """Add a validator. Returns self for chaining."""
        self.validators.append(validator)
        return self

    def use_many(self, *validators: Validator) -> Guard:
        """Add multiple validators."""
        self.validators.extend(validators)
        return self

    def validate(self, value: Any, **metadata: Any) -> GuardResult:
        """Validate a value through all validators.

        Applies on_fail actions but does NOT support reask (no LLM).
        For reask support, use __call__() with llm parameter.

        Args:
            value: The value to validate.
            **metadata: Extra context passed to validators.

        Returns:
            GuardResult with validation outcome.
        """
        return self._run_validators(value, original=value, metadata=metadata)

    def parse(
        self,
        llm_output: Any,
        *,
        max_reasks: int = 0,
        llm: Any = None,
        prompt: str = "",
        **metadata: Any,
    ) -> GuardResult:
        """Validate/fix an existing LLM output without making an initial LLM call.

        Similar to Guardrails AI's guard.parse(). If max_reasks > 0 and llm
        is provided, will reask on failure.

        Args:
            llm_output: The LLM output to validate.
            max_reasks: Number of reask attempts (0 = validate only).
            llm: LLM for reask (optional).
            prompt: Original prompt for reask context.
            **metadata: Extra context for validators.
        """
        start = time.monotonic()
        llm_calls = 0

        original = llm_output
        value = llm_output
        reask_count = 0

        for attempt in range(max_reasks + 1):
            result = self._run_validators(value, original=original, metadata=metadata)

            if result.is_valid or attempt == max_reasks or llm is None:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            needs_reask = any(
                isinstance(e, FailResult)
                for v, e in zip(self.validators, self._last_results, strict=False)
                if isinstance(e, FailResult)
                and v.on_fail in (OnFailAction.REASK, OnFailAction.FIX_REASK)
            )

            if not needs_reask:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            error_msgs = [e.error_message for e in result.errors]
            reask_prompt = self._build_reask_prompt(
                original_prompt=prompt,
                previous_output=str(value),
                errors=error_msgs,
            )
            value = self._call_llm(llm, prompt=reask_prompt)
            llm_calls += 1
            reask_count += 1

        result = self._run_validators(value, original=original, metadata=metadata)
        result.reask_count = reask_count
        self._record_history(original, result, llm_calls, reask_count, start)
        return result

    def __call__(
        self,
        value: Any = None,
        *,
        llm: Any = None,
        model: str | None = None,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        max_reasks: int = 1,
        **metadata: Any,
    ) -> GuardResult:
        """Validate with full on_fail strategy support including reask.

        Supports transparent LLM wrapping: if value is None and
        llm/model + messages are provided, calls the LLM first.

        Args:
            value: Value to validate (or None to call LLM first).
            llm: DSPy LM or callable for reask.
            model: Model name string — resolves to a DSPy LM automatically.
                Convenience alternative to passing llm directly.
            messages: Chat messages for initial LLM call.
            prompt: Simple prompt string (alternative to messages).
            max_reasks: Maximum number of reask attempts.
            **metadata: Extra context for validators.

        Returns:
            GuardResult with validation outcome.
        """
        start = time.monotonic()
        llm_calls = 0

        # Resolve model string to LLM
        if llm is None and model is not None:
            llm = self._resolve_model(model)

        # Initial LLM call if no value provided
        if value is None and (messages or prompt):
            value = self._call_llm(llm, messages=messages, prompt=prompt)
            llm_calls += 1

        original = value
        reask_count = 0

        for attempt in range(max_reasks + 1):
            result = self._run_validators(value, original=original, metadata=metadata)

            if result.is_valid:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            # Check if any failures need reask
            needs_reask = any(
                isinstance(e, FailResult)
                for v, e in zip(self.validators, self._last_results, strict=False)
                if isinstance(e, FailResult)
                and v.on_fail in (OnFailAction.REASK, OnFailAction.FIX_REASK)
            )

            if not needs_reask or llm is None or attempt == max_reasks:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            # Build reask prompt
            error_msgs = [e.error_message for e in result.errors]
            reask_prompt = self._build_reask_prompt(
                original_prompt=prompt or (messages[-1]["content"] if messages else ""),
                previous_output=str(value),
                errors=error_msgs,
            )

            value = self._call_llm(llm, prompt=reask_prompt)
            llm_calls += 1
            reask_count += 1

        result = self._run_validators(value, original=original, metadata=metadata)
        result.reask_count = reask_count
        self._record_history(original, result, llm_calls, reask_count, start)
        return result

    def _run_validators(
        self,
        value: Any,
        original: Any,
        metadata: dict,
    ) -> GuardResult:
        """Run all validators and apply on_fail actions."""
        errors: list[FailResult] = []
        current = value
        self._last_results: list[ValidatorResult] = []

        for validator in self.validators:
            result = validator.validate(current, **metadata)
            self._last_results.append(result)

            if isinstance(result, PassResult):
                continue

            # Validation failed
            fail: FailResult = result

            match validator.on_fail:
                case OnFailAction.EXCEPTION:
                    raise ValidationError([fail])

                case OnFailAction.FIX:
                    if fail.fix_value is not None:
                        current = fail.fix_value
                        logger.debug(
                            "Validator %s: auto-fixed value", validator.name
                        )
                    else:
                        current = validator.fix(current, **metadata)
                    # Don't add to errors if fix succeeded
                    re_result = validator.validate(current, **metadata)
                    if isinstance(re_result, FailResult):
                        errors.append(fail)

                case OnFailAction.FIX_REASK:
                    if fail.fix_value is not None:
                        current = fail.fix_value
                    else:
                        current = validator.fix(current, **metadata)
                    re_result = validator.validate(current, **metadata)
                    if isinstance(re_result, FailResult):
                        errors.append(fail)  # Will trigger reask in __call__

                case OnFailAction.REFRAIN:
                    return GuardResult(
                        is_valid=False,
                        output=None,
                        raw_output=original,
                        errors=[fail],
                    )

                case OnFailAction.FILTER:
                    current = None
                    # Continue with other validators

                case OnFailAction.REASK:
                    errors.append(fail)

                case OnFailAction.NOOP:
                    errors.append(fail)

        return GuardResult(
            is_valid=len(errors) == 0,
            output=current,
            raw_output=original,
            errors=errors,
        )

    def _record_history(
        self,
        input_value: Any,
        result: GuardResult,
        llm_calls: int,
        reask_count: int,
        start: float,
    ) -> None:
        self._history.append(
            GuardHistoryEntry(
                input_value=input_value,
                output=result,
                llm_calls=llm_calls,
                reask_count=reask_count,
                duration_ms=(time.monotonic() - start) * 1000,
                timestamp=time.time(),
            )
        )

    def _resolve_model(self, model: str) -> Any:
        """Resolve a model name string to a DSPy LM instance."""
        try:
            import dspy

            return dspy.LM(model)
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve model '{model}'. "
                "Install dspy and ensure model name is valid."
            ) from exc

    def _call_llm(
        self,
        llm: Any,
        messages: list[dict] | None = None,
        prompt: str | None = None,
    ) -> str:
        """Call LLM to generate text."""
        if llm is None:
            raise ValueError("LLM required for generation/reask but none provided")

        # DSPy LM
        if callable(llm):
            if messages:
                return str(llm(messages=messages)[0])
            if prompt:
                return str(llm(prompt)[0])

        raise TypeError(f"Unsupported LLM type: {type(llm)}")

    def _build_reask_prompt(
        self,
        original_prompt: str,
        previous_output: str,
        errors: list[str],
    ) -> str:
        """Build a prompt for reasking the LLM."""
        error_text = "\n".join(f"- {e}" for e in errors)
        return (
            f"Your previous response had the following issues:\n{error_text}\n\n"
            f"Original request: {original_prompt}\n\n"
            f"Your previous response: {previous_output}\n\n"
            "Please provide a corrected response that addresses all the issues above."
        )


# =============================================================================
# AsyncGuard - Async Guard with parallel validator execution
# =============================================================================


class AsyncGuard:
    """Async Guard that runs validators concurrently.

    Validators without ordering dependencies (noop, reask) run in parallel
    via asyncio.gather. Validators with immediate side effects (exception,
    refrain, fix, filter) still short-circuit.

    Usage:
        guard = AsyncGuard(validators=[
            NoInjection(on_fail="exception"),
            NoPII(on_fail="fix"),
            NoToxicity(on_fail="noop"),
        ])

        result = await guard.validate("some text")
        result = await guard(llm=my_llm, messages=[...])
    """

    def __init__(self, validators: list[Validator] | None = None, name: str = ""):
        self.validators = validators or []
        self.name = name
        self._history: list[GuardHistoryEntry] = []

    @property
    def history(self) -> list[GuardHistoryEntry]:
        return list(self._history)

    def use(self, validator: Validator) -> AsyncGuard:
        self.validators.append(validator)
        return self

    def use_many(self, *validators: Validator) -> AsyncGuard:
        self.validators.extend(validators)
        return self

    async def validate(self, value: Any, **metadata: Any) -> GuardResult:
        """Validate with parallel validator execution."""
        return await self._run_validators(value, original=value, metadata=metadata)

    async def parse(
        self,
        llm_output: Any,
        *,
        max_reasks: int = 0,
        llm: Any = None,
        prompt: str = "",
        **metadata: Any,
    ) -> GuardResult:
        """Async parse mode."""
        start = time.monotonic()
        llm_calls = 0
        value = llm_output
        original = llm_output
        reask_count = 0

        for attempt in range(max_reasks + 1):
            result = await self._run_validators(value, original=original, metadata=metadata)

            if result.is_valid or attempt == max_reasks or llm is None:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            needs_reask = self._needs_reask(result)
            if not needs_reask:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            error_msgs = [e.error_message for e in result.errors]
            reask_prompt = Guard._build_reask_prompt(
                None,
                original_prompt=prompt,
                previous_output=str(value),
                errors=error_msgs,
            )
            value = await asyncio.to_thread(self._call_llm_sync, llm, prompt=reask_prompt)
            llm_calls += 1
            reask_count += 1

        result = await self._run_validators(value, original=original, metadata=metadata)
        result.reask_count = reask_count
        self._record_history(original, result, llm_calls, reask_count, start)
        return result

    async def __call__(
        self,
        value: Any = None,
        *,
        llm: Any = None,
        model: str | None = None,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        max_reasks: int = 1,
        **metadata: Any,
    ) -> GuardResult:
        """Async validate with LLM wrapping and reask."""
        start = time.monotonic()
        llm_calls = 0

        if llm is None and model is not None:
            llm = Guard._resolve_model(None, model)

        if value is None and (messages or prompt):
            value = await asyncio.to_thread(
                self._call_llm_sync, llm, messages=messages, prompt=prompt
            )
            llm_calls += 1

        original = value
        reask_count = 0

        for attempt in range(max_reasks + 1):
            result = await self._run_validators(value, original=original, metadata=metadata)

            if result.is_valid:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            needs_reask = self._needs_reask(result)
            if not needs_reask or llm is None or attempt == max_reasks:
                result.reask_count = reask_count
                self._record_history(original, result, llm_calls, reask_count, start)
                return result

            error_msgs = [e.error_message for e in result.errors]
            reask_prompt = Guard._build_reask_prompt(
                None,
                original_prompt=prompt or (messages[-1]["content"] if messages else ""),
                previous_output=str(value),
                errors=error_msgs,
            )
            value = await asyncio.to_thread(
                self._call_llm_sync, llm, prompt=reask_prompt
            )
            llm_calls += 1
            reask_count += 1

        result = await self._run_validators(value, original=original, metadata=metadata)
        result.reask_count = reask_count
        self._record_history(original, result, llm_calls, reask_count, start)
        return result

    async def _run_validators(
        self,
        value: Any,
        original: Any,
        metadata: dict,
    ) -> GuardResult:
        """Run validators with parallel execution where possible.

        Strategy: run all validators concurrently, then process results
        in order to apply on_fail actions. If any validator has fix/fix_reask
        on_fail, fall back to sequential execution from that point since
        fixes modify the value for subsequent validators.
        """
        # Phase 1: Run all validators in parallel
        async def run_one(v: Validator) -> ValidatorResult:
            return await asyncio.to_thread(v.validate, value, **metadata)

        results = await asyncio.gather(
            *[run_one(v) for v in self.validators],
            return_exceptions=True,
        )

        # Phase 2: Process results in order, applying on_fail actions
        errors: list[FailResult] = []
        current = value
        need_sequential_from: int | None = None
        self._last_results: list[ValidatorResult] = []

        for i, (validator, result) in enumerate(zip(self.validators, results, strict=True)):
            if isinstance(result, BaseException):
                fail = FailResult(error_message=str(result))
                self._last_results.append(fail)
                errors.append(fail)
                continue

            self._last_results.append(result)

            if isinstance(result, PassResult):
                continue

            fail = result

            match validator.on_fail:
                case OnFailAction.EXCEPTION:
                    raise ValidationError([fail])

                case OnFailAction.REFRAIN:
                    return GuardResult(
                        is_valid=False,
                        output=None,
                        raw_output=original,
                        errors=[fail],
                    )

                case OnFailAction.FIX | OnFailAction.FIX_REASK:
                    # Fix modifies value — need to re-run remaining validators sequentially
                    if fail.fix_value is not None:
                        current = fail.fix_value
                    else:
                        current = await asyncio.to_thread(
                            validator.fix, current, **metadata
                        )
                    re_result = await asyncio.to_thread(
                        validator.validate, current, **metadata
                    )
                    if isinstance(re_result, FailResult):
                        errors.append(fail)
                    need_sequential_from = i + 1
                    break

                case OnFailAction.FILTER:
                    current = None

                case OnFailAction.REASK | OnFailAction.NOOP:
                    errors.append(fail)

        # Phase 3: If a fix happened, re-run remaining validators sequentially
        if need_sequential_from is not None:
            for j in range(need_sequential_from, len(self.validators)):
                validator = self.validators[j]
                result = await asyncio.to_thread(
                    validator.validate, current, **metadata
                )
                if j < len(self._last_results):
                    self._last_results[j] = result
                else:
                    self._last_results.append(result)

                if isinstance(result, PassResult):
                    continue

                fail = result
                match validator.on_fail:
                    case OnFailAction.EXCEPTION:
                        raise ValidationError([fail])
                    case OnFailAction.REFRAIN:
                        return GuardResult(
                            is_valid=False, output=None,
                            raw_output=original, errors=[fail],
                        )
                    case OnFailAction.FIX | OnFailAction.FIX_REASK:
                        if fail.fix_value is not None:
                            current = fail.fix_value
                        else:
                            current = await asyncio.to_thread(
                                validator.fix, current, **metadata
                            )
                        re_result = await asyncio.to_thread(
                            validator.validate, current, **metadata
                        )
                        if isinstance(re_result, FailResult):
                            errors.append(fail)
                    case OnFailAction.FILTER:
                        current = None
                    case OnFailAction.REASK | OnFailAction.NOOP:
                        errors.append(fail)

        return GuardResult(
            is_valid=len(errors) == 0,
            output=current,
            raw_output=original,
            errors=errors,
        )

    def _needs_reask(self, result: GuardResult) -> bool:
        return any(
            isinstance(r, FailResult)
            and v.on_fail in (OnFailAction.REASK, OnFailAction.FIX_REASK)
            for v, r in zip(self.validators, self._last_results, strict=False)
            if isinstance(r, FailResult)
        )

    def _record_history(
        self,
        input_value: Any,
        result: GuardResult,
        llm_calls: int,
        reask_count: int,
        start: float,
    ) -> None:
        self._history.append(
            GuardHistoryEntry(
                input_value=input_value,
                output=result,
                llm_calls=llm_calls,
                reask_count=reask_count,
                duration_ms=(time.monotonic() - start) * 1000,
                timestamp=time.time(),
            )
        )

    @staticmethod
    def _call_llm_sync(
        llm: Any,
        messages: list[dict] | None = None,
        prompt: str | None = None,
    ) -> str:
        if llm is None:
            raise ValueError("LLM required but none provided")
        if callable(llm):
            if messages:
                return str(llm(messages=messages)[0])
            if prompt:
                return str(llm(prompt)[0])
        raise TypeError(f"Unsupported LLM type: {type(llm)}")
