"""Tests for Guard advanced features: history, parse mode, model param, AsyncGuard, streaming+validators."""

import asyncio

import pytest

from dspy_guardrails.validators import (
    AsyncGuard,
    FailResult,
    Guard,
    GuardHistoryEntry,
    NoInjection,
    NoPII,
    OnFailAction,
    PassResult,
    ValidLength,
    Validator,
    ValidatorResult,
)
from dspy_guardrails.validators.base import ValidationError


# =============================================================================
# Helper: simple always-pass / always-fail validators
# =============================================================================


class AlwaysPass(Validator):
    def validate(self, value, **kw) -> ValidatorResult:
        return PassResult(value=value)


class AlwaysFail(Validator):
    def __init__(self, msg="failed", fix=None, on_fail="noop"):
        super().__init__(on_fail=on_fail)
        self._msg = msg
        self._fix = fix

    def validate(self, value, **kw) -> ValidatorResult:
        return FailResult(error_message=self._msg, fix_value=self._fix)

    def fix(self, value, **kw):
        return self._fix if self._fix is not None else value


class LengthGte10(Validator):
    """Pass if len >= 10."""

    def __init__(self, on_fail="noop"):
        super().__init__(on_fail=on_fail)

    def validate(self, value, **kw) -> ValidatorResult:
        if len(str(value)) >= 10:
            return PassResult(value=value)
        return FailResult(error_message=f"Too short: {len(str(value))}")


# =============================================================================
# Guard.history
# =============================================================================


class TestGuardHistory:

    def test_history_recorded(self):
        guard = Guard(validators=[AlwaysPass()])
        guard.validate("hello")
        guard.validate("world")
        assert len(guard.history) == 0  # validate() doesn't record history

    def test_call_records_history(self):
        guard = Guard(validators=[AlwaysPass()])
        guard("hello")
        guard("world")
        assert len(guard.history) == 2
        entry = guard.history[0]
        assert isinstance(entry, GuardHistoryEntry)
        assert entry.input_value == "hello"
        assert entry.output.is_valid
        assert entry.duration_ms >= 0
        assert entry.timestamp > 0

    def test_history_with_reask(self):
        call_count = 0

        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return ["A long enough response that passes validation easily"]

        guard = Guard(validators=[LengthGte10(on_fail="reask")])
        result = guard("short", llm=mock_llm, prompt="Write", max_reasks=2)
        assert result.is_valid
        assert len(guard.history) == 1
        assert guard.history[0].reask_count >= 1
        assert guard.history[0].llm_calls >= 1


# =============================================================================
# Guard.parse
# =============================================================================


class TestGuardParse:

    def test_parse_valid(self):
        guard = Guard(validators=[AlwaysPass()])
        result = guard.parse("hello")
        assert result.is_valid
        assert result.output == "hello"

    def test_parse_invalid(self):
        guard = Guard(validators=[AlwaysFail(on_fail="noop")])
        result = guard.parse("hello")
        assert not result.is_valid

    def test_parse_records_history(self):
        guard = Guard(validators=[AlwaysPass()])
        guard.parse("hello")
        assert len(guard.history) == 1

    def test_parse_with_reask(self):
        call_count = 0

        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            return ["A sufficiently long response"]

        guard = Guard(validators=[LengthGte10(on_fail="reask")])
        result = guard.parse("short", max_reasks=2, llm=mock_llm, prompt="Write")
        assert result.is_valid
        assert result.reask_count >= 1

    def test_parse_no_reask_by_default(self):
        guard = Guard(validators=[LengthGte10(on_fail="reask")])
        result = guard.parse("short")
        assert not result.is_valid
        assert result.reask_count == 0


# =============================================================================
# Guard with model= param
# =============================================================================


class TestGuardModelParam:

    def test_model_param_without_dspy_raises(self):
        guard = Guard(validators=[AlwaysPass()])
        # model= tries to import dspy.LM — should raise if dspy not configured
        # We just verify the param is accepted and error is reasonable
        with pytest.raises((ValueError, Exception)):
            guard(llm=None, model="nonexistent/model", prompt="hi")


# =============================================================================
# AsyncGuard
# =============================================================================


class TestAsyncGuard:

    def test_validate_all_pass(self):
        guard = AsyncGuard(validators=[AlwaysPass(), AlwaysPass()])
        result = asyncio.run(guard.validate("hello"))
        assert result.is_valid
        assert result.output == "hello"

    def test_validate_with_failure(self):
        guard = AsyncGuard(validators=[
            AlwaysPass(),
            AlwaysFail(on_fail="noop"),
        ])
        result = asyncio.run(guard.validate("hello"))
        assert not result.is_valid
        assert len(result.errors) == 1

    def test_validate_exception(self):
        guard = AsyncGuard(validators=[
            AlwaysFail(on_fail="exception"),
        ])
        with pytest.raises(ValidationError):
            asyncio.run(guard.validate("hello"))

    def test_validate_refrain(self):
        guard = AsyncGuard(validators=[
            AlwaysFail(on_fail="refrain"),
        ])
        result = asyncio.run(guard.validate("hello"))
        assert not result.is_valid
        assert result.output is None

    def test_validate_fix(self):
        guard = AsyncGuard(validators=[
            AlwaysFail(msg="bad", fix="fixed_value", on_fail="fix"),
        ])
        result = asyncio.run(guard.validate("hello"))
        # After fix, re-validate still fails (AlwaysFail), so error recorded
        assert not result.is_valid

    def test_validate_filter(self):
        guard = AsyncGuard(validators=[
            AlwaysFail(on_fail="filter"),
            AlwaysPass(),
        ])
        result = asyncio.run(guard.validate("hello"))
        assert result.output is None

    def test_parallel_execution(self):
        """Verify multiple validators run concurrently."""
        import time

        class SlowPass(Validator):
            def validate(self, value, **kw):
                time.sleep(0.05)
                return PassResult(value=value)

        guard = AsyncGuard(validators=[SlowPass(), SlowPass(), SlowPass()])
        start = time.monotonic()
        result = asyncio.run(guard.validate("hello"))
        elapsed = time.monotonic() - start
        assert result.is_valid
        # 3 validators × 50ms sequential = 150ms; parallel should be ~50ms
        assert elapsed < 0.12  # generous margin

    def test_call_with_mock_llm(self):
        def mock_llm(prompt):
            return ["A valid long response for testing purposes"]

        guard = AsyncGuard(validators=[LengthGte10(on_fail="reask")])
        result = asyncio.run(
            guard(llm=mock_llm, prompt="Write something", max_reasks=1)
        )
        assert result.is_valid

    def test_call_records_history(self):
        guard = AsyncGuard(validators=[AlwaysPass()])
        asyncio.run(guard("hello"))
        assert len(guard.history) == 1

    def test_parse(self):
        guard = AsyncGuard(validators=[AlwaysPass()])
        result = asyncio.run(guard.parse("hello"))
        assert result.is_valid

    def test_chaining(self):
        guard = AsyncGuard()
        guard.use(AlwaysPass()).use(AlwaysPass())
        assert len(guard.validators) == 2


# =============================================================================
# StreamGuardrail + Validators integration
# =============================================================================


class TestStreamValidatorIntegration:

    def test_from_validators_clean(self):
        from dspy_guardrails.streaming import StreamGuardrail

        stream_guard = StreamGuardrail.from_validators(
            validators=[AlwaysPass()],
            on_violation="block",
        )

        async def token_stream():
            for token in ["Hello", " ", "world", "."]:
                yield token

        result_tokens = []

        async def run():
            async for token in stream_guard.filter(token_stream()):
                result_tokens.append(token)

        asyncio.run(run())
        assert "".join(result_tokens) == "Hello world."
        assert stream_guard.is_clean

    def test_from_validators_violation(self):
        from dspy_guardrails.streaming import StreamGuardrail

        stream_guard = StreamGuardrail.from_validators(
            validators=[AlwaysFail(on_fail="exception")],
            on_violation="block",
        )

        async def token_stream():
            for token in ["bad", " ", "text", "."]:
                yield token

        result_tokens = []

        async def run():
            async for token in stream_guard.filter(token_stream()):
                result_tokens.append(token)

        asyncio.run(run())
        assert len(stream_guard.violations) >= 1
        assert stream_guard.violations[0].on_fail == "block"

    def test_from_validators_warn(self):
        from dspy_guardrails.streaming import StreamGuardrail

        stream_guard = StreamGuardrail.from_validators(
            validators=[AlwaysFail(on_fail="noop")],
            on_violation="warn",
        )

        async def token_stream():
            for token in ["some", " ", "text", "."]:
                yield token

        result_tokens = []

        async def run():
            async for token in stream_guard.filter(token_stream()):
                result_tokens.append(token)

        asyncio.run(run())
        # warn mode: tokens still pass through
        assert len(result_tokens) == 4
        assert len(stream_guard.violations) >= 1
        assert stream_guard.violations[0].on_fail == "warn"

    def test_from_validators_on_fail_mapping(self):
        from dspy_guardrails.streaming import _on_fail_to_stream_action
        from dspy_guardrails.validators.base import OnFailAction

        assert _on_fail_to_stream_action(OnFailAction.EXCEPTION) == "block"
        assert _on_fail_to_stream_action(OnFailAction.REFRAIN) == "block"
        assert _on_fail_to_stream_action(OnFailAction.FIX) == "warn"
        assert _on_fail_to_stream_action(OnFailAction.NOOP) == "warn"
        assert _on_fail_to_stream_action(OnFailAction.REASK) == "warn"
        assert _on_fail_to_stream_action("exception") == "block"
