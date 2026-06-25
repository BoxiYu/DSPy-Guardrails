"""
Structured Output Validation - Pydantic 集成

Provides field-level validation for Pydantic models, integrating with
DSPy TypedPredictor for structured output with guardrails.

Usage:
    from pydantic import BaseModel, Field
    from dspy_guardrails.validators.structured import validated_field, GuardedModel

    class Answer(GuardedModel):
        text: str = validated_field(NoToxicity(on_fail="reask"), description="Answer text")
        sources: list[str] = validated_field(ValidLength(min=1), description="Sources")

    # Validate an instance
    result = Answer.guard_validate(text="bad text", sources=["src1"])

    # With DSPy
    from dspy_guardrails.validators.structured import GuardedPredictor

    predictor = GuardedPredictor(
        signature="question -> answer: str, confidence: float",
        validators={"answer": [NoToxicity(), NoPII(on_fail="fix")]},
    )
    result = predictor(question="What is the capital of France?")
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from dspy_guardrails.validators.base import (
    FailResult,
    Validator,
)
from dspy_guardrails.validators.guard import Guard

logger = logging.getLogger(__name__)


# =============================================================================
# Field-level validator metadata
# =============================================================================

_FIELD_VALIDATORS_KEY = "__guardrail_validators__"


def validated_field(
    *validators: Validator,
    description: str = "",
    **field_kwargs: Any,
) -> Any:
    """Create a Pydantic field with attached guardrail validators.

    Usage:
        class MyModel(GuardedModel):
            name: str = validated_field(
                ValidLength(min=3, max=50, on_fail="reask"),
                description="User name",
            )
    """
    from pydantic import Field

    json_extra = field_kwargs.pop("json_schema_extra", {}) or {}
    json_extra[_FIELD_VALIDATORS_KEY] = list(validators)

    return Field(
        description=description,
        json_schema_extra=json_extra,
        **field_kwargs,
    )


def get_field_validators(model_cls: type[BaseModel], field_name: str) -> list[Validator]:
    """Get validators attached to a field."""
    field_info = model_cls.model_fields.get(field_name)
    if field_info is None:
        return []
    extra = field_info.json_schema_extra or {}
    return extra.get(_FIELD_VALIDATORS_KEY, [])


# =============================================================================
# GuardedModel - Pydantic model with field-level guardrails
# =============================================================================


class FieldValidationResult:
    """Validation result for a single field."""

    def __init__(self, field: str, is_valid: bool, value: Any, errors: list[FailResult]):
        self.field = field
        self.is_valid = is_valid
        self.value = value
        self.errors = errors


class ModelValidationResult:
    """Validation result for a full model."""

    def __init__(
        self,
        is_valid: bool,
        data: dict[str, Any],
        field_results: dict[str, FieldValidationResult],
    ):
        self.is_valid = is_valid
        self.data = data
        self.field_results = field_results

    @property
    def errors(self) -> dict[str, list[str]]:
        return {
            name: [e.error_message for e in r.errors]
            for name, r in self.field_results.items()
            if not r.is_valid
        }


class GuardedModel(BaseModel):
    """Pydantic BaseModel with field-level guardrail validation.

    Subclass this and use validated_field() to attach validators.

    Usage:
        class SafeResponse(GuardedModel):
            answer: str = validated_field(
                NoToxicity(on_fail="fix"),
                NoPII(on_fail="fix"),
                description="The answer",
            )
            confidence: float = validated_field(
                ValidRange(min=0, max=1, on_fail="fix"),
                description="Confidence score",
            )

        # Validate
        result = SafeResponse.guard_validate(answer="...", confidence=0.9)
        if result.is_valid:
            response = SafeResponse(**result.data)
    """

    @classmethod
    def guard_validate(cls, **field_values: Any) -> ModelValidationResult:
        """Validate field values against their attached validators.

        Applies on_fail actions (fix, filter, etc.) but not reask.

        Args:
            **field_values: Field name -> value pairs.

        Returns:
            ModelValidationResult with per-field results.
        """
        results: dict[str, FieldValidationResult] = {}
        fixed_data: dict[str, Any] = {}
        all_valid = True

        for name, value in field_values.items():
            validators = get_field_validators(cls, name)
            if not validators:
                fixed_data[name] = value
                results[name] = FieldValidationResult(name, True, value, [])
                continue

            guard = Guard(validators=validators, name=f"{cls.__name__}.{name}")
            guard_result = guard.validate(value)

            results[name] = FieldValidationResult(
                field=name,
                is_valid=guard_result.is_valid,
                value=guard_result.output,
                errors=guard_result.errors,
            )
            fixed_data[name] = guard_result.output

            if not guard_result.is_valid:
                all_valid = False

        return ModelValidationResult(
            is_valid=all_valid,
            data=fixed_data,
            field_results=results,
        )


# =============================================================================
# GuardedPredictor - DSPy integration for structured output
# =============================================================================


class GuardedPredictor:
    """DSPy predictor wrapper with per-field validators and reask support.

    Wraps a DSPy Predict/ChainOfThought call with field-level validation.
    On failure, can reask the LLM with error feedback.

    Usage:
        import dspy

        predictor = GuardedPredictor(
            signature="question -> answer: str",
            validators={"answer": [NoToxicity(on_fail="reask"), NoPII(on_fail="fix")]},
            max_reasks=2,
        )

        result = predictor(question="What is Python?")
        print(result.answer)
    """

    def __init__(
        self,
        signature: str | Any,
        validators: dict[str, list[Validator]] | None = None,
        max_reasks: int = 1,
        use_cot: bool = False,
    ):
        import dspy

        self.validators = validators or {}
        self.max_reasks = max_reasks
        self.signature = signature

        if use_cot:
            self._predictor = dspy.ChainOfThought(signature)
        else:
            self._predictor = dspy.Predict(signature)

    def __call__(self, **kwargs: Any) -> Any:
        """Run prediction with validation and reask loop."""
        for attempt in range(self.max_reasks + 1):
            prediction = self._predictor(**kwargs)
            errors = self._validate_prediction(prediction)

            if not errors:
                return prediction

            if attempt == self.max_reasks:
                # Return with fixes applied
                return self._apply_fixes(prediction, errors)

            # Add error feedback for reask
            error_text = "; ".join(
                f"{field}: {err.error_message}" for field, err in errors
            )
            # Augment the input with error feedback
            kwargs["_validation_feedback"] = (
                f"Your previous output had issues: {error_text}. "
                "Please regenerate with corrections."
            )

        return prediction

    def _validate_prediction(self, prediction: Any) -> list[tuple[str, FailResult]]:
        """Validate all fields in a prediction."""
        errors = []
        for field_name, field_validators in self.validators.items():
            value = getattr(prediction, field_name, None)
            if value is None:
                continue

            for validator in field_validators:
                result = validator.validate(value)
                if isinstance(result, FailResult):
                    errors.append((field_name, result))
                    break  # One error per field

        return errors

    def _apply_fixes(self, prediction: Any, errors: list[tuple[str, FailResult]]) -> Any:
        """Apply fixes from FailResults to prediction fields."""
        for field_name, fail in errors:
            if fail.fix_value is not None:
                try:
                    setattr(prediction, field_name, fail.fix_value)
                except (AttributeError, TypeError):
                    pass
        return prediction
