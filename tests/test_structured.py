"""Tests for structured output validation."""

import pytest

from dspy_guardrails.validators import NoPII, NoToxicity, ValidLength, ValidRange
from dspy_guardrails.validators.structured import (
    GuardedModel,
    get_field_validators,
    validated_field,
)


class TestValidatedField:

    def test_field_with_validators(self):
        class MyModel(GuardedModel):
            name: str = validated_field(
                ValidLength(min=3, max=50),
                description="User name",
            )

        validators = get_field_validators(MyModel, "name")
        assert len(validators) == 1
        assert isinstance(validators[0], ValidLength)

    def test_field_without_validators(self):
        class MyModel(GuardedModel):
            age: int = 0

        validators = get_field_validators(MyModel, "age")
        assert len(validators) == 0


class TestGuardedModel:

    def test_all_valid(self):
        class Response(GuardedModel):
            text: str = validated_field(
                ValidLength(min=1, max=500),
                description="Response text",
            )

        result = Response.guard_validate(text="Hello world")
        assert result.is_valid
        assert result.data["text"] == "Hello world"

    def test_fix_applied(self):
        class Response(GuardedModel):
            text: str = validated_field(
                NoPII(on_fail="fix"),
                description="Response text",
            )

        result = Response.guard_validate(text="Email: test@example.com")
        assert "[EMAIL]" in result.data["text"]

    def test_validation_failure(self):
        class Response(GuardedModel):
            text: str = validated_field(
                ValidLength(min=50, on_fail="noop"),
                description="Must be long",
            )

        result = Response.guard_validate(text="short")
        assert not result.is_valid
        assert "text" in result.errors

    def test_multiple_fields(self):
        class Answer(GuardedModel):
            text: str = validated_field(
                NoPII(on_fail="fix"),
                description="Answer",
            )
            confidence: float = validated_field(
                ValidRange(min=0, max=1, on_fail="fix"),
                description="Confidence",
            )

        result = Answer.guard_validate(
            text="Contact: test@example.com",
            confidence=1.5,
        )
        assert "[EMAIL]" in result.data["text"]
        assert result.data["confidence"] == 1.0

    def test_field_results(self):
        class Response(GuardedModel):
            text: str = validated_field(
                ValidLength(min=100, on_fail="noop"),
                description="Text",
            )

        result = Response.guard_validate(text="short")
        assert "text" in result.field_results
        assert not result.field_results["text"].is_valid
