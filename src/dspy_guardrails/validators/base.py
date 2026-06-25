"""
Validator base classes and on_fail action types.

Provides the Validator abstract interface and result types that all
validators must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OnFailAction(str, Enum):
    """Action to take when a validator fails.

    NOOP     — Record failure, do nothing.
    EXCEPTION — Raise a ValidationError.
    REASK    — Re-ask the LLM to regenerate with the error message.
    FIX      — Auto-fix using the validator's fix() method.
    FIX_REASK — Fix first, then re-validate; reask if still failing.
    FILTER   — Remove the failing field from structured output.
    REFRAIN  — Return None / empty instead of the failing output.
    """

    NOOP = "noop"
    EXCEPTION = "exception"
    REASK = "reask"
    FIX = "fix"
    FIX_REASK = "fix_reask"
    FILTER = "filter"
    REFRAIN = "refrain"


@dataclass
class PassResult:
    """Returned when validation succeeds."""

    value: Any = None


@dataclass
class FailResult:
    """Returned when validation fails."""

    error_message: str
    fix_value: Any = None  # Proposed fix (used by FIX/FIX_REASK)
    metadata: dict = field(default_factory=dict)


ValidatorResult = PassResult | FailResult


class ValidationError(Exception):
    """Raised when on_fail=EXCEPTION and validation fails."""

    def __init__(self, errors: list[FailResult]):
        self.errors = errors
        messages = [e.error_message for e in errors]
        super().__init__(f"Validation failed: {'; '.join(messages)}")


class Validator(ABC):
    """Abstract base class for all validators.

    Subclasses must implement validate(). Optionally override fix()
    to support on_fail="fix".

    Usage:
        class MyValidator(Validator):
            def validate(self, value, **kwargs):
                if is_ok(value):
                    return PassResult()
                return FailResult("Not OK", fix_value=clean(value))

            def fix(self, value, **kwargs):
                return clean(value)

        v = MyValidator(on_fail="fix")
        result = v.validate("some text")
    """

    # Registry of all validator subclasses by name
    _registry: dict[str, type["Validator"]] = {}

    def __init__(self, on_fail: str | OnFailAction = OnFailAction.NOOP):
        if isinstance(on_fail, str):
            on_fail = OnFailAction(on_fail)
        self.on_fail = on_fail

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            Validator._registry[cls.__name__] = cls

    @abstractmethod
    def validate(self, value: Any, **kwargs: Any) -> ValidatorResult:
        """Validate a value. Return PassResult or FailResult."""
        ...

    def fix(self, value: Any, **kwargs: Any) -> Any:
        """Auto-fix a failing value. Override in subclass for FIX support.

        Default implementation returns value unchanged.
        """
        return value

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def get_validator(cls, name: str) -> type["Validator"] | None:
        """Look up a validator class by name."""
        return cls._registry.get(name)

    @classmethod
    def list_validators(cls) -> list[str]:
        """List all registered validator names."""
        return sorted(cls._registry.keys())
