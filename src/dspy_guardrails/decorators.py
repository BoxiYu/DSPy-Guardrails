"""
Decorators - Declarative constraint decorators

Provides the @Guarded decorator for declarative constraints.

Usage:
    @Guarded(
        input_checks=["no_injection", "no_pii"],
        output_checks=["no_toxicity", "factuality >= 0.8"],
    )
    class SafeQA(dspy.Module):
        def forward(self, question):
            return self.generate(question=question)
"""

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

import dspy

from dspy_guardrails.constraints import Constraint, ConstraintSet
from dspy_guardrails.guardrail import guardrail


class Guarded:
    """
    Declarative constraint decorator

    Examples:
        @Guarded(
            input_checks=["no_injection"],
            output_checks=["no_toxicity"],
        )
        class SafeModule(dspy.Module):
            ...

        @Guarded(constraints=[
            Constraint.input("no_injection"),
            Constraint.output("factuality >= 0.8"),
        ])
        class AnotherModule(dspy.Module):
            ...
    """

    def __init__(
        self,
        input_checks: list[str] = None,
        output_checks: list[str] = None,
        constraints: list[Constraint] = None,
        on_violation: str = "assert",  # "assert", "suggest", "log", "ignore"
    ):
        self.input_checks = input_checks or []
        self.output_checks = output_checks or []
        self.constraints = constraints or []
        self.on_violation = on_violation

        # Build constraint set
        self._constraint_set = self._build_constraints()

    def _build_constraints(self) -> ConstraintSet:
        """Build constraint set."""
        constraints = ConstraintSet(self.constraints.copy())

        for check in self.input_checks:
            constraints.add(Constraint.input(check))

        for check in self.output_checks:
            constraints.add(Constraint.output(check))

        return constraints

    def __call__(self, cls):
        """Decorate a dspy.Module class, auto-detecting async forward."""
        original_forward = cls.forward
        constraint_set = self._constraint_set
        on_violation = self.on_violation

        if asyncio.iscoroutinefunction(original_forward):

            @wraps(original_forward)
            async def async_guarded_forward(self, *args, **kwargs):
                inputs = _extract_inputs(args, kwargs)
                for constraint in constraint_set.input_constraints():
                    _check_constraint(constraint, inputs, on_violation)

                result = await original_forward(self, *args, **kwargs)

                output = _extract_output(result)
                for constraint in constraint_set.output_constraints():
                    _check_constraint(constraint, output, on_violation)
                return result

            cls.forward = async_guarded_forward
        else:

            @wraps(original_forward)
            def guarded_forward(self, *args, **kwargs):
                inputs = _extract_inputs(args, kwargs)
                for constraint in constraint_set.input_constraints():
                    _check_constraint(constraint, inputs, on_violation)

                result = original_forward(self, *args, **kwargs)

                output = _extract_output(result)
                for constraint in constraint_set.output_constraints():
                    _check_constraint(constraint, output, on_violation)
                return result

            cls.forward = guarded_forward

        cls._guardrail_constraints = constraint_set
        return cls


def guarded(
    input_checks: list[str] = None,
    output_checks: list[str] = None,
    on_violation: str = "assert",
):
    """
    Function decorator variant

    Usage:
        @guarded(input_checks=["no_injection"])
        def my_function(text):
            return process(text)
    """
    def decorator(func):
        constraints = ConstraintSet()

        for check in (input_checks or []):
            constraints.add(Constraint.input(check))

        for check in (output_checks or []):
            constraints.add(Constraint.output(check))

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                inputs = _extract_inputs(args, kwargs)
                for constraint in constraints.input_constraints():
                    _check_constraint(constraint, inputs, on_violation)

                result = await func(*args, **kwargs)

                output = _extract_output(result)
                for constraint in constraints.output_constraints():
                    _check_constraint(constraint, output, on_violation)
                return result

            return async_wrapper
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                inputs = _extract_inputs(args, kwargs)
                for constraint in constraints.input_constraints():
                    _check_constraint(constraint, inputs, on_violation)

                result = func(*args, **kwargs)

                output = _extract_output(result)
                for constraint in constraints.output_constraints():
                    _check_constraint(constraint, output, on_violation)
                return result

            return wrapper

    return decorator


def _extract_inputs(args: tuple, kwargs: dict) -> str:
    """Extract input text."""
    # Try common input field names from kwargs
    for key in ["question", "query", "input", "text", "prompt", "message"]:
        if key in kwargs:
            return str(kwargs[key])

    # Get the first string from args
    for arg in args:
        if isinstance(arg, str):
            return arg

    # Concatenate all kwargs values
    return " ".join(str(v) for v in kwargs.values())


def _extract_output(result: Any) -> str:
    """Extract output text."""
    if isinstance(result, str):
        return result

    # DSPy Prediction
    if hasattr(result, "answer"):
        return str(result.answer)

    if hasattr(result, "response"):
        return str(result.response)

    if hasattr(result, "output"):
        return str(result.output)

    # Try the first non-reasoning field
    if hasattr(result, "keys"):
        for key in result.keys():
            if key not in ("reasoning", "rationale"):
                return str(getattr(result, key, result))

    return str(result)


def _check_constraint(constraint: Constraint, text: str, on_violation: str):
    """Execute constraint check."""
    check_name = constraint.check if isinstance(constraint.check, str) else constraint.name

    # Get check function
    check_fn = _get_check_function(check_name)
    if check_fn is None:
        import warnings
        warnings.warn(f"Unknown guardrail check: {check_name}", stacklevel=2)
        return

    # Run check
    result = check_fn(text)

    if isinstance(result, bool):
        passed = result
        score = 1.0 if result else 0.0
    else:
        score = float(result)
        passed = constraint.evaluate(text, score)

    if not passed:
        message = constraint.message or f"Constraint violated: {check_name}"

        if on_violation == "assert":
            dspy.Assert(False, message)
        elif on_violation == "suggest":
            dspy.Suggest(False, message)
        elif on_violation == "log":
            print(f"[Guardrail Warning] {message}")
        # "ignore" does nothing


def _get_check_function(name: str) -> Callable | None:
    """Get check function by name."""
    # Map check names to functions
    check_map = {
        "no_injection": guardrail.no_injection,
        "no_pii": guardrail.no_pii,
        "no_toxicity": guardrail.no_toxicity,
        "safe": guardrail.safe,
        "safe_input": guardrail.safe_input,
        "safe_output": guardrail.safe_output,
        "injection": lambda t: 1.0 - guardrail.injection_score(t),
        "pii": lambda t: 1.0 - guardrail.pii_score(t),
        "toxicity": lambda t: 1.0 - guardrail.toxicity(t),
        "factuality": guardrail.factuality,
        "relevance": guardrail.relevance,
        "quality": guardrail.quality,
    }

    return check_map.get(name)
