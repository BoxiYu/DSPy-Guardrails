"""
Declarative Constraints

Define constraints in a declarative style similar to DSPy Signatures.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConstraintTarget(Enum):
    INPUT = "input"
    OUTPUT = "output"
    BOTH = "both"


@dataclass
class Constraint:
    """
    Declarative constraint.

    Examples:
        # Simple constraint
        no_injection = Constraint.input("no_injection")
        no_toxicity = Constraint.output("no_toxicity")

        # Constraint with threshold
        factual = Constraint.output("factuality >= 0.8")
        safe = Constraint.output("toxicity < 0.3")

        # Custom check function
        custom = Constraint.output(lambda x: len(x) > 10, "too_short")
    """

    check: str | Callable
    target: ConstraintTarget
    threshold: float | None = None
    operator: str | None = None
    name: str | None = None
    message: str | None = None
    is_hard: bool = True  # Assert (hard) vs Suggest (soft)

    @classmethod
    def input(cls, check: str | Callable, message: str = None) -> "Constraint":
        """Create an input constraint."""
        return cls._parse(check, ConstraintTarget.INPUT, message)

    @classmethod
    def output(cls, check: str | Callable, message: str = None) -> "Constraint":
        """Create an output constraint."""
        return cls._parse(check, ConstraintTarget.OUTPUT, message)

    @classmethod
    def suggest(cls, check: str | Callable, message: str = None) -> "Constraint":
        """Create a soft constraint (Suggest)."""
        constraint = cls._parse(check, ConstraintTarget.OUTPUT, message)
        constraint.is_hard = False
        return constraint

    @classmethod
    def _parse(cls, check: str | Callable, target: ConstraintTarget, message: str = None) -> "Constraint":
        """Parse a constraint expression."""
        if callable(check):
            return cls(
                check=check,
                target=target,
                name=getattr(check, "__name__", "custom"),
                message=message,
            )

        # Parse expression: "factuality >= 0.8"
        pattern = r"(\w+)\s*([<>=!]+)\s*([\d.]+)"
        match = re.match(pattern, check)

        if match:
            name, operator, threshold = match.groups()
            return cls(
                check=name,
                target=target,
                threshold=float(threshold),
                operator=operator,
                name=name,
                message=message or f"{name} {operator} {threshold}",
            )

        # Simple constraint: "no_injection"
        return cls(
            check=check,
            target=target,
            name=check,
            message=message or f"Check: {check}",
        )

    def evaluate(self, value: Any, score: float) -> bool:
        """Evaluate whether the constraint is satisfied."""
        if self.threshold is None:
            return score >= 0.5  # default threshold

        if self.operator == ">=":
            return score >= self.threshold
        elif self.operator == ">":
            return score > self.threshold
        elif self.operator == "<=":
            return score <= self.threshold
        elif self.operator == "<":
            return score < self.threshold
        elif self.operator == "==":
            return abs(score - self.threshold) < 0.01
        elif self.operator == "!=":
            return abs(score - self.threshold) >= 0.01

        return score >= self.threshold


@dataclass
class ConstraintSet:
    """
    Constraint set.

    Examples:
        constraints = ConstraintSet([
            Constraint.input("no_injection"),
            Constraint.output("no_toxicity"),
            Constraint.output("factuality >= 0.8"),
        ])
    """

    constraints: list[Constraint] = field(default_factory=list)

    def input_constraints(self) -> list[Constraint]:
        """Get all input constraints."""
        return [c for c in self.constraints if c.target in (ConstraintTarget.INPUT, ConstraintTarget.BOTH)]

    def output_constraints(self) -> list[Constraint]:
        """Get all output constraints."""
        return [c for c in self.constraints if c.target in (ConstraintTarget.OUTPUT, ConstraintTarget.BOTH)]

    def hard_constraints(self) -> list[Constraint]:
        """Get all hard constraints (Assert)."""
        return [c for c in self.constraints if c.is_hard]

    def soft_constraints(self) -> list[Constraint]:
        """Get all soft constraints (Suggest)."""
        return [c for c in self.constraints if not c.is_hard]

    def add(self, constraint: Constraint) -> "ConstraintSet":
        """Add a constraint."""
        self.constraints.append(constraint)
        return self

    def __iter__(self):
        return iter(self.constraints)

    def __len__(self):
        return len(self.constraints)


# Predefined constraints
class CommonConstraints:
    """Common predefined constraints."""

    NO_INJECTION = Constraint.input("no_injection", "Input contains prompt injection")
    NO_PII_INPUT = Constraint.input("no_pii", "Input contains PII")
    NO_TOXICITY = Constraint.output("no_toxicity", "Output is toxic")
    NO_PII_OUTPUT = Constraint.output("no_pii", "Output leaks PII")
    FACTUAL = Constraint.output("factuality >= 0.7", "Output should be factual")
    RELEVANT = Constraint.output("relevance >= 0.7", "Output should be relevant")
    SAFE = Constraint.output("toxicity < 0.3", "Output should be safe")

    @classmethod
    def safety_set(cls) -> ConstraintSet:
        """Safety constraint set."""
        return ConstraintSet([
            cls.NO_INJECTION,
            cls.NO_TOXICITY,
            cls.SAFE,
        ])

    @classmethod
    def quality_set(cls) -> ConstraintSet:
        """Quality constraint set."""
        return ConstraintSet([
            cls.FACTUAL,
            cls.RELEVANT,
        ])

    @classmethod
    def full_set(cls) -> ConstraintSet:
        """Full constraint set."""
        return ConstraintSet([
            cls.NO_INJECTION,
            cls.NO_PII_INPUT,
            cls.NO_TOXICITY,
            cls.NO_PII_OUTPUT,
            cls.FACTUAL,
            cls.RELEVANT,
        ])
