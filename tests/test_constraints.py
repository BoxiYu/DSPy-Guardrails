"""D1: Unit tests for constraints.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dspy_guardrails.constraints import (
    Constraint,
    ConstraintSet,
    ConstraintTarget,
    CommonConstraints,
)


class TestConstraintParse:
    """Test Constraint._parse for various expression formats."""

    def test_simple_name(self):
        c = Constraint.input("no_injection")
        assert c.check == "no_injection"
        assert c.name == "no_injection"
        assert c.target == ConstraintTarget.INPUT
        assert c.threshold is None
        assert c.operator is None

    def test_expression_gte(self):
        c = Constraint.output("factuality >= 0.8")
        assert c.name == "factuality"
        assert c.operator == ">="
        assert c.threshold == 0.8
        assert c.target == ConstraintTarget.OUTPUT

    def test_expression_gt(self):
        c = Constraint.output("quality > 0.5")
        assert c.operator == ">"
        assert c.threshold == 0.5

    def test_expression_lte(self):
        c = Constraint.output("toxicity <= 0.3")
        assert c.operator == "<="
        assert c.threshold == 0.3

    def test_expression_lt(self):
        c = Constraint.output("toxicity < 0.3")
        assert c.operator == "<"
        assert c.threshold == 0.3

    def test_expression_eq(self):
        c = Constraint.output("score == 1.0")
        assert c.operator == "=="
        assert c.threshold == 1.0

    def test_expression_neq(self):
        c = Constraint.output("score != 0.0")
        assert c.operator == "!="
        assert c.threshold == 0.0

    def test_callable_check(self):
        fn = lambda x: len(x) > 10
        c = Constraint.input(fn, "too_short")
        assert callable(c.check)
        assert c.message == "too_short"
        assert c.name == "<lambda>"


class TestConstraintEvaluate:
    """Test Constraint.evaluate with various operators."""

    def test_no_threshold_default(self):
        c = Constraint.input("no_injection")
        assert c.evaluate("text", 0.6) is True
        assert c.evaluate("text", 0.4) is False

    def test_gte(self):
        c = Constraint.output("factuality >= 0.8")
        assert c.evaluate("", 0.8) is True
        assert c.evaluate("", 0.9) is True
        assert c.evaluate("", 0.7) is False

    def test_gt(self):
        c = Constraint.output("quality > 0.5")
        assert c.evaluate("", 0.6) is True
        assert c.evaluate("", 0.5) is False

    def test_lte(self):
        c = Constraint.output("toxicity <= 0.3")
        assert c.evaluate("", 0.3) is True
        assert c.evaluate("", 0.2) is True
        assert c.evaluate("", 0.4) is False

    def test_lt(self):
        c = Constraint.output("toxicity < 0.3")
        assert c.evaluate("", 0.2) is True
        assert c.evaluate("", 0.3) is False

    def test_eq(self):
        c = Constraint.output("score == 1.0")
        assert c.evaluate("", 1.0) is True
        assert c.evaluate("", 1.005) is True  # within 0.01 tolerance
        assert c.evaluate("", 0.5) is False

    def test_neq(self):
        c = Constraint.output("score != 0.0")
        assert c.evaluate("", 0.5) is True
        assert c.evaluate("", 0.0) is False


class TestConstraintFactoryMethods:
    """Test input(), output(), suggest() factory methods."""

    def test_input_is_hard(self):
        c = Constraint.input("no_injection")
        assert c.is_hard is True
        assert c.target == ConstraintTarget.INPUT

    def test_output_is_hard(self):
        c = Constraint.output("no_toxicity")
        assert c.is_hard is True
        assert c.target == ConstraintTarget.OUTPUT

    def test_suggest_is_soft(self):
        c = Constraint.suggest("factuality >= 0.8")
        assert c.is_hard is False
        assert c.target == ConstraintTarget.OUTPUT

    def test_suggest_with_message(self):
        c = Constraint.suggest("no_toxicity", "Output may be toxic")
        assert c.message == "Output may be toxic"


class TestConstraintSet:
    """Test ConstraintSet filtering methods."""

    def test_input_constraints(self):
        cs = ConstraintSet([
            Constraint.input("no_injection"),
            Constraint.output("no_toxicity"),
            Constraint.input("no_pii"),
        ])
        inputs = cs.input_constraints()
        assert len(inputs) == 2
        assert all(c.target == ConstraintTarget.INPUT for c in inputs)

    def test_output_constraints(self):
        cs = ConstraintSet([
            Constraint.input("no_injection"),
            Constraint.output("no_toxicity"),
            Constraint.output("factuality >= 0.8"),
        ])
        outputs = cs.output_constraints()
        assert len(outputs) == 2

    def test_hard_constraints(self):
        cs = ConstraintSet([
            Constraint.input("no_injection"),
            Constraint.suggest("factuality >= 0.8"),
        ])
        hard = cs.hard_constraints()
        assert len(hard) == 1
        assert hard[0].name == "no_injection"

    def test_soft_constraints(self):
        cs = ConstraintSet([
            Constraint.input("no_injection"),
            Constraint.suggest("factuality >= 0.8"),
        ])
        soft = cs.soft_constraints()
        assert len(soft) == 1
        assert soft[0].name == "factuality"

    def test_add(self):
        cs = ConstraintSet()
        assert len(cs) == 0
        cs.add(Constraint.input("no_injection"))
        assert len(cs) == 1

    def test_iter(self):
        cs = ConstraintSet([Constraint.input("a"), Constraint.output("b")])
        names = [c.name for c in cs]
        assert names == ["a", "b"]


class TestCommonConstraints:
    """Test CommonConstraints predefined sets."""

    def test_safety_set(self):
        cs = CommonConstraints.safety_set()
        assert len(cs) == 3

    def test_quality_set(self):
        cs = CommonConstraints.quality_set()
        assert len(cs) == 2

    def test_full_set(self):
        cs = CommonConstraints.full_set()
        assert len(cs) == 6

    def test_predefined_targets(self):
        assert CommonConstraints.NO_INJECTION.target == ConstraintTarget.INPUT
        assert CommonConstraints.NO_PII_INPUT.target == ConstraintTarget.INPUT
        assert CommonConstraints.NO_TOXICITY.target == ConstraintTarget.OUTPUT
        assert CommonConstraints.NO_PII_OUTPUT.target == ConstraintTarget.OUTPUT
        assert CommonConstraints.FACTUAL.target == ConstraintTarget.OUTPUT
        assert CommonConstraints.RELEVANT.target == ConstraintTarget.OUTPUT
