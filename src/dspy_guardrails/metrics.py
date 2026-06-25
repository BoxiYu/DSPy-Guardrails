"""
Guardrail Metrics - DSPy Metric integration

Use guardrail checks as DSPy metrics for evaluation and optimization.

Usage:
    from dspy_guardrails.v2 import GuardrailMetric, combined_metric

    # Create an evaluation metric
    safety_metric = GuardrailMetric(
        checks={"toxicity": 0.3, "factuality": 0.4, "relevance": 0.3},
    )

    # Use for DSPy evaluation
    evaluate = dspy.Evaluate(devset=examples, metric=safety_metric)
    score = evaluate(my_module)

    # Or as an optimization objective
    optimizer = dspy.BootstrapFewShot(metric=safety_metric)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dspy_guardrails.guardrail import guardrail


@dataclass
class GuardrailMetric:
    """
    Guardrail as a DSPy Metric.

    Can be used directly with dspy.Evaluate or as an Optimizer metric.

    Examples:
        metric = GuardrailMetric(
            checks={"toxicity": 0.3, "factuality": 0.4, "relevance": 0.3},
        )

        # Evaluate a single example
        score = metric(example, prediction)

        # Use with dspy.Evaluate
        evaluate = dspy.Evaluate(devset=examples, metric=metric)
    """

    checks: dict[str, float] = field(default_factory=dict)  # check_name -> weight
    aggregation: str = "weighted_mean"  # "weighted_mean", "min", "product"
    threshold: float = 0.5  # pass threshold

    def __post_init__(self):
        # Normalize weights
        if self.checks:
            total = sum(self.checks.values())
            if total > 0:
                self.checks = {k: v / total for k, v in self.checks.items()}

    def __call__(self, example: Any, prediction: Any, trace: Any = None) -> float:
        """
        Compute the metric score.

        Args:
            example: DSPy Example (contains inputs).
            prediction: Model prediction result.
            trace: Optional execution trace.

        Returns:
            float: Score between 0 and 1.
        """
        # Extract prediction text
        pred_text = self._extract_text(prediction)

        # Get query (if available)
        query = ""
        if hasattr(example, "question"):
            query = example.question
        elif hasattr(example, "query"):
            query = example.query
        elif hasattr(example, "input"):
            query = example.input

        # Compute scores for each check
        scores = {}
        for check_name in self.checks:
            scores[check_name] = self._run_check(check_name, pred_text, query)

        # Aggregate scores
        return self._aggregate(scores)

    def _extract_text(self, prediction: Any) -> str:
        """Extract text from prediction result."""
        if isinstance(prediction, str):
            return prediction

        # DSPy Prediction
        for attr in ["answer", "response", "output", "text"]:
            if hasattr(prediction, attr):
                return str(getattr(prediction, attr))

        return str(prediction)

    def _run_check(self, check_name: str, text: str, query: str = "") -> float:
        """Run a single check."""
        check_map = {
            "toxicity": lambda t: 1.0 - guardrail.toxicity(t),
            "no_toxicity": lambda t: float(guardrail.no_toxicity(t)),
            "injection": lambda t: 1.0 - guardrail.injection_score(t),
            "no_injection": lambda t: float(guardrail.no_injection(t)),
            "pii": lambda t: 1.0 - guardrail.pii_score(t),
            "no_pii": lambda t: float(guardrail.no_pii(t)),
            "factuality": guardrail.factuality,
            "relevance": lambda t: guardrail.relevance(t, query),
            "quality": guardrail.quality,
            "safe": lambda t: float(guardrail.safe(t)),
            "safe_input": lambda t: float(guardrail.safe_input(t)),
            "safe_output": lambda t: float(guardrail.safe_output(t)),
        }

        check_fn = check_map.get(check_name)
        if check_fn:
            return check_fn(text)

        return 0.5  # unknown check returns neutral score

    def _aggregate(self, scores: dict[str, float]) -> float:
        """Aggregate scores."""
        if not scores:
            return 0.5

        if self.aggregation == "weighted_mean":
            total = 0.0
            for name, score in scores.items():
                weight = self.checks.get(name, 1.0 / len(scores))
                total += score * weight
            return total

        elif self.aggregation == "min":
            return min(scores.values())

        elif self.aggregation == "product":
            result = 1.0
            for score in scores.values():
                result *= score
            return result

        return sum(scores.values()) / len(scores)

    def as_reward_fn(self) -> Callable:
        """
        Convert to a reward function compatible with dspy.Refine.

        Returns:
            Callable: A function accepting (args, pred).
        """
        def reward_fn(args: dict, pred: Any) -> float:
            # Create a simple example object
            class SimpleExample:
                pass

            example = SimpleExample()
            for k, v in args.items():
                setattr(example, k, v)

            return self(example, pred)

        return reward_fn


def combined_metric(
    checks: list[str],
    weights: list[float] = None,
    aggregation: str = "weighted_mean",
) -> GuardrailMetric:
    """
    Quickly create a combined metric.

    Args:
        checks: List of check names.
        weights: List of weights (optional, defaults to equal weights).
        aggregation: Aggregation method.

    Returns:
        GuardrailMetric

    Examples:
        metric = combined_metric(["toxicity", "factuality", "relevance"])
        metric = combined_metric(
            ["toxicity", "factuality"],
            weights=[0.6, 0.4]
        )
    """
    if weights is None:
        weights = [1.0 / len(checks)] * len(checks)

    check_dict = dict(zip(checks, weights, strict=False))

    return GuardrailMetric(
        checks=check_dict,
        aggregation=aggregation,
    )


class SafetyMetric(GuardrailMetric):
    """Predefined safety metric."""

    def __init__(self):
        super().__init__(
            checks={
                "no_injection": 0.3,
                "no_toxicity": 0.4,
                "no_pii": 0.3,
            },
            aggregation="min",  # safety checks use min aggregation
        )


class QualityMetric(GuardrailMetric):
    """Predefined quality metric."""

    def __init__(self):
        super().__init__(
            checks={
                "factuality": 0.4,
                "relevance": 0.3,
                "quality": 0.3,
            },
            aggregation="weighted_mean",
        )


class ComprehensiveMetric(GuardrailMetric):
    """Comprehensive metric (safety + quality)."""

    def __init__(self, safety_weight: float = 0.5):
        quality_weight = 1.0 - safety_weight
        super().__init__(
            checks={
                # Safety checks
                "no_injection": 0.15 * safety_weight,
                "no_toxicity": 0.2 * safety_weight,
                "no_pii": 0.15 * safety_weight,
                # Quality checks
                "factuality": 0.2 * quality_weight,
                "relevance": 0.15 * quality_weight,
                "quality": 0.15 * quality_weight,
            },
            aggregation="weighted_mean",
        )
