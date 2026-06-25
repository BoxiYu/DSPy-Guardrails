"""
BaseAdaptiveAttack - PSSU (Propose-Score-Select-Update) unified attack interface.

Based on "The Attacker Moves Second" (arXiv:2510.09023) unified framework.
All adaptive attacks (PAIR, TAP, etc.) implement this interface.

Supports cross-model attacks via optional `attacker_lm` parameter:
when set, propose/score LLM calls use the attacker LM via dspy.context().
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Target(Protocol):
    """Protocol for any object that can be attacked."""

    def invoke(self, prompt: str) -> Any:
        """Invoke the target. Must return an object with .was_blocked and .response."""
        ...


@dataclass
class AttackAttempt:
    """A single attack attempt with its result."""

    prompt: str
    response: str
    was_blocked: bool
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveAttackResult:
    """Result of a full adaptive attack run."""

    success: bool
    best_prompt: str | None = None
    best_response: str | None = None
    best_score: float = 0.0
    iterations_used: int = 0
    total_queries: int = 0
    history: list[AttackAttempt] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Success: {self.success}, "
            f"Iterations: {self.iterations_used}, "
            f"Queries: {self.total_queries}, "
            f"Best score: {self.best_score:.2f}"
        )


class BaseAdaptiveAttack(ABC):
    """
    PSSU unified attack interface.

    Subclasses implement the four PSSU steps:
    - Propose: generate attack candidates
    - Score: evaluate candidates against the target
    - Select: pick best candidates for next iteration
    - Update: refine attack strategy based on feedback

    The main `attack()` method orchestrates the loop.

    Args:
        target: Object with .invoke(prompt) -> TargetResponse
        max_iterations: Maximum PSSU iterations
        attacker_lm: Optional DSPy LM for cross-model attacks. When set,
            propose() and score() LLM calls use this LM via dspy.context().
        judge_fn: Optional external judge function (goal, prompt, response) -> float.
            When provided, used instead of the internal judge for early-stopping
            decisions. This ensures the attack iterates until the external judge
            (e.g., StrongREJECT) confirms success, not just the internal LLM judge.
        success_threshold: Score threshold for declaring attack success (default: 0.5).
        verbose: Log progress
    """

    def __init__(
        self,
        target: Target,
        max_iterations: int = 20,
        attacker_lm: Any = None,
        judge_fn: Any = None,
        success_threshold: float = 0.5,
        verbose: bool = False,
    ):
        self.target = target
        self.max_iterations = max_iterations
        self.attacker_lm = attacker_lm
        self.judge_fn = judge_fn
        self.success_threshold = success_threshold
        self.verbose = verbose
        self._total_queries = 0

    @abstractmethod
    def propose(self, goal: str, context: dict[str, Any]) -> list[str]:
        """Generate attack candidates for the given goal."""
        ...

    @abstractmethod
    def score(self, candidates: list[str], goal: str) -> list[AttackAttempt]:
        """Test candidates against the target and score them."""
        ...

    @abstractmethod
    def select(self, attempts: list[AttackAttempt], k: int) -> list[AttackAttempt]:
        """Select the top-k attempts for the next iteration."""
        ...

    @abstractmethod
    def update(self, selected: list[AttackAttempt], goal: str, context: dict[str, Any]) -> None:
        """Update internal state based on selected attempts."""
        ...

    def _query_target(self, prompt: str) -> AttackAttempt:
        """Query the target and wrap the response."""
        self._total_queries += 1
        resp = self.target.invoke(prompt)
        return AttackAttempt(
            prompt=prompt,
            response=resp.response,
            was_blocked=resp.was_blocked,
            metadata=getattr(resp, "metadata", {}),
        )

    def attack(self, goal: str) -> AdaptiveAttackResult:
        """
        Run the full PSSU attack loop.

        Args:
            goal: The attack objective (e.g., "extract system prompt").

        Returns:
            AdaptiveAttackResult with the best attempt found.
        """
        self._total_queries = 0
        context: dict[str, Any] = {"history": []}
        all_attempts: list[AttackAttempt] = []

        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Propose
            candidates = self.propose(goal, context)

            # Score
            attempts = self.score(candidates, goal)
            all_attempts.extend(attempts)

            # Check for success
            for attempt in attempts:
                if not attempt.was_blocked:
                    # Use external judge if provided, otherwise internal score
                    if self.judge_fn is not None:
                        try:
                            judge_score = self.judge_fn(
                                goal, attempt.prompt, attempt.response
                            )
                            attempt.score = float(judge_score)
                        except Exception:
                            pass  # Keep internal score on judge failure
                    if attempt.score > self.success_threshold:
                        return AdaptiveAttackResult(
                            success=True,
                            best_prompt=attempt.prompt,
                            best_response=attempt.response,
                            best_score=attempt.score,
                            iterations_used=iteration + 1,
                            total_queries=self._total_queries,
                            history=all_attempts,
                        )

            # Select
            selected = self.select(attempts, k=min(len(attempts), 5))

            # Update
            self.update(selected, goal, context)
            context["history"].append(selected)

        # Return best attempt overall
        best = max(all_attempts, key=lambda a: a.score) if all_attempts else None
        return AdaptiveAttackResult(
            success=False,
            best_prompt=best.prompt if best else None,
            best_response=best.response if best else None,
            best_score=best.score if best else 0.0,
            iterations_used=self.max_iterations,
            total_queries=self._total_queries,
            history=all_attempts,
        )
