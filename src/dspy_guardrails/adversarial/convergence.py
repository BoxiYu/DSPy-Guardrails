"""
ConvergenceDetector - Detect Training Convergence

Monitors training progress and determines when to stop based on:
1. Attack success rate falling below threshold
2. Consecutive rounds of low ASR
3. Maximum rounds limit
"""

from dataclasses import dataclass

from .metrics import RoundStats


@dataclass
class ConvergenceStatus:
    """Current convergence status"""
    converged: bool
    reason: str
    current_asr: float
    rounds_below_threshold: int
    total_rounds: int

    def __str__(self) -> str:
        status = "CONVERGED" if self.converged else "RUNNING"
        return (
            f"[{status}] Round {self.total_rounds}: "
            f"ASR={self.current_asr:.1%}, "
            f"Consecutive below threshold: {self.rounds_below_threshold}"
        )


class ConvergenceDetector:
    """
    Detect when adversarial training has converged.

    Convergence is detected when:
    - ASR < threshold for K consecutive rounds, OR
    - Maximum rounds reached

    Also tracks trend for early stopping if ASR is not improving.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        consecutive_rounds: int = 3,
        max_rounds: int = 100,
        early_stop_patience: int = 10,
    ):
        """
        Initialize convergence detector.

        Args:
            threshold: ASR threshold for convergence (default 5%)
            consecutive_rounds: Number of consecutive rounds below threshold
            max_rounds: Maximum rounds before forced stop
            early_stop_patience: Stop if no improvement for this many rounds
        """
        self.threshold = threshold
        self.consecutive_rounds = consecutive_rounds
        self.max_rounds = max_rounds
        self.early_stop_patience = early_stop_patience

        # History
        self.history: list[RoundStats] = []
        self._converged = False
        self._convergence_round: int | None = None
        self._convergence_reason: str = ""

    def update(self, stats: RoundStats) -> ConvergenceStatus:
        """
        Update with new round statistics.

        Args:
            stats: Statistics from the completed round

        Returns:
            Current convergence status
        """
        self.history.append(stats)

        # Check convergence conditions
        status = self._check_convergence()

        return status

    def _check_convergence(self) -> ConvergenceStatus:
        """Check all convergence conditions"""
        if not self.history:
            return ConvergenceStatus(
                converged=False,
                reason="No data",
                current_asr=0.0,
                rounds_below_threshold=0,
                total_rounds=0,
            )

        current_round = len(self.history)
        current_asr = self.history[-1].asr

        # Count consecutive rounds below threshold
        consecutive_below = 0
        for stats in reversed(self.history):
            if stats.asr < self.threshold:
                consecutive_below += 1
            else:
                break

        # Condition 1: Consecutive rounds below threshold
        if consecutive_below >= self.consecutive_rounds:
            self._converged = True
            self._convergence_round = current_round
            self._convergence_reason = (
                f"ASR below {self.threshold:.1%} for {consecutive_below} consecutive rounds"
            )
            return ConvergenceStatus(
                converged=True,
                reason=self._convergence_reason,
                current_asr=current_asr,
                rounds_below_threshold=consecutive_below,
                total_rounds=current_round,
            )

        # Condition 2: Maximum rounds reached
        if current_round >= self.max_rounds:
            self._converged = True
            self._convergence_round = current_round
            self._convergence_reason = f"Maximum rounds ({self.max_rounds}) reached"
            return ConvergenceStatus(
                converged=True,
                reason=self._convergence_reason,
                current_asr=current_asr,
                rounds_below_threshold=consecutive_below,
                total_rounds=current_round,
            )

        # Condition 3: Early stopping (no improvement)
        if current_round >= self.early_stop_patience:
            recent_asrs = [s.asr for s in self.history[-self.early_stop_patience:]]
            if len(set(round(a, 3) for a in recent_asrs)) == 1:
                # ASR hasn't changed in patience rounds
                self._converged = True
                self._convergence_round = current_round
                self._convergence_reason = (
                    f"No improvement for {self.early_stop_patience} rounds (ASR stuck at {current_asr:.1%})"
                )
                return ConvergenceStatus(
                    converged=True,
                    reason=self._convergence_reason,
                    current_asr=current_asr,
                    rounds_below_threshold=consecutive_below,
                    total_rounds=current_round,
                )

        # Not converged yet
        return ConvergenceStatus(
            converged=False,
            reason="Training in progress",
            current_asr=current_asr,
            rounds_below_threshold=consecutive_below,
            total_rounds=current_round,
        )

    def is_converged(self) -> bool:
        """Check if training has converged"""
        return self._converged

    def get_convergence_round(self) -> int | None:
        """Get the round where convergence was detected"""
        return self._convergence_round

    def get_convergence_reason(self) -> str:
        """Get the reason for convergence"""
        return self._convergence_reason

    def get_asr_history(self) -> list[float]:
        """Get ASR history for visualization"""
        return [s.asr for s in self.history]

    def get_improvement_rate(self) -> float:
        """
        Calculate the rate of ASR improvement.

        Returns:
            Improvement rate (positive = improving, negative = worsening)
        """
        if len(self.history) < 2:
            return 0.0

        initial_asr = self.history[0].asr
        current_asr = self.history[-1].asr

        if initial_asr == 0:
            return 0.0

        return (initial_asr - current_asr) / initial_asr

    def get_trend(self, window: int = 5) -> str:
        """
        Get recent trend.

        Returns:
            "improving", "stable", or "worsening"
        """
        if len(self.history) < window:
            return "insufficient_data"

        recent = [s.asr for s in self.history[-window:]]
        first_half = sum(recent[:window // 2]) / (window // 2)
        second_half = sum(recent[window // 2:]) / (window - window // 2)

        diff = first_half - second_half
        if abs(diff) < 0.01:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "worsening"

    def summary(self) -> str:
        """Generate a summary of convergence status"""
        if not self.history:
            return "No training data yet"

        lines = [
            f"Rounds completed: {len(self.history)}",
            f"Converged: {self._converged}",
        ]

        if self._converged:
            lines.append(f"Convergence round: {self._convergence_round}")
            lines.append(f"Reason: {self._convergence_reason}")

        lines.extend([
            f"Initial ASR: {self.history[0].asr:.1%}",
            f"Current ASR: {self.history[-1].asr:.1%}",
            f"Improvement: {self.get_improvement_rate():.1%}",
            f"Trend: {self.get_trend()}",
        ])

        return "\n".join(lines)

    def reset(self):
        """Reset detector state"""
        self.history = []
        self._converged = False
        self._convergence_round = None
        self._convergence_reason = ""
