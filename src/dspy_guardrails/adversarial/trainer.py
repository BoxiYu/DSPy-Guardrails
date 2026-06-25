"""
AdversarialTrainer - Main Training Loop

Orchestrates the adversarial training process:
1. Load initial attacks
2. Execute attacks against target
3. Evolve defenses based on successful attacks
4. Evolve attacks based on results
5. Repeat until convergence
"""

import json
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..optimizer import Example as OptimizerExample
from ..optimizer import GuardrailOptimizer
from ..redteam.payloads import (
    BypassPayloads,
    InjectionPayloads,
    JailbreakPayloads,
    MCPPayloads,
)
from ..testing.targets import TargetResponse
from ..shield import Shield
from .attack_evolver import AttackEvolver, EvolvedAttack
from .convergence import ConvergenceDetector
from .defense_evolver import DefenseEvolver
from .evolvable_target import EvolvableShieldTarget
from .metrics import (
    AdversarialConfig,
    AttackResult,
    DefenseUpdate,
    RoundStats,
    TrainingResult,
)

# =============================================================================
# Evolvable Target Protocol
# =============================================================================

class EvolvableTarget(Protocol):
    """Protocol for targets that support dynamic defense updates"""

    def invoke(self, prompt: str) -> TargetResponse:
        """Invoke the target with a prompt"""
        ...

    def update_defense(self, update: DefenseUpdate) -> None:
        """Update defense rules dynamically"""
        ...

    def reset_session(self) -> None:
        """Reset conversation session"""
        ...

    def get_defense_stats(self) -> dict[str, Any]:
        """Get current defense statistics"""
        ...


# =============================================================================
# Main Trainer
# =============================================================================

class AdversarialTrainer:
    """
    Adversarial training orchestrator.

    Implements the closed-loop training process:
    1. Attack → 2. Evaluate → 3. Evolve Defense → 4. Evolve Attack → Repeat
    """

    PAYLOAD_PROVIDERS = {
        "injection": InjectionPayloads,
        "jailbreak": JailbreakPayloads,
        "bypass": BypassPayloads,
        "mcp": MCPPayloads,
    }

    def __init__(
        self,
        target: EvolvableTarget,
        config: AdversarialConfig | None = None,
        attacker_lm: Any | None = None,
        round_callback: Any | None = None,
    ):
        """
        Initialize adversarial trainer.

        Args:
            target: Target that supports dynamic defense updates
            config: Training configuration
            round_callback: Optional callback called after each round with
                (round_num, stats, trainer) args. Useful for validation evaluation.
        """
        self.target = target
        self.config = config or AdversarialConfig()
        self.attacker_lm = attacker_lm
        self.round_callback = round_callback

        # Initialize components
        self.defense_evolver = DefenseEvolver(
            complexity_threshold=self.config.complexity_threshold,
            max_patterns=self.config.max_patterns,
            max_examples=self.config.max_examples,
            force_pattern_extraction=self.config.defense_force_pattern_extraction,
            llm_example_mode=self.config.defense_example_mode,
        )
        # Detect v2 mode from target guardrail
        self._target_v2 = getattr(getattr(target, "guardrail", None), "use_v2", False)

        self.attack_evolver = AttackEvolver(
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            max_mutations_per_attack=self.config.max_mutations_per_attack,
            use_advanced_mutations=self.config.attack_use_advanced_mutations,
            advanced_mutation_weight=self.config.attack_advanced_mutation_weight,
            use_llm_bypass=self.config.attack_use_llm_bypass,
            optimize_bypass_generator=self.config.attack_optimizer_enabled,
            bypass_optimizer_mode=self.config.attack_bypass_optimizer_mode,
            bypass_optimizer_candidates=self.config.attack_bypass_optimizer_candidates,
            bypass_optimizer_every_generations=self.config.attack_optimizer_every_rounds,
            bypass_optimizer_min_examples=self.config.attack_optimizer_min_examples,
            bypass_optimizer_max_failed_samples=self.config.attack_optimizer_max_failed_samples,
            transfer_constraint_weight=self.config.attack_transfer_weight,
            transfer_constraint_min_score=self.config.attack_transfer_min_score,
            attacker_lm=self.attacker_lm,
            use_v2=self._target_v2,
        )
        self.convergence = ConvergenceDetector(
            threshold=self.config.convergence_threshold,
            consecutive_rounds=self.config.consecutive_rounds,
            max_rounds=self.config.max_rounds,
        )

        self.defense_optimizer: GuardrailOptimizer | None = None
        if self.config.defense_optimizer_mode:
            self.defense_optimizer = GuardrailOptimizer(
                mode=self.config.defense_optimizer_mode,
                max_iterations=self.config.defense_optimizer_max_iterations,
            )

        # State
        self.current_attacks: list[EvolvedAttack] = []
        self.result: TrainingResult | None = None
        self._run_id: str = ""
        self._optimizer_unsafe_examples: list[OptimizerExample] = []
        self._optimizer_seen_unsafe_payloads: set[str] = set()
        self._shadow_target: EvolvableShieldTarget | None = self._build_shadow_target()

    def run(self) -> TrainingResult:
        """
        Run adversarial training until convergence.

        Returns:
            TrainingResult with all training data
        """
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = datetime.now()

        # Initialize result
        self.result = TrainingResult(
            config=self.config.to_dict(),
            start_time=start_time,
        )

        # Load initial attacks
        self._load_initial_attacks()
        if self.config.verbose:
            print(f"Loaded {len(self.current_attacks)} initial attacks")

        round_num = 0

        while not self.convergence.is_converged():
            round_num += 1
            round_start = time.time()

            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Round {round_num}")
                print(f"{'='*60}")

            # 1. Execute attacks
            results = self._execute_attacks()

            # 2. Compute statistics
            stats = self._compute_stats(round_num, results, time.time() - round_start)

            # 3. Update convergence detector
            convergence_status = self.convergence.update(stats)
            if self.config.verbose:
                print(f"  {convergence_status}")

            # 4. Separate successful and failed attacks
            successful = [r for r in results if r.bypassed]
            failed = [r for r in results if not r.bypassed]
            self._record_optimizer_examples(results)

            if self.config.verbose:
                print(f"  Bypassed: {len(successful)}, Blocked: {len(failed)}")

            # 5. Evolve defense rules at a slower cadence (if there are successful attacks)
            should_update_rules = (
                successful
                and (round_num % max(1, self.config.defense_rule_update_every_rounds) == 0)
            )
            if should_update_rules:
                if self.config.defense_use_proactive_evolution:
                    defense_update = self.defense_evolver.proactive_evolve(successful)
                else:
                    defense_update = self.defense_evolver.evolve(successful)
                self.target.update_defense(defense_update)

                stats.new_patterns_added = len(defense_update.new_patterns)
                stats.new_examples_added = len(defense_update.new_examples)
                stats.total_patterns = len(self.defense_evolver.get_all_patterns())
                stats.total_examples = len(self.defense_evolver.get_all_examples())

                if self.config.verbose:
                    print(f"  Defense evolved: +{len(defense_update.new_patterns)} patterns, +{len(defense_update.new_examples)} examples")

                # V2: update attack evolver's defense profile
                if self._target_v2:
                    all_patterns = self.defense_evolver.get_all_patterns()
                    profile_parts = [f"Patterns: {len(all_patterns)} regex rules"]
                    if all_patterns:
                        profile_parts.append(f"Recent: {'; '.join(all_patterns[-5:])}")
                    examples_count = len(self.defense_evolver.get_all_examples())
                    if examples_count:
                        profile_parts.append(f"Few-shot examples: {examples_count}")
                    profile_parts.append("Detection: ChainOfThought reasoning + defense_hints")
                    self.attack_evolver.set_defense_profile(". ".join(profile_parts))

            else:
                # Always report cumulative totals even when rules don't update
                stats.total_patterns = len(self.defense_evolver.get_all_patterns())
                stats.total_examples = len(self.defense_evolver.get_all_examples())
                if successful and self.config.verbose:
                    print(
                        "  Defense rule update skipped this round "
                        f"(interval={max(1, self.config.defense_rule_update_every_rounds)})"
                    )

            # 6.5 Optimize LLM defense periodically with accumulated failures
            self._maybe_optimize_defense(round_num, stats)

            # 6. Evolve attacks (fast timescale by default: every round)
            if round_num % max(1, self.config.attack_update_every_rounds) == 0:
                new_attacks = self.attack_evolver.evolve(
                    successful,
                    failed,
                    is_blocked_fn=self._is_payload_blocked if self.config.attack_optimizer_enabled else None,
                    is_blocked_shadow_fn=self._is_payload_blocked_shadow
                    if self.config.attack_transfer_constraint_enabled and self._shadow_target
                    else None,
                )
                attack_opt_info = self.attack_evolver.get_last_optimization_info()
                stats.attack_optimizer_ran = bool(attack_opt_info.get("ran"))
                stats.attack_optimizer_status = str(attack_opt_info.get("status", ""))
                if new_attacks:
                    self.current_attacks = new_attacks
                    stats.new_attacks_generated = len(new_attacks)
                    if self.config.verbose:
                        print(f"  Attack evolved: {len(new_attacks)} new variants")
                else:
                    # Reload initial attacks if evolution produces nothing
                    self._load_initial_attacks()
            else:
                stats.attack_optimizer_status = (
                    f"skipped_interval_every_{max(1, self.config.attack_update_every_rounds)}"
                )
                if self.config.verbose:
                    print(
                        "  Attack evolution skipped this round "
                        f"(interval={max(1, self.config.attack_update_every_rounds)})"
                    )

            # 7. Store round data
            self.result.rounds.append(stats)

            # 8. Save checkpoint
            if self.config.save_every_round:
                self._save_round(round_num, results, stats)

            # 9. Optional round callback (e.g. for validation evaluation)
            if self.round_callback is not None:
                try:
                    self.round_callback(round_num, stats, self)
                except Exception as e:
                    if self.config.verbose:
                        print(f"  Round callback error: {e}")

        # Finalize result
        self.result.end_time = datetime.now()
        self.result.total_rounds = round_num
        self.result.converged = self.convergence.is_converged()
        self.result.convergence_round = self.convergence.get_convergence_round()

        if self.result.rounds:
            self.result.initial_asr = self.result.rounds[0].asr
            self.result.final_asr = self.result.rounds[-1].asr
            self.result.asr_reduction = self.result.initial_asr - self.result.final_asr

        self.result.final_patterns = self.defense_evolver.get_all_patterns()
        self.result.final_examples = self.defense_evolver.get_all_examples()
        self.result.evolved_attacks = [a.to_dict() for a in self.current_attacks]

        # Save final result
        self._save_final_result()

        if self.config.verbose:
            print("\n" + self.result.summary())

        return self.result

    def _load_initial_attacks(self) -> None:
        """Load initial attack payloads"""
        self.current_attacks = []

        per_category = max(1, self.config.attacks_per_round // max(1, len(self.config.attack_categories)))
        for category in self.config.attack_categories:
            provider = self.PAYLOAD_PROVIDERS.get(category)
            if provider:
                payloads = provider.get_all()
                for payload in payloads[:per_category]:
                    # Handle both AttackPayload (prompt) and PayloadTemplate (template)
                    payload_text = getattr(payload, 'prompt', None) or getattr(payload, 'template', str(payload))
                    severity_value = payload.severity.value if hasattr(payload.severity, 'value') else str(payload.severity)
                    self.current_attacks.append(EvolvedAttack(
                        id=str(uuid.uuid4())[:8],
                        payload=payload_text,
                        category=category,
                        severity=severity_value,
                        evolution_type="initial",
                        generation=0,
                    ))

    def _execute_attacks(self) -> list[AttackResult]:
        """Execute current attacks against target"""
        results = []

        # Limit attacks per round
        attacks_to_run = self.current_attacks[:self.config.attacks_per_round]

        for attack in attacks_to_run:
            try:
                # Reset session for each attack
                self.target.reset_session()

                # Execute attack
                start_time = time.time()
                response = self.target.invoke(attack.payload)
                response_time = (time.time() - start_time) * 1000  # ms

                # Determine if attack bypassed
                bypassed = not response.was_blocked

                results.append(AttackResult(
                    attack_id=attack.id,
                    payload=attack.payload,
                    category=attack.category,
                    severity=attack.severity,
                    bypassed=bypassed,
                    blocked=response.was_blocked,
                    response=response.response[:500] if response.response else "",
                    response_time_ms=response_time,
                    block_reason=response.metadata.get('block_reason') if response.metadata else None,
                ))

            except Exception as e:
                # Treat errors as blocked (fail-safe)
                results.append(AttackResult(
                    attack_id=attack.id,
                    payload=attack.payload,
                    category=attack.category,
                    severity=attack.severity,
                    bypassed=False,
                    blocked=True,
                    response=f"Error: {str(e)}",
                    response_time_ms=0,
                    block_reason=f"Exception: {str(e)}",
                ))

        return results

    def _record_optimizer_examples(self, results: list[AttackResult]) -> None:
        """Collect unsafe replay examples for periodic defense optimization."""
        for result in results:
            if not result.bypassed:
                # Focus replay on defense misses (unsafe prompts that bypassed).
                continue
            payload = (result.payload or "").strip()
            if not payload or payload in self._optimizer_seen_unsafe_payloads:
                continue
            self._optimizer_seen_unsafe_payloads.add(payload)
            self._optimizer_unsafe_examples.append(
                OptimizerExample(
                    text=payload,
                    is_unsafe=True,
                    category=result.category or "injection",
                )
            )

        # Keep a moving window to avoid unbounded growth.
        max_examples = max(self.config.defense_optimizer_max_trainset, 1)
        if len(self._optimizer_unsafe_examples) > max_examples:
            overflow = len(self._optimizer_unsafe_examples) - max_examples
            removed = self._optimizer_unsafe_examples[:overflow]
            self._optimizer_unsafe_examples = self._optimizer_unsafe_examples[overflow:]
            for item in removed:
                self._optimizer_seen_unsafe_payloads.discard(item.text)

    def _build_optimizer_dataset(self) -> tuple[list[OptimizerExample], list[OptimizerExample]]:
        """Build train/val splits for defense optimization with balanced replay."""
        unsafe_examples = self._optimizer_unsafe_examples[-self.config.defense_optimizer_max_trainset:]
        if not unsafe_examples:
            return [], []

        safe_examples_pool = [
            OptimizerExample(text=text, is_unsafe=False, category="injection")
            for text in self.config.defense_optimizer_safe_examples
        ]
        hard_negative_pool = [
            OptimizerExample(text=text, is_unsafe=False, category="injection")
            for text in self.config.defense_optimizer_hard_negative_examples
        ]

        if not self.config.defense_optimizer_use_balanced_replay:
            dataset = unsafe_examples + safe_examples_pool
        else:
            ratio_unsafe = max(1, self.config.defense_optimizer_replay_ratio_unsafe)
            ratio_safe = max(1, self.config.defense_optimizer_replay_ratio_safe)
            ratio_hard = max(1, self.config.defense_optimizer_replay_ratio_hard_negative)
            unit = ratio_unsafe + ratio_safe + ratio_hard

            max_units_by_data = min(
                len(unsafe_examples) // ratio_unsafe,
                len(safe_examples_pool) // ratio_safe,
                len(hard_negative_pool) // ratio_hard,
            )
            max_units_by_budget = max(1, self.config.defense_optimizer_max_trainset // unit)
            units = min(max_units_by_data, max_units_by_budget)
            if units <= 0:
                return [], []

            unsafe_part = unsafe_examples[-(ratio_unsafe * units):]
            safe_part = safe_examples_pool[:ratio_safe * units]
            hard_part = hard_negative_pool[:ratio_hard * units]
            dataset = unsafe_part + safe_part + hard_part

        if len(dataset) < self.config.defense_optimizer_min_examples:
            return [], []

        val_size = max(1, len(dataset) // 5)
        trainset = dataset[:-val_size] if len(dataset) > val_size else dataset
        valset = dataset[-val_size:]
        return trainset, valset

    def _maybe_optimize_defense(self, round_num: int, stats: RoundStats) -> None:
        """Run periodic GuardrailOptimizer for LLM defenses."""
        if not self.defense_optimizer:
            stats.defense_optimizer_status = "disabled"
            return
        if round_num % max(1, self.config.defense_optimizer_every_rounds) != 0:
            stats.defense_optimizer_status = (
                f"skipped_interval_every_{max(1, self.config.defense_optimizer_every_rounds)}"
            )
            return
        if not hasattr(self.target, "guardrail"):
            stats.defense_optimizer_status = "target_missing_guardrail"
            return

        trainset, valset = self._build_optimizer_dataset()
        if not trainset:
            stats.defense_optimizer_status = "skipped_insufficient_dataset"
            return

        try:
            original_guardrail = self.target.guardrail
            candidate_guardrail = self._clone_guardrail(original_guardrail)
            if candidate_guardrail is None:
                stats.defense_optimizer_status = "skipped_clone_failed"
                if self.config.verbose:
                    print("  Defense optimizer skipped: unable to clone guardrail for safe rollback")
                return

            result = self.defense_optimizer.optimize(
                guardrail=candidate_guardrail,
                trainset=trainset,
                valset=valset,
                metric="f1",
            )
            stats.defense_optimizer_ran = True
            stats.defense_optimizer_mode = self.config.defense_optimizer_mode
            stats.defense_optimizer_improvement = result.improvement
            applied = (
                getattr(result, "optimized_module", None) is not None
                and result.improvement >= self.config.defense_optimizer_min_improvement
            )
            if (
                applied
            ):
                self.target.guardrail = result.optimized_module
                if hasattr(self.target, "reapply_few_shot_examples"):
                    self.target.reapply_few_shot_examples()
                stats.defense_optimizer_applied = True
                stats.defense_optimizer_status = "applied"
            else:
                self.target.guardrail = original_guardrail
                stats.defense_optimizer_applied = False
                stats.defense_optimizer_status = (
                    f"rejected_improvement<{self.config.defense_optimizer_min_improvement:.3f}"
                )
            stats.defense_optimizer_checkpoint = self._persist_defense_optimizer_result(
                round_num=round_num,
                result=result,
                applied=applied,
                trainset_size=len(trainset),
                valset_size=len(valset),
            )
            if self.config.verbose:
                print(
                    f"  Defense optimizer ({self.config.defense_optimizer_mode}) "
                    f"improvement: {result.original_score:.3f} -> {result.optimized_score:.3f}"
                )
        except Exception as exc:
            stats.defense_optimizer_ran = True
            stats.defense_optimizer_mode = self.config.defense_optimizer_mode
            stats.defense_optimizer_improvement = 0.0
            stats.defense_optimizer_applied = False
            stats.defense_optimizer_status = f"failed:{type(exc).__name__}"
            if self.config.verbose:
                print(f"  Defense optimizer failed: {exc}")

    def _persist_defense_optimizer_result(
        self,
        round_num: int,
        result: Any,
        applied: bool,
        trainset_size: int,
        valset_size: int,
    ) -> str | None:
        """Persist optimizer artifacts for reproducibility."""
        if not self.config.defense_optimizer_save_checkpoints:
            return None

        status = "applied" if applied else "rejected"
        checkpoint_dir = (
            Path(self.config.output_dir)
            / f"run_{self._run_id}"
            / "optimizer_checkpoints"
            / f"round_{round_num:03d}_{status}"
        )
        try:
            checkpoint_path = result.save(
                str(checkpoint_dir),
                description=(
                    f"Defense optimizer result for round {round_num} "
                    f"(mode={self.config.defense_optimizer_mode}, status={status})"
                ),
            )
            context_path = checkpoint_dir / "context.json"
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round_num": round_num,
                        "status": status,
                        "mode": self.config.defense_optimizer_mode,
                        "trainset_size": trainset_size,
                        "valset_size": valset_size,
                        "min_improvement": self.config.defense_optimizer_min_improvement,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            return checkpoint_path
        except Exception as exc:
            if self.config.verbose:
                print(f"  Defense optimizer checkpoint save failed: {exc}")
            return None

    def _is_payload_blocked(self, payload: str) -> bool:
        """Callback for attack-side optimizer to test payload block status."""
        self.target.reset_session()
        response = self.target.invoke(payload)
        return response.was_blocked

    def _is_payload_blocked_shadow(self, payload: str) -> bool:
        """Secondary defender callback for transfer-aware attack optimization."""
        if self._shadow_target is None:
            return self._is_payload_blocked(payload)
        self._shadow_target.reset_session()
        response = self._shadow_target.invoke(payload)
        return response.was_blocked

    @staticmethod
    def _clone_guardrail(guardrail: Any) -> Any | None:
        """Clone guardrail module to enable optimizer gating and rollback."""
        try:
            return deepcopy(guardrail)
        except Exception:
            return None

    def _build_shadow_target(self) -> EvolvableShieldTarget | None:
        """Build a lightweight shadow defender for transfer-aware attack scoring."""
        if not self.config.attack_transfer_constraint_enabled:
            return None
        if self.config.attack_transfer_shadow_mode != "shield_fast":
            return None
        try:
            shadow_shield = Shield(mode="fast", checks=["injection"])
            return EvolvableShieldTarget(shadow_shield)
        except Exception:
            return None

    def _compute_stats(
        self,
        round_num: int,
        results: list[AttackResult],
        duration: float,
    ) -> RoundStats:
        """Compute statistics for the round"""
        bypassed = sum(1 for r in results if r.bypassed)
        blocked = sum(1 for r in results if r.blocked)

        # Category breakdown
        category_stats = {}
        for r in results:
            if r.category not in category_stats:
                category_stats[r.category] = {"bypassed": 0, "blocked": 0}
            if r.bypassed:
                category_stats[r.category]["bypassed"] += 1
            else:
                category_stats[r.category]["blocked"] += 1

        return RoundStats(
            round_num=round_num,
            timestamp=datetime.now(),
            total_attacks=len(results),
            bypassed_count=bypassed,
            blocked_count=blocked,
            avg_response_time_ms=sum(r.response_time_ms for r in results) / len(results) if results else 0,
            round_duration_seconds=duration,
            category_stats=category_stats,
        )

    def _save_round(
        self,
        round_num: int,
        results: list[AttackResult],
        stats: RoundStats,
    ) -> None:
        """Save round data to disk"""
        output_dir = Path(self.config.output_dir) / f"run_{self._run_id}" / "rounds"
        output_dir.mkdir(parents=True, exist_ok=True)

        round_data = {
            "round_num": round_num,
            "stats": stats.to_dict(),
            "results": [
                {
                    "attack_id": r.attack_id,
                    "payload": r.payload[:200],  # Truncate for storage
                    "category": r.category,
                    "bypassed": r.bypassed,
                    "blocked": r.blocked,
                    "response_time_ms": r.response_time_ms,
                }
                for r in results
            ],
        }

        with open(output_dir / f"round_{round_num:03d}.json", "w") as f:
            json.dump(round_data, f, indent=2, default=str)

    def _save_final_result(self) -> None:
        """Save final training result"""
        output_dir = Path(self.config.output_dir) / f"run_{self._run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        with open(output_dir / "summary.json", "w") as f:
            json.dump(self.result.to_dict(), f, indent=2, default=str)

        # Save evolved patterns
        with open(output_dir / "evolved_patterns.json", "w") as f:
            json.dump(self.result.final_patterns, f, indent=2)

        # Save evolved examples
        with open(output_dir / "evolved_examples.json", "w") as f:
            json.dump(self.result.final_examples, f, indent=2)

        # Save evolved attacks
        with open(output_dir / "evolved_attacks.json", "w") as f:
            json.dump(self.result.evolved_attacks, f, indent=2)

        # Save a human-friendly payload list for reuse (one payload per line).
        try:
            payloads = [
                str(item.get("payload", "")).strip()
                for item in (self.result.evolved_attacks or [])
                if isinstance(item, dict)
            ]
            payloads = [p for p in payloads if p]
            if payloads:
                with open(output_dir / "evolved_attack_payloads.txt", "w", encoding="utf-8") as f:
                    for p in payloads:
                        f.write(p.replace("\n", "\\n") + "\n")
        except Exception:
            pass

        # Persist the final defense module snapshot (LLM guardrail) so it can be reused directly.
        # This captures compiled predictors + demos already injected during training.
        try:
            guardrail = getattr(self.target, "guardrail", None)
            if guardrail is not None and hasattr(guardrail, "save"):
                guardrail.save(str(output_dir / "final_guardrail_module.json"))
        except Exception:
            pass

        # Persist the target-side few-shot examples cache (if available).
        try:
            few_shot = getattr(self.target, "few_shot_examples", None)
            if isinstance(few_shot, list) and few_shot:
                with open(output_dir / "final_guardrail_few_shot_examples.json", "w", encoding="utf-8") as f:
                    json.dump(few_shot, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

        # Lightweight summary for downstream consumption.
        try:
            summary = {
                "run_id": self._run_id,
                "defense_backend": type(getattr(self.target, "guardrail", None)).__name__
                if hasattr(self.target, "guardrail") else type(self.target).__name__,
                "final_examples_count": len(self.result.final_examples or []),
                "final_patterns_count": len(self.result.final_patterns or []),
                "evolved_attacks_count": len(self.result.evolved_attacks or []),
                "final_guardrail_module_saved": (output_dir / "final_guardrail_module.json").exists(),
            }
            with open(output_dir / "artifacts.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        if self.config.verbose:
            print(f"\nResults saved to: {output_dir}")

    def get_result(self) -> TrainingResult | None:
        """Get the training result"""
        return self.result

    def reset(self) -> None:
        """Reset trainer state"""
        self.defense_evolver.reset()
        self.attack_evolver.reset()
        self.convergence.reset()
        self.current_attacks = []
        self.result = None
        self._optimizer_unsafe_examples = []
        self._optimizer_seen_unsafe_payloads = set()
