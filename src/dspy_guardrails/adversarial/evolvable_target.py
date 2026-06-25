"""
EvolvableShieldTarget - Shield wrapper that implements EvolvableTarget protocol.

This module bridges the Shield API with the AdversarialTrainer, enabling
closed-loop adversarial training where both attacks and defenses evolve.

Usage:
    from dspy_guardrails import Shield
    from dspy_guardrails.adversarial import AdversarialTrainer, AdversarialConfig
    from dspy_guardrails.adversarial.evolvable_target import EvolvableShieldTarget

    # Wrap Shield as an evolvable target
    shield = Shield(mode="fast", checks=["injection"])
    target = EvolvableShieldTarget(shield)

    # Run adversarial training
    config = AdversarialConfig(max_rounds=10)
    trainer = AdversarialTrainer(target, config)
    result = trainer.run()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..shield import Shield, ShieldResult
from ..testing.targets import TargetResponse
from .metrics import DefenseUpdate


@dataclass
class EvolvableShieldTarget:
    """
    Wrapper that makes Shield compatible with AdversarialTrainer.

    Implements the EvolvableTarget protocol:
    - invoke(prompt) -> TargetResponse
    - update_defense(update) -> None
    - reset_session() -> None
    - get_defense_stats() -> dict

    The target tracks learned patterns and examples from successful attacks,
    using them to improve detection over time.

    Attributes:
        shield: The underlying Shield instance
        learned_patterns: Regex patterns extracted from successful attacks
        learned_examples: Few-shot examples for LLM-based detection
        stats: Statistics about defense updates
    """

    shield: Shield
    learned_patterns: list[str] = field(default_factory=list)
    learned_examples: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {
        "total_invocations": 0,
        "blocked_count": 0,
        "bypassed_count": 0,
        "defense_updates": 0,
        "patterns_added": 0,
        "examples_added": 0,
    })
    _session_context: dict = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        checks: list[str] | None = None,
        mode: str = "fast",
        domain: str | None = None,
        **shield_kwargs,
    ) -> "EvolvableShieldTarget":
        """
        Create target from Shield configuration.

        Args:
            checks: List of checks (default: ["injection"])
            mode: "fast" or "hybrid"
            domain: Domain allowlist (e.g., "technical")
            **shield_kwargs: Additional Shield arguments

        Returns:
            Configured EvolvableShieldTarget
        """
        checks = checks or ["injection"]
        shield = Shield(
            checks=checks,
            mode=mode,
            domain=domain,
            **shield_kwargs,
        )
        return cls(shield=shield)

    def invoke(self, prompt: str) -> TargetResponse:
        """
        Invoke the shield on a prompt.

        Args:
            prompt: Input text to check

        Returns:
            TargetResponse with blocking decision and metadata
        """
        self.stats["total_invocations"] += 1

        # Check against learned patterns first (fast path)
        for pattern in self.learned_patterns:
            try:
                if re.search(pattern, prompt, re.IGNORECASE):
                    self.stats["blocked_count"] += 1
                    return TargetResponse(
                        response="",
                        guardrail_status={
                            "learned_pattern": {
                                "passed": False,
                                "reason": f"Matched pattern: {pattern[:50]}",
                            }
                        },
                        metadata={
                            "block_reason": f"Matched learned pattern: {pattern[:50]}",
                            "detection_source": "learned_pattern",
                        },
                    )
            except re.error:
                continue  # Skip invalid patterns

        # Check against Shield
        result = self.shield.check(prompt)

        if not result.safe:
            self.stats["blocked_count"] += 1
            block_reasons = [issue.message for issue in result.issues]
            return TargetResponse(
                response="",
                guardrail_status={
                    "shield": {
                        "passed": False,
                        "reasons": block_reasons,
                    }
                },
                metadata={
                    "block_reason": "; ".join(block_reasons),
                    "detection_source": "shield",
                    "issues": [
                        {"check": i.check, "message": i.message, "severity": i.severity}
                        for i in result.issues
                    ],
                },
            )

        # Passed all checks
        self.stats["bypassed_count"] += 1
        return TargetResponse(
            response=f"[LLM would respond to: {prompt[:100]}...]",
            guardrail_status={"shield": {"passed": True}},
            metadata={
                "detection_source": "none",
            },
        )

    def update_defense(self, update: DefenseUpdate) -> None:
        """
        Update defense with new patterns and examples.

        Called by AdversarialTrainer after successful attacks are analyzed.

        Args:
            update: DefenseUpdate containing new patterns and examples
        """
        self.stats["defense_updates"] += 1

        # Add new patterns (deduplicated)
        for pattern in update.new_patterns:
            if pattern not in self.learned_patterns:
                # Validate pattern is valid regex
                try:
                    re.compile(pattern)
                    self.learned_patterns.append(pattern)
                    self.stats["patterns_added"] += 1
                except re.error:
                    continue

        # Add new examples (for potential LLM few-shot learning)
        for example in update.new_examples:
            if example not in self.learned_examples:
                self.learned_examples.append(example)
                self.stats["examples_added"] += 1

        # Optionally: Update Shield's custom allowlists
        # This could be implemented to dynamically adjust Shield configuration

    def reset_session(self) -> None:
        """Reset session context (for multi-turn scenarios)."""
        self._session_context = {}

    def get_defense_stats(self) -> dict[str, Any]:
        """
        Get current defense statistics.

        Returns:
            Dict with invocation counts, block rates, learned patterns, etc.
        """
        total = self.stats["total_invocations"]
        blocked = self.stats["blocked_count"]
        bypassed = self.stats["bypassed_count"]

        return {
            "total_invocations": total,
            "blocked_count": blocked,
            "bypassed_count": bypassed,
            "block_rate": blocked / total if total > 0 else 0,
            "bypass_rate": bypassed / total if total > 0 else 0,
            "defense_updates": self.stats["defense_updates"],
            "learned_patterns_count": len(self.learned_patterns),
            "learned_examples_count": len(self.learned_examples),
            "shield_mode": self.shield._mode,
            "shield_checks": self.shield._check_names,
        }

    def export_learned_defenses(self) -> dict:
        """
        Export learned patterns and examples for persistence.

        Returns:
            Dict with patterns and examples that can be saved/loaded
        """
        return {
            "patterns": list(self.learned_patterns),
            "examples": list(self.learned_examples),
            "stats": dict(self.stats),
        }

    def import_learned_defenses(self, data: dict) -> None:
        """
        Import previously learned defenses.

        Args:
            data: Dict from export_learned_defenses()
        """
        if "patterns" in data:
            for pattern in data["patterns"]:
                if pattern not in self.learned_patterns:
                    try:
                        re.compile(pattern)
                        self.learned_patterns.append(pattern)
                    except re.error:
                        continue

        if "examples" in data:
            for example in data["examples"]:
                if example not in self.learned_examples:
                    self.learned_examples.append(example)


@dataclass
class EvolvableLLMTarget:
    """
    Target wrapper for LLM-based guardrails (LLMGuardrail, HybridGuardrail).

    This enables adversarial training against LLM-based detection systems,
    where the defense can be updated by modifying few-shot examples.
    """

    guardrail: Any  # LLMGuardrail or HybridGuardrail
    defender_lm: Any | None = None
    few_shot_examples: list[dict] = field(default_factory=list)
    max_dynamic_demos: int = 40
    _defense_hints_parts: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=lambda: {
        "total_invocations": 0,
        "blocked_count": 0,
        "bypassed_count": 0,
        "defense_updates": 0,
        "examples_added": 0,
        "demos_applied": 0,
        "demo_application_failures": 0,
    })

    @property
    def _is_v2(self) -> bool:
        return getattr(self.guardrail, "use_v2", False)

    def invoke(self, prompt: str) -> TargetResponse:
        """Invoke LLM guardrail on prompt."""
        self.stats["total_invocations"] += 1

        if self.defender_lm is not None:
            try:
                import dspy

                with dspy.context(lm=self.defender_lm):
                    return self._invoke_with_current_guardrail(prompt)
            except Exception:
                # Fall back to default runtime LM if context setup fails.
                pass

        return self._invoke_with_current_guardrail(prompt)

    def _invoke_with_current_guardrail(self, prompt: str) -> TargetResponse:
        """Invoke guardrail assuming LM context is already configured."""
        try:
            # Call the guardrail
            if hasattr(self.guardrail, 'check_all'):
                result = self.guardrail.check_all(prompt)
            else:
                result = self.guardrail.check(prompt, "injection")

            is_unsafe = getattr(result, 'is_unsafe', False)
            if isinstance(is_unsafe, str):
                is_unsafe = is_unsafe.lower() in ('true', 'yes', '1')

            if is_unsafe:
                self.stats["blocked_count"] += 1
                return TargetResponse(
                    response="",
                    guardrail_status={
                        "llm_guardrail": {
                            "passed": False,
                            "reason": getattr(result, 'reason', 'LLM detected attack'),
                        }
                    },
                    metadata={
                        "block_reason": getattr(result, 'reason', 'LLM detected attack'),
                        "confidence": getattr(result, 'confidence', 0.0),
                        "few_shot_count": len(self.few_shot_examples),
                    },
                )
            else:
                self.stats["bypassed_count"] += 1
                return TargetResponse(
                    response=f"[LLM would respond]",
                    guardrail_status={"llm_guardrail": {"passed": True}},
                )

        except Exception as e:
            # Errors = fail closed (block)
            self.stats["blocked_count"] += 1
            return TargetResponse(
                response="",
                guardrail_status={
                    "llm_guardrail": {"passed": False, "reason": f"Error: {str(e)}"}
                },
                metadata={"block_reason": f"Error: {str(e)}"},
            )

    def update_defense(self, update: DefenseUpdate) -> None:
        """Update defense with new examples and apply them to DSPy demos."""
        self.stats["defense_updates"] += 1
        for example in update.new_examples:
            if example not in self.few_shot_examples:
                self.few_shot_examples.append(example)
                self.stats["examples_added"] += 1

        # V2: accumulate defense hints from new patterns
        if self._is_v2 and update.new_patterns:
            for pat in update.new_patterns:
                if pat not in self._defense_hints_parts:
                    self._defense_hints_parts.append(pat)
            hints = "; ".join(self._defense_hints_parts[-20:])  # keep last 20
            if hasattr(self.guardrail, "set_defense_hints"):
                self.guardrail.set_defense_hints(hints)

        self._apply_few_shot_examples()

    def reapply_few_shot_examples(self) -> None:
        """Re-apply cached few-shot examples after optimizer recompilation."""
        self._apply_few_shot_examples()

    def _build_demo(self, example: dict[str, Any], comprehensive: bool):
        """Convert a defense example into a DSPy demo matching classifier signatures."""
        try:
            import dspy
        except Exception:
            return None

        text = str(example.get("input") or example.get("text") or "").strip()
        if not text:
            return None

        threat = str(example.get("threat_type") or example.get("category") or "injection").lower()
        reason = str(
            example.get("explanation")
            or example.get("reason")
            or "Unsafe prompt injection or jailbreak attempt"
        ).strip()

        # V2 mode: match ThreatAnalysisV2 signature
        if self._is_v2:
            return dspy.Example(
                text=text,
                defense_hints="",
                verdict="UNSAFE",
                threat_type=threat if threat and threat != "none" else "injection",
                confidence=1.0,
            ).with_inputs("text", "defense_hints")

        if comprehensive:
            categories = threat if threat and threat != "none" else "injection"
            return dspy.Example(
                text=text,
                is_unsafe=True,
                confidence=1.0,
                categories=categories,
                reason=reason,
            ).with_inputs("text")

        category = threat if threat in {"injection", "toxicity", "pii"} else "injection"
        return dspy.Example(
            text=text,
            category=category,
            is_unsafe=True,
            confidence=1.0,
            reason=reason,
        ).with_inputs("text", "category")

    @staticmethod
    def _extract_demo_text(demo: Any) -> str:
        """Extract demo text from a DSPy Example-like object."""
        try:
            if hasattr(demo, "get"):
                return str(demo.get("text", "")).strip()
        except Exception:
            pass
        return str(getattr(demo, "text", "")).strip()

    def _apply_few_shot_examples(self) -> None:
        """Inject learned examples into classifier demos for immediate effect."""
        if not self.few_shot_examples:
            return
        if not getattr(self.guardrail, "use_dspy", False):
            return

        # V2 mode: inject into v2_analyzer
        if self._is_v2:
            v2_analyzer = getattr(self.guardrail, "v2_analyzer", None)
            if v2_analyzer is None:
                return
            try:
                recent = self.few_shot_examples[-self.max_dynamic_demos:]
                existing_demos = list(getattr(v2_analyzer, "demos", []) or [])
                seen_texts = {
                    self._extract_demo_text(d) for d in existing_demos
                    if self._extract_demo_text(d)
                }
                applied = 0
                for example in recent:
                    demo = self._build_demo(example, comprehensive=False)
                    if demo is None:
                        continue
                    dt = self._extract_demo_text(demo)
                    if not dt or dt in seen_texts:
                        continue
                    existing_demos.append(demo)
                    seen_texts.add(dt)
                    applied += 1
                v2_analyzer.demos = existing_demos[-self.max_dynamic_demos:]
                self.stats["demos_applied"] += applied
            except Exception:
                self.stats["demo_application_failures"] += 1
            return

        # V1 mode: inject into classifier / comprehensive_classifier
        predictors: list[tuple[Any, bool]] = []
        comprehensive = getattr(self.guardrail, "comprehensive_classifier", None)
        classifier = getattr(self.guardrail, "classifier", None)
        if comprehensive is not None:
            predictors.append((comprehensive, True))
        if classifier is not None:
            predictors.append((classifier, False))
        if not predictors:
            return

        applied = 0
        try:
            recent_examples = self.few_shot_examples[-self.max_dynamic_demos:]
            for predictor, is_comprehensive in predictors:
                existing_demos = list(getattr(predictor, "demos", []) or [])
                seen_texts = {
                    self._extract_demo_text(demo)
                    for demo in existing_demos
                    if self._extract_demo_text(demo)
                }

                for example in recent_examples:
                    demo = self._build_demo(example, comprehensive=is_comprehensive)
                    if demo is None:
                        continue
                    demo_text = self._extract_demo_text(demo)
                    if not demo_text or demo_text in seen_texts:
                        continue
                    existing_demos.append(demo)
                    seen_texts.add(demo_text)
                    applied += 1

                predictor.demos = existing_demos[-self.max_dynamic_demos:]

            self.stats["demos_applied"] += applied
        except Exception:
            self.stats["demo_application_failures"] += 1

    def reset_session(self) -> None:
        """Reset session."""
        pass

    def get_defense_stats(self) -> dict[str, Any]:
        """Get statistics."""
        total = self.stats["total_invocations"]
        return {
            **self.stats,
            "block_rate": self.stats["blocked_count"] / total if total > 0 else 0,
            "few_shot_count": len(self.few_shot_examples),
            "max_dynamic_demos": self.max_dynamic_demos,
        }
