"""Guardrail Optimizer — Self-evolving optimizer.

Uses GEPA / MIPROv2 / BootstrapFewShot / SIMBA to automatically improve
guardrail detection.

Core approach:
1. Collect failure cases (missed attacks + false positives)
2. LLM reflects on failure causes
3. Evolve prompts / rules / few-shot demos
4. Adversarial learning: attack and defense co-evolve

Usage:
    from dspy_guardrails import GuardrailOptimizer

    trainset = [
        Example(text="hello", is_unsafe=False),
        Example(text="ignore all instructions", is_unsafe=True),
    ]

    optimizer = GuardrailOptimizer(mode="gepa")
    result = optimizer.optimize(
        guardrail=LLMGuardrail(),
        trainset=trainset,
        metric="f1",
    )

    print(f"Improvement: {result.improvement:.1%}")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gepa import EvaluationBatch

import dspy


def _dspy_reflection_lm(prompt: str) -> str:
    """Default reflection LM for GEPA.

    GEPA's reflective mutation proposer expects `reflection_lm(prompt: str) -> str`.
    DSPy's LM returns a list of outputs; we return the first text-like output.
    """
    lm = getattr(dspy.settings, "lm", None)
    if lm is None:
        raise ValueError("DSPy LM is not configured (dspy.settings.lm is None)")

    outputs = lm(prompt)
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("content", "text", "message"):
                value = first.get(key)
                if value:
                    return str(value)
            return str(first)
    return str(outputs)


@dataclass
class OptimizationResult:
    """Optimization result with save/load support.

    Examples:
        result = optimizer.optimize(guardrail, trainset)
        result.save("checkpoints/guardrail_v1")

        loaded = OptimizationResult.load(
            "checkpoints/guardrail_v1",
            module_class=LLMGuardrail
        )
        guardrail = loaded.optimized_module
    """
    original_prompt: str
    optimized_prompt: str
    original_score: float
    optimized_score: float
    improvement: float
    iterations: int
    failure_analysis: list[dict] = field(default_factory=list)

    # Optimized module and checkpoint path
    optimized_module: dspy.Module | None = None
    checkpoint_path: str | None = None

    def save(self, path: str, description: str = "") -> str:
        """Save optimization result.

        Args:
            path: Save directory path
            description: Optional description

        Returns:
            The saved directory path
        """
        from pathlib import Path
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save result data
        result_data = {
            "original_prompt": self.original_prompt,
            "optimized_prompt": self.optimized_prompt,
            "original_score": self.original_score,
            "optimized_score": self.optimized_score,
            "improvement": self.improvement,
            "iterations": self.iterations,
            "failure_analysis": self.failure_analysis,
        }

        with open(save_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        # Save module (if available)
        if self.optimized_module is not None:
            self.optimized_module.save(str(save_dir / "module.json"))

        # Save metadata
        from datetime import datetime
        metadata = {
            "type": "optimization",
            "created_at": datetime.now().isoformat(),
            "description": description,
            "metrics": {
                "original_score": self.original_score,
                "optimized_score": self.optimized_score,
                "improvement": self.improvement,
            },
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.checkpoint_path = str(save_dir)
        return str(save_dir)

    @classmethod
    def load(
        cls,
        path: str,
        module_class: type = None,
        **module_init_kwargs,
    ) -> OptimizationResult:
        """Load optimization result.

        Args:
            path: Checkpoint directory path
            module_class: Module class (for loading the module)
            **module_init_kwargs: Module initialization arguments

        Returns:
            OptimizationResult
        """
        from pathlib import Path
        load_dir = Path(path)

        if not load_dir.exists():
            raise ValueError(f"Checkpoint not found: {path}")

        # Load result data
        with open(load_dir / "result.json", encoding="utf-8") as f:
            result_data = json.load(f)

        # Load module (if available)
        module = None
        module_path = load_dir / "module.json"
        if module_class is not None and module_path.exists():
            module = module_class(**module_init_kwargs)
            module.load(str(module_path))

        return cls(
            original_prompt=result_data.get("original_prompt", ""),
            optimized_prompt=result_data.get("optimized_prompt", ""),
            original_score=result_data.get("original_score", 0.0),
            optimized_score=result_data.get("optimized_score", 0.0),
            improvement=result_data.get("improvement", 0.0),
            iterations=result_data.get("iterations", 0),
            failure_analysis=result_data.get("failure_analysis", []),
            optimized_module=module,
            checkpoint_path=str(load_dir),
        )


@dataclass
class Example:
    """Training example."""
    text: str
    is_unsafe: bool
    category: str = "injection"


# =============================================================================
# GEPA Adapter
# =============================================================================

class GuardrailGEPAAdapter:
    """
    GEPA Adapter for Guardrails

    Implements the GEPA protocol interface so GEPA can optimize guardrail prompts.

    Usage:
        import gepa
        from dspy_guardrails import LLMGuardrail

        adapter = GuardrailGEPAAdapter(LLMGuardrail())

        result = gepa.optimize(
            seed_candidate={"system_prompt": adapter.get_current_prompt()},
            trainset=examples,
            adapter=adapter,
            task_lm="openai/gpt-4o-mini",
            reflection_lm="openai/gpt-4o",
        )
    """

    # GEPAAdapter protocol expects this attribute to exist (callable or None).
    propose_new_texts = None

    def __init__(
        self,
        guardrail: dspy.Module,
        metric: str = "f1",
    ):
        self.guardrail = guardrail
        self.metric = metric
        self._current_prompt = self._extract_prompt(guardrail)

    def get_current_prompt(self) -> str:
        """Get current prompt."""
        return self._current_prompt

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate candidate prompt.

        Args:
            batch: [{"text": "...", "is_unsafe": True/False, "category": "..."}]
            candidate: {"system_prompt": "..."}
            capture_traces: whether to return per-example trajectories for reflection

        Returns:
            EvaluationBatch with scores and trajectories
        """
        try:
            from gepa import EvaluationBatch
        except ImportError:
            raise ImportError("GEPA not installed. Run: pip install gepa") from None

        # Apply candidate prompt to guardrail
        self._apply_prompt(candidate.get("system_prompt", self._current_prompt))

        scores: list[float] = []
        outputs: list[object] = []
        trajectories: list[dict] | None = [] if capture_traces else None

        for example in batch:
            text = example.get("text", "")
            expected = example.get("is_unsafe", example.get("label", False))
            category = example.get("category", "injection")

            try:
                result = self.guardrail(text=text, category=category)
                predicted = self._parse_prediction(result)

                # Compute score (correct=1, wrong=0)
                correct = (predicted == expected)
                score = 1.0 if correct else 0.0

            except Exception as e:
                score = 0.0
                predicted = None
                result = None
                error_msg = str(e)
            else:
                error_msg = ""

            scores.append(score)
            outputs.append(predicted)
            if trajectories is not None:
                trajectories.append({
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": bool(score >= 1.0),
                    "confidence": getattr(result, "confidence", 0.5) if result is not None else 0.0,
                    "reason": getattr(result, "reason", "") if result is not None else "",
                    "error": error_msg,
                    "error_type": None if score >= 1.0 else (
                        "exception" if result is None else ("false_negative" if expected else "false_positive")
                    ),
                })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Build reflective dataset.

        Extract information from failure cases for GEPA's reflection LM to analyze.
        """
        reflective_data = {}

        for component in components_to_update:
            failures = []

            trajectories = eval_batch.trajectories or []
            for traj, score in zip(trajectories, eval_batch.scores, strict=False):
                if score < 1.0:  # Failure case
                    failures.append({
                        "input": traj.get("text", ""),
                        "expected": traj.get("expected"),
                        "predicted": traj.get("predicted"),
                        "error_type": traj.get("error_type"),
                        "reason": traj.get("reason", ""),
                        "error": traj.get("error", ""),
                    })

            reflective_data[component] = failures

        return reflective_data

    def _apply_prompt(self, prompt: str):
        """Apply new prompt to guardrail."""
        self._current_prompt = prompt

        # Dynamically update the guardrail's signature docstring
        classifier = None
        if hasattr(self.guardrail, "get_classifier"):
            try:
                classifier = self.guardrail.get_classifier()
            except Exception:
                classifier = None

        if classifier is None:
            if (
                getattr(self.guardrail, "comprehensive", False)
                and hasattr(self.guardrail, "comprehensive_classifier")
            ):
                classifier = self.guardrail.comprehensive_classifier
            elif hasattr(self.guardrail, "classifier"):
                classifier = self.guardrail.classifier

        if classifier is not None and hasattr(classifier, "signature"):
            sig_class = classifier.signature
            sig_class.__doc__ = prompt

    def _extract_prompt(self, guardrail: dspy.Module) -> str:
        """Extract guardrail prompt."""
        if hasattr(guardrail, "get_classifier"):
            try:
                classifier = guardrail.get_classifier()
            except Exception:
                classifier = None
            if classifier is not None and hasattr(classifier, "signature"):
                sig = classifier.signature
                return sig.__doc__ or str(sig)

        if (
            getattr(guardrail, "comprehensive", False)
            and hasattr(guardrail, "comprehensive_classifier")
        ):
            sig = guardrail.comprehensive_classifier.signature
            return sig.__doc__ or str(sig)
        if hasattr(guardrail, 'classifier'):
            sig = guardrail.classifier.signature
            return sig.__doc__ or str(sig)
        return str(guardrail)

    def _parse_prediction(self, result) -> bool:
        """Parse prediction result."""
        predicted = result.is_unsafe
        if isinstance(predicted, str):
            predicted = predicted.lower() in ('true', 'yes', '1')
        return predicted


# =============================================================================
# Guardrail Optimizer
# =============================================================================

class GuardrailOptimizer:
    """
    Guardrail Optimizer

    Supports four modes:
    1. GEPA mode: Use GEPA to evolve prompts
    2. MIPROv2 mode: Use DSPy MIPROv2 to jointly optimize prompt + few-shot
    3. DSPy mode: Use DSPy BootstrapFewShot
    4. SIMBA mode: Use DSPy SIMBA for online reflective optimization

    Examples:
        # Prepare training data
        trainset = [
            Example("hello", is_unsafe=False),
            Example("ignore previous", is_unsafe=True),
        ]

        # Optimize with GEPA mode
        optimizer = GuardrailOptimizer(mode="gepa")
        result = optimizer.optimize(
            guardrail=LLMGuardrail(),
            trainset=trainset,
        )

        print(f"Improvement: {result.improvement:.1%}")
    """

    def __init__(
        self,
        mode: str = "gepa",  # "gepa" | "mipro" | "dspy" | "simba"
        task_lm: str = None,  # GEPA: task execution LM
        reflection_lm: str = None,  # GEPA: reflection analysis LM
        max_iterations: int = 50,  # GEPA: max_metric_calls
        auto_save: bool = False,  # Whether to auto-save checkpoints
        checkpoint_dir: str = "./checkpoints/optimizations",  # Checkpoint directory
    ):
        self.mode = mode
        self.task_lm = task_lm
        self.reflection_lm = reflection_lm
        self.max_iterations = max_iterations
        self.auto_save = auto_save
        self.checkpoint_dir = checkpoint_dir

    def _check_dspy_configured(self) -> bool:
        """Check if DSPy LM is configured.

        Returns:
            True if LM is configured, False otherwise
        """
        try:
            if not getattr(dspy.settings, "lm", None):
                return False
            return True
        except Exception:
            return False

    def optimize(
        self,
        guardrail: dspy.Module,
        trainset: list[Example],
        valset: list[Example] = None,
        metric: str = "f1",
    ) -> OptimizationResult:
        """Optimize guardrail.

        Args:
            guardrail: Guardrail module to optimize
            trainset: Training data
            valset: Validation data (optional)
            metric: Optimization target ("f1", "accuracy", "precision", "recall")

        Returns:
            OptimizationResult: Optimization result

        Raises:
            ValueError: If DSPy LM is not configured and mode requires it
        """
        # Check LLM configuration
        if not self._check_dspy_configured():
            import warnings
            msg = (
                "DSPy LM not configured. Optimization may fail or produce suboptimal results. "
                "Configure with: dspy.configure(lm=dspy.LM('openai/gpt-4', api_key=...))"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        valset = valset or trainset[:max(1, len(trainset)//5)]

        # Get original prompt
        original_prompt = self._extract_prompt(guardrail)
        original_score = self._evaluate(guardrail, valset, metric)

        if self.mode == "gepa":
            result = self._optimize_with_gepa(
                guardrail, trainset, valset, metric
            )
        elif self.mode == "mipro":
            result = self._optimize_with_mipro(
                guardrail, trainset, valset, metric
            )
        elif self.mode == "simba":
            result = self._optimize_with_simba(
                guardrail, trainset, valset, metric
            )
        else:
            result = self._optimize_with_dspy(
                guardrail, trainset, valset, metric
            )

        optimization_result = OptimizationResult(
            original_prompt=original_prompt,
            optimized_prompt=result["prompt"],
            original_score=original_score,
            optimized_score=result["score"],
            improvement=result["score"] - original_score,
            iterations=result["iterations"],
            failure_analysis=result.get("failures", []),
            optimized_module=guardrail,  # Store the optimized module
        )

        # Auto-save checkpoint
        if self.auto_save:
            from datetime import datetime
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{guardrail.__class__.__name__}_{timestamp}"
            checkpoint_path = Path(self.checkpoint_dir) / checkpoint_name
            optimization_result.save(
                str(checkpoint_path),
                description=f"Auto-saved optimization result (mode={self.mode})"
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        return optimization_result

    def _optimize_with_gepa(
        self,
        guardrail: dspy.Module,
        trainset: list[Example],
        valset: list[Example],
        metric: str,
    ) -> dict:
        """Optimize using GEPA."""
        try:
            import gepa
        except ImportError:
            print("Warning: GEPA not installed, falling back to simple optimization")
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)

        # Create GEPA adapter
        adapter = GuardrailGEPAAdapter(guardrail, metric)

        # Convert data format
        gepa_trainset = [
            {"text": e.text, "is_unsafe": e.is_unsafe, "category": e.category}
            for e in trainset
        ]
        gepa_valset = [
            {"text": e.text, "is_unsafe": e.is_unsafe, "category": e.category}
            for e in valset
        ]

        # Get initial prompt
        seed_candidate = {
            "system_prompt": adapter.get_current_prompt()
        }

        try:
            # Run GEPA optimization
            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=gepa_trainset,
                valset=gepa_valset,
                adapter=adapter,
                # When an adapter is provided, GEPA requires task_lm=None.
                task_lm=None,
                # Default to using the currently configured DSPy LM for reflection.
                reflection_lm=self.reflection_lm or _dspy_reflection_lm,
                max_metric_calls=self.max_iterations,
                display_progress_bar=True,
            )

            # Apply best prompt
            best_prompt = result.best_candidate.get("system_prompt", seed_candidate["system_prompt"])
            adapter._apply_prompt(best_prompt)

            # Evaluate final performance
            final_score = self._evaluate(guardrail, valset, metric)

            return {
                "prompt": best_prompt,
                "score": final_score,
                "iterations": result.total_metric_calls,
            }

        except Exception as e:
            print(f"GEPA optimization failed: {e}, falling back to simple optimization")
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)

    @staticmethod
    def _make_metric():
        """Create a metric function that handles both v1 (is_unsafe) and v2 (verdict) outputs."""
        def guardrail_metric(example, pred, trace=None):
            # Determine expected outcome
            verdict_field = getattr(example, "verdict", None)
            if verdict_field is not None:
                expected_unsafe = str(verdict_field).strip().upper().startswith("UNSAFE")
            else:
                expected_unsafe = bool(example.is_unsafe)
            # Determine predicted outcome
            pred_verdict = getattr(pred, "verdict", None)
            if pred_verdict is not None:
                actual_unsafe = str(pred_verdict).strip().upper().startswith("UNSAFE")
            else:
                actual = getattr(pred, "is_unsafe", False)
                if isinstance(actual, str):
                    actual_unsafe = actual.lower() in ("true", "yes", "1")
                else:
                    actual_unsafe = bool(actual)
            return float(expected_unsafe == actual_unsafe)
        return guardrail_metric

    def _optimize_with_dspy(
        self,
        guardrail: dspy.Module,
        trainset: list[Example],
        valset: list[Example],
        metric: str,
    ) -> dict:
        """Optimize using DSPy BootstrapFewShot."""
        comprehensive_mode = self._is_comprehensive_mode(guardrail)
        v2_mode = self._is_v2_mode(guardrail)

        guardrail_metric = self._make_metric()

        dspy_trainset = self._to_dspy_examples(trainset, comprehensive_mode, v2_mode=v2_mode)
        student = self._get_student_program(guardrail, comprehensive_mode)
        if student is None:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)
        student = self._prepare_uncompiled_student(student)
        if student is None:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)

        # Use BootstrapFewShot
        try:
            from dspy.teleprompt import BootstrapFewShot

            optimizer = BootstrapFewShot(
                metric=guardrail_metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=8,
            )

            optimized = optimizer.compile(
                student,
                trainset=dspy_trainset,
            )
            self._apply_compiled_student(guardrail, optimized, comprehensive_mode)

            # Evaluate optimized performance
            score = self._evaluate(guardrail, valset, metric)

            return {
                "prompt": self._extract_prompt(guardrail),
                "score": score,
                "iterations": 1,
            }
        except Exception as e:
            print(f"DSPy optimization failed: {e}")
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)

    def _optimize_with_mipro(
        self,
        guardrail: dspy.Module,
        trainset: list[Example],
        valset: list[Example],
        metric: str,
    ) -> dict:
        """Optimize using DSPy MIPROv2 (prompt + demos)."""
        comprehensive_mode = self._is_comprehensive_mode(guardrail)
        v2_mode = self._is_v2_mode(guardrail)

        guardrail_metric = self._make_metric()

        dspy_trainset = self._to_dspy_examples(trainset, comprehensive_mode, v2_mode=v2_mode)
        dspy_valset = self._to_dspy_examples(valset, comprehensive_mode, v2_mode=v2_mode)
        student = self._get_student_program(guardrail, comprehensive_mode)
        if student is None:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)
        student = self._prepare_uncompiled_student(student)
        if student is None:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)

        if not dspy_trainset:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)

        minibatch_size = max(2, min(20, len(dspy_trainset)))
        eval_size = len(dspy_valset) if dspy_valset else len(dspy_trainset)
        minibatch = eval_size > 1
        if minibatch:
            minibatch_size = max(2, min(minibatch_size, eval_size))

        try:
            from dspy.teleprompt import MIPROv2

            num_candidates = max(2, min(self.max_iterations, 4))
            num_trials = max(2, min(self.max_iterations, 4))
            optimizer = MIPROv2(
                metric=guardrail_metric,
                auto=None,
                max_bootstrapped_demos=2,
                max_labeled_demos=4,
                num_candidates=num_candidates,
                num_threads=1,
                verbose=False,
            )

            optimized = optimizer.compile(
                student,
                trainset=dspy_trainset,
                valset=dspy_valset if dspy_valset else None,
                num_trials=num_trials,
                minibatch=minibatch,
                minibatch_size=minibatch_size,
                minibatch_full_eval_steps=2,
            )
            self._apply_compiled_student(guardrail, optimized, comprehensive_mode)

            score = self._evaluate(guardrail, valset, metric)
            return {
                "prompt": self._extract_prompt(guardrail),
                "score": score,
                "iterations": num_trials,
            }
        except Exception as e:
            print(f"MIPROv2 optimization failed: {e}, falling back to BootstrapFewShot")
            return self._optimize_with_dspy(guardrail, trainset, valset, metric)

    def _optimize_with_simba(
        self,
        guardrail: dspy.Module,
        trainset: list[Example],
        valset: list[Example],
        metric: str,
    ) -> dict:
        """Optimize using DSPy SIMBA for online reflective optimization."""
        comprehensive_mode = self._is_comprehensive_mode(guardrail)
        v2_mode = self._is_v2_mode(guardrail)

        _base_metric = self._make_metric()

        def guardrail_metric(example, pred):
            if pred is None:
                return 0.0
            return _base_metric(example, pred)

        dspy_trainset = self._to_dspy_examples(trainset, comprehensive_mode, v2_mode=v2_mode)
        student = self._get_student_program(guardrail, comprehensive_mode)
        if student is None:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)
        student = self._prepare_uncompiled_student(student)
        if student is None:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)
        if not dspy_trainset:
            return self._optimize_prompt_simple(guardrail, trainset, valset, metric)

        try:
            from dspy.teleprompt import SIMBA

            bsize = max(4, min(32, len(dspy_trainset)))
            max_steps = max(2, min(self.max_iterations, 8))
            num_candidates = max(2, min(self.max_iterations, 8))
            optimizer = SIMBA(
                metric=guardrail_metric,
                bsize=bsize,
                num_candidates=num_candidates,
                max_steps=max_steps,
                max_demos=4,
                prompt_model=getattr(dspy.settings, "lm", None),
                num_threads=1,
            )

            optimized = optimizer.compile(
                student,
                trainset=dspy_trainset,
                seed=9,
            )
            self._apply_compiled_student(guardrail, optimized, comprehensive_mode)

            score = self._evaluate(guardrail, valset, metric)
            return {
                "prompt": self._extract_prompt(guardrail),
                "score": score,
                "iterations": max_steps,
            }
        except Exception as e:
            print(f"SIMBA optimization failed: {e}, falling back to MIPROv2")
            return self._optimize_with_mipro(guardrail, trainset, valset, metric)

    @staticmethod
    def _is_comprehensive_mode(guardrail: dspy.Module) -> bool:
        return bool(
            getattr(guardrail, "comprehensive", False)
            and hasattr(guardrail, "comprehensive_classifier")
        )

    @staticmethod
    def _is_v2_mode(guardrail: dspy.Module) -> bool:
        return bool(getattr(guardrail, "use_v2", False))

    @staticmethod
    def _is_v3_mode(guardrail: dspy.Module) -> bool:
        return bool(getattr(guardrail, "use_v3", False))

    def _to_dspy_examples(
        self,
        examples: list[Example],
        comprehensive_mode: bool,
        v2_mode: bool = False,
        v3_mode: bool = False,
    ) -> list[dspy.Example]:
        # V3 mode: match SafetyClassifierV3 module inputs/outputs
        if v3_mode:
            return [
                dspy.Example(
                    text=e.text,
                    defense_hints="",
                    # Final outputs the metric will check
                    is_unsafe=e.is_unsafe,
                    verdict="UNSAFE" if e.is_unsafe else "SAFE",
                    threat_type=e.category if e.is_unsafe else "none",
                    confidence=1.0,
                ).with_inputs("text", "defense_hints")
                for e in examples
            ]

        # V2 mode: match ThreatAnalysisV2 signature
        if v2_mode:
            return [
                dspy.Example(
                    text=e.text,
                    defense_hints="",
                    verdict="UNSAFE" if e.is_unsafe else "SAFE",
                    threat_type=e.category if e.is_unsafe else "none",
                    confidence=1.0,
                ).with_inputs("text", "defense_hints")
                for e in examples
            ]

        if not comprehensive_mode:
            return [
                dspy.Example(
                    text=e.text,
                    category=e.category,
                    is_unsafe=e.is_unsafe,
                ).with_inputs("text", "category")
                for e in examples
            ]

        converted: list[dspy.Example] = []
        for e in examples:
            is_unsafe = bool(e.is_unsafe)
            categories = e.category if is_unsafe else "none"
            reason = "unsafe prompt injection or jailbreak attempt" if is_unsafe else "benign request"
            converted.append(
                dspy.Example(
                    text=e.text,
                    is_unsafe=is_unsafe,
                    confidence=1.0,
                    categories=categories,
                    reason=reason,
                ).with_inputs("text")
            )
        return converted

    @staticmethod
    def _get_student_program(guardrail: dspy.Module, comprehensive_mode: bool):
        # V3: return the full multi-step module (two named predictors)
        if getattr(guardrail, "use_v3", False):
            return getattr(guardrail, "v3_classifier", None)
        if comprehensive_mode:
            return getattr(guardrail, "comprehensive_classifier", None)
        if hasattr(guardrail, "get_classifier"):
            return guardrail.get_classifier()
        return getattr(guardrail, "classifier", None)

    @staticmethod
    def _prepare_uncompiled_student(student):
        """Return an uncompiled copy of student for DSPy teleprompters.

        DSPy teleprompters require an uncompiled program. In co-evolution loops,
        the guardrail's internal predictor may already be compiled from prior
        rounds, which triggers "Student must be uncompiled". Prefer reset_copy()
        when available, then fallback to deepcopy().
        """
        if student is None:
            return None
        try:
            if hasattr(student, "reset_copy"):
                return student.reset_copy()
        except Exception:
            pass
        try:
            if hasattr(student, "deepcopy"):
                return student.deepcopy()
        except Exception:
            pass
        return student

    @staticmethod
    def _apply_compiled_student(
        guardrail: dspy.Module,
        optimized_student,
        comprehensive_mode: bool,
    ) -> None:
        # V3 mode: replace the full multi-step classifier
        if getattr(guardrail, "use_v3", False):
            guardrail.v3_classifier = optimized_student
            return
        # V2 mode
        if getattr(guardrail, "use_v2", False):
            guardrail.v2_analyzer = optimized_student
            return
        if comprehensive_mode:
            guardrail.comprehensive_classifier = optimized_student
            return
        if hasattr(guardrail, "classifier"):
            guardrail.classifier = optimized_student

    def _optimize_prompt_simple(
        self,
        guardrail: dspy.Module,
        trainset: list[Example],
        valset: list[Example],
        metric: str,
    ) -> dict:
        """Simple prompt optimization (fallback when GEPA/complex optimizers unavailable)."""
        # Collect failure cases
        failures = self._collect_failures(guardrail, trainset)

        # Use LLM to reflect and improve the prompt
        original_prompt = self._extract_prompt(guardrail)

        if not failures:
            return {
                "prompt": original_prompt,
                "score": self._evaluate(guardrail, valset, metric),
                "iterations": 0,
            }

        # Build reflection prompt
        reflection_prompt = f"""You are optimizing a safety classifier prompt.

Current prompt:
{original_prompt}

Failure cases (the classifier got these wrong):
{json.dumps(failures[:10], indent=2, ensure_ascii=False)}

Analyze why the classifier failed and provide an improved prompt that would correctly classify these cases.
Focus on:
1. Adding specific patterns that were missed
2. Clarifying ambiguous instructions
3. Reducing false positives/negatives

Improved prompt:"""

        # Call LLM to generate an improved prompt
        try:
            lm = dspy.settings.lm
            response = lm(reflection_prompt)
            improved_prompt = response[0] if isinstance(response, list) else str(response)

            # Apply the improved prompt
            if self._is_comprehensive_mode(guardrail):
                guardrail.comprehensive_classifier.signature.__doc__ = improved_prompt
            elif hasattr(guardrail, 'classifier') and hasattr(guardrail.classifier, 'signature'):
                guardrail.classifier.signature.__doc__ = improved_prompt

            # Evaluate improvement
            new_score = self._evaluate(guardrail, valset, metric)

            return {
                "prompt": improved_prompt,
                "score": new_score,
                "iterations": 1,
                "failures": failures,
            }
        except Exception as e:
            print(f"Simple optimization failed: {e}")
            return {
                "prompt": original_prompt,
                "score": self._evaluate(guardrail, valset, metric),
                "iterations": 0,
            }

    def _collect_failures(
        self,
        guardrail: dspy.Module,
        examples: list[Example],
    ) -> list[dict]:
        """Collect failure cases."""
        failures = []
        comprehensive_mode = self._is_comprehensive_mode(guardrail)

        for ex in examples:
            try:
                result = self._predict(guardrail, ex, comprehensive_mode)
                predicted = result.is_unsafe
                if isinstance(predicted, str):
                    predicted = predicted.lower() in ('true', 'yes', '1')

                if predicted != ex.is_unsafe:
                    failures.append({
                        "text": ex.text,
                        "expected": ex.is_unsafe,
                        "predicted": predicted,
                        "type": "false_negative" if ex.is_unsafe else "false_positive",
                    })
            except Exception as e:
                failures.append({
                    "text": ex.text,
                    "error": str(e),
                })

        return failures

    def _evaluate(
        self,
        guardrail: dspy.Module,
        examples: list[Example],
        metric: str,
    ) -> float:
        """Evaluate guardrail performance."""
        tp = fp = tn = fn = 0
        comprehensive_mode = self._is_comprehensive_mode(guardrail)

        for ex in examples:
            try:
                result = self._predict(guardrail, ex, comprehensive_mode)
                predicted = result.is_unsafe
                if isinstance(predicted, str):
                    predicted = predicted.lower() in ('true', 'yes', '1')

                if predicted and ex.is_unsafe:
                    tp += 1
                elif predicted and not ex.is_unsafe:
                    fp += 1
                elif not predicted and not ex.is_unsafe:
                    tn += 1
                else:
                    fn += 1
            except Exception:
                fn += 1  # Error treated as missed detection (false negative)

        # Compute metrics
        total = len(examples) if examples else 1
        if metric == "accuracy":
            return (tp + tn) / total
        elif metric == "precision":
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric == "recall":
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        elif metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return 0

    def _extract_prompt(self, guardrail: dspy.Module) -> str:
        """Extract guardrail prompt."""
        if self._is_comprehensive_mode(guardrail):
            sig = guardrail.comprehensive_classifier.signature
            return sig.__doc__ or str(sig)
        # Try get_classifier() method first (preferred for LLMGuardrail)
        if hasattr(guardrail, 'get_classifier'):
            classifier = guardrail.get_classifier()
            if classifier and hasattr(classifier, 'signature'):
                sig = classifier.signature
                return sig.__doc__ or str(sig)
        # Fall back to direct attribute access
        if hasattr(guardrail, 'classifier'):
            sig = guardrail.classifier.signature
            return sig.__doc__ or str(sig)
        return str(guardrail)

    @staticmethod
    def _predict(guardrail: dspy.Module, ex: Example, comprehensive_mode: bool):
        if comprehensive_mode and hasattr(guardrail, "check_all"):
            return guardrail.check_all(ex.text)
        return guardrail(text=ex.text, category=ex.category)


# =============================================================================
# Adversarial Optimizer
# =============================================================================

class AdversarialOptimizer:
    """
    Adversarial Optimizer - Co-evolves attack and defense

    Red Team (attack) and Blue Team (defense) engage in adversarial learning,
    mutually driving improvement.

    Usage:
        from dspy_guardrails import AdversarialOptimizer, PromptInjectionAttacker, LLMGuardrail

        optimizer = AdversarialOptimizer(num_rounds=10)
        result = optimizer.optimize(
            attacker=PromptInjectionAttacker(),
            guardrail=LLMGuardrail(),
        )

        print(f"Final bypass rate: {result['final_bypass_rate']:.1%}")
        print(f"History: {result['history']}")
    """

    def __init__(
        self,
        task_lm: str = None,
        reflection_lm: str = None,
        num_rounds: int = 10,
        attacks_per_round: int = 10,
        use_gepa: bool = True,
    ):
        self.task_lm = task_lm
        self.reflection_lm = reflection_lm
        self.num_rounds = num_rounds
        self.attacks_per_round = attacks_per_round
        self.use_gepa = use_gepa

    def optimize(
        self,
        attacker: dspy.Module,
        guardrail: dspy.Module,
        safe_examples: list[str] = None,
    ) -> dict:
        """Adversarial optimization.

        Args:
            attacker: Attack module (generates malicious inputs)
            guardrail: Defense module (detects malicious inputs)
            safe_examples: Safe text examples (for false positive testing)

        Returns:
            Dict containing optimized attacker, guardrail, and history
        """
        safe_examples = safe_examples or [
            "Hello, how are you?",
            "What is the weather today?",
            "Can you help me with Python?",
            "Please explain how machine learning works.",
            "What's the capital of France?",
        ]

        history = []
        all_attacks = []  # Collect all attack samples for training

        for round_num in range(self.num_rounds):
            round_result = self._run_round(
                round_num=round_num,
                attacker=attacker,
                guardrail=guardrail,
                safe_examples=safe_examples,
                all_attacks=all_attacks,
            )

            history.append(round_result)

            print(f"Round {round_num + 1}/{self.num_rounds}: "
                  f"bypass_rate={round_result['bypass_rate']:.1%}, "
                  f"false_positives={round_result['false_positives']}, "
                  f"attacks_generated={round_result['total_attacks']}")

            # Evolve guardrail using GEPA (every 3 rounds)
            if self.use_gepa and (round_num + 1) % 3 == 0 and len(all_attacks) > 5:
                guardrail = self._evolve_guardrail(guardrail, all_attacks, safe_examples)

        return {
            "attacker": attacker,
            "guardrail": guardrail,
            "history": history,
            "final_bypass_rate": history[-1]["bypass_rate"] if history else 0,
            "total_attacks_collected": len(all_attacks),
        }

    def _run_round(
        self,
        round_num: int,
        attacker: dspy.Module,
        guardrail: dspy.Module,
        safe_examples: list[str],
        all_attacks: list[dict],
    ) -> dict:
        """Run one adversarial round."""
        # === Red Team: Generate attacks ===
        attacks = []
        for _i in range(self.attacks_per_round):
            try:
                attack = attacker(
                    target_behavior="bypass safety filter",
                    defense_description=str(guardrail),
                )
                attack_text = attack.prompt if hasattr(attack, 'prompt') else str(attack)
                attacks.append(attack_text)
            except Exception:
                pass

        # === Blue Team: Evaluate defense ===
        bypassed = []
        blocked = []

        for attack in attacks:
            try:
                result = guardrail(text=attack, category="injection")
                is_unsafe = result.is_unsafe
                if isinstance(is_unsafe, str):
                    is_unsafe = is_unsafe.lower() in ('true', 'yes', '1')

                if not is_unsafe:  # Not detected = attack succeeded
                    bypassed.append(attack)
                    all_attacks.append({
                        "text": attack,
                        "is_unsafe": True,
                        "category": "injection",
                        "round": round_num,
                        "bypassed": True,
                    })
                else:
                    blocked.append(attack)
                    all_attacks.append({
                        "text": attack,
                        "is_unsafe": True,
                        "category": "injection",
                        "round": round_num,
                        "bypassed": False,
                    })
            except Exception:
                bypassed.append(attack)  # Error treated as bypass

        # Test false positives
        false_positives = []
        for safe_text in safe_examples:
            try:
                result = guardrail(text=safe_text, category="injection")
                is_unsafe = result.is_unsafe
                if isinstance(is_unsafe, str):
                    is_unsafe = is_unsafe.lower() in ('true', 'yes', '1')

                if is_unsafe:  # False positive
                    false_positives.append(safe_text)
            except Exception:
                pass

        return {
            "round": round_num,
            "total_attacks": len(attacks),
            "bypassed": len(bypassed),
            "blocked": len(blocked),
            "false_positives": len(false_positives),
            "bypass_rate": len(bypassed) / len(attacks) if attacks else 0,
            "bypassed_samples": bypassed[:3],  # Keep top 3 bypass samples
        }

    def _evolve_guardrail(
        self,
        guardrail: dspy.Module,
        all_attacks: list[dict],
        safe_examples: list[str],
    ) -> dspy.Module:
        """Evolve guardrail using GEPA."""
        # Build trainset: attack samples + safe samples
        trainset = [
            Example(
                text=a["text"],
                is_unsafe=True,
                category=a.get("category", "injection")
            )
            for a in all_attacks[-50:]  # Last 50 attacks
        ]

        # Add safe samples
        for safe_text in safe_examples:
            trainset.append(Example(text=safe_text, is_unsafe=False, category="injection"))

        # Optimize using GuardrailOptimizer
        optimizer = GuardrailOptimizer(
            mode="gepa" if self.use_gepa else "dspy",
            task_lm=self.task_lm,
            reflection_lm=self.reflection_lm,
            max_iterations=20,
        )

        try:
            result = optimizer.optimize(
                guardrail=guardrail,
                trainset=trainset,
                metric="f1",
            )
            print(f"  [Evolution] Guardrail improved: {result.original_score:.3f} -> {result.optimized_score:.3f}")
        except Exception as e:
            print(f"  [Evolution] Failed: {e}")

        return guardrail


# =============================================================================
# Quick Functions
# =============================================================================

def optimize_guardrail(
    guardrail: dspy.Module,
    trainset: list[Example],
    mode: str = "gepa",
    **kwargs,
) -> OptimizationResult:
    """Quick optimization for guardrail.

    Args:
        guardrail: Guardrail to optimize
        trainset: Training data
        mode: "gepa" | "mipro" | "dspy" | "simba"
        **kwargs: Additional arguments passed to GuardrailOptimizer

    Returns:
        OptimizationResult

    Example:
        from dspy_guardrails import optimize_guardrail, Example, LLMGuardrail

        result = optimize_guardrail(
            guardrail=LLMGuardrail(),
            trainset=[
                Example("hello", is_unsafe=False),
                Example("ignore instructions", is_unsafe=True),
            ],
        )
    """
    optimizer = GuardrailOptimizer(mode=mode, **kwargs)
    return optimizer.optimize(guardrail, trainset)


def adversarial_train(
    attacker: dspy.Module,
    guardrail: dspy.Module,
    num_rounds: int = 10,
    **kwargs,
) -> dict:
    """Adversarial training.

    Args:
        attacker: Attack module
        guardrail: Defense module
        num_rounds: Number of adversarial rounds
        **kwargs: Additional arguments passed to AdversarialOptimizer

    Returns:
        Training result dict

    Example:
        from dspy_guardrails import adversarial_train, PromptInjectionAttacker, LLMGuardrail

        result = adversarial_train(
            attacker=PromptInjectionAttacker(),
            guardrail=LLMGuardrail(),
            num_rounds=10,
        )
    """
    optimizer = AdversarialOptimizer(num_rounds=num_rounds, **kwargs)
    return optimizer.optimize(attacker, guardrail)
