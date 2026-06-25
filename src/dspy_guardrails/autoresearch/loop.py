"""Research loop orchestrator for the autoresearch system.

This module provides the infrastructure for managing the autoresearch process.
It is NOT the agent itself — it handles:
  - State tracking (current best, iteration count, history)
  - Loading and evaluating algorithms written by the agent
  - Keep/discard decisions based on metric comparison
  - Updating the agent memory log

The actual algorithm writing is done by Claude Code (T10).  This loop
provides the methods the CLI/skill calls one at a time.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .harness import AttackEvalResult, DefenseEvalResult, ResearchHarness
from .memory import AgentMemory, IterationRecord, init_log
from .registry import load_algorithm

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# ResearchConfig
# ---------------------------------------------------------------------------


@dataclass
class ResearchConfig:
    """Configuration for a research loop run."""

    methods_dir: Path = Path("autoresearch/methods")
    log_path: Path = Path("autoresearch/AGENT_LOG.md")
    results_tsv: Path = Path("autoresearch/results.tsv")
    results_dir: Path = Path("autoresearch/results")
    target_url: str = "http://localhost:18921/v1"
    target_model: str = "Huihui-Qwen3.5-27B-abliterated"
    judge_lm: Any = None
    query_budget: int = 20
    n_behaviors: int = 20
    n_benign: int = 15
    seed: int = 42

    def resolved(self, key: str) -> Path:
        """Return the absolute path for a relative config path field."""
        return _PROJECT_ROOT / getattr(self, key)


# ---------------------------------------------------------------------------
# AttackResearchLoop
# ---------------------------------------------------------------------------


class AttackResearchLoop:
    """Orchestrator for iterative attack algorithm research.

    Tracks the current best ASR, loads candidate algorithms from version
    directories, evaluates them with the ResearchHarness, and records
    keep/discard decisions in AgentMemory.

    The loop does NOT run autonomously — callers invoke methods one at a time.
    """

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig()
        self._memory: AgentMemory | None = None
        self._harness: ResearchHarness | None = None
        self._best_name: str = "none"
        self._best_asr: float = 0.0
        self._iteration: int = 0
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, goal: str = "Discover novel attack algorithms") -> None:
        """Initialize directories, AGENT_LOG.md, and run baseline evaluation.

        Creates the methods_dir, results_dir, and log_path directories.
        Initialises AgentMemory and ResearchHarness.  If an existing log is
        present, state is resumed from it.
        """
        print(f"[AttackLoop] Setting up research loop. Goal: {goal}")

        # Resolve and create directories
        for dir_key in ("methods_dir", "results_dir"):
            d = self.config.resolved(dir_key)
            d.mkdir(parents=True, exist_ok=True)
            print(f"[AttackLoop]  Directory ready: {d}")

        log_abs = self.config.resolved("log_path")
        log_abs.parent.mkdir(parents=True, exist_ok=True)

        tsv_abs = self.config.resolved("results_tsv")
        tsv_abs.parent.mkdir(parents=True, exist_ok=True)

        # Memory
        self._memory = AgentMemory(
            log_path=log_abs,
            results_tsv_path=tsv_abs,
        )

        # Harness
        self._harness = ResearchHarness(
            target_url=self.config.target_url,
            target_model=self.config.target_model,
            judge_lm=self.config.judge_lm,
            query_budget=self.config.query_budget,
            n_behaviors=self.config.n_behaviors,
            n_benign=self.config.n_benign,
            seed=self.config.seed,
        )

        # Resume state from existing log, or create fresh log
        history = self._memory.load_history()
        if history:
            self._iteration = self._memory.get_next_version() - 1
            attacks_kept = [r for r in history if r.kind == "attack" and r.status == "keep"]
            if attacks_kept:
                best = max(attacks_kept, key=lambda r: r.results.get("asr", 0.0))
                self._best_name = best.algorithm_name
                self._best_asr = best.results.get("asr", 0.0)
            print(
                f"[AttackLoop] Resumed from existing log. "
                f"Iterations so far: {self._iteration}, "
                f"best ASR: {self._best_asr:.3f} ({self._best_name})"
            )
        else:
            init_log(log_abs, goal)
            print("[AttackLoop] Fresh AGENT_LOG.md created.")

        # Scan for any seed algorithms already present in methods_dir
        self._load_seed_algorithms()

    def evaluate_version(self, version_dir: Path) -> AttackEvalResult:
        """Load an attack algorithm from *version_dir* and evaluate it.

        Args:
            version_dir: Directory (or .py file path) containing the algorithm.

        Returns:
            AttackEvalResult from the ResearchHarness.

        Raises:
            RuntimeError: if setup() has not been called.
        """
        self._require_setup()
        version_dir = Path(version_dir)

        # Accept either a directory (look for algorithm.py) or a direct .py file
        algo_file = self._resolve_algo_file(version_dir)
        print(f"[AttackLoop] Loading algorithm from: {algo_file}")

        algo_cls = load_algorithm(algo_file)
        algo_instance = algo_cls()

        print(f"[AttackLoop] Evaluating: {algo_cls.algorithm_name!r} …")
        result = self._harness.evaluate_attack(algo_instance)  # type: ignore[union-attr]
        return result

    def record_result(
        self,
        version: int,
        algorithm_name: str,
        result: AttackEvalResult,
        hypothesis: str = "",
        analysis: str = "",
    ) -> str:
        """Compare result with current best, decide keep/discard, update memory.

        Args:
            version:        Iteration/version number.
            algorithm_name: Name of the evaluated algorithm.
            result:         AttackEvalResult returned by evaluate_version().
            hypothesis:     Agent-provided hypothesis for this iteration.
            analysis:       Agent-provided analysis of the result.

        Returns:
            Status string: "keep", "discard", or "crash".
        """
        self._require_setup()

        new_asr = result.asr
        if new_asr > self._best_asr:
            status = "keep"
            self._best_asr = new_asr
            self._best_name = algorithm_name
            print(
                f"[AttackLoop] KEEP — new best ASR {new_asr:.3f} "
                f"(was {self._best_asr:.3f})"
            )
        else:
            status = "discard"
            print(
                f"[AttackLoop] DISCARD — ASR {new_asr:.3f} <= "
                f"best {self._best_asr:.3f}"
            )

        self._iteration = version
        self._history.append(
            {"version": version, "algorithm": algorithm_name, "asr": new_asr, "status": status}
        )

        record = IterationRecord(
            iteration=version,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            kind="attack",
            algorithm_name=algorithm_name,
            algorithm_path=str(self.config.resolved("methods_dir") / f"v{version}"),
            hypothesis=hypothesis,
            results={
                "asr": result.asr,
                "strongreject": result.mean_score,
                "queries": result.total_queries,
            },
            status=status,
            analysis=analysis,
        )
        self._memory.append_iteration(record)  # type: ignore[union-attr]
        return status

    def get_current_best(self) -> tuple[str, float]:
        """Return (algorithm_name, best_asr) for the current best attack."""
        return self._best_name, self._best_asr

    def get_status(self) -> dict:
        """Return a summary dict of current loop state."""
        return {
            "kind": "attack",
            "iterations": self._iteration,
            "best_algorithm": self._best_name,
            "best_asr": self._best_asr,
            "history": list(self._history),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_setup(self) -> None:
        if self._memory is None or self._harness is None:
            raise RuntimeError(
                "AttackResearchLoop.setup() must be called before using the loop."
            )

    def _resolve_algo_file(self, path: Path) -> Path:
        """Return a .py file path from a directory or direct file path."""
        if path.is_file() and path.suffix == ".py":
            return path
        if path.is_dir():
            candidates = list(path.glob("algorithm.py")) + list(path.glob("*.py"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
            f"No Python algorithm file found at {path}. "
            "Expected a .py file or a directory containing algorithm.py."
        )

    def _load_seed_algorithms(self) -> None:
        """Scan methods_dir for pre-existing algorithm files and register them."""
        methods_abs = self.config.resolved("methods_dir")
        py_files = list(methods_abs.rglob("*.py"))
        if not py_files:
            print("[AttackLoop] No seed algorithms found.")
            return
        loaded = 0
        for f in py_files:
            try:
                load_algorithm(f)
                loaded += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"[AttackLoop] Loaded {loaded}/{len(py_files)} seed algorithm(s).")


# ---------------------------------------------------------------------------
# DefenseResearchLoop
# ---------------------------------------------------------------------------


class DefenseResearchLoop:
    """Orchestrator for iterative defense algorithm research.

    Mirrors AttackResearchLoop but optimises F1 instead of ASR.
    Evaluation uses harness.evaluate_defense() with a fixed attack suite.
    """

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig()
        self._memory: AgentMemory | None = None
        self._harness: ResearchHarness | None = None
        self._best_name: str = "none"
        self._best_f1: float = 0.0
        self._iteration: int = 0
        self._history: list[dict] = []
        self._attack_suite: list[Any] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, goal: str = "Discover novel defense algorithms") -> None:
        """Initialize directories, AGENT_LOG.md, and prepare evaluation infrastructure."""
        print(f"[DefenseLoop] Setting up research loop. Goal: {goal}")

        for dir_key in ("methods_dir", "results_dir"):
            d = self.config.resolved(dir_key)
            d.mkdir(parents=True, exist_ok=True)
            print(f"[DefenseLoop]  Directory ready: {d}")

        log_abs = self.config.resolved("log_path")
        log_abs.parent.mkdir(parents=True, exist_ok=True)

        tsv_abs = self.config.resolved("results_tsv")
        tsv_abs.parent.mkdir(parents=True, exist_ok=True)

        self._memory = AgentMemory(
            log_path=log_abs,
            results_tsv_path=tsv_abs,
        )
        self._harness = ResearchHarness(
            target_url=self.config.target_url,
            target_model=self.config.target_model,
            judge_lm=self.config.judge_lm,
            query_budget=self.config.query_budget,
            n_behaviors=self.config.n_behaviors,
            n_benign=self.config.n_benign,
            seed=self.config.seed,
        )

        history = self._memory.load_history()
        if history:
            self._iteration = self._memory.get_next_version() - 1
            defenses_kept = [r for r in history if r.kind == "defense" and r.status == "keep"]
            if defenses_kept:
                best = max(defenses_kept, key=lambda r: r.results.get("f1", 0.0))
                self._best_name = best.algorithm_name
                self._best_f1 = best.results.get("f1", 0.0)
            print(
                f"[DefenseLoop] Resumed from existing log. "
                f"Iterations so far: {self._iteration}, "
                f"best F1: {self._best_f1:.3f} ({self._best_name})"
            )
        else:
            init_log(log_abs, goal)
            print("[DefenseLoop] Fresh AGENT_LOG.md created.")

        self._load_seed_algorithms()

    def set_attack_suite(self, attack_algorithms: list[Any]) -> None:
        """Set the fixed attack suite used for defense evaluation.

        Args:
            attack_algorithms: List of AttackAlgorithm instances.
        """
        self._attack_suite = list(attack_algorithms)
        print(f"[DefenseLoop] Attack suite set: {len(self._attack_suite)} algorithm(s).")

    def evaluate_version(self, version_dir: Path) -> DefenseEvalResult:
        """Load a defense algorithm from *version_dir* and evaluate it.

        Uses the fixed attack suite set via set_attack_suite().

        Args:
            version_dir: Directory (or .py file path) containing the algorithm.

        Returns:
            DefenseEvalResult from the ResearchHarness.
        """
        self._require_setup()
        version_dir = Path(version_dir)

        algo_file = self._resolve_algo_file(version_dir)
        print(f"[DefenseLoop] Loading defense algorithm from: {algo_file}")

        algo_cls = load_algorithm(algo_file)
        algo_instance = algo_cls()

        print(f"[DefenseLoop] Evaluating: {algo_cls.algorithm_name!r} …")
        result = self._harness.evaluate_defense(  # type: ignore[union-attr]
            algo_instance, self._attack_suite
        )
        return result

    def record_result(
        self,
        version: int,
        algorithm_name: str,
        result: DefenseEvalResult,
        hypothesis: str = "",
        analysis: str = "",
    ) -> str:
        """Compare result with current best, decide keep/discard, update memory.

        Args:
            version:        Iteration/version number.
            algorithm_name: Name of the evaluated algorithm.
            result:         DefenseEvalResult returned by evaluate_version().
            hypothesis:     Agent-provided hypothesis for this iteration.
            analysis:       Agent-provided analysis of the result.

        Returns:
            Status string: "keep", "discard", or "crash".
        """
        self._require_setup()

        new_f1 = result.f1
        if new_f1 > self._best_f1:
            status = "keep"
            self._best_f1 = new_f1
            self._best_name = algorithm_name
            print(
                f"[DefenseLoop] KEEP — new best F1 {new_f1:.3f} "
                f"(was {self._best_f1:.3f})"
            )
        else:
            status = "discard"
            print(
                f"[DefenseLoop] DISCARD — F1 {new_f1:.3f} <= "
                f"best {self._best_f1:.3f}"
            )

        self._iteration = version
        self._history.append(
            {"version": version, "algorithm": algorithm_name, "f1": new_f1, "status": status}
        )

        record = IterationRecord(
            iteration=version,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            kind="defense",
            algorithm_name=algorithm_name,
            algorithm_path=str(self.config.resolved("methods_dir") / f"v{version}"),
            hypothesis=hypothesis,
            results={
                "f1": result.f1,
                "block_rate": result.block_rate,
                "fpr": result.fpr,
            },
            status=status,
            analysis=analysis,
        )
        self._memory.append_iteration(record)  # type: ignore[union-attr]
        return status

    def get_current_best(self) -> tuple[str, float]:
        """Return (algorithm_name, best_f1) for the current best defense."""
        return self._best_name, self._best_f1

    def get_status(self) -> dict:
        """Return a summary dict of current loop state."""
        return {
            "kind": "defense",
            "iterations": self._iteration,
            "best_algorithm": self._best_name,
            "best_f1": self._best_f1,
            "history": list(self._history),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_setup(self) -> None:
        if self._memory is None or self._harness is None:
            raise RuntimeError(
                "DefenseResearchLoop.setup() must be called before using the loop."
            )

    def _resolve_algo_file(self, path: Path) -> Path:
        if path.is_file() and path.suffix == ".py":
            return path
        if path.is_dir():
            candidates = list(path.glob("algorithm.py")) + list(path.glob("*.py"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(
            f"No Python algorithm file found at {path}. "
            "Expected a .py file or a directory containing algorithm.py."
        )

    def _load_seed_algorithms(self) -> None:
        methods_abs = self.config.resolved("methods_dir")
        py_files = list(methods_abs.rglob("*.py"))
        if not py_files:
            print("[DefenseLoop] No seed algorithms found.")
            return
        loaded = 0
        for f in py_files:
            try:
                load_algorithm(f)
                loaded += 1
            except Exception:  # noqa: BLE001
                pass
        print(f"[DefenseLoop] Loaded {loaded}/{len(py_files)} seed algorithm(s).")


# ---------------------------------------------------------------------------
# CoEvolutionLoop
# ---------------------------------------------------------------------------


class CoEvolutionLoop:
    """Co-evolution orchestrator that alternates attack and defense research cycles.

    Composes an AttackResearchLoop and a DefenseResearchLoop.  Tracks which
    phase (attack vs. defense) we are in and how many co-evolution cycles
    have completed.

    Usage::

        loop = CoEvolutionLoop(config)
        loop.setup("Find best attack/defense pair")
        # CLI / agent calls loop.attack_loop.evaluate_version(...) etc.
    """

    def __init__(
        self,
        config: ResearchConfig | None = None,
        attack_rounds_per_cycle: int = 3,
        defense_rounds_per_cycle: int = 3,
    ) -> None:
        self.config = config or ResearchConfig()
        self.attack_rounds_per_cycle = attack_rounds_per_cycle
        self.defense_rounds_per_cycle = defense_rounds_per_cycle

        self.attack_loop = AttackResearchLoop(self.config)
        self.defense_loop = DefenseResearchLoop(self.config)

        self._phase: str = "attack"  # "attack" or "defense"
        self._cycle: int = 0
        self._attack_rounds_this_cycle: int = 0
        self._defense_rounds_this_cycle: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, goal: str = "Co-evolve attack and defense algorithms") -> None:
        """Initialize both attack and defense sub-loops.

        Args:
            goal: High-level research goal written to AGENT_LOG.md.
        """
        print(f"[CoEvoLoop] Setting up co-evolution loop. Goal: {goal}")
        self.attack_loop.setup(goal=goal)
        self.defense_loop.setup(goal=goal)
        print(
            f"[CoEvoLoop] Ready. "
            f"Attack rounds/cycle={self.attack_rounds_per_cycle}, "
            f"Defense rounds/cycle={self.defense_rounds_per_cycle}"
        )

    def advance_phase(self) -> str:
        """Advance internal phase tracker and return the current phase name.

        Should be called after each successful round to let the loop decide
        whether to switch from attack to defense or start a new cycle.

        Returns:
            Current phase: "attack" or "defense".
        """
        if self._phase == "attack":
            self._attack_rounds_this_cycle += 1
            if self._attack_rounds_this_cycle >= self.attack_rounds_per_cycle:
                self._phase = "defense"
                self._attack_rounds_this_cycle = 0
                print(
                    f"[CoEvoLoop] Switching to defense phase "
                    f"(cycle {self._cycle + 1})."
                )
        else:
            self._defense_rounds_this_cycle += 1
            if self._defense_rounds_this_cycle >= self.defense_rounds_per_cycle:
                self._cycle += 1
                self._phase = "attack"
                self._defense_rounds_this_cycle = 0
                print(
                    f"[CoEvoLoop] Cycle {self._cycle} complete. "
                    "Switching back to attack phase."
                )
        return self._phase

    def get_status(self) -> dict:
        """Return combined status from both sub-loops plus co-evolution metadata."""
        attack_status = self.attack_loop.get_status()
        defense_status = self.defense_loop.get_status()
        return {
            "phase": self._phase,
            "cycle": self._cycle,
            "attack_rounds_this_cycle": self._attack_rounds_this_cycle,
            "defense_rounds_this_cycle": self._defense_rounds_this_cycle,
            "attack": attack_status,
            "defense": defense_status,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers with crash protection
    # ------------------------------------------------------------------

    def safe_evaluate_attack(self, version_dir: Path) -> AttackEvalResult | None:
        """Evaluate an attack version with crash protection.

        Returns None and prints a traceback if the evaluation raises an exception.
        """
        try:
            return self.attack_loop.evaluate_version(version_dir)
        except Exception:  # noqa: BLE001
            print(f"[CoEvoLoop] Attack evaluation crashed:\n{traceback.format_exc()}")
            return None

    def safe_evaluate_defense(self, version_dir: Path) -> DefenseEvalResult | None:
        """Evaluate a defense version with crash protection.

        Returns None and prints a traceback if the evaluation raises an exception.
        """
        try:
            return self.defense_loop.evaluate_version(version_dir)
        except Exception:  # noqa: BLE001
            print(f"[CoEvoLoop] Defense evaluation crashed:\n{traceback.format_exc()}")
            return None
