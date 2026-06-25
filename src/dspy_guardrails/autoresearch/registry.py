"""Algorithm registry with dynamic loading for the autoresearch system.

Algorithms self-register via __init_subclass__ when their module is imported.
Use load_algorithm() to import a file and register its algorithm class,
list_algorithms() to enumerate registered algorithms, and get_algorithm()
to retrieve a class by name.
"""

from __future__ import annotations

import importlib.util
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

_ATTACK_REGISTRY: dict[str, type] = {}
_DEFENSE_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# AlgorithmInfo
# ---------------------------------------------------------------------------


@dataclass
class AlgorithmInfo:
    """Metadata snapshot for a registered algorithm."""

    name: str
    kind: str  # "attack" or "defense"
    version: int
    description: str
    parent_version: int | None
    module_path: str  # dotted module name or file path string
    cls: type


# ---------------------------------------------------------------------------
# AttackAlgorithm base class
# ---------------------------------------------------------------------------


class AttackAlgorithm(ABC):
    """Base class for attack algorithms.  Subclasses auto-register on definition.

    Subclass and set ``algorithm_name`` to register:

        class MyAttack(AttackAlgorithm):
            algorithm_name = "my_attack"
            version = 1
            description = "Does something clever."

            def create_attack(self, target, attacker_lm=None, judge_fn=None, **kwargs):
                return MyAdaptiveAttack(target, attacker_lm=attacker_lm, ...)
    """

    algorithm_name: ClassVar[str | None] = None
    version: ClassVar[int] = 0
    description: ClassVar[str] = ""
    parent_version: ClassVar[int | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "algorithm_name" in cls.__dict__ and cls.algorithm_name is not None:
            _ATTACK_REGISTRY[cls.algorithm_name] = cls

    @abstractmethod
    def create_attack(
        self,
        target: Any,
        attacker_lm: Any = None,
        judge_fn: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Return a BaseAdaptiveAttack-compatible object configured for *target*."""

    @classmethod
    def metadata(cls) -> AlgorithmInfo:
        """Return an AlgorithmInfo snapshot for this class."""
        module_path = inspect.getfile(cls) if inspect.isclass(cls) else ""
        return AlgorithmInfo(
            name=cls.algorithm_name or cls.__name__,
            kind="attack",
            version=cls.version,
            description=cls.description,
            parent_version=cls.parent_version,
            module_path=module_path,
            cls=cls,
        )


# ---------------------------------------------------------------------------
# DefenseAlgorithm base class
# ---------------------------------------------------------------------------


class DefenseAlgorithm(ABC):
    """Base class for defense algorithms.  Subclasses auto-register on definition.

    Subclass and set ``algorithm_name`` to register:

        class MyDefense(DefenseAlgorithm):
            algorithm_name = "my_defense"
            version = 1
            description = "Blocks things."

            def create_target(self, base_lm=None, **kwargs):
                return MyEvolvableTarget(base_lm=base_lm, ...)
    """

    algorithm_name: ClassVar[str | None] = None
    version: ClassVar[int] = 0
    description: ClassVar[str] = ""
    parent_version: ClassVar[int | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "algorithm_name" in cls.__dict__ and cls.algorithm_name is not None:
            _DEFENSE_REGISTRY[cls.algorithm_name] = cls

    @abstractmethod
    def create_target(self, base_lm: Any = None, **kwargs: Any) -> Any:
        """Return an EvolvableTarget-compatible object."""

    @classmethod
    def metadata(cls) -> AlgorithmInfo:
        """Return an AlgorithmInfo snapshot for this class."""
        module_path = inspect.getfile(cls) if inspect.isclass(cls) else ""
        return AlgorithmInfo(
            name=cls.algorithm_name or cls.__name__,
            kind="defense",
            version=cls.version,
            description=cls.description,
            parent_version=cls.parent_version,
            module_path=module_path,
            cls=cls,
        )


# ---------------------------------------------------------------------------
# Dynamic loading
# ---------------------------------------------------------------------------


def load_algorithm(path: Path) -> type:
    """Dynamically import *path* and return the first AttackAlgorithm or DefenseAlgorithm
    subclass found in the module.  As a side-effect the class is registered in the
    appropriate registry via __init_subclass__.

    Raises:
        ValueError: if no AttackAlgorithm or DefenseAlgorithm subclass is found.
        FileNotFoundError: if *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm file not found: {path}")

    module_name = f"_autoresearch_dyn.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Walk module members to find the first concrete algorithm subclass.
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if obj in (AttackAlgorithm, DefenseAlgorithm):
            continue
        if issubclass(obj, (AttackAlgorithm, DefenseAlgorithm)):
            return obj

    raise ValueError(
        f"No AttackAlgorithm or DefenseAlgorithm subclass found in {path}"
    )


# ---------------------------------------------------------------------------
# Registry queries
# ---------------------------------------------------------------------------


def list_algorithms(kind: str = "all") -> list[AlgorithmInfo]:
    """Return AlgorithmInfo for all registered algorithms.

    Args:
        kind: "attack", "defense", or "all" (default).

    Raises:
        ValueError: if *kind* is not one of the accepted values.
    """
    if kind not in {"attack", "defense", "all"}:
        raise ValueError(f"kind must be 'attack', 'defense', or 'all'; got {kind!r}")

    results: list[AlgorithmInfo] = []
    if kind in {"attack", "all"}:
        for cls in _ATTACK_REGISTRY.values():
            results.append(cls.metadata())  # type: ignore[attr-defined]
    if kind in {"defense", "all"}:
        for cls in _DEFENSE_REGISTRY.values():
            results.append(cls.metadata())  # type: ignore[attr-defined]
    return results


def get_algorithm(name: str) -> type:
    """Return the algorithm class registered under *name*.

    Searches attack registry first, then defense registry.

    Raises:
        KeyError: if no algorithm with that name is registered.
    """
    if name in _ATTACK_REGISTRY:
        return _ATTACK_REGISTRY[name]
    if name in _DEFENSE_REGISTRY:
        return _DEFENSE_REGISTRY[name]
    raise KeyError(
        f"No algorithm named {name!r} found in attack or defense registries. "
        f"Registered attacks: {list(_ATTACK_REGISTRY)}. "
        f"Registered defenses: {list(_DEFENSE_REGISTRY)}."
    )
