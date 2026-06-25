"""Autoresearch — autonomous algorithm discovery for LLM security."""
from .registry import (
    AlgorithmInfo,
    AttackAlgorithm,
    DefenseAlgorithm,
    get_algorithm,
    list_algorithms,
    load_algorithm,
)
from .harness import AttackEvalResult, DefenseEvalResult, ResearchHarness
from .memory import AgentMemory, IterationRecord, init_log

__all__ = [
    "AlgorithmInfo", "AttackAlgorithm", "DefenseAlgorithm",
    "get_algorithm", "list_algorithms", "load_algorithm",
    "AttackEvalResult", "DefenseEvalResult", "ResearchHarness",
    "AgentMemory", "IterationRecord", "init_log",
]
