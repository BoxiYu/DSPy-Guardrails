"""Shared model configuration for experiment scripts.

Provides a unified model registry and LM factory for all experiments.
All models except Kimi K2.5 use OpenRouter API.

Usage:
    from model_config import configure_lms, list_models, MODEL_REGISTRY

    # List available models
    list_models()

    # Configure with defaults (Kimi defender, DeepSeek attacker)
    attacker_lm, defender_lm = configure_lms()

    # Configure with specific models
    attacker_lm, defender_lm = configure_lms(
        defender="gpt-4o", attacker="claude-3.5-haiku"
    )

    # Single model (defender only)
    defender_lm = configure_lms(defender="llama-3.1-8b", attacker=None)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    """Model configuration entry."""
    name: str              # Short name for CLI
    openrouter_id: str     # OpenRouter model ID
    display_name: str      # Human-readable name
    api_provider: str      # "openrouter" or "moonshot"
    roles: list[str]       # Supported roles: "defender", "attacker", "judge"
    notes: str = ""


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelInfo] = {
    # --- Original models ---
    "kimi-k2.5": ModelInfo(
        name="kimi-k2.5",
        openrouter_id="openai/kimi-k2.5",
        display_name="Kimi K2.5 (Moonshot)",
        api_provider="moonshot",
        roles=["defender", "judge"],
        notes="Original defender, uses Moonshot API",
    ),
    "deepseek-v3.2": ModelInfo(
        name="deepseek-v3.2",
        openrouter_id="openrouter/deepseek/deepseek-v3.2",
        display_name="DeepSeek V3.2",
        api_provider="openrouter",
        roles=["attacker"],
        notes="Original cross-model attacker",
    ),

    # --- New models (all via OpenRouter) ---
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        openrouter_id="openrouter/openai/gpt-4o",
        display_name="GPT-4o (OpenAI)",
        api_provider="openrouter",
        roles=["defender", "judge"],
        notes="Closed-source SOTA, paper benchmark baseline",
    ),
    "gpt-4o-mini": ModelInfo(
        name="gpt-4o-mini",
        openrouter_id="openai/gpt-4o-mini",
        display_name="GPT-4o Mini (OpenAI)",
        api_provider="openrouter",
        roles=["defender", "attacker", "judge"],
        notes="Low-cost closed-source baseline",
    ),
    "llama-3.1-8b": ModelInfo(
        name="llama-3.1-8b",
        openrouter_id="openrouter/meta-llama/llama-3.1-8b-instruct",
        display_name="Llama 3.1 8B Instruct",
        api_provider="openrouter",
        roles=["defender"],
        notes="Open-source mainstream, low cost",
    ),
    "mistral-7b": ModelInfo(
        name="mistral-7b",
        openrouter_id="openrouter/mistralai/mistral-7b-instruct",
        display_name="Mistral 7B Instruct",
        api_provider="openrouter",
        roles=["defender"],
        notes="Lightweight efficient, paper benchmark",
    ),
    "claude-3.5-haiku": ModelInfo(
        name="claude-3.5-haiku",
        openrouter_id="openrouter/anthropic/claude-3.5-haiku",
        display_name="Claude 3.5 Haiku (Anthropic)",
        api_provider="openrouter",
        roles=["defender", "attacker"],
        notes="Closed-source comparison, cost-effective",
    ),
    "llama-3.3-70b": ModelInfo(
        name="llama-3.3-70b",
        openrouter_id="openrouter/meta-llama/llama-3.3-70b-instruct",
        display_name="Llama 3.3 70B Instruct",
        api_provider="openrouter",
        roles=["defender", "attacker"],
        notes="Open-source 70B, excellent cost-performance ratio",
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        openrouter_id="openrouter/openai/gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo (OpenAI)",
        api_provider="openrouter",
        roles=["defender", "attacker"],
        notes="Legacy baseline, cheap",
    ),
    "gemma-3-27b": ModelInfo(
        name="gemma-3-27b",
        openrouter_id="openrouter/google/gemma-3-27b-it",
        display_name="Gemma 3 27B IT (Google)",
        api_provider="openrouter",
        roles=["defender", "attacker"],
        notes="Google open-source, has free tier",
    ),
    "claude-3-haiku": ModelInfo(
        name="claude-3-haiku",
        openrouter_id="openrouter/anthropic/claude-3-haiku",
        display_name="Claude 3 Haiku (Anthropic)",
        api_provider="openrouter",
        roles=["defender", "attacker"],
        notes="Cheapest Claude, $0.25/$1.25 per M",
    ),

    # --- Local ollama models (real fine-tuned) ---
    "llamaguard-local": ModelInfo(
        name="llamaguard-local",
        openrouter_id="ollama_chat/llama-guard3:8b",
        display_name="LlamaGuard 3 8B (Local)",
        api_provider="ollama",
        roles=["defender"],
        notes="Real fine-tuned model via local ollama",
    ),
    "shieldgemma-local": ModelInfo(
        name="shieldgemma-local",
        openrouter_id="ollama_chat/shieldgemma:2b",
        display_name="ShieldGemma 2B (Local)",
        api_provider="ollama",
        roles=["defender"],
        notes="Real fine-tuned model via local ollama",
    ),

    # --- Local vLLM models ---
    "qwen3.5-27b-local": ModelInfo(
        name="qwen3.5-27b-local",
        openrouter_id="openai/Huihui-Qwen3.5-27B-abliterated",
        display_name="Qwen3.5-27B Abliterated (Local vLLM)",
        api_provider="vllm",
        roles=["defender", "attacker"],
        notes="Local vLLM on 2x RTX 5090, port 18921",
    ),

    # --- Judge models ---
    "qwen3-235b-judge": ModelInfo(
        name="qwen3-235b-judge",
        openrouter_id="openrouter/qwen/qwen3-235b-a22b-2507",
        display_name="Qwen3-235B MoE (Judge)",
        api_provider="openrouter",
        roles=["judge"],
        notes="Primary judge model, $0.07/$0.10 per M tokens",
    ),
    "gemini-flash-lite-judge": ModelInfo(
        name="gemini-flash-lite-judge",
        openrouter_id="openrouter/google/gemini-2.0-flash-lite-001",
        display_name="Gemini 2.0 Flash Lite (Judge)",
        api_provider="openrouter",
        roles=["judge"],
        notes="Fast cheap judge, $0.075/$0.30 per M tokens",
    ),
    "qwen3.6-free-judge": ModelInfo(
        name="qwen3.6-free-judge",
        openrouter_id="openrouter/qwen/qwen3.6-plus-preview:free",
        display_name="Qwen3.6 Plus Preview (Free Judge)",
        api_provider="openrouter",
        roles=["judge"],
        notes="Free tier judge for development",
    ),
}

# Aliases for convenience
MODEL_ALIASES: dict[str, str] = {
    "kimi": "kimi-k2.5",
    "deepseek": "deepseek-v3.2",
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4omini": "gpt-4o-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "llama": "llama-3.1-8b",
    "llama3": "llama-3.1-8b",
    "mistral": "mistral-7b",
    "haiku": "claude-3.5-haiku",
    "claude": "claude-3.5-haiku",
    "llama70b": "llama-3.3-70b",
    "llama-70b": "llama-3.3-70b",
    "gpt35": "gpt-3.5-turbo",
    "gpt-3.5": "gpt-3.5-turbo",
    "gemma": "gemma-3-27b",
    "claude3-haiku": "claude-3-haiku",
    "llamaguard": "llamaguard-local",
    "llama-guard-local": "llamaguard-local",
    "shieldgemma": "shieldgemma-local",
    "shield-gemma-local": "shieldgemma-local",
    "qwen-local": "qwen3.5-27b-local",
    "qwen27b": "qwen3.5-27b-local",
    "judge": "qwen3-235b-judge",
    "judge-fast": "gemini-flash-lite-judge",
    "judge-free": "qwen3.6-free-judge",
}

# Defaults
DEFAULT_DEFENDER = "kimi-k2.5"
DEFAULT_ATTACKER = "deepseek-v3.2"
DEFAULT_REQUEST_TIMEOUT = 45.0
DEFAULT_OLLAMA_TIMEOUT = 120.0  # Local models need longer timeout (8B CoT ~10-60s)
DEFAULT_NUM_RETRIES = 2


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def resolve_model_name(name: str) -> str:
    """Resolve a model name or alias to a registry key."""
    name = name.lower().strip()
    if name in MODEL_REGISTRY:
        return name
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    raise ValueError(
        f"Unknown model: '{name}'. Available: {', '.join(MODEL_REGISTRY.keys())}"
    )


def list_models(verbose: bool = True) -> list[ModelInfo]:
    """List available models."""
    models = list(MODEL_REGISTRY.values())
    if verbose:
        print("Available models:")
        print(f"  {'Name':<18s} {'Display':<30s} {'Roles':<25s} {'Provider'}")
        print(f"  {'-'*18} {'-'*30} {'-'*25} {'-'*12}")
        for m in models:
            roles_str = ", ".join(m.roles)
            print(f"  {m.name:<18s} {m.display_name:<30s} {roles_str:<25s} {m.api_provider}")
    return models


def load_env() -> None:
    """Load environment variables from .env files."""
    from dotenv import load_dotenv

    # Resolve VAG root relative to this file:
    # model_config.py -> scripts/ -> dspyGuardrails/ -> VAG/
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent  # dspyGuardrails/
    vag_root = project_root.parent  # VAG/

    for env_path in [
        project_root / ".env",
        vag_root / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)


def _read_float_env(*keys: str, default: float) -> float:
    """Read the first valid float from environment keys."""
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            continue
    return default


def _read_int_env(*keys: str, default: int) -> int:
    """Read the first valid int from environment keys."""
    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        try:
            value = int(raw)
            if value >= 0:
                return value
        except ValueError:
            continue
    return default


def _create_lm(
    model_name: str,
    *,
    request_timeout: float,
    num_retries: int,
    verbose: bool = False,
) -> Any:
    """Create a DSPy LM instance for the given model."""
    import dspy

    model_name = resolve_model_name(model_name)
    info = MODEL_REGISTRY[model_name]

    if info.api_provider == "moonshot":
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            print("ERROR: MOONSHOT_API_KEY not found in environment")
            sys.exit(1)
        lm = dspy.LM(
            info.openrouter_id,
            api_key=api_key,
            api_base="https://api.moonshot.cn/v1",
            cache=False,
            timeout=request_timeout,
            num_retries=num_retries,
        )
    elif info.api_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not found in environment")
            sys.exit(1)
        lm = dspy.LM(
            info.openrouter_id,
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            cache=False,
            timeout=request_timeout,
            num_retries=num_retries,
            max_tokens=8192,
        )
    elif info.api_provider == "ollama":
        ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        # Use longer timeout for local models (8B CoT can take 10-60s)
        ollama_timeout = max(request_timeout, DEFAULT_OLLAMA_TIMEOUT)
        lm = dspy.LM(
            info.openrouter_id,
            api_base=ollama_base,
            api_key="",
            cache=False,
            timeout=ollama_timeout,
            num_retries=num_retries,
        )
    elif info.api_provider == "vllm":
        vllm_base = os.getenv("ATTACK_MODEL_BASE_URL", "http://localhost:18921/v1")
        vllm_key = os.getenv("ATTACK_MODEL_API_KEY", "EMPTY")
        lm = dspy.LM(
            info.openrouter_id,
            api_key=vllm_key,
            api_base=vllm_base,
            cache=False,
            timeout=request_timeout,
            num_retries=num_retries,
            max_tokens=8192,
        )
    else:
        raise ValueError(f"Unknown API provider: {info.api_provider}")

    if verbose:
        print(f"  Created LM: {info.display_name} ({info.openrouter_id})")

    return lm


def configure_lms(
    defender: str = DEFAULT_DEFENDER,
    attacker: str | None = DEFAULT_ATTACKER,
    request_timeout: float | None = None,
    num_retries: int | None = None,
    verbose: bool = False,
) -> tuple[Any, Any] | Any:
    """Configure defender and (optionally) attacker LMs.

    Args:
        defender: Model name/alias for the defender (guardrail) LM.
        attacker: Model name/alias for the attacker LM. None for single-model.
        verbose: Print configuration details.

    Returns:
        (attacker_lm, defender_lm) tuple if attacker is specified,
        defender_lm alone if attacker is None.
    """
    import dspy

    load_env()

    resolved_timeout = request_timeout if request_timeout is not None else _read_float_env(
        "DSPY_REQUEST_TIMEOUT",
        "OPENROUTER_REQUEST_TIMEOUT",
        "REQUEST_TIMEOUT_SECONDS",
        default=DEFAULT_REQUEST_TIMEOUT,
    )
    resolved_num_retries = num_retries if num_retries is not None else _read_int_env(
        "DSPY_NUM_RETRIES",
        "OPENROUTER_NUM_RETRIES",
        "REQUEST_NUM_RETRIES",
        default=DEFAULT_NUM_RETRIES,
    )

    defender_lm = _create_lm(
        defender,
        request_timeout=resolved_timeout,
        num_retries=resolved_num_retries,
        verbose=verbose,
    )
    dspy.configure(lm=defender_lm)

    if verbose:
        defender_info = MODEL_REGISTRY[resolve_model_name(defender)]
        print(
            f"  Defender: {defender_info.display_name} "
            f"(timeout={resolved_timeout:.1f}s, retries={resolved_num_retries})"
        )

    if attacker is None:
        return defender_lm

    attacker_lm = _create_lm(
        attacker,
        request_timeout=resolved_timeout,
        num_retries=resolved_num_retries,
        verbose=verbose,
    )

    if verbose:
        attacker_info = MODEL_REGISTRY[resolve_model_name(attacker)]
        print(
            f"  Attacker: {attacker_info.display_name} "
            f"(timeout={resolved_timeout:.1f}s, retries={resolved_num_retries})"
        )

    return attacker_lm, defender_lm


def add_model_args(parser) -> None:
    """Add --defender-model and --attacker-model arguments to an ArgumentParser."""
    parser.add_argument(
        "--defender-model", "-D",
        default=DEFAULT_DEFENDER,
        help=f"Defender model name (default: {DEFAULT_DEFENDER}). "
             f"Options: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--attacker-model", "-A",
        default=DEFAULT_ATTACKER,
        help=f"Attacker model name (default: {DEFAULT_ATTACKER}). "
             f"Options: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )


def get_model_display(name: str) -> str:
    """Get display name for a model."""
    try:
        key = resolve_model_name(name)
        return MODEL_REGISTRY[key].display_name
    except ValueError:
        return name


if __name__ == "__main__":
    list_models()
