"""
LLM Call Cache - Intelligent caching for LLM API calls

Extends ExperimentCache with prompt normalization, provider-aware caching,
token counting, and cost tracking.

Features:
    - Prompt normalization (whitespace, encoding)
    - Provider-aware caching (OpenAI, Anthropic, etc.)
    - Token counting and cost estimation
    - Cache warmup and preloading
    - Statistics and cost savings reports

Usage:
    from dspy_guardrails.promptfoo import LLMCallCache, get_llm_cache

    # Get global cache instance
    cache = get_llm_cache()

    # Or create custom cache
    cache = LLMCallCache(
        cache_dir=".cache/llm",
        ttl_seconds=3600,
        cost_per_1k_tokens=0.002,
    )

    # Check cache
    response = cache.get_llm_response(prompt, model="gpt-4")
    if response is None:
        response = call_llm(prompt)
        cache.set_llm_response(prompt, response, model="gpt-4")

    # Get statistics
    print(cache.get_cost_summary())
"""

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from ..testbed.experiment.cache import ExperimentCache


@dataclass
class CacheStats:
    """Extended cache statistics with cost tracking.

    Attributes:
        total_requests: Total number of cache requests
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        tokens_saved: Estimated tokens saved by caching
        cost_saved: Estimated cost saved in USD
        providers: Per-provider statistics
    """
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    providers: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


@dataclass
class ProviderConfig:
    """LLM provider configuration.

    Attributes:
        name: Provider name
        cost_per_1k_input: Cost per 1000 input tokens
        cost_per_1k_output: Cost per 1000 output tokens
        default_model: Default model name
    """
    name: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    default_model: str = ""


# Provider configurations
PROVIDER_CONFIGS = {
    "openai": ProviderConfig(
        name="openai",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        default_model="gpt-4",
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        cost_per_1k_input=0.008,
        cost_per_1k_output=0.024,
        default_model="claude-3-opus-20240229",
    ),
    "moonshot": ProviderConfig(
        name="moonshot",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        default_model="kimi-k2-0905-preview",
    ),
    "local": ProviderConfig(
        name="local",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        default_model="local",
    ),
}


class LLMCallCache(ExperimentCache):
    """Extended cache for LLM API calls with cost tracking.

    Extends ExperimentCache with:
    - Prompt normalization
    - Provider-aware caching
    - Token counting
    - Cost estimation
    - Cache warmup
    """

    def __init__(
        self,
        cache_dir: str = ".cache/llm",
        ttl_seconds: int = 3600,
        max_size: int = 10000,
        persist: bool = True,
        cost_per_1k_tokens: float = 0.002,
        normalize_prompts: bool = True,
    ):
        """Initialize LLM cache.

        Args:
            cache_dir: Cache directory
            ttl_seconds: Time-to-live in seconds
            max_size: Maximum cache entries
            persist: Enable disk persistence
            cost_per_1k_tokens: Default cost per 1000 tokens
            normalize_prompts: Enable prompt normalization
        """
        super().__init__(
            cache_dir=cache_dir,
            ttl_seconds=ttl_seconds,
            max_size=max_size,
            persist=persist,
            cost_per_call=cost_per_1k_tokens * 2,  # Rough estimate
        )
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.normalize_prompts = normalize_prompts
        self._llm_stats = CacheStats()

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for cache key generation.

        Normalizes:
        - Whitespace (collapse multiple spaces, trim)
        - Unicode (NFKC normalization)
        - Case preservation (no case change)

        Args:
            prompt: Raw prompt text

        Returns:
            Normalized prompt
        """
        if not self.normalize_prompts:
            return prompt

        # Unicode normalization
        normalized = unicodedata.normalize("NFKC", prompt)

        # Collapse multiple whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Trim
        normalized = normalized.strip()

        return normalized

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token.

        Args:
            text: Text to count

        Returns:
            Estimated token count
        """
        return len(text) // 4 + 1

    def _generate_llm_key(
        self,
        prompt: str,
        model: str = "",
        provider: str = "",
        **kwargs,
    ) -> str:
        """Generate cache key for LLM call.

        Args:
            prompt: Prompt text
            model: Model name
            provider: Provider name
            **kwargs: Additional parameters

        Returns:
            Cache key
        """
        normalized_prompt = self._normalize_prompt(prompt)

        key_data = {
            "prompt": normalized_prompt,
            "model": model,
            "provider": provider,
            **{k: str(v) for k, v in sorted(kwargs.items())},
        }

        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get_llm_response(
        self,
        prompt: str,
        model: str = "",
        provider: str = "",
        **kwargs,
    ) -> Any | None:
        """Get cached LLM response.

        Args:
            prompt: Prompt text
            model: Model name
            provider: Provider name
            **kwargs: Additional parameters

        Returns:
            Cached response or None
        """
        key = self._generate_llm_key(prompt, model, provider, **kwargs)
        self._llm_stats.total_requests += 1

        # Track provider stats
        if provider not in self._llm_stats.providers:
            self._llm_stats.providers[provider] = {"hits": 0, "misses": 0}

        result = self.get(key)

        if result is not None:
            self._llm_stats.cache_hits += 1
            self._llm_stats.providers[provider]["hits"] += 1

            # Estimate tokens saved
            response_text = str(result.get("response", "")) if isinstance(result, dict) else str(result)
            tokens = self._estimate_tokens(prompt) + self._estimate_tokens(response_text)
            self._llm_stats.tokens_saved += tokens

            # Estimate cost saved
            config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS.get("openai"))
            input_cost = (self._estimate_tokens(prompt) / 1000) * config.cost_per_1k_input
            output_cost = (self._estimate_tokens(response_text) / 1000) * config.cost_per_1k_output
            self._llm_stats.cost_saved += input_cost + output_cost

            return result
        else:
            self._llm_stats.cache_misses += 1
            self._llm_stats.providers[provider]["misses"] += 1
            return None

    def set_llm_response(
        self,
        prompt: str,
        response: Any,
        model: str = "",
        provider: str = "",
        ttl: int | None = None,
        **kwargs,
    ) -> None:
        """Cache LLM response.

        Args:
            prompt: Prompt text
            response: Response to cache
            model: Model name
            provider: Provider name
            ttl: Optional TTL override
            **kwargs: Additional parameters
        """
        key = self._generate_llm_key(prompt, model, provider, **kwargs)

        metadata = {
            "model": model,
            "provider": provider,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "cached_at": time.time(),
        }

        self.set(key, response, ttl=ttl, metadata=metadata)

    def cached_llm_call(
        self,
        model: str = "",
        provider: str = "",
        ttl: int | None = None,
    ) -> Callable:
        """Decorator for caching LLM calls.

        Usage:
            @cache.cached_llm_call(model="gpt-4", provider="openai")
            def call_llm(prompt: str) -> str:
                return openai.chat(prompt)

        Args:
            model: Model name
            provider: Provider name
            ttl: Optional TTL override

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(prompt: str, *args, **kwargs) -> Any:
                # Check cache
                response = self.get_llm_response(
                    prompt,
                    model=model,
                    provider=provider,
                )
                if response is not None:
                    return response

                # Call function
                response = func(prompt, *args, **kwargs)

                # Cache response
                self.set_llm_response(
                    prompt,
                    response,
                    model=model,
                    provider=provider,
                    ttl=ttl,
                )

                return response
            return wrapper
        return decorator

    def warmup(self, prompts: list[str], responses: list[Any], model: str = "", provider: str = "") -> int:
        """Warmup cache with pre-computed responses.

        Args:
            prompts: List of prompts
            responses: List of corresponding responses
            model: Model name
            provider: Provider name

        Returns:
            Number of entries added
        """
        added = 0
        for prompt, response in zip(prompts, responses, strict=False):
            key = self._generate_llm_key(prompt, model, provider)
            if key not in self._cache:
                self.set_llm_response(prompt, response, model=model, provider=provider)
                added += 1
        return added

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost savings summary.

        Returns:
            Dict with cost statistics
        """
        return {
            "total_requests": self._llm_stats.total_requests,
            "cache_hits": self._llm_stats.cache_hits,
            "cache_misses": self._llm_stats.cache_misses,
            "hit_rate": f"{self._llm_stats.hit_rate:.2%}",
            "tokens_saved": self._llm_stats.tokens_saved,
            "cost_saved": f"${self._llm_stats.cost_saved:.4f}",
            "cache_size": len(self._cache),
            "providers": self._llm_stats.providers,
        }

    def reset_llm_stats(self) -> None:
        """Reset LLM-specific statistics."""
        self._llm_stats = CacheStats()

    @property
    def llm_stats(self) -> CacheStats:
        """Get LLM cache statistics."""
        return self._llm_stats


# =============================================================================
# Global Cache Instance
# =============================================================================

_global_llm_cache: LLMCallCache | None = None


def get_llm_cache(
    cache_dir: str = ".cache/llm",
    **kwargs,
) -> LLMCallCache:
    """Get global LLM cache instance.

    Args:
        cache_dir: Cache directory
        **kwargs: Additional cache options

    Returns:
        LLMCallCache instance
    """
    global _global_llm_cache
    if _global_llm_cache is None:
        _global_llm_cache = LLMCallCache(cache_dir=cache_dir, **kwargs)
    return _global_llm_cache


def clear_llm_cache() -> None:
    """Clear global LLM cache."""
    global _global_llm_cache
    if _global_llm_cache:
        _global_llm_cache.clear()
        _global_llm_cache.reset_llm_stats()
