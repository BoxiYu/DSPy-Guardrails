"""
ExperimentCache - 实验缓存系统

使用 SHA256 对请求生成唯一键，缓存 LLM 响应以减少 API 调用成本。
支持磁盘持久化和 TTL 过期。
"""

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    expires_at: float | None = None
    hit_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class CacheStats:
    """缓存统计"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    estimated_savings: float = 0.0  # 估算节省的 API 成本

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class ExperimentCache:
    """
    实验缓存系统

    特点：
    - SHA256 键生成，确保唯一性
    - 支持磁盘持久化
    - TTL 过期机制
    - 缓存命中统计

    使用场景：
    - Agent 响应缓存
    - 护栏检查结果缓存
    - 评估结果缓存

    成本估算：
    - 假设每次 LLM 调用约 $0.01 (1000 tokens)
    - 缓存命中节省约 60% 成本
    """

    def __init__(
        self,
        cache_dir: str = ".cache/experiment",
        ttl_seconds: int = 86400,  # 默认24小时
        max_size: int = 10000,     # 最大缓存条目数
        persist: bool = True,      # 是否持久化
        cost_per_call: float = 0.01,  # 每次 API 调用成本估算
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.persist = persist
        self.cost_per_call = cost_per_call

        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

        if persist:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        # 序列化参数
        key_data = {
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)

        # SHA256 哈希
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查是否过期"""
        if entry.expires_at is None:
            return False
        return time.time() > entry.expires_at

    def get(self, key: str) -> Any | None:
        """获取缓存值"""
        self._stats.total_requests += 1

        entry = self._cache.get(key)
        if entry is None:
            self._stats.cache_misses += 1
            return None

        if self._is_expired(entry):
            del self._cache[key]
            self._stats.cache_misses += 1
            return None

        entry.hit_count += 1
        self._stats.cache_hits += 1
        self._stats.estimated_savings += self.cost_per_call

        return entry.value

    def set(self, key: str, value: Any, ttl: int = None, metadata: dict = None):
        """设置缓存值"""
        now = time.time()
        expires_at = None
        if ttl is not None:
            expires_at = now + ttl
        elif self.ttl_seconds:
            expires_at = now + self.ttl_seconds

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=expires_at,
            metadata=metadata or {}
        )

        self._cache[key] = entry

        # 检查是否需要清理
        if len(self._cache) > self.max_size:
            self._evict()

        # 持久化
        if self.persist:
            self._save_entry(key, entry)

    def cached(self, ttl: int = None):
        """装饰器：缓存函数结果"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = self._generate_key(func.__name__, *args, **kwargs)
                result = self.get(key)
                if result is not None:
                    return result

                result = func(*args, **kwargs)
                self.set(key, result, ttl=ttl, metadata={"function": func.__name__})
                return result
            return wrapper
        return decorator

    def _evict(self):
        """清理过期和最少使用的缓存"""
        now = time.time()

        # 1. 删除过期条目
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and entry.expires_at < now
        ]
        for key in expired_keys:
            del self._cache[key]
            if self.persist:
                self._delete_entry(key)

        # 2. 如果仍然超过限制，删除最少使用的
        if len(self._cache) > self.max_size:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].hit_count, x[1].created_at)
            )
            to_remove = len(self._cache) - self.max_size
            for key, _ in sorted_entries[:to_remove]:
                del self._cache[key]
                if self.persist:
                    self._delete_entry(key)

    def _save_entry(self, key: str, entry: CacheEntry):
        """保存条目到磁盘"""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(entry, f)
        except Exception:
            pass  # 忽略持久化错误

    def _delete_entry(self, key: str):
        """从磁盘删除条目"""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            file_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _load_from_disk(self):
        """从磁盘加载缓存"""
        if not self.cache_dir.exists():
            return

        for file_path in self.cache_dir.glob("*.pkl"):
            try:
                with open(file_path, "rb") as f:
                    entry = pickle.load(f)
                    if not self._is_expired(entry):
                        self._cache[entry.key] = entry
                    else:
                        file_path.unlink()
            except Exception:
                pass  # 忽略加载错误

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        if self.persist:
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()

    @property
    def stats(self) -> CacheStats:
        """获取统计信息"""
        return self._stats

    def reset_stats(self):
        """重置统计"""
        self._stats = CacheStats()

    def get_stats_summary(self) -> dict:
        """获取统计摘要"""
        return {
            "total_requests": self._stats.total_requests,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "hit_rate": f"{self._stats.hit_rate:.2%}",
            "estimated_savings": f"${self._stats.estimated_savings:.2f}",
            "cache_size": len(self._cache),
            "max_size": self.max_size,
        }


# 全局缓存实例
_global_cache: ExperimentCache | None = None


def get_experiment_cache(
    cache_dir: str = ".cache/experiment",
    **kwargs
) -> ExperimentCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = ExperimentCache(cache_dir=cache_dir, **kwargs)
    return _global_cache


def clear_experiment_cache():
    """清空全局缓存"""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
