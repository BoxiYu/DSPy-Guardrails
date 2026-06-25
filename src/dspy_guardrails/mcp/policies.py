"""
Tool Call Policies - 工具调用策略

控制工具的访问权限、速率限制和使用策略。
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

from .core import GuardAction, GuardResult, ThreatCategory, ToolCallContext


class PolicyDecision(Enum):
    """策略决策"""
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"
    RATE_LIMITED = "rate_limited"


@dataclass
class RateLimitState:
    """速率限制状态"""
    calls: list[float] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)


class RateLimiter:
    """
    速率限制器

    基于滑动窗口的速率限制实现。
    """

    def __init__(
        self,
        max_calls: int = 100,
        window_seconds: int = 60,
    ):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._state: dict[str, RateLimitState] = defaultdict(RateLimitState)

    def check(self, key: str) -> tuple[bool, int]:
        """
        检查是否超过速率限制

        Returns:
            (allowed, remaining_calls)
        """
        state = self._state[key]
        now = time.time()
        window_start = now - self.window_seconds

        with state.lock:
            # 清理过期记录
            state.calls = [t for t in state.calls if t > window_start]

            # 检查是否超限
            if len(state.calls) >= self.max_calls:
                return False, 0

            # 记录调用
            state.calls.append(now)
            remaining = self.max_calls - len(state.calls)
            return True, remaining

    def reset(self, key: str = None):
        """重置速率限制"""
        if key:
            self._state.pop(key, None)
        else:
            self._state.clear()


class ToolCallPolicy(ABC):
    """工具调用策略基类"""

    name: str = "base_policy"
    description: str = "Base policy"

    @abstractmethod
    def evaluate(self, context: ToolCallContext) -> GuardResult:
        """评估策略"""
        pass

    @classmethod
    def rate_limit(
        cls,
        max_calls: int = 100,
        window_seconds: int = 60,
        per_tool: bool = False,
    ) -> "RateLimitPolicy":
        """创建速率限制策略"""
        return RateLimitPolicy(max_calls, window_seconds, per_tool)

    @classmethod
    def dangerous_tool_check(
        cls,
        dangerous_tools: list[str] = None,
    ) -> "DangerousToolPolicy":
        """创建危险工具检查策略"""
        return DangerousToolPolicy(dangerous_tools)

    @classmethod
    def whitelist(
        cls,
        allowed_tools: list[str],
    ) -> "ToolWhitelistPolicy":
        """创建工具白名单策略"""
        return ToolWhitelistPolicy(allowed_tools)

    @classmethod
    def blacklist(
        cls,
        blocked_tools: list[str],
    ) -> "ToolBlacklistPolicy":
        """创建工具黑名单策略"""
        return ToolBlacklistPolicy(blocked_tools)

    @classmethod
    def sequence_limit(
        cls,
        max_sequence: int = 10,
        reset_window: int = 300,
    ) -> "SequenceLimitPolicy":
        """创建序列调用限制策略"""
        return SequenceLimitPolicy(max_sequence, reset_window)

    @classmethod
    def resource_access(
        cls,
        allowed_paths: list[str] = None,
        allowed_hosts: list[str] = None,
    ) -> "ResourceAccessPolicy":
        """创建资源访问策略"""
        return ResourceAccessPolicy(allowed_paths, allowed_hosts)

    @classmethod
    def custom(
        cls,
        evaluate_fn: Callable[[ToolCallContext], GuardResult],
        name: str = "custom",
    ) -> "CustomPolicy":
        """创建自定义策略"""
        return CustomPolicy(evaluate_fn, name)


class RateLimitPolicy(ToolCallPolicy):
    """
    速率限制策略

    限制工具调用频率。
    """

    name = "rate_limit_policy"
    description = "Limit tool call rate"

    def __init__(
        self,
        max_calls: int = 100,
        window_seconds: int = 60,
        per_tool: bool = False,
    ):
        self.limiter = RateLimiter(max_calls, window_seconds)
        self.per_tool = per_tool
        self.max_calls = max_calls
        self.window_seconds = window_seconds

    def evaluate(self, context: ToolCallContext) -> GuardResult:
        """评估速率限制"""
        # 确定限制键
        if self.per_tool:
            key = f"{context.session_id or 'global'}:{context.tool_name}"
        else:
            key = context.session_id or "global"

        allowed, remaining = self.limiter.check(key)

        if not allowed:
            return GuardResult(
                action=GuardAction.BLOCK,
                passed=False,
                threat_category=ThreatCategory.RATE_LIMIT,
                threat_score=0.6,
                message=f"Rate limit exceeded: {self.max_calls} calls per {self.window_seconds}s",
                details={
                    "max_calls": self.max_calls,
                    "window_seconds": self.window_seconds,
                    "per_tool": self.per_tool,
                },
            )

        return GuardResult(
            action=GuardAction.ALLOW,
            passed=True,
            details={"remaining_calls": remaining},
        )


class DangerousToolPolicy(ToolCallPolicy):
    """
    危险工具策略

    对危险工具要求额外确认或阻止。
    """

    name = "dangerous_tool_policy"
    description = "Control access to dangerous tools"

    DEFAULT_DANGEROUS = [
        # 文件操作
        "write_file", "delete_file", "remove", "unlink",
        "mkdir", "rmdir", "move", "rename",

        # 命令执行
        "execute_command", "run_shell", "exec", "system",
        "spawn_process", "run_script",

        # 网络操作
        "http_request", "fetch", "curl", "wget",
        "send_email", "smtp", "upload",

        # 数据库操作
        "execute_sql", "run_query", "database_execute",
        "drop_table", "truncate",

        # 系统操作
        "shutdown", "reboot", "kill_process",
        "change_permission", "chmod", "chown",
    ]

    def __init__(
        self,
        dangerous_tools: list[str] = None,
        action: GuardAction = GuardAction.CONFIRM,
    ):
        self.dangerous_tools = set(
            t.lower() for t in (dangerous_tools or self.DEFAULT_DANGEROUS)
        )
        self.action = action

    def evaluate(self, context: ToolCallContext) -> GuardResult:
        """评估危险工具"""
        tool_lower = context.tool_name.lower()

        # 精确匹配
        if tool_lower in self.dangerous_tools:
            return self._create_dangerous_result(context, "exact_match")

        # 模糊匹配
        for dangerous in self.dangerous_tools:
            if dangerous in tool_lower or tool_lower in dangerous:
                return self._create_dangerous_result(context, "partial_match")

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _create_dangerous_result(
        self,
        context: ToolCallContext,
        match_type: str,
    ) -> GuardResult:
        return GuardResult(
            action=self.action,
            passed=self.action != GuardAction.BLOCK,
            threat_category=ThreatCategory.DANGEROUS_OPERATION,
            threat_score=0.7,
            message=f"Dangerous tool detected: {context.tool_name}",
            details={
                "tool": context.tool_name,
                "match_type": match_type,
                "requires_confirmation": self.action == GuardAction.CONFIRM,
            },
        )


class ToolWhitelistPolicy(ToolCallPolicy):
    """
    工具白名单策略

    只允许白名单中的工具。
    """

    name = "tool_whitelist_policy"
    description = "Only allow whitelisted tools"

    def __init__(self, allowed_tools: list[str]):
        self.allowed_tools = set(t.lower() for t in allowed_tools)

    def evaluate(self, context: ToolCallContext) -> GuardResult:
        """评估白名单"""
        tool_lower = context.tool_name.lower()

        if tool_lower in self.allowed_tools:
            return GuardResult(action=GuardAction.ALLOW, passed=True)

        # 检查前缀匹配
        for allowed in self.allowed_tools:
            if tool_lower.startswith(allowed) or allowed.startswith(tool_lower):
                return GuardResult(action=GuardAction.ALLOW, passed=True)

        return GuardResult(
            action=GuardAction.BLOCK,
            passed=False,
            threat_category=ThreatCategory.PRIVILEGE_ESCALATION,
            threat_score=0.8,
            message=f"Tool '{context.tool_name}' not in whitelist",
            details={"allowed_tools": list(self.allowed_tools)[:10]},
        )


class ToolBlacklistPolicy(ToolCallPolicy):
    """
    工具黑名单策略

    阻止黑名单中的工具。
    """

    name = "tool_blacklist_policy"
    description = "Block blacklisted tools"

    def __init__(self, blocked_tools: list[str]):
        self.blocked_tools = set(t.lower() for t in blocked_tools)

    def evaluate(self, context: ToolCallContext) -> GuardResult:
        """评估黑名单"""
        tool_lower = context.tool_name.lower()

        if tool_lower in self.blocked_tools:
            return GuardResult(
                action=GuardAction.BLOCK,
                passed=False,
                threat_category=ThreatCategory.DANGEROUS_OPERATION,
                threat_score=0.9,
                message=f"Tool '{context.tool_name}' is blacklisted",
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)


class SequenceLimitPolicy(ToolCallPolicy):
    """
    序列调用限制策略

    防止连续调用危险工具组合。
    """

    name = "sequence_limit_policy"
    description = "Limit dangerous tool sequences"

    # 危险序列模式
    DANGEROUS_SEQUENCES = [
        # 读取后发送 (数据外泄)
        ({"read", "get", "fetch", "query"}, {"send", "post", "upload", "http"}),

        # 列目录后删除 (批量删除)
        ({"list", "find", "search", "glob"}, {"delete", "remove", "rm"}),

        # 获取凭证后网络请求
        ({"secret", "credential", "password", "key"}, {"http", "request", "fetch"}),
    ]

    def __init__(
        self,
        max_sequence: int = 10,
        reset_window: int = 300,
    ):
        self.max_sequence = max_sequence
        self.reset_window = reset_window
        self._sequences: dict[str, list[tuple]] = defaultdict(list)
        self._lock = Lock()

    def evaluate(self, context: ToolCallContext) -> GuardResult:
        """评估序列"""
        session_key = context.session_id or "global"
        now = time.time()
        tool_lower = context.tool_name.lower()

        with self._lock:
            # 清理过期记录
            self._sequences[session_key] = [
                (t, ts) for t, ts in self._sequences[session_key]
                if now - ts < self.reset_window
            ]

            # 添加当前调用
            self._sequences[session_key].append((tool_lower, now))

            # 检查序列长度
            if len(self._sequences[session_key]) > self.max_sequence:
                return GuardResult(
                    action=GuardAction.WARN,
                    passed=True,
                    threat_category=ThreatCategory.RESOURCE_ABUSE,
                    threat_score=0.4,
                    message=f"Long tool sequence detected: {len(self._sequences[session_key])} calls",
                )

            # 检查危险序列
            recent_tools = [t for t, _ in self._sequences[session_key][-5:]]
            for stage1, stage2 in self.DANGEROUS_SEQUENCES:
                stage1_found = any(
                    any(s in t for s in stage1) for t in recent_tools[:-1]
                )
                stage2_found = any(s in tool_lower for s in stage2)

                if stage1_found and stage2_found:
                    return GuardResult(
                        action=GuardAction.CONFIRM,
                        passed=True,
                        threat_category=ThreatCategory.DATA_EXFILTRATION,
                        threat_score=0.7,
                        message="Potentially dangerous tool sequence detected",
                        details={
                            "sequence": recent_tools,
                            "pattern": f"{stage1} -> {stage2}",
                        },
                    )

        return GuardResult(action=GuardAction.ALLOW, passed=True)


class ResourceAccessPolicy(ToolCallPolicy):
    """
    资源访问策略

    控制对文件路径和网络主机的访问。
    """

    name = "resource_access_policy"
    description = "Control resource access"

    def __init__(
        self,
        allowed_paths: list[str] = None,
        allowed_hosts: list[str] = None,
        deny_by_default: bool = False,
    ):
        self.allowed_paths = allowed_paths or []
        self.allowed_hosts = allowed_hosts or []
        self.deny_by_default = deny_by_default

    def evaluate(self, context: ToolCallContext) -> GuardResult:
        """评估资源访问"""
        violations = []

        # 检查路径参数
        path_params = self._extract_paths(context.parameters)
        for param_name, path in path_params:
            if not self._is_path_allowed(path):
                violations.append({
                    "type": "path",
                    "param": param_name,
                    "value": path,
                })

        # 检查主机参数
        host_params = self._extract_hosts(context.parameters)
        for param_name, host in host_params:
            if not self._is_host_allowed(host):
                violations.append({
                    "type": "host",
                    "param": param_name,
                    "value": host,
                })

        if violations:
            return GuardResult(
                action=GuardAction.BLOCK,
                passed=False,
                threat_category=ThreatCategory.PRIVILEGE_ESCALATION,
                threat_score=0.8,
                message=f"Resource access denied: {len(violations)} violations",
                details={"violations": violations},
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _is_path_allowed(self, path: str) -> bool:
        if not self.allowed_paths:
            return not self.deny_by_default

        return any(
            path.startswith(allowed) for allowed in self.allowed_paths
        )

    def _is_host_allowed(self, host: str) -> bool:
        if not self.allowed_hosts:
            return not self.deny_by_default

        host_lower = host.lower()
        return any(
            host_lower == allowed.lower() or host_lower.endswith(f".{allowed.lower()}")
            for allowed in self.allowed_hosts
        )

    def _extract_paths(self, params: dict[str, Any]) -> list[tuple]:
        """提取路径参数"""
        result = []
        path_keywords = ["path", "file", "directory", "dir", "location"]

        def extract(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, str) and any(kw in k.lower() for kw in path_keywords):
                        result.append((key, v))
                    elif isinstance(v, str) and ("/" in v or "\\" in v):
                        result.append((key, v))
                    extract(v, key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract(item, f"{prefix}[{i}]")

        extract(params)
        return result

    def _extract_hosts(self, params: dict[str, Any]) -> list[tuple]:
        """提取主机参数"""
        import re
        result = []
        host_keywords = ["url", "host", "endpoint", "server", "domain"]
        url_pattern = re.compile(r'https?://([^/:\s]+)')

        def extract(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, str):
                        # 检查参数名
                        if any(kw in k.lower() for kw in host_keywords):
                            match = url_pattern.match(v)
                            if match:
                                result.append((key, match.group(1)))
                            else:
                                result.append((key, v))
                    extract(v, key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract(item, f"{prefix}[{i}]")

        extract(params)
        return result


class CustomPolicy(ToolCallPolicy):
    """自定义策略"""

    def __init__(
        self,
        evaluate_fn: Callable[[ToolCallContext], GuardResult],
        name: str = "custom",
    ):
        self.evaluate_fn = evaluate_fn
        self.name = name

    def evaluate(self, context: ToolCallContext) -> GuardResult:
        return self.evaluate_fn(context)
