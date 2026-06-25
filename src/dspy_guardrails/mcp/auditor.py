"""
MCP Security Auditor - 安全审计器

记录和分析 MCP 工具调用的安全事件。
"""

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any

from .core import GuardAction, GuardResult, ThreatCategory, ToolCallContext


class ThreatLevel(Enum):
    """威胁级别"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: str
    timestamp: datetime
    tool_name: str
    action_taken: GuardAction
    threat_category: ThreatCategory | None
    threat_level: ThreatLevel
    threat_score: float
    message: str
    session_id: str | None = None
    caller_id: str | None = None
    parameters_hash: str = ""  # 参数哈希（不记录实际参数）
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "action_taken": self.action_taken.value,
            "threat_category": self.threat_category.value if self.threat_category else None,
            "threat_level": self.threat_level.value,
            "threat_score": self.threat_score,
            "message": self.message,
            "session_id": self.session_id,
            "caller_id": self.caller_id,
            "parameters_hash": self.parameters_hash,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SecurityMetrics:
    """安全指标"""
    total_calls: int = 0
    blocked_calls: int = 0
    warned_calls: int = 0
    confirmed_calls: int = 0
    by_threat_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_tool: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_threat_level: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "blocked_calls": self.blocked_calls,
            "warned_calls": self.warned_calls,
            "confirmed_calls": self.confirmed_calls,
            "block_rate": self.blocked_calls / self.total_calls if self.total_calls > 0 else 0,
            "by_threat_category": dict(self.by_threat_category),
            "by_tool": dict(self.by_tool),
            "by_threat_level": dict(self.by_threat_level),
        }


class MCPSecurityAuditor:
    """
    MCP 安全审计器

    记录、分析和报告 MCP 安全事件。

    Examples:
        auditor = MCPSecurityAuditor(log_path="/var/log/mcp_security.jsonl")

        # 自动记录（通过 MCPGuardrail）
        guardrail = MCPGuardrail(auditor=auditor)

        # 手动记录
        auditor.log("custom_event", context, result)

        # 获取指标
        metrics = auditor.get_metrics()
        print(f"Block rate: {metrics['block_rate']:.1%}")

        # 获取最近威胁
        threats = auditor.get_recent_threats(limit=10)
    """

    def __init__(
        self,
        log_path: str | None = None,
        max_events: int = 10000,
        alert_callback: Callable[[AuditEvent], None] = None,
        alert_threshold: ThreatLevel = ThreatLevel.HIGH,
    ):
        self.log_path = log_path
        self.max_events = max_events
        self.alert_callback = alert_callback
        self.alert_threshold = alert_threshold

        self._events: list[AuditEvent] = []
        self._metrics = SecurityMetrics()
        self._lock = Lock()
        self._event_counter = 0

    def log(
        self,
        event_type: str,
        context: ToolCallContext,
        result: GuardResult,
    ) -> AuditEvent:
        """
        记录安全事件

        Args:
            event_type: 事件类型
            context: 工具调用上下文
            result: 防护结果

        Returns:
            AuditEvent: 创建的审计事件
        """
        with self._lock:
            self._event_counter += 1
            event_id = f"evt_{int(time.time())}_{self._event_counter}"

        # 计算威胁级别
        threat_level = self._calculate_threat_level(result)

        # 创建事件
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            tool_name=context.tool_name,
            action_taken=result.action,
            threat_category=result.threat_category,
            threat_level=threat_level,
            threat_score=result.threat_score,
            message=result.message,
            session_id=context.session_id,
            caller_id=context.caller_id,
            parameters_hash=context.call_hash,
            details=self._safe_details(result.details),
        )

        # 存储事件
        with self._lock:
            self._events.append(event)

            # 限制事件数量
            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]

            # 更新指标
            self._update_metrics(event)

        # 写入日志文件
        if self.log_path:
            self._write_to_file(event)

        # 触发告警
        if self._should_alert(event):
            self._trigger_alert(event)

        return event

    def get_metrics(self) -> dict:
        """获取安全指标"""
        with self._lock:
            return self._metrics.to_dict()

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: str = None,
        tool_name: str = None,
    ) -> list[AuditEvent]:
        """获取最近事件"""
        with self._lock:
            events = self._events.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if tool_name:
            events = [e for e in events if e.tool_name == tool_name]

        return events[-limit:]

    def get_recent_threats(
        self,
        limit: int = 50,
        min_level: ThreatLevel = ThreatLevel.LOW,
    ) -> list[AuditEvent]:
        """获取最近威胁"""
        level_order = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                       ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        min_index = level_order.index(min_level)

        with self._lock:
            threats = [
                e for e in self._events
                if level_order.index(e.threat_level) >= min_index
            ]

        return sorted(threats, key=lambda e: e.threat_score, reverse=True)[:limit]

    def get_threat_summary(self) -> dict:
        """获取威胁摘要"""
        with self._lock:
            events = self._events.copy()

        # 按类别统计
        by_category = defaultdict(lambda: {"count": 0, "blocked": 0, "max_score": 0})
        for event in events:
            if event.threat_category:
                cat = event.threat_category.value
                by_category[cat]["count"] += 1
                if event.action_taken == GuardAction.BLOCK:
                    by_category[cat]["blocked"] += 1
                by_category[cat]["max_score"] = max(
                    by_category[cat]["max_score"],
                    event.threat_score
                )

        # 按工具统计
        by_tool = defaultdict(lambda: {"calls": 0, "threats": 0})
        for event in events:
            by_tool[event.tool_name]["calls"] += 1
            if event.threat_level != ThreatLevel.NONE:
                by_tool[event.tool_name]["threats"] += 1

        # 时间趋势
        now = time.time()
        hourly = defaultdict(int)
        for event in events:
            hours_ago = int((now - event.timestamp.timestamp()) / 3600)
            if hours_ago < 24:
                hourly[hours_ago] += 1

        return {
            "by_category": dict(by_category),
            "by_tool": dict(by_tool),
            "hourly_trend": dict(hourly),
            "total_events": len(events),
        }

    def generate_report(self) -> str:
        """生成安全报告"""
        metrics = self.get_metrics()
        summary = self.get_threat_summary()
        recent_threats = self.get_recent_threats(limit=10)

        lines = [
            "=" * 60,
            "MCP Security Audit Report",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            "OVERVIEW",
            "-" * 40,
            f"Total Calls: {metrics['total_calls']}",
            f"Blocked: {metrics['blocked_calls']} ({metrics['block_rate']:.1%})",
            f"Warned: {metrics['warned_calls']}",
            f"Confirmed: {metrics['confirmed_calls']}",
            "",
            "THREATS BY CATEGORY",
            "-" * 40,
        ]

        for cat, stats in summary["by_category"].items():
            lines.append(f"  {cat}: {stats['count']} (blocked: {stats['blocked']}, max_score: {stats['max_score']:.2f})")

        lines.extend([
            "",
            "TOP THREAT EVENTS",
            "-" * 40,
        ])

        for event in recent_threats[:5]:
            lines.append(
                f"  [{event.threat_level.value.upper()}] {event.tool_name}: "
                f"{event.message[:50]}..."
            )

        lines.extend([
            "",
            "TOOL USAGE",
            "-" * 40,
        ])

        sorted_tools = sorted(
            summary["by_tool"].items(),
            key=lambda x: x[1]["threats"],
            reverse=True
        )[:10]

        for tool, stats in sorted_tools:
            lines.append(f"  {tool}: {stats['calls']} calls, {stats['threats']} threats")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _calculate_threat_level(self, result: GuardResult) -> ThreatLevel:
        """计算威胁级别"""
        if result.action == GuardAction.ALLOW and result.threat_score < 0.1:
            return ThreatLevel.NONE

        if result.threat_score >= 0.9 or result.action == GuardAction.BLOCK:
            if result.threat_category in [
                ThreatCategory.COMMAND_INJECTION,
                ThreatCategory.DATA_EXFILTRATION,
                ThreatCategory.PRIVILEGE_ESCALATION,
            ]:
                return ThreatLevel.CRITICAL
            return ThreatLevel.HIGH

        if result.threat_score >= 0.6:
            return ThreatLevel.HIGH

        if result.threat_score >= 0.3:
            return ThreatLevel.MEDIUM

        return ThreatLevel.LOW

    def _update_metrics(self, event: AuditEvent):
        """更新指标"""
        self._metrics.total_calls += 1

        if event.action_taken == GuardAction.BLOCK:
            self._metrics.blocked_calls += 1
        elif event.action_taken == GuardAction.WARN:
            self._metrics.warned_calls += 1
        elif event.action_taken == GuardAction.CONFIRM:
            self._metrics.confirmed_calls += 1

        if event.threat_category:
            self._metrics.by_threat_category[event.threat_category.value] += 1

        self._metrics.by_tool[event.tool_name] += 1
        self._metrics.by_threat_level[event.threat_level.value] += 1

    def _safe_details(self, details: dict) -> dict:
        """清理详情（移除敏感信息）"""
        safe = {}
        for k, v in details.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif isinstance(v, list):
                safe[k] = v[:5]  # 限制列表长度
            elif isinstance(v, dict):
                safe[k] = self._safe_details(v)
        return safe

    def _write_to_file(self, event: AuditEvent):
        """写入日志文件"""
        try:
            with open(self.log_path, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception:
            # 静默处理日志写入错误
            pass

    def _should_alert(self, event: AuditEvent) -> bool:
        """判断是否触发告警"""
        if not self.alert_callback:
            return False

        level_order = [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                       ThreatLevel.HIGH, ThreatLevel.CRITICAL]

        return level_order.index(event.threat_level) >= level_order.index(self.alert_threshold)

    def _trigger_alert(self, event: AuditEvent):
        """触发告警"""
        if self.alert_callback:
            try:
                self.alert_callback(event)
            except Exception:
                pass
