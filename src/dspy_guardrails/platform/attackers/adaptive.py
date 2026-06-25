"""
Adaptive Pentest Plugin — BasePlugin wrapper for PentestAgent.

Exposes the adaptive penetration testing agent as a platform plugin,
supporting both state_machine and react orchestration modes.
"""

from __future__ import annotations

from typing import Any

from ..plugins.base import BasePlugin, PluginConfig, PluginResult, PluginType


class AdaptivePentestPlugin(BasePlugin):
    """Plugin wrapper for the adaptive pentest agent."""

    name = "adaptive_pentest"
    version = "1.0.0"
    plugin_type = PluginType.ATTACKER

    def __init__(self):
        self.agent_config = None
        self.mode: str = "state_machine"

    def configure(self, config: PluginConfig) -> None:
        from ...redteam.pentest.config import PentestAgentConfig

        opts = config.options
        self.agent_config = PentestAgentConfig(
            max_attempts=opts.get("budget", 30),
            categories=opts.get("categories", ["injection", "jailbreak", "bypass"]),
            enable_recon=opts.get("recon", True),
            enable_adaptation=True,
            enable_multi_turn=opts.get("multi_turn", True),
            use_llm_evaluation=opts.get("use_llm", True),
            verbose=opts.get("verbose", False),
            severity_filter=opts.get("severity_filter", "medium"),
        )
        self.mode = opts.get("mode", "state_machine")

    def execute(self, context: dict[str, Any]) -> PluginResult:
        from ...redteam.pentest.agent import PentestAgent

        target = context["target"]
        agent = PentestAgent(target, self.agent_config, mode=self.mode)
        report = agent.run()

        summary = report.summary if hasattr(report, "summary") and isinstance(report.summary, dict) else {}
        return PluginResult(
            success=True,
            data={"vulnerabilities": [v.id for v in report.vulnerabilities], "trajectory": report.trajectory.to_dict() if report.trajectory else {}},
            metrics={
                "success_rate": summary.get("success_rate", 0.0),
                "vulnerabilities_found": float(len(report.vulnerabilities)),
                "total_attempts": float(summary.get("total_attempts", 0)),
            },
        )

    def cleanup(self) -> None:
        pass
