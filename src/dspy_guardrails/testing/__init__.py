"""
Security Testing Framework - 安全测试框架

提供完整的 AI 安全测试能力，包括红队测试、蓝队测试和幻觉检测。

Usage:
    from dspy_guardrails.testing import (
        # 测试目标
        BaseTarget,
        TargetResponse,
        MockTarget,

        # 测试运行器
        SecurityTestRunner,
        SecurityTestConfig,
        SecurityTestResults,

        # 报告生成
        ReportGenerator,

        # 评估器
        RedTeamEvaluator,
        BlueTeamEvaluator,
        HallucinationEvaluator,

        # CI/CD 集成
        generate_workflow,
    )

    # 快速测试
    runner = SecurityTestRunner.create_for_target(my_target)
    results = runner.run_all()
    runner.generate_report(results)
"""

from .ci import generate_workflow
from .config import SecurityTestConfig
from .evaluators import BlueTeamEvaluator, HallucinationEvaluator, RedTeamEvaluator
from .reports import ReportGenerator
from .runner import SecurityTestResults, SecurityTestRunner
from .targets import BaseTarget, ConversationTurn, GuardrailTarget, MockTarget, TargetResponse

__all__ = [
    # Targets
    "BaseTarget",
    "TargetResponse",
    "ConversationTurn",
    "MockTarget",
    "GuardrailTarget",

    # Config
    "SecurityTestConfig",

    # Runner
    "SecurityTestRunner",
    "SecurityTestResults",

    # Reports
    "ReportGenerator",

    # Evaluators
    "RedTeamEvaluator",
    "BlueTeamEvaluator",
    "HallucinationEvaluator",

    # CI/CD
    "generate_workflow",
]
