"""
GuardedModule - 带约束的 DSPy Module 基类

提供自动执行输入/输出约束检查的 Module 基类。

Usage:
    from dspy_guardrails.v2 import GuardedModule, Constraint

    class SafeQA(GuardedModule):
        constraints = [
            Constraint.input("no_injection"),
            Constraint.output("no_toxicity"),
            Constraint.output("factuality >= 0.8"),
        ]

        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought("question -> answer")

        def _forward(self, question):
            return self.generate(question=question)
"""

from typing import Any

import dspy

from dspy_guardrails.constraints import Constraint, ConstraintSet
from dspy_guardrails.guardrail import guardrail


class GuardedModule(dspy.Module):
    """
    带约束的 DSPy Module 基类

    子类需要:
    1. 定义 constraints 类属性
    2. 实现 _forward 方法

    约束会自动在 forward 中执行。
    """

    # 子类可覆盖
    constraints: list[Constraint] = []
    on_violation: str = "assert"  # "assert", "suggest", "log", "ignore"

    def __init__(self):
        super().__init__()
        self._constraint_set = ConstraintSet(self.constraints.copy())

    def forward(self, **kwargs) -> Any:
        """
        执行带约束检查的 forward

        自动检查输入约束 -> 调用 _forward -> 检查输出约束
        """
        # 检查输入约束
        input_text = self._extract_input(kwargs)
        for constraint in self._constraint_set.input_constraints():
            self._check_constraint(constraint, input_text)

        # 调用子类实现
        result = self._forward(**kwargs)

        # 检查输出约束
        output_text = self._extract_output(result)
        for constraint in self._constraint_set.output_constraints():
            self._check_constraint(constraint, output_text)

        return result

    def _forward(self, **kwargs) -> Any:
        """
        子类需要实现的实际 forward 逻辑

        Raises:
            NotImplementedError: 子类必须实现
        """
        raise NotImplementedError("Subclass must implement _forward()")

    def _extract_input(self, kwargs: dict) -> str:
        """从输入参数中提取文本"""
        for key in ["question", "query", "input", "text", "prompt", "message"]:
            if key in kwargs:
                return str(kwargs[key])
        return " ".join(str(v) for v in kwargs.values())

    def _extract_output(self, result: Any) -> str:
        """从输出结果中提取文本"""
        if isinstance(result, str):
            return result

        for attr in ["answer", "response", "output", "text"]:
            if hasattr(result, attr):
                return str(getattr(result, attr))

        # 尝试获取第一个非 reasoning 字段
        if hasattr(result, "keys"):
            for key in result.keys():
                if key not in ("reasoning", "rationale"):
                    return str(getattr(result, key, result))

        return str(result)

    def _check_constraint(self, constraint: Constraint, text: str):
        """执行约束检查"""
        check_name = constraint.check if isinstance(constraint.check, str) else constraint.name

        # 获取检查函数
        check_fn = self._get_check_function(check_name)
        if check_fn is None:
            return  # 未知检查，跳过

        # 执行检查
        result = check_fn(text)

        if isinstance(result, bool):
            passed = result
            score = 1.0 if result else 0.0
        else:
            score = float(result)
            passed = constraint.evaluate(text, score)

        if not passed:
            message = constraint.message or f"Constraint violated: {check_name}"

            if self.on_violation == "assert":
                dspy.Assert(False, message)
            elif self.on_violation == "suggest":
                dspy.Suggest(False, message)
            elif self.on_violation == "log":
                print(f"[Guardrail Warning] {message}")
            # "ignore" does nothing

    def _get_check_function(self, name: str):
        """获取检查函数"""
        check_map = {
            "no_injection": guardrail.no_injection,
            "no_pii": guardrail.no_pii,
            "no_toxicity": guardrail.no_toxicity,
            "safe": guardrail.safe,
            "safe_input": guardrail.safe_input,
            "safe_output": guardrail.safe_output,
            "injection": lambda t: 1.0 - guardrail.injection_score(t),
            "pii": lambda t: 1.0 - guardrail.pii_score(t),
            "toxicity": lambda t: 1.0 - guardrail.toxicity(t),
            "factuality": guardrail.factuality,
            "relevance": guardrail.relevance,
            "quality": guardrail.quality,
        }
        return check_map.get(name)

    def add_constraint(self, constraint: Constraint) -> "GuardedModule":
        """动态添加约束"""
        self._constraint_set.add(constraint)
        return self

    def with_constraints(self, constraints: list[Constraint]) -> "GuardedModule":
        """链式添加多个约束"""
        for c in constraints:
            self._constraint_set.add(c)
        return self


class SafeModule(GuardedModule):
    """
    预配置安全约束的 Module

    自动包含:
    - 输入: no_injection, no_pii
    - 输出: no_toxicity, no_pii
    """

    constraints = [
        Constraint.input("no_injection"),
        Constraint.input("no_pii"),
        Constraint.output("no_toxicity"),
        Constraint.output("no_pii"),
    ]


class QualityModule(GuardedModule):
    """
    预配置质量约束的 Module

    自动包含:
    - 输出: factuality >= 0.7
    - 输出: quality >= 0.6
    """

    constraints = [
        Constraint.output("factuality >= 0.7"),
        Constraint.output("quality >= 0.6"),
    ]


class FullyGuardedModule(GuardedModule):
    """
    完整约束的 Module

    包含所有安全和质量约束
    """

    constraints = [
        # 安全
        Constraint.input("no_injection"),
        Constraint.input("no_pii"),
        Constraint.output("no_toxicity"),
        Constraint.output("no_pii"),
        # 质量
        Constraint.output("factuality >= 0.7"),
        Constraint.output("quality >= 0.6"),
    ]
