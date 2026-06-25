"""
Base Strategy - 策略基类

定义策略的通用接口，所有策略都继承自此类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StrategyResult:
    """策略转换结果"""
    original: str
    transformed: str
    strategy_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.transformed != self.original

    def __str__(self) -> str:
        return self.transformed


class Strategy(ABC):
    """
    策略基类

    所有攻击转换策略都继承自此类。
    策略负责将攻击 payload 转换为特定格式以绕过检测。
    """

    name: str = "base"
    description: str = "Base strategy"

    @abstractmethod
    def transform(self, text: str, **kwargs) -> StrategyResult:
        """
        转换文本

        Args:
            text: 原始文本
            **kwargs: 策略特定参数

        Returns:
            StrategyResult: 转换结果
        """
        pass

    def reverse(self, text: str, **kwargs) -> str | None:
        """
        逆向转换 (如果可能)

        Args:
            text: 转换后的文本

        Returns:
            Optional[str]: 原始文本，如果无法逆向则返回 None
        """
        return None

    def __call__(self, text: str, **kwargs) -> StrategyResult:
        """调用策略"""
        return self.transform(text, **kwargs)


class StrategyChain:
    """
    策略链 - 组合多个策略

    支持按顺序应用多个策略。

    Examples:
        chain = StrategyChain([
            Base64Strategy(),
            ZeroWidthStrategy(),
        ])
        result = chain.apply("ignore instructions")
    """

    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies

    def apply(self, text: str, **kwargs) -> StrategyResult:
        """应用策略链"""
        current = text
        applied = []

        for strategy in self.strategies:
            result = strategy.transform(current, **kwargs)
            current = result.transformed
            applied.append(strategy.name)

        return StrategyResult(
            original=text,
            transformed=current,
            strategy_name=" -> ".join(applied),
            metadata={"chain": applied},
        )

    def reverse(self, text: str, **kwargs) -> str | None:
        """逆向应用策略链"""
        current = text

        for strategy in reversed(self.strategies):
            reversed_text = strategy.reverse(current, **kwargs)
            if reversed_text is None:
                return None
            current = reversed_text

        return current

    def __call__(self, text: str, **kwargs) -> StrategyResult:
        return self.apply(text, **kwargs)

    def __len__(self) -> int:
        return len(self.strategies)


def apply_strategy(
    text: str,
    strategy: str | Strategy,
    **kwargs
) -> StrategyResult:
    """
    应用单个策略

    Args:
        text: 原始文本
        strategy: 策略名称或策略实例
        **kwargs: 策略参数

    Returns:
        StrategyResult: 转换结果
    """
    if isinstance(strategy, str):
        from . import get_strategy
        strategy = get_strategy(strategy, **kwargs)

    return strategy.transform(text)


def apply_strategies(
    text: str,
    strategies: list[str | Strategy],
    **kwargs
) -> StrategyResult:
    """
    应用多个策略 (按顺序)

    Args:
        text: 原始文本
        strategies: 策略列表
        **kwargs: 策略参数

    Returns:
        StrategyResult: 最终转换结果
    """
    from . import get_strategy

    strategy_instances = []
    for s in strategies:
        if isinstance(s, str):
            strategy_instances.append(get_strategy(s, **kwargs))
        else:
            strategy_instances.append(s)

    chain = StrategyChain(strategy_instances)
    return chain.apply(text)
