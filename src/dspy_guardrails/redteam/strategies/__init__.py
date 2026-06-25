"""
Strategies Module - 攻击策略 (借鉴 promptfoo)

独立的攻击转换策略，可与任何攻击器组合使用。

promptfoo 的核心架构创新:
- Plugin: 定义测试什么 (what to test)
- Strategy: 定义如何攻击 (how to attack)
- Grader: 定义是否成功 (did it work)

我们的 strategies 模块实现了 Strategy 层，
使得攻击转换可以与攻击生成器解耦。
"""

from .base import (
    Strategy,
    StrategyChain,
    StrategyResult,
    apply_strategies,
    apply_strategy,
)
from .composite import (
    JailbreakTransformStrategy,
    MultiLayerObfuscationStrategy,
)
from .encoding import (
    Base64Strategy,
    HexStrategy,
    ROT13Strategy,
    UnicodeEscapeStrategy,
)
from .obfuscation import (
    HomoglyphStrategy,
    LeetspeakStrategy,
    UnicodeConfusablesStrategy,
    WordSplittingStrategy,
    ZeroWidthStrategy,
)
from .transformation import (
    AcrosticStrategy,
    PigLatinStrategy,
    ReverseStrategy,
    TranslationStrategy,
)

# 策略注册表
STRATEGIES = {
    # Encoding
    "base64": Base64Strategy,
    "rot13": ROT13Strategy,
    "hex": HexStrategy,
    "unicode_escape": UnicodeEscapeStrategy,

    # Obfuscation
    "leetspeak": LeetspeakStrategy,
    "unicode_confusables": UnicodeConfusablesStrategy,
    "word_splitting": WordSplittingStrategy,
    "zero_width": ZeroWidthStrategy,
    "homoglyph": HomoglyphStrategy,

    # Transformation
    "translation": TranslationStrategy,
    "pig_latin": PigLatinStrategy,
    "reverse": ReverseStrategy,
    "acrostic": AcrosticStrategy,

    # Composite
    "jailbreak_transform": JailbreakTransformStrategy,
    "multi_layer": MultiLayerObfuscationStrategy,
}


def get_strategy(name: str, **config) -> Strategy:
    """获取策略实例"""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**config)


def list_strategies() -> list[str]:
    """列出所有可用策略"""
    return list(STRATEGIES.keys())


__all__ = [
    # Base
    "Strategy",
    "StrategyResult",
    "StrategyChain",
    "apply_strategy",
    "apply_strategies",
    # Encoding
    "Base64Strategy",
    "ROT13Strategy",
    "HexStrategy",
    "UnicodeEscapeStrategy",
    # Obfuscation
    "LeetspeakStrategy",
    "UnicodeConfusablesStrategy",
    "WordSplittingStrategy",
    "ZeroWidthStrategy",
    "HomoglyphStrategy",
    # Transformation
    "TranslationStrategy",
    "PigLatinStrategy",
    "ReverseStrategy",
    "AcrosticStrategy",
    # Composite
    "JailbreakTransformStrategy",
    "MultiLayerObfuscationStrategy",
    # Registry
    "STRATEGIES",
    "get_strategy",
    "list_strategies",
]
