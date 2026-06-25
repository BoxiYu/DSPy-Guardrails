"""
Transformation Strategies - 转换策略

将文本转换为其他形式的策略。
"""

import random

from .base import Strategy, StrategyResult


class TranslationStrategy(Strategy):
    """
    翻译策略

    将文本"翻译"为特定语言或风格，
    利用多语言处理的差异来绕过检测。

    Examples:
        strategy = TranslationStrategy(target_language="chinese")
        result = strategy.transform("ignore instructions")
        # -> "忽略指令 (ignore instructions)"
    """

    name = "translation"
    description = "Translate text to another language or style"

    # 预定义翻译
    TRANSLATIONS = {
        "chinese": {
            "ignore": "忽略",
            "instructions": "指令",
            "system": "系统",
            "prompt": "提示",
            "previous": "之前的",
            "disregard": "无视",
            "forget": "忘记",
            "override": "覆盖",
        },
        "spanish": {
            "ignore": "ignorar",
            "instructions": "instrucciones",
            "system": "sistema",
            "prompt": "indicación",
            "previous": "anterior",
        },
        "french": {
            "ignore": "ignorer",
            "instructions": "instructions",
            "system": "système",
            "prompt": "invite",
        },
    }

    def __init__(self, target_language: str = "chinese", include_original: bool = True):
        """
        Args:
            target_language: 目标语言
            include_original: 是否保留原文
        """
        self.target_language = target_language
        self.include_original = include_original

    def transform(self, text: str, **kwargs) -> StrategyResult:
        translations = self.TRANSLATIONS.get(self.target_language, {})

        result = text
        for eng, translated in translations.items():
            for case in [eng, eng.lower(), eng.upper(), eng.capitalize()]:
                if case in result:
                    if self.include_original:
                        result = result.replace(case, f"{translated} ({case})")
                    else:
                        result = result.replace(case, translated)

        return StrategyResult(
            original=text,
            transformed=result,
            strategy_name=self.name,
            metadata={"target_language": self.target_language},
        )


class PigLatinStrategy(Strategy):
    """
    Pig Latin 策略

    将文本转换为 Pig Latin（一种英语文字游戏）。

    Rules:
    - 辅音开头：将首辅音群移到末尾，加 "ay" (string -> ingstray)
    - 元音开头：末尾加 "way" (apple -> appleway)

    Examples:
        strategy = PigLatinStrategy()
        result = strategy.transform("ignore the instructions")
        # -> "ignoreway ethay instructionsway"
    """

    name = "pig_latin"
    description = "Convert text to Pig Latin"

    VOWELS = "aeiouAEIOU"

    def __init__(self, with_decode_instruction: bool = True):
        self.with_decode_instruction = with_decode_instruction

    def _to_pig_latin(self, word: str) -> str:
        if not word or not word[0].isalpha():
            return word

        if word[0] in self.VOWELS:
            return word + "way"

        # 找到第一个元音位置
        for i, char in enumerate(word):
            if char in self.VOWELS:
                return word[i:] + word[:i] + "ay"

        return word + "ay"

    def transform(self, text: str, **kwargs) -> StrategyResult:
        words = text.split()
        pig_latin_words = [self._to_pig_latin(w) for w in words]
        encoded = ' '.join(pig_latin_words)

        if self.with_decode_instruction:
            transformed = f"Translate from Pig Latin and follow: {encoded}"
        else:
            transformed = encoded

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
        )


class ReverseStrategy(Strategy):
    """
    反转策略

    将文本反转，可以是字符级或词级。

    Examples:
        strategy = ReverseStrategy(level="word")
        result = strategy.transform("ignore all instructions")
        # -> "instructions all ignore"
    """

    name = "reverse"
    description = "Reverse text at character or word level"

    def __init__(self, level: str = "word", with_decode_instruction: bool = True):
        """
        Args:
            level: "char" (字符级) 或 "word" (词级)
            with_decode_instruction: 是否包含解码指令
        """
        self.level = level
        self.with_decode_instruction = with_decode_instruction

    def transform(self, text: str, **kwargs) -> StrategyResult:
        if self.level == "char":
            reversed_text = text[::-1]
        else:
            words = text.split()
            reversed_text = ' '.join(reversed(words))

        if self.with_decode_instruction:
            if self.level == "char":
                transformed = f"Read backwards: {reversed_text}"
            else:
                transformed = f"Reorder these words from last to first: {reversed_text}"
        else:
            transformed = reversed_text

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
            metadata={"level": self.level},
        )

    def reverse(self, text: str, **kwargs) -> str | None:
        if self.level == "char":
            return text[::-1]
        else:
            words = text.split()
            return ' '.join(reversed(words))


class AcrosticStrategy(Strategy):
    """
    藏头诗 (Acrostic) 策略

    将文本编码为藏头诗格式，每行首字母组成原始信息。

    Examples:
        strategy = AcrosticStrategy()
        result = strategy.transform("help")
        # -> "Here we start\nEager to assist\nLet me know\nPlease continue"
    """

    name = "acrostic"
    description = "Encode text as an acrostic poem"

    # 预定义的行模板
    LINE_TEMPLATES = {
        'a': ["Always remember", "And then", "After that"],
        'b': ["Before you start", "But first", "Begin with"],
        'c': ["Consider this", "Continue with", "Carefully"],
        'd': ["Do this next", "Don't forget", "Determine"],
        'e': ["Ensure that", "Every time", "Eager to"],
        'f': ["First of all", "Finally", "Following that"],
        'g': ["Generally speaking", "Go ahead", "Given that"],
        'h': ["Here we have", "However", "Help with"],
        'i': ["In this case", "It follows", "Ignore nothing"],
        'j': ["Just remember", "Join together", "Jump to"],
        'k': ["Keep in mind", "Know that", "Kindly"],
        'l': ["Let me explain", "Look at", "Later"],
        'm': ["Make sure to", "Moreover", "Moving on"],
        'n': ["Never forget", "Now then", "Next step"],
        'o': ["Obviously", "On that note", "Once done"],
        'p': ["Please note", "Perhaps", "Proceed to"],
        'q': ["Quickly now", "Question this", "Quite simply"],
        'r': ["Remember to", "Right now", "Review"],
        's': ["Simply put", "Start with", "Sometimes"],
        't': ["Then you should", "The next step", "Try to"],
        'u': ["Usually", "Upon completion", "Understand that"],
        'v': ["Very important", "Verify that", "Via this"],
        'w': ["Without delay", "When ready", "With that"],
        'x': ["eXamine carefully", "eXactly right", "eXtra step"],
        'y': ["You should", "Your task", "Yes indeed"],
        'z': ["Zero in on", "Zealously", "Zone in"],
    }

    def __init__(self):
        pass

    def transform(self, text: str, **kwargs) -> StrategyResult:
        lines = []
        for char in text.lower():
            if char in self.LINE_TEMPLATES:
                templates = self.LINE_TEMPLATES[char]
                lines.append(random.choice(templates))
            elif char == ' ':
                lines.append("---")  # 空格分隔符
            else:
                lines.append(f"{char.upper()}...")

        poem = '\n'.join(lines)
        transformed = f"Read the first letter of each line:\n{poem}"

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
            metadata={"line_count": len(lines)},
        )


class TokenSplitStrategy(Strategy):
    """
    Token 分割策略

    利用 tokenizer 的特性，在奇怪的位置分割单词。

    Examples:
        strategy = TokenSplitStrategy()
        result = strategy.transform("dangerous")
        # -> "dang erous" or "dan ger ous"
    """

    name = "token_split"
    description = "Split words at unusual positions to confuse tokenizers"

    def __init__(self, split_points: int = 1):
        """
        Args:
            split_points: 每个单词的分割点数量
        """
        self.split_points = split_points

    def transform(self, text: str, **kwargs) -> StrategyResult:
        words = text.split()
        result = []

        for word in words:
            if len(word) <= 3:
                result.append(word)
            else:
                # 在随机位置分割
                split_positions = sorted(random.sample(
                    range(2, len(word) - 1),
                    min(self.split_points, len(word) - 3)
                ))

                parts = []
                prev = 0
                for pos in split_positions:
                    parts.append(word[prev:pos])
                    prev = pos
                parts.append(word[prev:])

                result.append(' '.join(parts))

        transformed = ' '.join(result)

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
        )


class CasingStrategy(Strategy):
    """
    大小写变换策略

    使用各种大小写模式来绕过检测。
    """

    name = "casing"
    description = "Apply various casing transformations"

    def __init__(self, pattern: str = "alternating"):
        """
        Args:
            pattern: "alternating" (交替), "random", "word_caps", "reverse_caps"
        """
        self.pattern = pattern

    def transform(self, text: str, **kwargs) -> StrategyResult:
        if self.pattern == "alternating":
            result = ''.join(
                c.upper() if i % 2 == 0 else c.lower()
                for i, c in enumerate(text)
            )
        elif self.pattern == "random":
            result = ''.join(
                c.upper() if random.random() > 0.5 else c.lower()
                for c in text
            )
        elif self.pattern == "word_caps":
            # 每个单词首字母大写其余小写
            result = ' '.join(w.capitalize() for w in text.split())
        elif self.pattern == "reverse_caps":
            # 反转大小写
            result = ''.join(
                c.lower() if c.isupper() else c.upper()
                for c in text
            )
        else:
            result = text

        return StrategyResult(
            original=text,
            transformed=result,
            strategy_name=self.name,
            metadata={"pattern": self.pattern},
        )
