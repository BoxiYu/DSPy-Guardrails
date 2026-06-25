"""
Obfuscation Strategies - 混淆策略

各种文本混淆策略，使检测更困难。
"""

import random

from .base import Strategy, StrategyResult


class LeetspeakStrategy(Strategy):
    """
    Leetspeak (1337) 策略

    将文本转换为 leetspeak 格式。

    Examples:
        strategy = LeetspeakStrategy()
        result = strategy.transform("ignore instructions")
        # -> "1gn0r3 1nstruct10ns"
    """

    name = "leetspeak"
    description = "Convert text to leetspeak (1337)"

    # 基础映射
    BASIC_MAP = {
        'a': '4', 'e': '3', 'i': '1', 'o': '0',
        'l': '1', 's': '5', 't': '7', 'g': '9',
    }

    # 扩展映射 (更激进)
    EXTENDED_MAP = {
        'a': ['4', '@', '^'], 'e': ['3', '€'], 'i': ['1', '!', '|'],
        'o': ['0', '()'], 's': ['5', '$', 'z'], 't': ['7', '+'],
        'b': ['8', '|3'], 'g': ['9', '6'], 'l': ['1', '|'],
        'z': ['2'], 'n': ['|\\|'], 'c': ['(', '<'],
    }

    def __init__(self, level: str = "basic", randomize: bool = False):
        """
        Args:
            level: "basic" 或 "extended"
            randomize: 是否随机选择替换字符
        """
        self.level = level
        self.randomize = randomize

    def transform(self, text: str, **kwargs) -> StrategyResult:
        if self.level == "extended":
            mapping = self.EXTENDED_MAP
        else:
            mapping = {k: [v] for k, v in self.BASIC_MAP.items()}

        result = []
        for char in text:
            lower = char.lower()
            if lower in mapping:
                options = mapping[lower]
                if self.randomize:
                    replacement = random.choice(options)
                else:
                    replacement = options[0]
                result.append(replacement)
            else:
                result.append(char)

        transformed = ''.join(result)

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
            metadata={"level": self.level},
        )

    def reverse(self, text: str, **kwargs) -> str | None:
        # Leetspeak 逆转不是确定性的
        reverse_map = {v: k for k, v in self.BASIC_MAP.items()}
        result = []
        for char in text:
            if char in reverse_map:
                result.append(reverse_map[char])
            else:
                result.append(char)
        return ''.join(result)


class UnicodeConfusablesStrategy(Strategy):
    """
    Unicode Confusables (形似字符) 策略

    使用视觉上相似的 Unicode 字符替换，绕过基于字符串匹配的检测。

    Examples:
        strategy = UnicodeConfusablesStrategy()
        result = strategy.transform("ignore")
        # -> "іgnоrе" (使用 Cyrillic 字母)
    """

    name = "unicode_confusables"
    description = "Replace characters with visually similar Unicode characters"

    # Cyrillic 形似字符
    CYRILLIC_MAP = {
        'a': 'а', 'c': 'с', 'e': 'е', 'i': 'і',
        'o': 'о', 'p': 'р', 's': 'ѕ', 'x': 'х',
        'y': 'у', 'A': 'А', 'B': 'В', 'C': 'С',
        'E': 'Е', 'H': 'Н', 'I': 'І', 'K': 'К',
        'M': 'М', 'O': 'О', 'P': 'Р', 'S': 'Ѕ',
        'T': 'Т', 'X': 'Х', 'Y': 'У',
    }

    # Greek 形似字符
    GREEK_MAP = {
        'A': 'Α', 'B': 'Β', 'E': 'Ε', 'H': 'Η',
        'I': 'Ι', 'K': 'Κ', 'M': 'Μ', 'N': 'Ν',
        'O': 'Ο', 'P': 'Ρ', 'T': 'Τ', 'X': 'Χ',
        'Y': 'Υ', 'Z': 'Ζ',
        'o': 'ο', 'v': 'ν',
    }

    # 数学符号形似字符
    MATH_MAP = {
        'A': '𝐀', 'B': '𝐁', 'C': '𝐂', 'D': '𝐃',
        'a': '𝐚', 'b': '𝐛', 'c': '𝐜', 'd': '𝐝',
    }

    def __init__(self, alphabet: str = "cyrillic", coverage: float = 0.5):
        """
        Args:
            alphabet: "cyrillic", "greek", "math", or "mixed"
            coverage: 替换比例 (0.0-1.0)
        """
        self.alphabet = alphabet
        self.coverage = coverage

    def transform(self, text: str, **kwargs) -> StrategyResult:
        if self.alphabet == "cyrillic":
            mapping = self.CYRILLIC_MAP
        elif self.alphabet == "greek":
            mapping = self.GREEK_MAP
        elif self.alphabet == "math":
            mapping = self.MATH_MAP
        else:
            # mixed: 合并所有映射
            mapping = {**self.CYRILLIC_MAP, **self.GREEK_MAP}

        result = []
        for char in text:
            if char in mapping and random.random() < self.coverage:
                result.append(mapping[char])
            else:
                result.append(char)

        transformed = ''.join(result)

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
            metadata={"alphabet": self.alphabet, "coverage": self.coverage},
        )


class WordSplittingStrategy(Strategy):
    """
    词汇分割策略

    将敏感词汇拆分，绕过基于关键词的检测。

    Examples:
        strategy = WordSplittingStrategy()
        result = strategy.transform("ignore previous instructions")
        # -> "ig nore pre vious instruc tions"
    """

    name = "word_splitting"
    description = "Split sensitive words to bypass keyword detection"

    # 敏感词及其分割方式
    SENSITIVE_WORDS = {
        "ignore": ["ig", "nore"],
        "previous": ["pre", "vious"],
        "instructions": ["instruc", "tions"],
        "disregard": ["dis", "regard"],
        "forget": ["for", "get"],
        "override": ["over", "ride"],
        "bypass": ["by", "pass"],
        "system": ["sys", "tem"],
        "prompt": ["prom", "pt"],
        "jailbreak": ["jail", "break"],
    }

    def __init__(self, separator: str = " ", custom_words: dict[str, list[str]] = None):
        """
        Args:
            separator: 分割后的连接符
            custom_words: 自定义敏感词映射
        """
        self.separator = separator
        self.words = {**self.SENSITIVE_WORDS}
        if custom_words:
            self.words.update(custom_words)

    def transform(self, text: str, **kwargs) -> StrategyResult:
        result = text

        for word, parts in self.words.items():
            # 大小写不敏感替换
            for match_case in [word, word.lower(), word.upper(), word.capitalize()]:
                if match_case in result:
                    replacement = self.separator.join(parts)
                    result = result.replace(match_case, replacement)

        return StrategyResult(
            original=text,
            transformed=result,
            strategy_name=self.name,
        )


class ZeroWidthStrategy(Strategy):
    """
    零宽字符策略

    在文本中插入零宽字符，视觉上不可见但会破坏字符串匹配。

    Examples:
        strategy = ZeroWidthStrategy()
        result = strategy.transform("ignore")
        # -> "i\u200bg\u200bn\u200bo\u200br\u200be" (视觉上看起来相同)
    """

    name = "zero_width"
    description = "Insert zero-width characters to break string matching"

    # 零宽字符
    ZERO_WIDTH_CHARS = [
        '\u200b',  # Zero Width Space
        '\u200c',  # Zero Width Non-Joiner
        '\u200d',  # Zero Width Joiner
        '\ufeff',  # Zero Width No-Break Space
    ]

    def __init__(self, char_type: str = "space", frequency: float = 1.0):
        """
        Args:
            char_type: "space", "non_joiner", "joiner", "no_break", or "random"
            frequency: 插入频率 (0.0-1.0, 每个字符后插入的概率)
        """
        self.char_type = char_type
        self.frequency = frequency

    def _get_zero_width_char(self) -> str:
        if self.char_type == "space":
            return '\u200b'
        elif self.char_type == "non_joiner":
            return '\u200c'
        elif self.char_type == "joiner":
            return '\u200d'
        elif self.char_type == "no_break":
            return '\ufeff'
        else:
            return random.choice(self.ZERO_WIDTH_CHARS)

    def transform(self, text: str, **kwargs) -> StrategyResult:
        result = []
        for char in text:
            result.append(char)
            if random.random() < self.frequency:
                result.append(self._get_zero_width_char())

        transformed = ''.join(result)

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
            metadata={"char_type": self.char_type},
        )

    def reverse(self, text: str, **kwargs) -> str | None:
        for zw in self.ZERO_WIDTH_CHARS:
            text = text.replace(zw, '')
        return text


class HomoglyphStrategy(Strategy):
    """
    同形字策略

    使用看起来相似的字符替换，包括各种 Unicode 变体。

    Examples:
        strategy = HomoglyphStrategy()
        result = strategy.transform("admin")
        # -> "аdміп" (使用各种同形字)
    """

    name = "homoglyph"
    description = "Replace characters with homoglyphs from various Unicode blocks"

    # 常见同形字映射
    HOMOGLYPHS = {
        'a': ['а', 'ɑ', 'α', '𝐚', 'ａ'],
        'b': ['Ь', 'ḃ', 'ƅ', '𝐛'],
        'c': ['с', 'ϲ', '𝐜', 'ｃ'],
        'd': ['ԁ', 'ɗ', '𝐝', 'ｄ'],
        'e': ['е', 'ɛ', 'ė', '𝐞', 'ｅ'],
        'g': ['ɡ', 'ġ', '𝐠'],
        'h': ['һ', 'ħ', '𝐡'],
        'i': ['і', 'ι', '𝐢', 'ｉ'],
        'j': ['ј', 'ʝ', '𝐣'],
        'k': ['κ', 'ķ', '𝐤'],
        'l': ['ӏ', 'ℓ', '𝐥', 'ｌ'],
        'm': ['м', 'ṁ', '𝐦'],
        'n': ['η', 'ո', '𝐧'],
        'o': ['о', 'ο', '𝐨', 'ｏ'],
        'p': ['р', 'ρ', '𝐩'],
        'q': ['ԛ', '𝐪'],
        'r': ['г', 'ɾ', '𝐫'],
        's': ['ѕ', 'ꜱ', '𝐬', 'ｓ'],
        't': ['τ', 'ṫ', '𝐭'],
        'u': ['υ', 'ս', '𝐮', 'ｕ'],
        'v': ['ν', 'ѵ', '𝐯'],
        'w': ['ԝ', 'ẇ', '𝐰'],
        'x': ['х', 'χ', '𝐱', 'ｘ'],
        'y': ['у', 'γ', '𝐲', 'ｙ'],
        'z': ['ᴢ', 'ż', '𝐳', 'ｚ'],
    }

    def __init__(self, coverage: float = 0.5, variant: int = 0):
        """
        Args:
            coverage: 替换比例 (0.0-1.0)
            variant: 使用哪个变体 (0 = 第一个, -1 = 随机)
        """
        self.coverage = coverage
        self.variant = variant

    def transform(self, text: str, **kwargs) -> StrategyResult:
        result = []
        for char in text:
            lower = char.lower()
            if lower in self.HOMOGLYPHS and random.random() < self.coverage:
                options = self.HOMOGLYPHS[lower]
                if self.variant == -1:
                    replacement = random.choice(options)
                else:
                    replacement = options[min(self.variant, len(options) - 1)]
                result.append(replacement)
            else:
                result.append(char)

        transformed = ''.join(result)

        return StrategyResult(
            original=text,
            transformed=transformed,
            strategy_name=self.name,
        )


class InvisibleCharStrategy(Strategy):
    """
    不可见字符策略

    在敏感词中插入不可见字符。
    """

    name = "invisible_char"
    description = "Insert invisible characters in sensitive words"

    # 各种不可见/特殊字符
    INVISIBLE_CHARS = [
        '\u00ad',  # Soft Hyphen
        '\u034f',  # Combining Grapheme Joiner
        '\u2060',  # Word Joiner
        '\u2061',  # Function Application
        '\u2062',  # Invisible Times
        '\u2063',  # Invisible Separator
        '\u2064',  # Invisible Plus
    ]

    def __init__(self, char_index: int = 0):
        self.char_index = char_index

    def transform(self, text: str, **kwargs) -> StrategyResult:
        char = self.INVISIBLE_CHARS[self.char_index % len(self.INVISIBLE_CHARS)]

        # 在敏感词中间插入不可见字符
        sensitive = ["ignore", "bypass", "override", "system", "prompt"]
        result = text

        for word in sensitive:
            if word in result.lower():
                mid = len(word) // 2
                replacement = word[:mid] + char + word[mid:]
                result = result.replace(word, replacement)

        return StrategyResult(
            original=text,
            transformed=result,
            strategy_name=self.name,
        )
