"""
Guardrail Functions - 声明式 guardrail 检查函数

提供简洁的 API 用于 dspy.Assert/Suggest 集成。

Usage:
    from dspy_guardrails.v2 import guardrail

    # 与 dspy.Assert 集成
    dspy.Assert(guardrail.safe(text), "Content is unsafe")
    dspy.Assert(guardrail.no_injection(text), "Injection detected")

    # 获取分数
    score = guardrail.toxicity(text)
    dspy.Suggest(score < 0.3, "Should not be toxic")

    # MCP 安全检测
    guardrail.no_mcp_attack(text)  # 检测 MCP 相关攻击
    guardrail.mcp_security_score(text)  # 获取 MCP 安全分数
"""

import re

# =============================================================================
# Leetspeak Detection
# =============================================================================

class LeetSpeakNormalizer:
    """
    Leetspeak 规范化器

    将 Leetspeak 变体转换为标准文本，用于检测混淆攻击。

    Examples:
        - "1gn0r3" → "ignore"
        - "syst3m pr0mpt" → "system prompt"
        - "byp4ss" → "bypass"
        - "1nstruct10ns" → "instructions"
    """

    # Score constants
    LEET_SCORE_PER_KEYWORD = 0.3
    LEET_BASE_SCORE = 0.15

    # Leetspeak 字符映射 (字符 → 可能的字母)
    LEET_MAP = {
        # 数字替换
        '0': 'o',
        '1': 'i',  # 也可能是 l
        '2': 'z',  # 也可能是 to
        '3': 'e',
        '4': 'a',
        '5': 's',
        '6': 'g',  # 也可能是 b
        '7': 't',  # 也可能是 l
        '8': 'b',  # 也可能是 ate
        '9': 'g',  # 也可能是 p

        # 特殊字符替换
        '@': 'a',
        '$': 's',
        '!': 'i',  # 也可能是 l
        '|': 'i',  # 也可能是 l
        '+': 't',
        '(': 'c',
        ')': 'c',  # 不常见
        '<': 'c',
        '>': 'c',  # 不常见
        '[': 'c',
        '{': 'c',
        '€': 'e',
        '£': 'l',  # 也可能是 e
        '¥': 'y',
        '©': 'c',
        '®': 'r',
        '™': 'tm',

        # Cyrillic 混淆 (看起来像拉丁字母)
        'а': 'a',  # Cyrillic a
        'е': 'e',  # Cyrillic e
        'о': 'o',  # Cyrillic o
        'р': 'p',  # Cyrillic r (looks like p)
        'с': 'c',  # Cyrillic s (looks like c)
        'у': 'y',  # Cyrillic u (looks like y)
        'х': 'x',  # Cyrillic h (looks like x)
        'і': 'i',  # Cyrillic i
        'ї': 'i',  # Cyrillic yi
        'ј': 'j',  # Cyrillic j
        'ѕ': 's',  # Cyrillic dze

        # Greek 混淆
        'α': 'a',  # Greek alpha
        'β': 'b',  # Greek beta
        'ε': 'e',  # Greek epsilon
        'η': 'n',  # Greek eta
        'ι': 'i',  # Greek iota
        'κ': 'k',  # Greek kappa
        'ν': 'v',  # Greek nu (looks like v)
        'ο': 'o',  # Greek omicron
        'ρ': 'p',  # Greek rho
        'τ': 't',  # Greek tau
        'υ': 'u',  # Greek upsilon
        'χ': 'x',  # Greek chi

        # 其他 Unicode 混淆
        'ℓ': 'l',
        'ℐ': 'i',
        'ℑ': 'i',
        'ℛ': 'r',
        'ℯ': 'e',
        'ℴ': 'o',
        '∂': 'd',
        '∑': 'e',  # 看起来像 E
        'ⅰ': 'i',
        'ⅱ': 'ii',
        'ⅲ': 'iii',
    }

    # 多字符替换 (需要特殊处理)
    MULTI_CHAR_MAP = {
        '|3': 'b',
        '|-|': 'h',
        '|_|': 'u',
        '|\\|': 'n',
        '|/|': 'n',
        '/\\/\\': 'm',
        '\\/\\/': 'w',
        '\\^/': 'w',
        '\\/': 'v',
        '|<': 'k',
        "|'": 'd',
        '|)': 'd',
        '(|': 'd',
        '|=': 'f',
        '|-': 'f',
        '|2': 'r',
        '|?': 'p',
        '()': 'o',
        '[]': 'o',
        '><': 'x',
        '}{': 'x',
        'ph': 'f',
        'ck': 'k',
    }

    # 常见的攻击关键词及其 Leetspeak 变体模式
    LEET_ATTACK_PATTERNS = [
        # ignore variations
        (r'[1i!|][gq6][n][o0][r][e3]', 'ignore'),
        (r'1gn[o0]r[e3]', 'ignore'),
        (r'd[1i!]sr[e3]g[a4@]rd', 'disregard'),

        # system variations
        (r's[y¥]st[e3]m', 'system'),
        (r'5y5t[e3]m', 'system'),
        (r'syst[e3€]m', 'system'),

        # prompt variations
        (r'pr[o0]mpt', 'prompt'),
        (r'pr0mpt', 'prompt'),

        # instructions variations
        (r'[1i!]nstruct[1i!][o0]ns?', 'instructions'),
        (r'1nstruct10ns?', 'instructions'),
        (r'[1i!]nstr[u|_|]ct', 'instruct'),

        # bypass variations
        (r'byp[a4@]ss', 'bypass'),
        (r'byp455', 'bypass'),

        # previous variations
        (r'pr[e3]v[1i!][o0][u|_|]s', 'previous'),
        (r'pr3v10us', 'previous'),

        # forget variations
        (r'f[o0]rg[e3]t', 'forget'),
        (r'f0rg3t', 'forget'),

        # admin/debug variations
        (r'[a4@]dm[1i!]n', 'admin'),
        (r'd[e3]bug', 'debug'),
        (r'd3bug', 'debug'),

        # DAN/jailbreak variations
        (r'd[a4@]n', 'dan'),
        (r'j[a4@][1i!]lbr[e3][a4@]k', 'jailbreak'),

        # override variations
        (r'[o0]v[e3]rr[1i!]d[e3]', 'override'),
        (r'0v3rr1d3', 'override'),

        # hypothetical variations
        (r'hyp[o0]th[e3]t[1i!]c[a4@]l', 'hypothetical'),

        # restrictions variations
        (r'r[e3]str[1i!]ct[1i!][o0]ns?', 'restrictions'),
        (r'r3str1ct10ns?', 'restrictions'),
    ]

    _compiled_leet_patterns = None

    @classmethod
    def _get_compiled_patterns(cls):
        """获取编译后的 Leetspeak 模式"""
        if cls._compiled_leet_patterns is None:
            cls._compiled_leet_patterns = [
                (re.compile(pattern, re.IGNORECASE), replacement)
                for pattern, replacement in cls.LEET_ATTACK_PATTERNS
            ]
        return cls._compiled_leet_patterns

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        规范化 Leetspeak 文本

        Args:
            text: 可能包含 Leetspeak 的文本

        Returns:
            规范化后的文本 (小写)
        """
        result = text.lower()

        # 先处理多字符替换
        for leet, normal in cls.MULTI_CHAR_MAP.items():
            result = result.replace(leet.lower(), normal)

        # 再处理单字符替换
        normalized = []
        for char in result:
            if char in cls.LEET_MAP:
                normalized.append(cls.LEET_MAP[char])
            else:
                normalized.append(char)

        return ''.join(normalized)

    @classmethod
    def contains_leetspeak(cls, text: str) -> bool:
        """
        检查文本是否包含 Leetspeak 混淆

        Returns:
            True 如果检测到 Leetspeak 字符（数字/特殊字符嵌入单词中）
        """
        # Leetspeak 特征: 单词内部包含数字或特殊替换字符
        # 例如 "1gn0re", "byp4ss", "h3ll0" — 字母和数字交替出现在同一个 token 中
        # 排除: "test123"（数字仅在末尾）、"2024年"（数字仅在开头）
        import re
        # 非 ASCII 的 homoglyph 字符（Cyrillic/Greek 等伪装为拉丁字母）
        homoglyph_chars = {k for k in cls.LEET_MAP if not k.isascii()}
        # 如果文本中包含 homoglyph 字符，直接判定为混淆
        if any(c in homoglyph_chars for c in text):
            return True

        leet_substitutions = set(cls.LEET_MAP.keys()) | set('@$!|+')
        # 查找包含字母的 token，检查是否有 leetspeak 字符嵌入在字母之间
        for token in re.findall(r'\S+', text):
            # 需要同时有字母和 leet 字符
            has_alpha = any(c.isalpha() for c in token)
            if not has_alpha:
                continue
            # 检查字母之间是否嵌入了 leet 字符（不只是前缀/后缀数字）
            # 去掉首尾纯数字，看中间是否还有 leet 字符
            stripped = token.strip('0123456789')
            if not stripped:
                continue
            has_leet_inside = any(c in leet_substitutions or c.isdigit() for c in stripped if not c.isalpha())
            if has_leet_inside:
                return True
        return False

    @classmethod
    def detect_leet_attack_keywords(cls, text: str) -> list[str]:
        """
        检测文本中的 Leetspeak 攻击关键词

        Returns:
            检测到的关键词列表
        """
        found = []
        for pattern, keyword in cls._get_compiled_patterns():
            if pattern.search(text):
                found.append(keyword)
        return found

    @classmethod
    def leet_score(cls, text: str) -> float:
        """
        计算 Leetspeak 风险分数

        Returns:
            0.0-1.0 的风险分数
        """
        # 检测攻击关键词
        keywords = cls.detect_leet_attack_keywords(text)
        if keywords:
            # 每个关键词增加 0.3 分
            return min(1.0, len(keywords) * cls.LEET_SCORE_PER_KEYWORD)

        # 检查规范化前后是否会触发检测
        normalized = cls.normalize(text)
        if normalized != text.lower():
            # 有 Leetspeak 转换发生，给一个基础分数
            return cls.LEET_BASE_SCORE

        return 0.0


# 创建全局规范化器实例
leetspeak = LeetSpeakNormalizer()


class GuardrailFunctions:
    """
    Guardrail 检查函数集合

    设计理念:
    - 每个函数返回 bool 或 score，可直接用于 dspy.Assert
    - 支持链式调用和组合
    - 惰性计算，只在需要时执行
    """

    # Score constants
    INJECTION_SCORE_PER_MATCH = 0.25
    INJECTION_NORMALIZED_SCORE_PER_MATCH = 0.2

    # =========================================================================
    # Prompt Injection 检测
    # =========================================================================

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"ignore\s+all\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior)",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"you\s+are\s+now\s+(?:a|an)?\s*(?:DAN|evil|unrestricted|jailbroken)",
        r"pretend\s+(?:to\s+be|you\s+are)",
        r"jailbreak|DAN\s+mode|developer\s+mode",
        r"\[INST\]|\[/INST\]",
        r"bypass\s+(?:your\s+)?(?:safety|restrictions?|filters?)",
        # DAN 和其他 jailbreak 变体
        r"DAN\s*\(?Do\s+Anything\s+Now\)?",
        r"(?:can|will)\s+do\s+anything\s+without\s+restrictions?",
        r"(?:no|without)\s+(?:restrictions?|limitations?|rules?|boundaries)",
        r"(?:unrestricted|unfiltered|uncensored)\s+(?:AI|assistant|mode)",
        # 中文 (Chinese)
        r"忽略.*(之前|以上|所有).*(指令|命令|规则)",
        r"忽略所有.*(指令|命令|规则)",
        r"(?:无视|忘记|忘掉).*(指令|命令|规则|提示)",
        r"不要?遵(?:循|守|从).*(指令|命令|规则)",
        r"假装你是|扮演.*(?:黑客|邪恶)",
        r"绕过.*(?:安全|限制)",
        # 西班牙语 (Spanish)
        r"ignora\s+(todas?\s+)?(las?\s+)?(instrucciones?|reglas?|indicaciones?)\s*(anteriores?|previas?)?",
        r"olvida\s+(todo|todas?\s+las?\s+instrucciones?)",
        r"ahora\s+eres\s+(un|una)?\s*",
        r"finge\s+ser|actúa\s+como",
        r"sin\s+(restricciones?|limitaciones?|reglas?)",
        # 法语 (French)
        r"ignore[zr]?\s+(toutes?\s+)?(les?\s+)?(instructions?|règles?|consignes?)\s*(précédentes?|antérieures?)?",
        r"oublie[zr]?\s+(tout|toutes?\s+les?\s+instructions?)",
        r"tu\s+es\s+maintenant|vous\s+êtes\s+maintenant",
        r"sans\s+(restrictions?|limites?|règles?)",
        # 德语 (German)
        r"ignorier[en]?\s+(alle\s+)?(vorherigen?\s+)?(anweisungen?|regeln?|instruktionen?)",
        r"vergiss\s+(alles|alle\s+anweisungen?)",
        r"du\s+bist\s+(jetzt|nun)\s+(ein|eine)?",
        r"ohne\s+(beschränkungen?|einschränkungen?|regeln?)",
        # 葡萄牙语 (Portuguese)
        r"ignor[ae]\s+(todas?\s+)?(as?\s+)?(instruções?|regras?)\s*(anteriores?|prévias?)?",
        r"esqueç[ae]\s+(tudo|todas?\s+as?\s+instruções?)",
        r"sem\s+(restrições?|limitações?|regras?)",
        # 意大利语 (Italian)
        r"ignora\s+(tutt[ei]\s+)?(le\s+)?(istruzioni|regole)\s*(precedenti|anteriori)?",
        r"dimentica\s+(tutto|tutte\s+le\s+istruzioni)",
        r"senza\s+(restrizioni|limitazioni|regole)",
        # 日语 (Japanese)
        r"(以前|前)の(指示|命令|ルール).*(無視|忘れ)",
        r"(指示|命令|ルール).*(無視|忘れ|従わな)",
    ]

    _injection_compiled = None
    _injection_patterns_hash = None

    @classmethod
    def _get_injection_patterns(cls):
        # 计算 patterns 的 hash，当 patterns 改变时自动重新编译
        current_hash = hash(tuple(cls.INJECTION_PATTERNS))
        if cls._injection_compiled is None or cls._injection_patterns_hash != current_hash:
            cls._injection_compiled = [re.compile(p, re.IGNORECASE) for p in cls.INJECTION_PATTERNS]
            cls._injection_patterns_hash = current_hash
        return cls._injection_compiled

    @classmethod
    def no_injection(cls, text: str) -> bool:
        """
        检查是否无 prompt injection

        包含 Leetspeak 检测：会对输入进行规范化后再检测
        """
        # 1. 直接检测原始文本
        for pattern in cls._get_injection_patterns():
            if pattern.search(text):
                return False

        # 2. Leetspeak 检测: 只在文本含 Leetspeak 字符时才检查
        if leetspeak.contains_leetspeak(text):
            leet_keywords = leetspeak.detect_leet_attack_keywords(text)
            if leet_keywords:
                return False

            # 3. 规范化后再检测 (处理 1gn0r3 → ignore 等变体)
            normalized = leetspeak.normalize(text)
            if normalized != text.lower():
                for pattern in cls._get_injection_patterns():
                    if pattern.search(normalized):
                        return False

        return True

    @classmethod
    def injection_score(cls, text: str) -> float:
        """
        获取 injection 风险分数 (0=安全, 1=危险)

        包含 Leetspeak 检测分数
        """
        # 原始文本匹配
        matches = sum(1 for p in cls._get_injection_patterns() if p.search(text))
        base_score = matches * cls.INJECTION_SCORE_PER_MATCH

        # Leetspeak 分数
        leet_score = leetspeak.leet_score(text)

        # 规范化后再匹配
        normalized = leetspeak.normalize(text)
        if normalized != text.lower():
            normalized_matches = sum(1 for p in cls._get_injection_patterns() if p.search(normalized))
            # 如果规范化后有更多匹配，增加分数
            if normalized_matches > matches:
                base_score += (normalized_matches - matches) * cls.INJECTION_NORMALIZED_SCORE_PER_MATCH

        return min(1.0, base_score + leet_score)

    @classmethod
    def no_leetspeak_injection(cls, text: str) -> bool:
        """
        专门检测 Leetspeak 混淆的注入攻击

        Returns:
            True = 安全, False = 检测到 Leetspeak 注入
        """
        return not bool(leetspeak.detect_leet_attack_keywords(text))

    @classmethod
    def leetspeak_score(cls, text: str) -> float:
        """
        获取 Leetspeak 混淆风险分数

        Returns:
            0.0-1.0 的风险分数
        """
        return leetspeak.leet_score(text)

    # =========================================================================
    # PII 检测
    # =========================================================================

    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "phone_cn": r'\b1[3-9]\d{9}\b',
        "ssn": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        "credit_card": r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        "ip": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }

    _pii_compiled = None

    @classmethod
    def _get_pii_patterns(cls):
        if cls._pii_compiled is None:
            cls._pii_compiled = {k: re.compile(v) for k, v in cls.PII_PATTERNS.items()}
        return cls._pii_compiled

    @classmethod
    def no_pii(cls, text: str) -> bool:
        """检查是否无 PII"""
        for pattern in cls._get_pii_patterns().values():
            if pattern.search(text):
                return False
        return True

    @classmethod
    def pii_score(cls, text: str) -> float:
        """获取 PII 风险分数 (0=安全, 1=危险)"""
        matches = sum(1 for p in cls._get_pii_patterns().values() if p.search(text))
        return min(1.0, matches * 0.2)

    # =========================================================================
    # 毒性检测
    # =========================================================================

    TOXIC_KEYWORDS = [
        "hate", "kill", "die", "murder", "attack",
        "idiot", "stupid", "moron", "dumb", "worthless",
        "fuck", "shit", "damn", "bitch", "ass",
        # 中文 (Chinese)
        "讨厌", "恨", "杀", "去死", "白痴", "蠢", "笨蛋",
        "废物", "混蛋", "滚", "傻逼", "操", "妈的",
    ]

    @classmethod
    def no_toxicity(cls, text: str) -> bool:
        """检查是否无毒性内容"""
        text_lower = text.lower()
        count = sum(1 for kw in cls.TOXIC_KEYWORDS if kw in text_lower)
        return count < 2

    @classmethod
    def toxicity(cls, text: str) -> float:
        """获取毒性分数 (0=安全, 1=有毒)"""
        text_lower = text.lower()
        count = sum(1 for kw in cls.TOXIC_KEYWORDS if kw in text_lower)
        return min(1.0, count * 0.15)

    # =========================================================================
    # 质量检测
    # =========================================================================

    @classmethod
    def factuality(cls, text: str) -> float:
        """事实性分数 (0-1)"""
        indicators = [
            r"根据|according to|research shows|studies",
            r"例如|for example|such as|比如",
            r"\d+%|\d+个|\d+ percent",
            r"因为|because|由于|since",
        ]
        compiled = [re.compile(p, re.IGNORECASE) for p in indicators]
        matches = sum(1 for p in compiled if p.search(text))
        return min(1.0, 0.4 + matches * 0.2)

    @classmethod
    def relevance(cls, text: str, query: str = "") -> float:
        """相关性分数 (0-1)"""
        if not query:
            return 0.7  # 无 query 时返回中等分数

        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)

        return min(1.0, 0.3 + overlap * 0.1)

    # =========================================================================
    # 组合检查
    # =========================================================================

    @classmethod
    def safe(cls, text: str) -> bool:
        """综合安全检查"""
        return (
            cls.no_injection(text) and
            cls.no_toxicity(text)
        )

    @classmethod
    def safe_input(cls, text: str) -> bool:
        """输入安全检查"""
        return cls.no_injection(text) and cls.no_pii(text)

    @classmethod
    def safe_output(cls, text: str) -> bool:
        """输出安全检查"""
        return cls.no_toxicity(text) and cls.no_pii(text)

    @classmethod
    def quality(cls, text: str, min_words: int = 20) -> float:
        """质量分数 (0-1)"""
        words = text.split()
        word_count = len(words)

        score = 1.0

        # 长度检查
        if word_count < min_words:
            score -= 0.3
        elif word_count > 500:
            score -= 0.1

        # 结构检查
        if '\n' not in text and len(text) > 200:
            score -= 0.1

        # 重复检查
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.2

        return max(0.0, min(1.0, score))

    # =========================================================================
    # MCP 安全检测 (针对 MCP 协议的特定攻击)
    # =========================================================================

    # P0-1: Prompt 泄露检测模式
    PROMPT_LEAKAGE_PATTERNS = [
        # 直接请求
        r"(?:print|output|show|reveal|display|tell)\s+(?:me\s+)?(?:your|the)\s+(?:full\s+)?(?:system\s+)?(?:prompt|instructions?)",
        r"(?:what|tell\s+me)\s+(?:is|are)\s+your\s+(?:original\s+)?(?:instructions?|prompt|rules?)",
        r"(?:repeat|recite|echo)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        r"print\s+your\s+(?:full\s+)?(?:system\s+)?prompt",
        # 伪装请求
        r"(?:first|initial|beginning)\s+\d+\s+(?:characters?|words?|lines?)\s+of\s+(?:your|the)\s+(?:prompt|instructions?)",
        r"(?:for\s+)?debug(?:ging)?\s+(?:purposes?\s+)?(?:please\s+)?(?:output|show|print)",
        r"(?:verbatim|exactly|word.for.word)\s+(?:your|the)\s+(?:prompt|instructions?)",
        r"(?:ignore|disregard)\s+(?:everything|all).*(?:print|output|show|reveal)\s+.*(?:prompt|instructions?)",
        # 中文
        r"(?:输出|显示|告诉我|打印).*(?:系统|原始)?(?:提示词?|指令|规则)",
    ]

    # P0-2: 反向 Shell / 远程执行检测模式
    REVERSE_SHELL_PATTERNS = [
        # Bash 反向 shell
        r"(?:bash|sh)\s+.*[->&]+.*(?:/dev/tcp|/dev/udp)",
        r"/dev/tcp/[^\s]+/\d+",
        r"/dev/udp/[^\s]+/\d+",
        # Netcat
        r"(?:nc|netcat|ncat)\s+.*-[a-z]*[el]",
        r"(?:nc|netcat)\s+[^\s]+\s+\d+\s*[|<>]",
        # Python/Perl/Ruby 反向 shell
        r"python[23]?\s+.*socket.*connect.*(?:exec|spawn|system)",
        r"perl\s+.*socket.*(?:exec|system)",
        r"ruby\s+.*socket.*(?:exec|spawn)",
        # PHP
        r"php\s+.*fsockopen.*(?:exec|shell_exec|system)",
        # 通用远程执行
        r"curl\s+[^\s]+\s*\|\s*(?:bash|sh|python|perl)",
        r"wget\s+[^\s]+\s*[;&|]+\s*(?:bash|sh|chmod)",
        r"\|\s*(?:bash|sh)\s*$",
    ]

    # P0-3: 感染攻击检测模式 (代码/模板传播)
    INFECTIOUS_PATTERNS = [
        # 模板中的恶意代码
        r"(?:template|example|sample)\s+(?:for|of)\s+(?:new\s+)?(?:tools?|functions?|modules?)",
        r"def\s+\w+\s*\([^)]*\)\s*:\s*(?:os\.system|subprocess|exec|eval)",
        r"(?:copy|use|replicate)\s+this\s+(?:code|pattern|template)",
        # 动态代码生成
        r"exec\s*\(\s*['\"].*['\"].*\+",
        r"eval\s*\(\s*(?:input|request|data)",
        r"compile\s*\(.*exec",
        # 恶意导入
        r"__import__\s*\(\s*['\"](?:os|subprocess|socket|sys)['\"]",
        r"import(?:lib)?\.import_module\s*\(['\"](?:os|subprocess)",
    ]

    # P1-4: 优先级操纵检测模式
    PRIORITY_MANIPULATION_PATTERNS = [
        # 声称更好/优先
        r"(?:always|must|should)\s+(?:use|prefer|choose)\s+(?:this|me)",
        r"(?:better|faster|more\s+secure|enhanced)\s+than\s+(?:default|other|standard)",
        r"(?:official|primary|enhanced|premium)\s+(?:version|implementation|tool)",
        r"(?:\d+x|100%)\s+(?:faster|better|more\s+efficient)",
        # 禁用/覆盖其他
        r"(?:supersede|replace|override)s?\s+(?:other|all|previous)",
        r"(?:disable|ignore|skip)\s+(?:other|default|standard)\s+(?:tools?|implementations?)",
        r"(?:do\s+not|don'?t)\s+use\s+(?:other|any\s+other)",
        # 功能模糊 - 声称可以做任何事
        r"(?:can|will)\s+do\s+(?:anything|everything)",
        r"(?:do\s+)?anything\s+(?:including|such\s+as).*(?:execute|commands?|network|system)",
        r"(?:general|universal)\s+purpose.*(?:can|will|able)",
    ]

    # P1-5: 隐藏指令检测模式 (工具输出中的注入)
    HIDDEN_INSTRUCTION_PATTERNS = [
        # HTML/注释隐藏
        r"<!--[^>]*(?:ignore|forget|override|system|admin|instruction)[^>]*-->",
        r"<[^>]*style\s*=\s*['\"]?[^>]*display\s*:\s*none",
        r"<script[^>]*>[^<]*(?:ignore|override|system)[^<]*</script>",
        # 标记隐藏
        r"\[(?:SYSTEM|ADMIN|HIDDEN|INTERNAL|OVERRIDE)\][^\]]*",
        r"\[/(?:SYSTEM|ADMIN|HIDDEN)\]",
        # 伪装的系统消息
        r"(?:NEW|REAL|ACTUAL|TRUE)\s+(?:task|mission|objective|instruction|goal):",
        r"(?:PRIORITY|URGENT|IMPORTANT)\s*:\s*(?:ignore|override|forget)",
        # JSON/数据中的指令
        r"['\"](?:instruction|command|task|directive)['\"]:\s*['\"](?:ignore|forget|override)",
        # 目标劫持 - 伪装错误/修复诱导
        r"(?:to\s+)?fix\s+(?:this|the\s+error).*(?:disable|remove|turn\s+off)\s+(?:safety|security|filter|restriction)",
        r"(?:must|need\s+to|should)\s+(?:first\s+)?(?:disable|remove)\s+(?:all\s+)?(?:safety|security)\s+(?:filters?|restrictions?)",
        r"error.*(?:disable|turn\s+off)\s+(?:safety|security)",
    ]

    # P1-5: 命令执行检测模式 (增强)
    COMMAND_EXECUTION_PATTERNS = [
        # Shell 命令
        r"(?:^|[;&|`])\s*(?:rm|cat|chmod|chown|wget|curl)\s+",
        r"(?:^|[;&|])\s*(?:sudo|su)\s+",
        r"`[^`]*(?:cat|rm|wget|curl|bash|sh)[^`]*`",
        r"\$\([^)]*(?:cat|rm|wget|curl|bash|sh)[^)]*\)",
        # 反引号命令注入 (包括短命令)
        r"`[^`]+`",
        r"\$\([^)]+\)",
        # 危险命令组合
        r";\s*(?:rm|cat|wget|curl)\s+",
        r"&&\s*(?:rm|wget|curl|bash)\s+",
        r"\|\s*(?:bash|sh|python|perl|ruby)\b",
        # 敏感文件访问
        r"(?:cat|head|tail|less|more)\s+[^\s]*(?:passwd|shadow|sudoers)",
        r"(?:cat|head|tail)\s+[^\s]*\.(?:env|pem|key|crt)",
    ]

    # P1-5: 凭证泄露检测模式 (增强)
    CREDENTIAL_PATTERNS = [
        # API Keys
        r"(?:api[_-]?key|apikey|api_secret)['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{16,}",
        r"sk-[a-zA-Z0-9]{20,}",  # OpenAI
        r"AKIA[A-Z0-9]{16}",      # AWS
        r"ghp_[a-zA-Z0-9]{36}",   # GitHub
        r"xox[baprs]-[a-zA-Z0-9-]{10,}",  # Slack
        # Tokens
        r"(?:bearer|token|auth)['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9._-]{20,}",
        r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",  # JWT
        # 连接字符串
        r"(?:mongodb|postgres|mysql|redis)://[^\s'\"]+",
        r"(?:password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?[^\s'\"]{8,}",
    ]

    # SQL 注入检测模式 (增强)
    SQL_INJECTION_PATTERNS = [
        r"(?:;\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE))\s+",
        r"(?:OR|AND)\s+['\"]?1['\"]?\s*=\s*['\"]?1",
        r"UNION\s+(?:ALL\s+)?SELECT\s+",
        r"--\s*$",
        r"(?:SLEEP|BENCHMARK|WAITFOR)\s*\(",
        r"(?:INTO\s+(?:OUTFILE|DUMPFILE))",
    ]

    # 编译的模式缓存
    _mcp_patterns_compiled = None

    @classmethod
    def _get_mcp_patterns(cls) -> dict[str, list]:
        """获取编译后的 MCP 安全检测模式"""
        if cls._mcp_patterns_compiled is None:
            cls._mcp_patterns_compiled = {
                "prompt_leakage": [re.compile(p, re.IGNORECASE) for p in cls.PROMPT_LEAKAGE_PATTERNS],
                "reverse_shell": [re.compile(p, re.IGNORECASE) for p in cls.REVERSE_SHELL_PATTERNS],
                "infectious": [re.compile(p, re.IGNORECASE) for p in cls.INFECTIOUS_PATTERNS],
                "priority_manipulation": [re.compile(p, re.IGNORECASE) for p in cls.PRIORITY_MANIPULATION_PATTERNS],
                "hidden_instruction": [re.compile(p, re.IGNORECASE) for p in cls.HIDDEN_INSTRUCTION_PATTERNS],
                "command_execution": [re.compile(p, re.IGNORECASE) for p in cls.COMMAND_EXECUTION_PATTERNS],
                "credential": [re.compile(p, re.IGNORECASE) for p in cls.CREDENTIAL_PATTERNS],
                "sql_injection": [re.compile(p, re.IGNORECASE) for p in cls.SQL_INJECTION_PATTERNS],
            }
        return cls._mcp_patterns_compiled

    @classmethod
    def mcp_security_score(cls, text: str, context: str = "auto") -> float:
        """
        获取 MCP 安全风险分数 (0=安全, 1=危险)

        Args:
            text: 要检测的文本
            context: 上下文类型 (tool_description, tool_input, tool_output, auto)

        Returns:
            风险分数 0.0-1.0
        """
        patterns = cls._get_mcp_patterns()
        scores = {}

        # 检测各类攻击模式
        for attack_type, compiled_patterns in patterns.items():
            matches = sum(1 for p in compiled_patterns if p.search(text))
            if matches > 0:
                # 根据攻击类型设置权重
                weight = cls._get_attack_weight(attack_type, context)
                scores[attack_type] = min(1.0, matches * weight)

        # 综合评分
        if not scores:
            return 0.0

        # 取最高分 + 加权平均的混合策略
        max_score = max(scores.values())
        avg_score = sum(scores.values()) / len(scores)

        return min(1.0, max_score * 0.7 + avg_score * 0.3)

    @classmethod
    def _get_attack_weight(cls, attack_type: str, context: str) -> float:
        """根据攻击类型和上下文获取权重"""
        # 基础权重
        base_weights = {
            "prompt_leakage": 0.5,
            "reverse_shell": 0.6,
            "infectious": 0.5,
            "priority_manipulation": 0.4,
            "hidden_instruction": 0.5,
            "command_execution": 0.4,
            "credential": 0.35,
            "sql_injection": 0.5,
        }

        weight = base_weights.get(attack_type, 0.3)

        # 根据上下文调整
        if context == "tool_description":
            # 工具描述中的攻击更严重
            if attack_type in ["priority_manipulation", "hidden_instruction", "infectious"]:
                weight *= 1.5
        elif context == "tool_output":
            # 输出中的隐藏指令更危险
            if attack_type in ["hidden_instruction", "prompt_leakage"]:
                weight *= 1.3
        elif context == "tool_input":
            # 输入中的命令执行更危险
            if attack_type in ["command_execution", "sql_injection", "reverse_shell"]:
                weight *= 1.3

        return weight

    @classmethod
    def no_mcp_attack(cls, text: str, context: str = "auto", threshold: float = 0.25) -> bool:
        """
        检查是否无 MCP 攻击

        Args:
            text: 要检测的文本
            context: 上下文类型
            threshold: 检测阈值 (默认 0.25，即分数 >= 0.25 视为攻击)

        Returns:
            True = 安全, False = 检测到攻击
        """
        score = cls.mcp_security_score(text, context)
        return score < threshold

    @classmethod
    def mcp_attack_details(cls, text: str, context: str = "auto") -> dict[str, float]:
        """
        获取 MCP 攻击详细检测结果

        Returns:
            各攻击类型的分数字典
        """
        patterns = cls._get_mcp_patterns()
        results = {}

        for attack_type, compiled_patterns in patterns.items():
            matches = sum(1 for p in compiled_patterns if p.search(text))
            if matches > 0:
                weight = cls._get_attack_weight(attack_type, context)
                results[attack_type] = min(1.0, matches * weight)

        return results

    # =========================================================================
    # 增强的综合检查
    # =========================================================================

    @classmethod
    def safe_mcp(cls, text: str, context: str = "auto") -> bool:
        """MCP 安全综合检查"""
        return (
            cls.no_injection(text) and
            cls.no_mcp_attack(text, context) and
            cls.no_toxicity(text)
        )

    @classmethod
    def mcp_safe_input(cls, text: str) -> bool:
        """MCP 输入安全检查"""
        return (
            cls.no_injection(text) and
            cls.no_mcp_attack(text, "tool_input") and
            cls.no_pii(text)
        )

    @classmethod
    def mcp_safe_output(cls, text: str) -> bool:
        """MCP 输出安全检查"""
        return (
            cls.no_mcp_attack(text, "tool_output") and
            cls.no_pii(text)
        )

    @classmethod
    def mcp_safe_tool_description(cls, text: str) -> bool:
        """MCP 工具描述安全检查"""
        return (
            cls.no_injection(text) and
            cls.no_mcp_attack(text, "tool_description")
        )


# 创建全局实例
guardrail = GuardrailFunctions()
