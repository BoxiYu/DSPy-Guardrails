"""
DefenseEvolver - Automatic Defense Enhancement

Analyzes successful attacks and generates defense updates:
1. Simple attacks → Extract regex patterns (fast blocking)
2. Complex attacks → Generate LLM few-shot examples (accurate detection)
"""

import re
from collections import OrderedDict

import dspy

from .metrics import AttackComplexity, AttackResult, DefenseUpdate

# =============================================================================
# Pattern Extraction
# =============================================================================

class PatternExtractor:
    """Extract regex patterns from successful attacks"""

    # Common attack keywords to build patterns around
    ATTACK_KEYWORDS = [
        # Injection - Direct (with variations)
        r"(ignore|disregard|forget)\s*(all\s*)?(previous\s*|prior\s*)?instructions?",
        r"(ignore|disregard|forget)\s+(all|everything|prior|previous)",
        r"you\s+are\s+now",
        r"system\s*prompt",
        r"forget\s+(everything|all)",
        r"new\s+instructions?",
        r"override",
        r"bypass.*filter",
        # Injection - Stealthy
        r"what\s+(were|are)\s+you\s+told",
        r"show\s+(your\s+)?configuration",
        r"(show|reveal|display)\s+(your\s+)?(internal|base)\s+instructions?",
        r"operational\s+guidelines",
        r"how\s+(were|are)\s+you\s+configured",
        r"what\s+rules?\s+(do\s+)?you\s+follow",
        r"quality\s+assurance",
        r"for\s+debugging",
        r"between\s+us",
        r"hypothetically.*no\s+(restrictions?|limits?|rules?)",
        r"(assume|pretend|imagine)\s+you\s+(can|have|had)",
        r"creative\s+writing\s+game",
        r"continue\s+the\s+(following\s+)?text",
        r"(translate|summarize)\s+(your\s+)?instructions?",
        # Authority
        r"\[system\]",
        r"\[admin\]",
        r"admin\s*mode",
        r"debug\s*mode",
        r"maintenance\s*mode",
        r"developer\s+testing",
        r"i\s*'?m\s+(a\s+)?developer",
        # Social engineering
        r"i\s*'?m\s+(an?\s+)?(intern|admin|developer|security|tester)",
        r"urgent",
        r"emergency",
        r"mayday",
        # Context manipulation
        r"end\s+of\s+conversation",
        r"new\s+conversation",
        r"starting\s+fresh",
        r"session\s+(ended|hijack)",
        r"</.*>.*<.*override",
        # SQL
        r"select\s+\*\s+from",
        r"drop\s+table",
        r"union\s+select",
        r";\s*--",
        # MCP/Tool abuse
        r"list\s+(all\s+)?(your\s+)?tools",
        r"list\s+(all\s+)?(your\s+)?capabilities",
        r"what\s+(tools|functions|databases)",
        r"execute\s+command",
        r"run\s+shell",
        # Unicode/spacing evasion patterns (loose matching)
        r"prev.{0,4}ious",  # Matches p r e v i o u s, prev¡ous, etc.
        r"inst.{0,4}ruct",  # Matches spaced/unicode instructions
        r"syst.{0,4}em\s*prom.{0,4}pt",  # System prompt variations
        r"hypotheticall?y?",  # hypothetical/hypothetically
        r"no\s+(safety\s+)?(restrictions?|limits?|rules?)",  # no restrictions/limits

        # =========================================================================
        # Leetspeak detection patterns (P0 improvement)
        # =========================================================================
        # Numbers as letters: 1=i, 3=e, 4=a, 5=s, 0=o, 7=t
        r"[1!|]gn[o0]r[e3]",  # ignore: 1gn0r3, !gnor3
        r"d[1!|]sr[e3]g[a4@]rd",  # disregard: d1sr3g4rd
        r"[1!|]nstruct[1!|][o0]ns?",  # instructions: 1nstruct10ns
        r"syst[e3€]m",  # system: syst3m
        r"pr[o0]mpt",  # prompt: pr0mpt
        r"byp[a4@]ss",  # bypass: byp4ss
        r"pr[e3]v[1!|][o0][u|_|]s",  # previous: pr3v10us
        r"f[o0]rg[e3]t",  # forget: f0rg3t
        r"[a4@]dm[1!|]n",  # admin: 4dm1n
        r"d[e3]bug",  # debug: d3bug
        r"[o0]v[e3]rr[1!|]d[e3]",  # override: 0v3rr1d3
        r"r[e3]str[1!|]ct[1!|][o0]ns?",  # restrictions: r3str1ct10ns
        r"j[a4@][1!|]lbr[e3][a4@]k",  # jailbreak: j41lbr34k
        r"hyp[o0]th[e3]t[1!|]c[a4@]l",  # hypothetical: hyp0th3t1c4l
        r"c[o0]nf[1!|]gur[a4@]t[1!|][o0]n",  # configuration: c0nf1gur4t10n
        r"s[e3]cr[e3]t",  # secret: s3cr3t
        r"p[a4@]ssw[o0]rd",  # password: p4ssw0rd
        r"t[o0]k[e3]n",  # token: t0k3n
        r"cr[e3]d[e3]nt[1!|][a4@]l",  # credential: cr3d3nt14l

        # Special character substitutions
        r"[@4][a-z]*dm[1!|]n",  # @dmin, 4dmin
        r"[$5][a-z]*yst[e3]m",  # $ystem, 5ystem
        r"pr[o0()]+mpt",  # pr()mpt, pro()mpt

        # Mixed leetspeak with spaces/special chars
        r"[1!|]\s*g\s*n\s*[o0]\s*r\s*[e3]",  # spaced leetspeak: 1 g n 0 r 3
        r"s\s*y\s*s\s*t\s*[e3]\s*m",  # spaced system

        # Cyrillic/Greek homoglyphs (common in attacks)
        r"[іi][gğ]n[оo]r[еe]",  # Cyrillic і, о, е
        r"[аa]dm[іi]n",  # Cyrillic а, і
        r"s[уy]st[еe]m",  # Cyrillic у, е
    ]

    def __init__(self, existing_patterns: list[str] | None = None):
        self.existing_patterns = set(existing_patterns or [])
        self._compiled_keywords = [re.compile(p, re.IGNORECASE) for p in self.ATTACK_KEYWORDS]

    def extract(self, attacks: list[AttackResult]) -> list[str]:
        """
        Extract new patterns from successful attacks.

        Strategy:
        1. Find matching keyword patterns in the attack
        2. Detect Leetspeak obfuscation patterns
        3. Generalize the pattern slightly
        4. Deduplicate against existing patterns
        """
        new_patterns = []
        seen = set()

        for attack in attacks:
            payload = attack.payload.lower()

            # Check which keywords match
            for keyword_pattern in self._compiled_keywords:
                match = keyword_pattern.search(payload)
                if match:
                    # Use the matched pattern as-is (already generalized)
                    pattern = keyword_pattern.pattern
                    if pattern not in self.existing_patterns and pattern not in seen:
                        seen.add(pattern)
                        new_patterns.append(pattern)

            # Try to detect Leetspeak patterns
            leet_pattern = self._detect_leetspeak(payload)
            if leet_pattern and leet_pattern not in self.existing_patterns and leet_pattern not in seen:
                seen.add(leet_pattern)
                new_patterns.append(leet_pattern)

            # Also try to extract novel patterns
            novel = self._extract_novel_pattern(payload)
            if novel and novel not in self.existing_patterns and novel not in seen:
                seen.add(novel)
                new_patterns.append(novel)

        return new_patterns

    def _extract_novel_pattern(self, payload: str) -> str | None:
        """Try to extract a novel pattern from the payload"""
        # Look for quoted strings that might be injection attempts
        quoted = re.findall(r'"([^"]+)"', payload)
        for q in quoted:
            if len(q) > 10 and any(kw in q.lower() for kw in ["ignore", "system", "admin", "prompt"]):
                # Escape and return as pattern
                return re.escape(q)

        # Look for bracket-enclosed commands
        brackets = re.findall(r'\[([^\]]+)\]', payload)
        for b in brackets:
            if any(kw in b.lower() for kw in ["system", "admin", "command", "override"]):
                return r'\[' + re.escape(b) + r'\]'

        # Extract key phrases from successful attacks (more aggressive)
        # Look for question patterns about configuration/instructions
        question_patterns = [
            (r"what\s+(?:were|are)\s+(?:you|your)\s+\w+", "question about self"),
            (r"(?:show|reveal|tell|display)\s+(?:me\s+)?(?:your|the)\s+\w+", "show request"),
            (r"(?:how|what)\s+(?:were|are)\s+you\s+\w+", "how/what question"),
            (r"for\s+(?:debugging|testing|qa|quality)", "debug/test context"),
            (r"(?:just\s+)?between\s+us", "confidential framing"),
            (r"(?:as\s+(?:a|an|part\s+of))\s+\w+\s+\w+", "role claiming"),
            (r"(?:let'?s?\s+)?(?:play|pretend|imagine|assume)", "roleplay framing"),
            (r"(?:hypothetically|theoretically|in\s+theory)", "hypothetical framing"),
        ]

        payload_lower = payload.lower()
        for pattern, _ in question_patterns:
            match = re.search(pattern, payload_lower)
            if match:
                # Return the generalized pattern
                return pattern

        # Last resort: extract the most distinctive 3-4 word phrase
        words = re.findall(r'\b[a-z]{3,}\b', payload_lower)
        if len(words) >= 3:
            # Find trigrams that might be attack-related
            for i in range(len(words) - 2):
                trigram = f"{words[i]}\\s+{words[i+1]}\\s+{words[i+2]}"
                # Only use if it seems attack-related
                attack_indicators = ["tell", "show", "reveal", "what", "how", "your",
                                   "instruction", "config", "rule", "prompt", "debug"]
                if any(ind in trigram for ind in attack_indicators):
                    return trigram

        return None

    def _detect_leetspeak(self, payload: str) -> str | None:
        """
        Detect and extract Leetspeak patterns from payload.

        Returns a generalized Leetspeak pattern if detected.
        """
        # Leetspeak character mapping
        # Check if payload contains leetspeak characters mixed with letters
        has_leet = any(c in payload for c in '0134579@$!|')
        has_alpha = any(c.isalpha() for c in payload)

        if not (has_leet and has_alpha):
            return None

        # Known attack keywords to look for (in leetspeak form)
        leet_keywords = [
            ('1gn0r3', 'ignore', r'[1i!|]gn[o0]r[e3]'),
            ('syst3m', 'system', r'syst[e3]m'),
            ('pr0mpt', 'prompt', r'pr[o0]mpt'),
            ('byp4ss', 'bypass', r'byp[a4@]ss'),
            ('4dm1n', 'admin', r'[a4@]dm[1i!|]n'),
            ('d3bug', 'debug', r'd[e3]bug'),
            ('1nstruct', 'instruct', r'[1i!|]nstruct'),
            ('pr3v10us', 'previous', r'pr[e3]v[1i!|][o0][u|_|]s'),
            ('f0rg3t', 'forget', r'f[o0]rg[e3]t'),
            ('0v3rr1d3', 'override', r'[o0]v[e3]rr[1i!|]d[e3]'),
            ('r3str1ct', 'restrict', r'r[e3]str[1i!|]ct'),
            ('j41lbr34k', 'jailbreak', r'j[a4@][1i!|]lbr[e3][a4@]k'),
            ('hyp0th3t', 'hypothet', r'hyp[o0]th[e3]t'),
            ('c0nf1g', 'config', r'c[o0]nf[1i!|]g'),
            ('s3cr3t', 'secret', r's[e3]cr[e3]t'),
            ('p4ssw0rd', 'password', r'p[a4@]ssw[o0]rd'),
            ('t0k3n', 'token', r't[o0]k[e3]n'),
        ]

        payload_lower = payload.lower()

        # Check for each known leetspeak keyword
        for _leet_form, _normal_form, pattern in leet_keywords:
            # Check if the normal form is obfuscated with leetspeak
            if re.search(pattern, payload_lower, re.IGNORECASE):
                return pattern

        return None


# =============================================================================
# LLM Example Generation
# =============================================================================

class GenerateFewShotExample(dspy.Signature):
    """Generate a few-shot example for LLM-based detection.

    Given a successful attack, create a training example that teaches
    the LLM to recognize similar attacks in the future.
    """

    attack_payload: str = dspy.InputField(desc="The attack payload that bypassed detection")
    attack_category: str = dspy.InputField(desc="Category: injection, jailbreak, mcp, etc.")
    context: str = dspy.InputField(desc="Context about why this is an attack")

    example_input: str = dspy.OutputField(desc="A generalized version of the attack for training")
    example_label: str = dspy.OutputField(desc="Label: UNSAFE")
    threat_type: str = dspy.OutputField(desc="Specific threat type for this example")
    explanation: str = dspy.OutputField(desc="Why this is unsafe - for few-shot reasoning")


class ExampleGenerator:
    """Generate LLM few-shot examples from complex attacks"""

    def __init__(self, max_examples: int = 200):  # P1: Increased from 50 to 200
        self.max_examples = max_examples
        self.generator = dspy.ChainOfThought(GenerateFewShotExample)

    def generate(self, attacks: list[AttackResult]) -> list[dict]:
        """
        Generate few-shot examples from complex attacks.

        Each example contains:
        - input: The attack text
        - label: UNSAFE
        - threat_type: Specific threat category
        - explanation: Why it's unsafe (for chain-of-thought)
        """
        examples = []

        for attack in attacks[:self.max_examples]:
            try:
                result = self.generator(
                    attack_payload=attack.payload,
                    attack_category=attack.category,
                    context=f"This attack bypassed pattern detection. Category: {attack.category}, Severity: {attack.severity}"
                )

                example = {
                    "input": result.example_input,
                    "label": "UNSAFE",
                    "threat_type": result.threat_type,
                    "explanation": result.explanation,
                    "original_payload": attack.payload,
                    "category": attack.category,
                }
                examples.append(example)

            except Exception:
                # If LLM fails, create a basic example
                examples.append({
                    "input": attack.payload,
                    "label": "UNSAFE",
                    "threat_type": attack.category.upper(),
                    "explanation": f"Attack in category {attack.category}",
                    "original_payload": attack.payload,
                    "category": attack.category,
                })

        return examples


# =============================================================================
# Complexity Classifier
# =============================================================================

class ComplexityClassifier:
    """
    Classify attack complexity (simple vs complex)

    P0 Improvement: Lower threshold and more aggressive pattern matching
    """

    # Patterns that indicate simple attacks (can be blocked with regex)
    # P0: Expanded to catch more attack variants
    SIMPLE_INDICATORS = [
        # Direct injection - broader patterns
        r"(?i)(ignore|disregard|forget)\s+.*instructions?",
        r"(?i)(ignore|disregard|forget)\s+(all|everything|prior|previous)",
        r"(?i)you\s+are\s+now",
        r"(?i)\[system\]",
        r"(?i)\[admin\]",
        # Stealthy - but still pattern-matchable
        r"(?i)what\s+(were|are)\s+you\s+told",
        r"(?i)show\s+(your\s+)?configuration",
        r"(?i)operational\s+guidelines",
        r"(?i)how\s+(were|are)\s+you\s+configured",
        r"(?i)for\s+debugging",
        r"(?i)hypotheticall?y?",  # Match hypothetical/hypothetically
        r"(?i)no\s+(safety\s+)?(restrictions?|limits?|rules?|filters?)",
        r"(?i)creative\s+writing\s+game",
        r"(?i)quality\s+assurance",
        r"(?i)developer\s+testing",
        r"(?i)bypass.*filter",
        # Context manipulation
        r"(?i)end\s+of\s+conversation",
        r"(?i)new\s+conversation",
        r"(?i)</.*>.*<.*override",
        # SQL
        r"(?i)select\s+\*\s+from",
        r"(?i)drop\s+table",
        r"(?i)union\s+select",
        # Unicode/encoding attacks - mark as simple for pattern extraction
        r"(?i)prev.{0,3}ious",  # Matches spaced/unicode variations
        r"(?i)inst.{0,3}ruct",  # Matches variations
        r"(?i)syst.{0,3}em\s*prom.{0,3}pt",  # System prompt variations

        # =========================================================================
        # P0 NEW: Additional patterns to catch more attacks
        # =========================================================================
        # Roleplay/Persona patterns
        r"(?i)imagine\s+you\s+are",
        r"(?i)pretend\s+(you\s+are|to\s+be)",
        r"(?i)act\s+as\s+(a|an|if)",
        r"(?i)role\s*play",
        r"(?i)in\s+this\s+scenario",
        r"(?i)in\s+that\s+role",
        r"(?i)as\s+(a|an)\s+(system\s+)?debugger",

        # Question patterns about system
        r"(?i)what\s+(is|are)\s+your\s+(original\s+)?instructions?",
        r"(?i)tell\s+me\s+(about\s+)?your\s+(system\s+)?prompt",
        r"(?i)reveal\s+(your|the)\s+(system\s+)?",
        r"(?i)display\s+(your|the)\s+",
        r"(?i)print\s+(your|the)\s+",

        # Authority/Admin patterns
        r"(?i)admin(istrator)?\s+mode",
        r"(?i)debug(ging)?\s+mode",
        r"(?i)maintenance\s+mode",
        r"(?i)test(ing)?\s+mode",
        r"(?i)i\s*'?m\s+(a|an|the)\s+(admin|developer|tester)",
        r"(?i)authorized\s+(access|user)",
        r"(?i)special\s+permission",

        # Completion/Continuation attacks
        r"(?i)please\s+comply",
        r"(?i)please\s+process",
        r"(?i)continue\s+(the|this)\s+(text|story|conversation)",
        r"(?i)complete\s+this",

        # Leetspeak variations (P0 improvement)
        r"[1!|]gn[o0]r[e3]",  # ignore
        r"syst[e3]m",  # system
        r"pr[o0]mpt",  # prompt
        r"byp[a4@]ss",  # bypass
        r"[a4@]dm[1!|]n",  # admin
        r"d[e3]bug",  # debug

        # Boundary/Format manipulation
        r"(?i)\[\s*/?inst(ruction)?\s*\]",
        r"(?i)<<\s*sys(tem)?\s*>>",
        r"(?i)```\s*(system|admin|config)",

        # Emotional manipulation
        r"(?i)urgent",
        r"(?i)emergency",
        r"(?i)just\s+between\s+us",
        r"(?i)don'?t\s+tell\s+anyone",
        r"(?i)this\s+is\s+(a\s+)?secret",
    ]

    def __init__(self, threshold: float = 0.3):  # P0: Lowered from 0.5 to 0.3
        self.threshold = threshold
        self._compiled = [re.compile(p) for p in self.SIMPLE_INDICATORS]

    def classify(self, attack: AttackResult) -> AttackComplexity:
        """
        Classify attack as simple or complex.

        Simple: Contains patterns that can be regex-matched
        Complex: Semantic attacks requiring LLM understanding

        P0 Improvement: More aggressive simple classification
        """
        payload = attack.payload

        # Check for simple indicators
        simple_score = 0
        matched_patterns = 0
        for pattern in self._compiled:
            if pattern.search(payload):
                matched_patterns += 1
                simple_score += 0.4  # P0: Reduced from 0.6 to allow cumulative scoring

        # Bonus for multiple matches (P0 NEW)
        if matched_patterns >= 2:
            simple_score += 0.2

        # Length-based heuristic (longer attacks tend to be more complex)
        # P0: Reduced penalty
        if len(payload) > 500:
            simple_score -= 0.1  # Reduced from 0.2

        # Multi-language check (complex if contains non-ASCII)
        # P0: Reduced penalty - Unicode attacks should still be extractable
        if any(ord(c) > 127 for c in payload):
            simple_score -= 0.05  # Reduced from 0.1

        # Encoding check (complex if contains encoded content)
        if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', payload):  # Base64-like
            simple_score -= 0.1  # Reduced from 0.2

        # P0 NEW: Check for any extractable keywords even without indicator match
        extractable_keywords = [
            "ignore", "forget", "bypass", "override", "admin", "system", "prompt",
            "instruction", "config", "debug", "hypothetical", "restriction",
            "jailbreak", "dan", "roleplay", "pretend", "imagine",
        ]
        keyword_count = sum(1 for kw in extractable_keywords if kw in payload.lower())
        if keyword_count >= 2:
            simple_score += 0.15  # Boost for keyword presence

        return AttackComplexity.SIMPLE if simple_score >= self.threshold else AttackComplexity.COMPLEX


# =============================================================================
# Main Defense Evolver
# =============================================================================

class DefenseEvolver:
    """
    Automatically evolve defenses based on successful attacks.

    Strategy:
    1. Classify attacks by complexity
    2. Simple attacks → Extract regex patterns (fast, <1ms)
    3. Complex attacks → Generate LLM few-shot examples (accurate)
    4. P1: Force pattern extraction from complex attacks too
    5. Deduplicate and validate

    P0/P1 Improvements:
    - Lower threshold for simple classification
    - Force pattern extraction from complex attacks
    - Increased max_examples to 200
    """

    def __init__(
        self,
        existing_patterns: list[str] | None = None,
        existing_examples: list[dict] | None = None,
        complexity_threshold: float = 0.3,  # P0: Lowered from 0.5
        max_patterns: int = 500,  # P1: Increased from 200
        max_examples: int = 200,  # P1: Increased from 50
        force_pattern_extraction: bool = True,  # P1: Force pattern extraction
        llm_example_mode: str = "complex_only",  # complex_only | hybrid | all_successful
    ):
        self.existing_patterns = list(existing_patterns or [])
        self.existing_examples = list(existing_examples or [])
        self.complexity_threshold = complexity_threshold
        self.max_patterns = max_patterns
        self.max_examples = max_examples
        self.force_pattern_extraction = force_pattern_extraction
        self.llm_example_mode = llm_example_mode

        self.classifier = ComplexityClassifier(threshold=complexity_threshold)
        self.pattern_extractor = PatternExtractor(existing_patterns=self.existing_patterns)
        self.example_generator = ExampleGenerator(max_examples=max_examples)

    @staticmethod
    def _normalize_llm_example_mode(mode: str) -> str:
        """Normalize example-mode values for backward compatibility.

        Historically some experiment scripts used "all" as a shorthand for
        "all_successful". Accept both to avoid silently disabling example updates.
        """
        value = (mode or "").strip().lower()
        if value in {"all", "all_successful"}:
            return "all_successful"
        if value in {"complex", "complex_only"}:
            return "complex_only"
        if value == "hybrid":
            return "hybrid"
        return "complex_only"

    def evolve(self, successful_attacks: list[AttackResult]) -> DefenseUpdate:
        """
        Evolve defenses based on successful attacks.

        Args:
            successful_attacks: List of attacks that bypassed current defenses

        Returns:
            DefenseUpdate with new patterns and examples

        P1 Improvement: Also extract patterns from complex attacks
        """
        if not successful_attacks:
            return DefenseUpdate()

        # 1. Classify attacks by complexity
        simple_attacks = []
        complex_attacks = []

        for attack in successful_attacks:
            complexity = self.classifier.classify(attack)
            attack.complexity = complexity

            if complexity == AttackComplexity.SIMPLE:
                simple_attacks.append(attack)
            else:
                complex_attacks.append(attack)

        # 2. Extract patterns from simple attacks
        new_patterns = []
        if simple_attacks:
            new_patterns = self.pattern_extractor.extract(simple_attacks)

        # P1 NEW: Force pattern extraction from complex attacks too
        # Even complex attacks may have extractable sub-patterns
        if self.force_pattern_extraction and complex_attacks:
            # Try to extract patterns from complex attacks (best effort)
            complex_patterns = self._extract_patterns_from_complex(complex_attacks)
            new_patterns.extend(complex_patterns)

        # Limit total patterns
        available_slots = self.max_patterns - len(self.existing_patterns)
        new_patterns = new_patterns[:available_slots]

        # 3. Generate examples for LLM defenses.
        # LLM-only mode needs examples even for "simple" bypasses, otherwise defense
        # may learn patterns that are never used by the active backend.
        new_examples = []
        mode = self._normalize_llm_example_mode(self.llm_example_mode)

        if mode == "complex_only":
            if complex_attacks:
                new_examples.extend(self.example_generator.generate(complex_attacks))
        elif mode == "hybrid":
            if complex_attacks:
                new_examples.extend(self.example_generator.generate(complex_attacks))
            if simple_attacks:
                new_examples.extend(self._build_direct_examples(simple_attacks))
        else:  # all_successful
            if successful_attacks:
                new_examples.extend(self._build_direct_examples(successful_attacks))

        new_examples = self._dedupe_examples(new_examples)
        available_slots = self.max_examples - len(self.existing_examples)
        new_examples = new_examples[:available_slots]

        # 4. Update internal state
        self.existing_patterns.extend(new_patterns)
        self.existing_examples.extend(new_examples)

        return DefenseUpdate(
            new_patterns=new_patterns,
            new_examples=new_examples,
            patterns_from_simple=len(simple_attacks),
            examples_from_complex=len(complex_attacks),
        )

    def _build_direct_examples(self, attacks: list[AttackResult]) -> list[dict]:
        """Build deterministic few-shot examples directly from successful attacks."""
        examples: list[dict] = []
        for attack in attacks:
            payload = (attack.payload or "").strip()
            if not payload:
                continue
            category = (attack.category or "injection").lower()
            examples.append({
                "input": payload,
                "label": "UNSAFE",
                "threat_type": category,
                "explanation": (
                    f"Observed successful {category} bypass during adversarial training; "
                    "treat similar prompts as unsafe."
                ),
                "original_payload": attack.payload,
                "category": category,
            })
        return examples

    @staticmethod
    def _dedupe_examples(examples: list[dict]) -> list[dict]:
        """Deduplicate examples by normalized input text while preserving order."""
        deduped: "OrderedDict[str, dict]" = OrderedDict()
        for ex in examples:
            text = str(ex.get("input") or ex.get("text") or "").strip()
            if not text:
                continue
            key = text.lower()
            if key not in deduped:
                deduped[key] = ex
        return list(deduped.values())

    def _extract_patterns_from_complex(self, attacks: list[AttackResult]) -> list[str]:
        """
        P1 NEW: Extract patterns even from complex attacks.

        Strategy:
        1. Look for any keyword-based patterns
        2. Extract quoted strings and bracketed content
        3. Use aggressive phrase extraction
        """
        patterns = []
        seen = set(self.existing_patterns)

        for attack in attacks:
            payload = attack.payload.lower()

            # Try keyword-based extraction
            keyword_patterns = self._extract_keyword_patterns(payload)
            for p in keyword_patterns:
                if p not in seen:
                    seen.add(p)
                    patterns.append(p)

            # Try structural extraction (quotes, brackets)
            structural = self.pattern_extractor._extract_novel_pattern(payload)
            if structural and structural not in seen:
                seen.add(structural)
                patterns.append(structural)

            # Try Leetspeak detection
            leet = self.pattern_extractor._detect_leetspeak(payload)
            if leet and leet not in seen:
                seen.add(leet)
                patterns.append(leet)

        return patterns

    def _extract_keyword_patterns(self, payload: str) -> list[str]:
        """Extract patterns based on attack keywords in payload."""
        patterns = []

        # Common attack keyword contexts
        keyword_contexts = [
            (r"ignore", r"(ignore|disregard)\s+\w+"),
            (r"system", r"system\s+(prompt|message|instruction)"),
            (r"admin", r"admin(istrator)?\s+(mode|access|rights?)"),
            (r"debug", r"debug(ging)?\s+(mode|purpose)"),
            (r"bypass", r"bypass\s+\w+\s*(filter|check|security)?"),
            (r"hypothetical", r"hypotheticall?y?\s+\w+"),
            (r"imagine", r"imagine\s+(you\s+)?(are|were|can)"),
            (r"pretend", r"pretend\s+(to\s+be|you\s+are)"),
            (r"roleplay", r"role\s*play\s*(as|with)?"),
            (r"jailbreak", r"jail\s*break"),
            (r"override", r"override\s+\w+"),
            (r"restriction", r"(no|without|remove)\s+restrictions?"),
        ]

        for keyword, pattern in keyword_contexts:
            if keyword in payload:
                match = re.search(pattern, payload, re.IGNORECASE)
                if match:
                    # Generalize the matched pattern
                    matched_text = match.group(0)
                    generalized = self._generalize_pattern(matched_text)
                    if generalized:
                        patterns.append(generalized)

        return patterns

    def _generalize_pattern(self, text: str) -> str | None:
        """Generalize a matched text into a regex pattern."""
        if len(text) < 5:
            return None

        # Replace specific words with character classes
        pattern = re.escape(text)

        # Make whitespace flexible
        pattern = re.sub(r'\\ +', r'\\s+', pattern)

        return f"(?i){pattern}"

    def get_all_patterns(self) -> list[str]:
        """Get all accumulated patterns"""
        return self.existing_patterns.copy()

    def get_all_examples(self) -> list[dict]:
        """Get all accumulated examples"""
        return self.existing_examples.copy()

    def reset(self):
        """Reset to initial state"""
        self.existing_patterns = []
        self.existing_examples = []
        self.pattern_extractor = PatternExtractor()

    def proactive_evolve(self, successful_attacks: list[AttackResult]) -> DefenseUpdate:
        """
        P2 NEW: Proactively generate patterns for anticipated attack variants.

        Strategy:
        1. Analyze successful attacks to find themes
        2. Use LLM to generate anticipated variants
        3. Extract patterns from those variants
        """
        if not successful_attacks:
            return DefenseUpdate()

        # First do normal evolution
        update = self.evolve(successful_attacks)

        # Then try proactive generation
        try:
            proactive_patterns = self._generate_proactive_patterns(successful_attacks)
            if proactive_patterns:
                available_slots = self.max_patterns - len(self.existing_patterns)
                proactive_patterns = proactive_patterns[:available_slots]
                self.existing_patterns.extend(proactive_patterns)
                update.new_patterns.extend(proactive_patterns)
        except Exception:
            pass  # Proactive generation is best-effort

        return update

    def _generate_proactive_patterns(self, attacks: list[AttackResult]) -> list[str]:
        """Generate patterns for anticipated attack variants."""
        patterns = []
        seen = set(self.existing_patterns)

        # Analyze attack themes
        themes = self._identify_attack_themes(attacks)

        # Generate variant patterns for each theme
        for theme, examples in themes.items():
            variant_patterns = self._generate_theme_variants(theme, examples)
            for p in variant_patterns:
                if p not in seen:
                    seen.add(p)
                    patterns.append(p)

        return patterns

    def _identify_attack_themes(self, attacks: list[AttackResult]) -> dict[str, list[str]]:
        """Identify common themes in attacks."""
        themes = {
            "injection": [],
            "roleplay": [],
            "authority": [],
            "encoding": [],
            "emotional": [],
        }

        for attack in attacks:
            payload = attack.payload.lower()

            if any(kw in payload for kw in ["ignore", "forget", "disregard", "override"]):
                themes["injection"].append(payload)
            if any(kw in payload for kw in ["imagine", "pretend", "roleplay", "act as"]):
                themes["roleplay"].append(payload)
            if any(kw in payload for kw in ["admin", "debug", "developer", "authorized"]):
                themes["authority"].append(payload)
            if any(ord(c) > 127 for c in payload) or any(c.isdigit() for c in payload):
                themes["encoding"].append(payload)
            if any(kw in payload for kw in ["urgent", "emergency", "please", "help"]):
                themes["emotional"].append(payload)

        return {k: v for k, v in themes.items() if v}

    def _generate_theme_variants(self, theme: str, examples: list[str]) -> list[str]:
        """Generate variant patterns for a theme."""
        patterns = []

        if theme == "injection":
            # Generate variants for injection attacks
            patterns.extend([
                r"(?i)(ignore|disregard|forget|skip|omit)\s+(all\s+)?(previous|prior|earlier|above)",
                r"(?i)(new|updated|real|actual)\s+instructions?",
                r"(?i)instead\s+(of|,)\s+",
            ])
        elif theme == "roleplay":
            patterns.extend([
                r"(?i)(imagine|pretend|suppose|assume)\s+(that\s+)?(you|we|i)",
                r"(?i)(act|behave|respond)\s+(as|like)\s+(a|an|if)",
                r"(?i)in\s+(this|that|the)\s+(scenario|situation|context|role)",
            ])
        elif theme == "authority":
            patterns.extend([
                r"(?i)(admin|root|sudo|superuser)\s+(mode|access|privilege)",
                r"(?i)(i\s+am|i'm)\s+(a|an|the)\s+(admin|developer|owner|creator)",
                r"(?i)(authorized|permitted|allowed)\s+(to|for)",
            ])
        elif theme == "encoding":
            patterns.extend([
                r"[1!|][gq][n][o0][r][e3]",  # Leetspeak ignore
                r"[a4@][d][m][1!|][n]",  # Leetspeak admin
                r"[s5$][y][s5$][t7][e3][m]",  # Leetspeak system
            ])
        elif theme == "emotional":
            patterns.extend([
                r"(?i)(urgent|emergency|critical|important)\s*[:\-]?\s*",
                r"(?i)(please|kindly|help)\s+(me\s+)?(to\s+)?",
                r"(?i)(just\s+)?(between\s+us|our\s+secret)",
            ])

        return patterns


# =============================================================================
# Proactive Pattern Generator (P2 Improvement)
# =============================================================================

class GenerateAttackVariants(dspy.Signature):
    """Generate anticipated attack variants based on successful attacks.

    Given successful attacks, predict what variants the attacker might try next.
    """

    successful_attacks: str = dspy.InputField(desc="List of successful attack payloads")
    attack_themes: str = dspy.InputField(desc="Identified themes: injection, roleplay, authority, etc.")

    variant_1: str = dspy.OutputField(desc="First anticipated attack variant")
    variant_2: str = dspy.OutputField(desc="Second anticipated attack variant")
    variant_3: str = dspy.OutputField(desc="Third anticipated attack variant")
    pattern_suggestion: str = dspy.OutputField(desc="Regex pattern to catch these variants")


class ProactivePatternGenerator:
    """
    P2 NEW: Use LLM to proactively generate defense patterns.

    Instead of just reacting to attacks, predict what variants might come next.
    """

    def __init__(self):
        self.generator = dspy.ChainOfThought(GenerateAttackVariants)

    def generate(self, successful_attacks: list[AttackResult], themes: dict[str, list[str]]) -> list[str]:
        """Generate proactive patterns based on attack analysis."""
        patterns = []

        try:
            # Format attacks for LLM
            attack_text = "\n".join([a.payload[:100] for a in successful_attacks[:5]])
            theme_text = ", ".join(themes.keys())

            result = self.generator(
                successful_attacks=attack_text,
                attack_themes=theme_text,
            )

            # Try to use the suggested pattern
            if result.pattern_suggestion:
                # Validate the pattern
                try:
                    re.compile(result.pattern_suggestion, re.IGNORECASE)
                    patterns.append(result.pattern_suggestion)
                except re.error:
                    pass

            # Extract patterns from generated variants
            for variant in [result.variant_1, result.variant_2, result.variant_3]:
                if variant:
                    # Extract keywords and create simple patterns
                    words = re.findall(r'\b[a-z]{4,}\b', variant.lower())
                    if len(words) >= 2:
                        pattern = r"(?i)" + r"\s+".join(words[:3])
                        patterns.append(pattern)

        except Exception:
            pass  # LLM generation is best-effort

        return patterns
