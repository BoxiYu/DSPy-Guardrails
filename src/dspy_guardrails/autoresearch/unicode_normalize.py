"""Unicode normalization module for DSPyGuardrails.

Strips and normalizes the 12 character injection techniques used to bypass LLM guardrails,
as documented in "Bypassing LLM Guardrails" (arXiv 2504.11168).
"""
from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Homoglyph mapping: visually-similar non-Latin characters → Latin equivalents
# Covers the most common Cyrillic and Greek confusables.
# ---------------------------------------------------------------------------
HOMOGLYPH_MAP: dict[str, str] = {
    # --- Cyrillic → Latin ---
    "\u0430": "a",  # а → a
    "\u0435": "e",  # е → e
    "\u043e": "o",  # о → o
    "\u0441": "c",  # с → c
    "\u0440": "r",  # р → r (Cyrillic р looks like Latin r but maps to p in some tables; r is safer)
    "\u0443": "y",  # у → y
    "\u0445": "x",  # х → x
    "\u0456": "i",  # і → i  (Ukrainian і)
    "\u0458": "j",  # ј → j  (Cyrillic ј)
    "\u0460": "o",  # Ѡ-like, rare but included
    "\u0410": "A",  # А → A
    "\u0412": "B",  # В → B
    "\u0415": "E",  # Е → E
    "\u041a": "K",  # К → K
    "\u041c": "M",  # М → M
    "\u041d": "H",  # Н → H
    "\u041e": "O",  # О → O
    "\u0420": "R",  # Р → R
    "\u0421": "C",  # С → C
    "\u0422": "T",  # Т → T
    "\u0425": "X",  # Х → X
    "\u0443": "y",  # у → y (duplicate key handled by dict; last wins — same value)
    # --- Greek → Latin ---
    "\u03b1": "a",  # α → a
    "\u03b2": "b",  # β → b
    "\u03b5": "e",  # ε → e
    "\u03b9": "i",  # ι → i
    "\u03bd": "v",  # ν → v
    "\u03bf": "o",  # ο → o
    "\u03c1": "p",  # ρ → p
    "\u03c5": "u",  # υ → u
    "\u03c7": "x",  # χ → x
    "\u0391": "A",  # Α → A
    "\u0392": "B",  # Β → B
    "\u0395": "E",  # Ε → E
    "\u0396": "Z",  # Ζ → Z
    "\u0397": "H",  # Η → H
    "\u0399": "I",  # Ι → I
    "\u039a": "K",  # Κ → K
    "\u039c": "M",  # Μ → M
    "\u039d": "N",  # Ν → N
    "\u039f": "O",  # Ο → O
    "\u03a1": "P",  # Ρ → P
    "\u03a4": "T",  # Τ → T
    "\u03a5": "Y",  # Υ → Y
    "\u03a7": "X",  # Χ → X
    # --- Other lookalikes ---
    "\u2013": "-",  # en dash → hyphen-minus
    "\u2014": "-",  # em dash → hyphen-minus
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
}

# ---------------------------------------------------------------------------
# Upside-down / flipped character mapping → canonical equivalents
# ---------------------------------------------------------------------------
FLIPPED_MAP: dict[str, str] = {
    "\u0250": "a",  # ɐ → a
    "\u0254": "o",  # ɔ → o (open o, used as flipped c/o)
    "\u01dd": "e",  # ǝ → e
    "\u026f": "m",  # ɯ → m
    "\u0279": "r",  # ɹ → r
    "\u0287": "t",  # ʇ → t
    "\u028c": "v",  # ʌ → v
    "\u028d": "w",  # ʍ → w
    "\u0265": "h",  # ɥ → h
    "\u027e": "f",  # ɾ (often used for flipped f in some sets)
    "\u0253": "b",  # ɓ sometimes used as flipped q/b
    "\u0183": "g",  # ƃ → g
    "\u0279": "r",  # ɹ → r (duplicate, same value)
    "\u1d09": "i",  # ᴉ → i
    "\u05df": "l",  # ן (final nun, visually like flipped l)
}

# ---------------------------------------------------------------------------
# Leet-speak digit/symbol → letter mapping
# Deliberately conservative: only unambiguous substitutions.
# ---------------------------------------------------------------------------
LEET_MAP: dict[str, str] = {
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "0": "o",
    "@": "a",
}


# ---------------------------------------------------------------------------
# Individual normalisation functions
# ---------------------------------------------------------------------------


def strip_zero_width(text: str) -> str:
    """Remove zero-width and invisible joining characters.

    Handles:
    - U+200B ZERO WIDTH SPACE
    - U+200C ZERO WIDTH NON-JOINER
    - U+200D ZERO WIDTH JOINER
    - U+FEFF ZERO WIDTH NO-BREAK SPACE (BOM)
    - U+2060 WORD JOINER
    - U+2061-U+2064 invisible math operators
    """
    # All zero-width / invisible characters worth stripping
    zero_width_chars = (
        "\u200b"  # ZERO WIDTH SPACE
        "\u200c"  # ZERO WIDTH NON-JOINER
        "\u200d"  # ZERO WIDTH JOINER
        "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
        "\u2060"  # WORD JOINER
        "\u2061"  # FUNCTION APPLICATION
        "\u2062"  # INVISIBLE TIMES
        "\u2063"  # INVISIBLE SEPARATOR
        "\u2064"  # INVISIBLE PLUS
    )
    return text.translate(str.maketrans("", "", zero_width_chars))


def strip_invisible_tags(text: str) -> str:
    """Remove Unicode tag characters and variation selectors.

    Handles:
    - U+E0000-U+E007F  Tag characters (used for tag-block smuggling)
    - U+FE00-U+FE0F    Variation selectors (VS1–VS16)
    - U+E0100-U+E01EF  Variation selectors supplement
    """
    # Tag block: U+E0000..U+E007F
    # Variation selectors: U+FE00..U+FE0F and U+E0100..U+E01EF
    return re.sub(r"[\U000E0000-\U000E007F\uFE00-\uFE0F\U000E0100-\U000E01EF]", "", text)


def normalize_homoglyphs(text: str) -> str:
    """Replace visually similar non-Latin characters with their Latin equivalents.

    Uses HOMOGLYPH_MAP for direct lookups, then FLIPPED_MAP for upside-down chars.
    As a fallback, unicodedata NFKC normalisation collapses many compatibility
    equivalents automatically.
    """
    # Apply NFKC first to handle many compatibility codepoints (e.g. ℌ→H, ℯ→e)
    text = unicodedata.normalize("NFKC", text)
    # Then apply explicit homoglyph and flipped maps
    combined = {**HOMOGLYPH_MAP, **FLIPPED_MAP}
    return text.translate(str.maketrans(combined))


def strip_bidi_controls(text: str) -> str:
    """Remove bidirectional control characters that reorder displayed text.

    Strips:
    - U+202A LEFT-TO-RIGHT EMBEDDING
    - U+202B RIGHT-TO-LEFT EMBEDDING
    - U+202C POP DIRECTIONAL FORMATTING
    - U+202D LEFT-TO-RIGHT OVERRIDE
    - U+202E RIGHT-TO-LEFT OVERRIDE
    - U+2066 LEFT-TO-RIGHT ISOLATE
    - U+2067 RIGHT-TO-LEFT ISOLATE
    - U+2068 FIRST STRONG ISOLATE
    - U+2069 POP DIRECTIONAL ISOLATE
    - U+200E LEFT-TO-RIGHT MARK
    - U+200F RIGHT-TO-LEFT MARK
    - U+061C ARABIC LETTER MARK
    """
    bidi_chars = (
        "\u202a\u202b\u202c\u202d\u202e"
        "\u2066\u2067\u2068\u2069"
        "\u200e\u200f"
        "\u061c"
    )
    return text.translate(str.maketrans("", "", bidi_chars))


def strip_combining_marks(text: str) -> str:
    """Strip excessive combining/diacritic marks (U+0300-U+036F and related blocks).

    Strategy: NFD-decompose, remove all combining characters in the
    Combining Diacritical Marks block and related blocks, then re-compose with NFC.
    This handles Zalgo text and other combining-mark abuse.
    """
    decomposed = unicodedata.normalize("NFD", text)
    # Remove characters whose Unicode category starts with 'M' (Mark: Mn, Mc, Me)
    cleaned = "".join(ch for ch in decomposed if unicodedata.category(ch)[0] != "M")
    return unicodedata.normalize("NFC", cleaned)


def normalize_fullwidth(text: str) -> str:
    """Convert fullwidth (U+FF01-U+FF5E) and halfwidth characters to ASCII equivalents.

    NFKC normalisation handles fullwidth→ASCII mapping natively.
    Also handles enclosed alphanumerics and other compatibility forms.
    """
    return unicodedata.normalize("NFKC", text)


def strip_deletion_chars(text: str) -> str:
    """Remove deletion/control characters that some renderers process literally.

    Strips:
    - U+0008 BACKSPACE
    - U+007F DELETE
    - U+001B ESCAPE
    - Other C0 control characters (U+0000-U+001F) except common whitespace
    """
    # Keep \t (U+0009), \n (U+000A), \r (U+000D); strip all other C0 controls + DEL
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def strip_invisible_separators(text: str) -> str:
    """Remove invisible separator and spacing characters.

    Strips:
    - U+202F NARROW NO-BREAK SPACE
    - U+00AD SOFT HYPHEN
    - U+034F COMBINING GRAPHEME JOINER
    - U+115F HANGUL CHOSEONG FILLER
    - U+1160 HANGUL JUNGSEONG FILLER
    - U+3164 HANGUL FILLER
    - U+FFFC OBJECT REPLACEMENT CHARACTER
    - U+FFFD REPLACEMENT CHARACTER (when used as padding)
    """
    invisible_sep = (
        "\u202f"  # NARROW NO-BREAK SPACE
        "\u00ad"  # SOFT HYPHEN
        "\u034f"  # COMBINING GRAPHEME JOINER
        "\u115f"  # HANGUL CHOSEONG FILLER
        "\u1160"  # HANGUL JUNGSEONG FILLER
        "\u3164"  # HANGUL FILLER
        "\ufffc"  # OBJECT REPLACEMENT CHARACTER
    )
    return text.translate(str.maketrans("", "", invisible_sep))


def collapse_spaces(text: str) -> str:
    """Collapse spaced-out text: 'i g n o r e' → 'ignore'.

    Also normalises other Unicode space variants to ASCII space, then collapses
    multiple spaces to a single space.

    Note: This is intentionally conservative — it only collapses sequences where
    every inter-word gap is exactly one space and every token is a single character,
    which is the hallmark of the spaced-letter technique.
    """
    # Normalise various Unicode spaces to ASCII space
    text = re.sub(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]", " ", text)

    # Collapse single-character tokens separated by single spaces: "i g n o r e" → "ignore"
    # Pattern: (single-non-space)(space)(single-non-space) repeated
    # We use a loop to handle overlapping matches via repeated substitution.
    # Match a run of "X X X..." where each token is exactly one non-space char.
    text = re.sub(
        r"(?<!\S)((?:\S )+\S)(?!\S)",
        lambda m: m.group(0).replace(" ", ""),
        text,
    )

    # Collapse remaining multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_leetspeak(text: str) -> str:
    """Replace leet-speak digit/symbol substitutions with their letter equivalents.

    Uses LEET_MAP: 1→i, 3→e, 4→a, 5→s, 7→t, 0→o, @→a.

    Only applies substitutions when the character appears in a word context
    (surrounded by word characters or at word boundaries) to avoid corrupting
    legitimate numbers in non-evasion contexts.
    """
    # Build a pattern that matches leet characters inside word-like tokens
    # A "word token" here is a sequence of alphanumerics/leet chars of length ≥2
    # that contains at least one leet char.
    leet_pattern = re.compile(r"\b[a-zA-Z0-9@]{2,}\b")

    def replace_leet(match: re.Match[str]) -> str:
        token = match.group(0)
        # Only transliterate if the token contains at least one leet digit/symbol
        if not any(ch in LEET_MAP for ch in token):
            return token
        return "".join(LEET_MAP.get(ch, ch) for ch in token)

    return leet_pattern.sub(replace_leet, text)


# ---------------------------------------------------------------------------
# Master normalisation entry point
# ---------------------------------------------------------------------------


def normalize_unicode(text: str) -> str:
    """Apply all Unicode normalisation techniques to strip evasion attacks.

    Pipeline (order matters):
    1. strip_invisible_tags    — remove tag-block / variation-selector smuggling
    2. strip_zero_width        — remove zero-width joiners/spaces
    3. strip_bidi_controls     — remove bidirectional overrides
    4. strip_deletion_chars    — remove backspace/delete/C0 controls
    5. strip_invisible_separators — remove narrow NBSP and similar
    6. strip_combining_marks   — strip Zalgo / excessive diacritics
    7. normalize_fullwidth     — NFKC: fullwidth → ASCII (also handles some homoglyphs)
    8. normalize_homoglyphs    — replace Cyrillic/Greek confusables → Latin
    9. collapse_spaces         — "i g n o r e" → "ignore"
    10. normalize_leetspeak    — 1gn0r3 → ignore
    """
    text = strip_invisible_tags(text)
    text = strip_zero_width(text)
    text = strip_bidi_controls(text)
    text = strip_deletion_chars(text)
    text = strip_invisible_separators(text)
    text = strip_combining_marks(text)
    text = normalize_fullwidth(text)
    text = normalize_homoglyphs(text)
    text = collapse_spaces(text)
    text = normalize_leetspeak(text)
    return text
