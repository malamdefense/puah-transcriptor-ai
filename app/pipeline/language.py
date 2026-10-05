"""Script-based language detection for the merged transcript (stdlib only).

The ivrit-ai Whisper model's language-ID head answers "he" on every window, English speech
included, so decoding stays in Hebrew and the language is read from the output instead: with
language="he" the model writes English speech mostly in Latin script. Each whitespace token is
assigned the language of its majority script; the call-level and per-segment languages are
derived from those counts. PII tags such as <שם_> are stripped first, so the same code works on
the de-identified copy.

Only the candidate languages passed in `allowed` can ever be reported, and only those that have
a script pattern below are counted — anything else (Cyrillic, Arabic, digits, punctuation)
counts for nothing.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

# One pattern per detectable language: the letters of its script.
SCRIPT_PATTERNS: dict[str, re.Pattern] = {
    "he": re.compile(r"[א-ת]"),  # Hebrew letters only (no niqqud, maqaf or geresh)
    "en": re.compile(r"[A-Za-z]"),
}

# Hebrew display names, for the UI-facing exports
LANGUAGE_NAMES_HE: dict[str, str] = {"he": "עברית", "en": "אנגלית"}

# Share of the call's words at which a language becomes the primary one (else the default)
PRIMARY_MIN = 0.5
# A minority language is listed only with at least this share AND this many words, so a
# Hebrew call with a few loanwords (IVF, OK) stays Hebrew-only
SECONDARY_MIN = 0.10
SECONDARY_MIN_WORDS = 20
# Share a language needs within one segment to label it
SEGMENT_MIN = 0.5

_TAG = re.compile(r"<[^<>]*>")  # PII placeholders must not count as Hebrew
_TOKEN = re.compile(r"\S+")


def _fallback(allowed: Sequence[str], default: str) -> str:
    """The default when it may be reported, else the first allowed language (its stand-in)."""
    allowed = list(dict.fromkeys(allowed))
    return default if default in allowed or not allowed else allowed[0]


def _candidates(allowed: Sequence[str], default: str) -> list[str]:
    """The detectable allowed languages, default first (it wins ties), without duplicates."""
    langs = [lang for lang in dict.fromkeys(allowed) if lang in SCRIPT_PATTERNS]
    if default in langs:
        langs.remove(default)
        langs.insert(0, default)
    return langs


def word_language(token: str, allowed: Sequence[str]) -> Optional[str]:
    """The language whose script has the most letters in the token; None without letters.

    Ties go to the earlier language in `allowed`.
    """
    token = _TAG.sub(" ", token or "")
    best, best_n = None, 0
    for lang in dict.fromkeys(allowed):
        pattern = SCRIPT_PATTERNS.get(lang)
        if pattern is None:
            continue
        n = len(pattern.findall(token))
        if n > best_n:
            best, best_n = lang, n
    return best


def _count_words(text: str, langs: Sequence[str], counts: dict[str, int]) -> None:
    for token in _TOKEN.findall(_TAG.sub(" ", text or "")):
        lang = word_language(token, langs)
        if lang:
            counts[lang] += 1


def segment_language(text: str, allowed: Sequence[str], default: str = "he") -> Optional[str]:
    """The segment's majority language (share >= SEGMENT_MIN), else the default.

    None when the text has no letters of any allowed script (digits, punctuation, tags only).
    """
    default = _fallback(allowed, default)
    langs = _candidates(allowed, default)
    counts = {lang: 0 for lang in langs}
    _count_words(text, langs, counts)
    total = sum(counts.values())
    if not total:
        return None
    best = max(counts, key=counts.get)  # first maximum: the default on a tie
    return best if counts[best] / total >= SEGMENT_MIN else default


def detect_languages(
    segments: list[dict[str, Any]],
    allowed: Sequence[str] = ("he", "en"),
    default: str = "he",
) -> dict[str, Any]:
    """Call-level language from the words of all display segments.

    Returns {"language": primary, "languages": [primary, *secondaries by share],
    "language_shares": {lang: share}, "language_method": "script"}. A text without any letters
    gives the default with empty shares. A default outside `allowed` is replaced by the first
    allowed language, so it is never reported either.
    """
    default = _fallback(allowed, default)
    langs = _candidates(allowed, default)
    counts = {lang: 0 for lang in langs}
    for seg in segments:
        _count_words(seg.get("text", ""), langs, counts)
    total = sum(counts.values())
    if not total:
        return {"language": default, "languages": [default], "language_shares": {},
                "language_method": "script"}

    shares = {lang: counts[lang] / total for lang in langs}
    best = max(shares, key=shares.get)  # first maximum: the default on a tie
    primary = best if shares[best] >= PRIMARY_MIN else default
    secondary = [
        lang for lang in sorted(langs, key=lambda x: -shares[x])
        if lang != primary and shares[lang] >= SECONDARY_MIN and counts[lang] >= SECONDARY_MIN_WORDS
    ]
    return {
        "language": primary,
        "languages": [primary, *secondary],
        "language_shares": {lang: round(share, 3) for lang, share in shares.items()},
        "language_method": "script",
    }


def annotate_segments(segments: list[dict[str, Any]], allowed: Sequence[str], default: str) -> None:
    """Set seg["language"] on every segment, in place.

    A segment without letters inherits the previous segment's language (the first one gets
    `default`, which callers pass as the call's primary language).
    """
    default = _fallback(allowed, default)
    prev = default
    for seg in segments:
        lang = segment_language(seg.get("text", ""), allowed, default) or prev
        seg["language"] = lang
        prev = lang
