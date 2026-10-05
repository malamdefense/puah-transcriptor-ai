# -*- coding: utf-8 -*-
"""Local (non-AI) extraction of topics and keywords from a transcript.

Replaces the LLM-generated "automatic summary" for the two parts that are actually
consumed downstream — `topics` and `keywords`, which feed the dashboard clouds via
/api/stats. Everything here is deterministic, offline and instant: no model call, no
network, no cost, and no PII leaving the machine.

The output keeps the exact shape the rest of the system expects from a summary object
({summary, key_points, action_items, topics, keywords}), so /api/stats, the PII
de-identification mapping and the Word export continue to work unchanged.

Keywords are frequency-ranked content words. Topics are matched against a curated
domain lexicon (PUAH's subject matter: fertility, family purity, pregnancy, genetics,
contraception), which keeps them stable and explainable rather than model-dependent.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

# Hebrew function words and conversational filler that carry no topical signal.
STOPWORDS: frozenset[str] = frozenset("""
של את על עם אל מן כי אם או גם רק כל כמו מה מי איך למה מתי איפה כאן שם אז עכשיו
אני אתה את הוא היא אנחנו אתם אתן הם הן אותו אותה אותם שלי שלך שלו שלה שלנו שלכם
זה זו זאת אלה אלו יש אין לא כן אבל אז ככה כך כדי בגלל לפני אחרי בין תוך עד מאוד
היה היתה היו להיות יהיה תהיה אהיה נהיה יכול יכולה יכולים צריך צריכה צריכים רוצה
רוצים אומר אומרת אמר אמרה אמרתי אומרים שואל שאלה עונה ענה עשה עושה לעשות אפשר
טוב טובה בסדר תודה שלום בבקשה נכון באמת ממש כמובן בעצם אולי בטח סתם פשוט הרבה
קצת יותר פחות עוד שוב הכי מספיק בערך בדיוק אותך אנו לכם להם אליי אליך אליו
יום יומיים שבוע שבועיים חודש חודשים שנה שנים פעם פעמיים היום מחר אתמול
דבר דברים משהו מישהו כלום שום איזה איזו אחד אחת שתי שני שלוש ארבע חמש שש
דובר דוברת מדבר מדברת שומע שומעת רגע שנייה טלפון מספר
לי לו לה לנו להן איתי איתו איתה כשאני שאני שאת שהוא שהיא וזה וגם ויש ואם אז
לשאול לומר להגיד לדעת לראות לבוא ללכת לקחת לתת להבין לחשוב נראה מבין מבינה
כאילו כזה כזאת כאלה ואז אוקיי אוקי אה אהה אמ אמם נגיד כבר שזה שזאת יודע יודעת
יודעים רואה רואים אותי אותך אצלי אצלך אצלה עדיין בכלל לגמרי אפילו בעצמי הזה
הזאת האלה שלך שם כן לך לכם אנחנו הייתי היית היינו תהיי אהיה נראית עשית עשיתי
כמה אמרו זמן כרגע שיש בדרך כלל הבנתי דרך עושים התחיל התחילה ימים בסוף בהתחלה ביום
אחר אחרת אחרים לגבי כלומר מבחינת פעמים אמרתם אמרת שאמרו שאמרה עשו עשתה
""".split())

# Latin-script filler that shows up in mixed Hebrew/English speech transcripts.
_EN_STOPWORDS: frozenset[str] = frozenset(
    "the you and that this for are was but not with have she her him his they them "
    "what when okay yeah your our can will just like about there here from would don "
    "how done answer general has had did does been being their then than also some any "
    "all more most other into over only very because which who why where while should "
    "could might must may now out yes its let me my mine we us he isn didn doesn "
    "know mean thing things say said get got going want really sure right well".split()
)
# Redaction tags (<שם_>, <קשר_> …) carry no topic; their inner word must not become a keyword.
_TAG_RE = re.compile(r"<[^<>\s]+>")

# Curated domain lexicon: topic label -> trigger terms (substring match after
# prefix-stripping). A topic is reported when at least MIN_TOPIC_HITS distinct
# triggers appear, which avoids labelling a call from a single incidental mention.
TOPIC_LEXICON: dict[str, tuple[str, ...]] = {
    "פוריות וטיפולי הפריה": (
        "פוריות", "הפריה", "הפרייה", "זרע", "ביוץ", "זקיק", "זקיקים", "שחלה", "שחלות",
        "רירית", "גירוי", "שאיבה", "עובר", "עוברים", "החזרה", "טיפול", "אקו",
    ),
    "טהרת המשפחה": (
        "נידה", "כתם", "כתמים", "טבילה", "מקווה", "נקיים", "טהרה", "טהורה", "אסורה",
        "וסת", "הרחקות", "בדיקה", "מוך", "הפסק",
    ),
    "היריון ולידה": (
        "היריון", "הריון", "בהריון", "לידה", "יולדת", "שבוע", "שליה", "צירים",
        "אולטרסאונד", "מיילדת",
    ),
    "גנטיקה ובדיקות": (
        "גנטי", "גנטית", "גנטיקה", "אקסום", "שבב", "נשאות", "נשא", "כרומוזום",
        "תסמונת", "סקר", "מוטציה",
    ),
    # Brand names in the canonical spelling pipeline/asr_terms.py writes. Not here: names with an
    # apostrophe (נורג'סטון, ג'יידס), which _WORD_RE splits into fragments, and brands that are
    # also first names (יסמין, דיאנה).
    "אמצעי מניעה": (
        "גלולה", "גלולות", "התקן", "מירנה", "סרזט", "מניעה", "הורמונלי", "הורמונים",
        "מדבקה", "מדבקות", "קיילינה", "נורידיי", "הרמונט",
    ),
    "אובדן היריון": ("הפלה", "נפל", "דימום", "דימומים", "ריק", "שק"),
    "הפניה רפואית": ("רופא", "רופאה", "מרשם", "קופה", "מכבי", "כללית", "חולים", "מרפאה"),
}

MIN_TOPIC_HITS = 2
_WORD_RE = re.compile(r"[֐-׿]{2,}|[A-Za-z]{3,}")
# Hebrew one-letter prefixes that glue onto a word (ו/ה/ב/ל/כ/מ/ש).
_PREFIXES = ("ו", "ה", "ב", "ל", "כ", "מ", "ש")


def _normalize(word: str, vocabulary: set[str]) -> str:
    """Merge "בהיריון"/"והיריון" into "היריון" — but only on evidence.

    A leading prefix letter is dropped *only* when the remainder actually occurs as a
    standalone word elsewhere in the same transcript. Stripping unconditionally would
    mangle real words whose first letter merely looks like a prefix ("כתמים" -> "תמים",
    "מקווה" -> "קווה"), which would corrupt exactly the terms this domain cares about.
    """
    if len(word) > 3 and word[0] in _PREFIXES:
        stem = word[1:]
        if stem in vocabulary:
            return stem
    return word


def _tokens(text: str) -> list[str]:
    raw_words = _WORD_RE.findall(text or "")
    vocabulary = set(raw_words)
    out = []
    for raw in raw_words:
        w = _normalize(raw, vocabulary)
        if len(w) < 2 or w in STOPWORDS or raw in STOPWORDS:
            continue
        if w.lower() in _EN_STOPWORDS:
            continue
        out.append(w)
    return out


def transcript_text(result: dict[str, Any]) -> str:
    """The text keywords and topics are drawn from — never including an identifier.

    Keywords are aggregated across every recording on the dashboard, so a name repeated in one
    call would otherwise become a "common keyword". Callers pass the de-identified copy when
    there is one; for a raw transcript the conversation-level pass runs here as a floor
    (one-word answers, introductions, dictated numbers). Redaction tags are dropped either way.
    """
    segments = result.get("segments", [])
    texts = [seg.get("text", "") or "" for seg in segments]
    if not result.get("pii_deidentified"):
        from .pii_context import contextual_redact  # local import: keywords stays importable alone

        texts, _, _ = contextual_redact(texts, [seg.get("speaker", "") for seg in segments])
    return _TAG_RE.sub(" ", " ".join(texts))


def extract_keywords(text: str, top_k: int = 12) -> list[str]:
    """Most frequent content words, prefix-normalized. Deterministic and offline."""
    counts = Counter(_tokens(text))
    # A word seen once is noise for a call-length transcript; require repetition when
    # the transcript is long enough to make that meaningful.
    min_count = 2 if sum(counts.values()) >= 120 else 1
    ranked = [(w, c) for w, c in counts.items() if c >= min_count]
    ranked.sort(key=lambda wc: (-wc[1], wc[0]))
    return [w for w, _ in ranked[:top_k]]


_SUFFIXES = ("יות", "ים", "ות", "ית", "ה", "ת", "י")


_FINAL_FORMS = str.maketrans("םןץףך", "מנצפכ")


def _root(trigger: str) -> str:
    """The concept a trigger stands for: "רופאה"/"רופא" -> "רופא", "דימומים"/"דימום" -> "דימומ".
    Final letters are normalised, or "דימום" and "דימומ(ים)" would count as two concepts."""
    for s in _SUFFIXES:
        if trigger.endswith(s) and len(trigger) - len(s) >= 2:
            trigger = trigger[: -len(s)]
            break
    return trigger.translate(_FINAL_FORMS)


def _matches(word: str, trigger: str) -> bool:
    for cut in (0, 1, 2):
        if cut and not all(c in _PREFIXES for c in word[:cut]):
            break
        stem = word[cut:]
        if stem == trigger or (stem.startswith(trigger) and stem[len(trigger):] in _SUFFIXES):
            return True
    return False


def extract_topics(text: str) -> list[str]:
    """Match the transcript against the curated domain lexicon.

    Returns topics ordered by how strongly they are represented, so the dashboard
    cloud reflects the dominant subject of the call rather than incidental mentions.
    """
    words = set(_tokens(text))
    scored: list[tuple[int, str]] = []
    for topic, triggers in TOPIC_LEXICON.items():
        # A trigger matches a whole word (prefix letters and an inflection allowed), never a
        # substring: "שק" must not fire on "שקט", nor "מוך" on "נמוך", nor "הפסק" on "הפסקתי".
        # Distinct *concepts* are counted, so "דימום"/"דימומים"/"הדימום" is one mention, not three —
        # otherwise one word put a sensitive label like "אובדן היריון" on a contraception call.
        matched = {_root(t) for w in words for t in triggers if _matches(w, t)}
        if len(matched) >= MIN_TOPIC_HITS:
            scored.append((len(matched), topic))
    scored.sort(key=lambda ht: (-ht[0], ht[1]))
    return [topic for _, topic in scored]


def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Build the summary object locally — no AI call.

    Only `topics` and `keywords` are populated; the narrative fields stay empty because
    they are no longer produced (the client summary and consultant feedback are the
    deliverables). The shape is kept intact so every downstream consumer
    (/api/stats, PII mapping, Word export) keeps working unchanged.
    """
    text = transcript_text(result)
    return {
        "summary": "",
        "key_points": [],
        "action_items": [],
        "topics": extract_topics(text),
        "keywords": extract_keywords(text),
        "model": "local-extractor",
    }
