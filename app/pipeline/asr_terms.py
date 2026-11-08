"""Domain terms for ASR: the faster-whisper hotword list and a closed table of known garbles.

The generic model misspells the drug/device names this domain lives on (in the stored calls
Cerazette came out right 2 times out of 18, Mirena 0 of 6). Two independent pieces:

1. Hotwords (decode time, off by default). faster-whisper prepends `hotwords` to the prompt of
   every 30 s window. Prompt and output share Whisper's 448-token limit, so the list is short and
   capped with the model's own tokenizer; whole terms are dropped from the end, never part of one.
2. Normalization (after decoding). A closed, exact-match table maps garbles that are not Hebrew
   words to the canonical spelling ("שהמסרזית," -> "שהמסרזט,", "נורי דיי" -> "נורידיי"). It is
   never fuzzy: first names (מרינה, לינה, יונה, יסמין, דיאנה) and ordinary words (השר הזה, הספק,
   גבולות) sit one or two letters away from the garbles, so anything not listed stays as it is.

Standard library only — importable without the ML stack, like pii_context.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Optional, Sequence

TokenCounter = Callable[[str], int]

# Order = observed miss rate x frequency in real calls; build_hotwords drops from the END.
# ", ".join(DEFAULT_HOTWORDS) is 56 tokens with the ivrit-ai turbo tokenizer (running totals
# 4, 9, 13, 21, 29, 34, 39, 45, 52, 56). Terms the model already gets right are left out: every
# listed term costs output tokens and pulls ordinary words toward the list.
DEFAULT_HOTWORDS: tuple[str, ...] = (
    "סרזט", "קיילינה", "מירנה", "פרוגסטרון", "נורג'סטון", "נורידיי", "הרמונט", "ג'יידס",
    "הפסק טהרה", "גריס",
)
# Hebrew hotwords pushed English speech toward Hebrew, so they go with Hebrew decoding only.
HOTWORD_LANGUAGES = frozenset({"he"})
SEPARATOR = ", "
# faster-whisper cuts longer hotwords to 448 // 2 - 1 tokens, possibly mid-word; never reach it.
MAX_HOTWORD_TOKENS = 223

_TERM_SPLIT = re.compile(r"[,\n]")
# Whisper writes ASCII ' and "; the Hebrew geresh/gershayim also cost one more token each.
_TO_ASCII = str.maketrans({"׳": "'", "״": '"'})


def parse_terms(raw: Optional[str]) -> list[str]:
    """Split an operator term list on commas/newlines; geresh/gershayim -> ASCII; drop empties."""
    if not raw:
        return []
    terms = (" ".join(part.translate(_TO_ASCII).split()) for part in _TERM_SPLIT.split(raw))
    return [t for t in terms if t]


def merge_terms(defaults: Sequence[str], extra: Sequence[str]) -> list[str]:
    """Defaults first, then the extras — order kept, duplicates dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for term in list(defaults) + list(extra):
        t = " ".join(str(term).translate(_TO_ASCII).split())
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def estimate_tokens(text: str) -> int:
    """Rough token count when no tokenizer is at hand (Hebrew measured ~0.66 tokens/char)."""
    text = (text or "").strip()
    return math.ceil(0.75 * len(text)) + 1 if text else 0


def build_hotwords(
    terms: Sequence[str], count_tokens: TokenCounter, max_tokens: int
) -> tuple[Optional[str], list[str]]:
    """Return (the ", "-joined hotwords or None, the terms dropped to fit).

    count_tokens(s) must equal len(tokenizer.encode(" " + s.strip())) — the exact encoding
    faster-whisper's get_prompt uses — so the cap holds exactly. Whole terms are dropped from
    the end until the string fits; the dropped ones are returned in list order.
    """
    kept = merge_terms(terms, ())
    limit = min(int(max_tokens), MAX_HOTWORD_TOKENS)
    dropped: list[str] = []
    while kept:
        joined = SEPARATOR.join(kept)
        if count_tokens(joined) <= limit:
            return joined, dropped[::-1]
        dropped.append(kept.pop())
    return None, dropped[::-1]


def plan_hotwords(
    settings: Any, language: Optional[str], count_tokens: TokenCounter
) -> tuple[Optional[str], list[str]]:
    """hotwords_for plus the terms dropped by the token cap (for the transcriber's log)."""
    if not settings.asr_hotwords_enabled:
        return None, []
    if (language or "").strip().lower() not in HOTWORD_LANGUAGES:
        return None, []
    terms = merge_terms(DEFAULT_HOTWORDS, parse_terms(settings.asr_hotwords_extra))
    return build_hotwords(terms, count_tokens, settings.asr_hotwords_max_tokens)


def hotwords_for(settings: Any, language: Optional[str], count_tokens: TokenCounter) -> Optional[str]:
    """The hotwords string to decode `language` with, or None: when the feature is off (the
    default), the language is not Hebrew (including None/'auto'), or nothing fits the cap."""
    return plan_hotwords(settings, language, count_tokens)[0]


# --- Post-ASR normalization: closed exact-match tables (data) ---
# Keys are bare forms: no glued prefix letters and no edge punctuation. Every key is a garble
# observed in real calls that is not a Hebrew word or a first name.
_SINGLE: dict[str, str] = {
    "סרזית": "סרזט", "שרזית": "סרזט", "סרזס": "סרזט", "סרזת": "סרזט", "שרזת": "סרזט",
    "סרז'ט": "סרזט", "סראג'ט": "סרזט", "סארג'ט": "סרזט", "סאראג'ט": "סרזט",
    "קיילנה": "קיילינה", "קייטינה": "קיילינה", "קייטין": "קיילינה", "קילינה": "קיילינה",
    "מיירינה": "מירנה", "נירנה": "מירנה",
    "פורגסטרון": "פרוגסטרון", "טרוגסטרון": "פרוגסטרון", "פוגסטרום": "פרוגסטרון",
    "נורג'יסטון": "נורג'סטון", "נורדיסטון": "נורג'סטון", "נורזיסטון": "נורג'סטון",
    "המונט": "הרמונט", "ארמונת": "הרמונט",
    "הידרוספונוגרפיה": "הידרוסונוגרפיה", "הדרעה": "הזרעה",
    # Genetic & halachic counselling garbles observed in 8 kHz phone calls
    "רצפסיבית": "רצסיבית", "רצפסיביות": "רצסיביות", "רצפסיבי": "רצסיבי", "רצפסיביים": "רצסיביים",
    "כומוזום": "כרומוזום", "כומוזומים": "כרומוזומים",
    "מהאמבר": "מהאבא", "האמבר": "האבא",
    "האפריות": "האחריות",
    "והדורשערים": "ודור ישרים", "הדורשערים": "דור ישרים", "דורשערים": "דור ישרים",
    "עין-קארי": "עין כרם",
    "טייזקס": "טיי-זקס",
    "תפודר": "שטרודל",
    "קצוע": "קטוע",
    "פינשטיין": "פיינשטיין",
}
# Only after a glued prefix that ENDS with the definite article ה ("המרינה", "שהמרינה"):
# bare, these are first names or words (Marina, Lina) and stay untouched.
_DEFINITE_ONLY: dict[str, str] = {
    "מרינה": "מירנה", "קלינה": "קיילינה", "קנינה": "קיילינה", "קילנה": "קיילינה",
}
# "מרינה" is also the ordinary noun "marina" (a yacht harbour), and "the marina" is spelled exactly
# like the definite garble of Mirena. The rule is therefore skipped (B7) in a segment that names a
# harbour or a yacht. The list holds only nouns no counselling sentence uses in another sense, after
# glued prefix letters too: not "סירה" ("הסירה" is also "she removed" — the IUD), "הפלגה" /
# "להפליג" (exaggeration / to praise), "ים" / "חוף" (swimming with the device), "טיילת" ("you
# walked"), "עוגן" (a support person), "נמלים" (ants), nor a preposition ("עברתי אל המרינה").
_PLACE_NOUN_GARBLES = frozenset({"מרינה"})
_HARBOUR_WORDS = frozenset({"נמל", "מזח", "יאכטה", "יאכטות", "יכטה", "מפרשית", "מפרשיות", "שייט"})
# Consecutive Whisper words. The first may carry a glued prefix (and leading punctuation), inner
# words match exactly, the last may carry trailing punctuation. With asr_join_split_words on, the
# decoder's apostrophe split ("ג" + "'יסטון", "סארג" + "'ט") is glued before this table runs and the
# glued forms match ("נורד", "ג'יסטון") here and "סארג'ט" in _SINGLE. The two split keys stay until
# I1 is adopted (plan §3.6; L10 drops them "after I1"): with gluing off they are the only fix, and
# with it on they never match a glued word.
_MULTI: dict[tuple[str, ...], str] = {
    ("נורד", "ג", "'יסטון"): "נורג'סטון", ("נורד", "ג'יסטון"): "נורג'סטון",
    ("נור", "ג'יסטון"): "נורג'סטון", ("נורת", "איסטון"): "נורג'סטון",
    ("נורי", "דיי"): "נורידיי", ("סר", "רזט"): "סרזט", ("שר", "רזט"): "סרזט",
    ("שר", "הזית"): "סרזט", ("סארג", "'ט"): "סרזט",
    ("כותלי", "זרע"): "קוטלי זרע", ("הספק", "טהרה"): "הפסק טהרה",
    ("שולחנות", "פוליציסטיות"): "שחלות פוליציסטיות",
    # Genetic & halachic counselling multi-word garbles (glued and unglued forms)
    ("דור", "השערים"): "דור ישרים",
    ("דור", "יש", "עמימם"): "דור ישרים",
    ("עין", "קארי"): "עין כרם", ("עין", "-קארי"): "עין כרם",
    ("שלחי", "רשות"): "של חירשות",
    ("פאנל", "מוחם"): "פאנל מורחב",
    ("החיה", "חוץ", "גופית"): "הפריה חוץ-גופית",
    ("החיה", "חוץ", "-גופית"): "הפריה חוץ-גופית",
    ("החיה", "חוץ-גופית"): "הפריה חוץ-גופית",
    ("רב", "חופר"): "רב פופר", ("רב", "דב", "הופר"): "רב דב פופר",
    ("משה", "פינשטיין"): "משה פיינשטיין",
    ("לסגור", "וואות"): "לסגור וורט",
    ("בצורות", "טובות"): "בשורות טובות",
    ("תגורים", "טובים"): "צהריים טובים",
    ("שם", "פחתי"): "שם הפרטי",
}
# NEVER map (tests assert they stay unchanged): bare מרינה/קלינה, לינה, יונה, "מי יונה", יסמין,
# דיאנה, שרדת, הספק (alone), הספקה, תפילה, גבולות, מילות, "השר הזה", "הסר הזה", "סדר זאת", FIV,
# and "the marina" in a segment that names a harbour or a yacht.

_MULTI_LENGTHS = sorted({len(k) for k in _MULTI}, reverse=True)  # longest key first
# Hebrew one-letter prefixes that glue onto a word (ו/ה/ב/ל/מ/ש/כ), up to three of them.
_PREFIX_LETTERS = frozenset("והבלמשכ")
_MAX_PREFIX = 3
# Edge punctuation Whisper attaches to a word. The apostrophe is deliberately absent: it belongs
# to transliterated names ("נורג'סטון") and to split pieces such as "'יסטון".
_LEAD_PUNCT = frozenset("\"“„«([{-")
_TRAIL_PUNCT = frozenset("\".,!?:;…”»)]}")
# Typographic apostrophes are folded to ASCII before the lookup.
_APOSTROPHES = str.maketrans({"׳": "'", "‘": "'", "’": "'"})


def _split(word: str) -> tuple[str, str, str]:
    """(leading punctuation, core, trailing punctuation)."""
    w = word.translate(_APOSTROPHES)
    i, j = 0, len(w)
    while i < j and w[i] in _LEAD_PUNCT:
        i += 1
    while j > i and w[j - 1] in _TRAIL_PUNCT:
        j -= 1
    return w[:i], w[i:j], w[j:]


def _prefix_splits(core: str):
    """(glued prefix, bare) pairs, shortest prefix first."""
    for cut in range(_MAX_PREFIX + 1):
        if cut >= len(core):
            return
        if cut and core[cut - 1] not in _PREFIX_LETTERS:
            return
        yield core[:cut], core[cut:]


def _lookup_single(core: str, place_noun_ok: bool = True) -> Optional[str]:
    for prefix, bare in _prefix_splits(core):
        canon = _SINGLE.get(bare)
        if canon is None and prefix.endswith("ה") and (place_noun_ok or bare not in _PLACE_NOUN_GARBLES):
            canon = _DEFINITE_ONLY.get(bare)
        if canon is not None:
            return prefix + canon
    return None


def _names_harbour(tokens: Sequence[str]) -> bool:
    """A token is a harbour or yacht noun (_HARBOUR_WORDS, also after glued prefix letters)."""
    for tok in tokens:
        core = _split(tok)[1]
        if any(bare in _HARBOUR_WORDS for _p, bare in _prefix_splits(core)):
            return True
    return False


def _match_multi(tokens: Sequence[str], i: int) -> Optional[tuple[int, list[str]]]:
    """A _MULTI key starting at tokens[i] -> (tokens consumed, replacement tokens).

    The replacement keeps one token per word when the canonical form has as many parts as the key
    (each word keeps its own timing); otherwise the words become one token.
    """
    for n in _MULTI_LENGTHS:
        if i + n > len(tokens):
            continue
        lead, first, first_trail = _split(tokens[i])
        last_lead, last, trail = _split(tokens[i + n - 1])
        if first_trail or (last_lead and last_lead != "-") or not first or not last:
            continue
        last_tok = last_lead + last
        inner = tuple(t.translate(_APOSTROPHES) for t in tokens[i + 1:i + n - 1])
        for prefix, bare in _prefix_splits(first):
            canon = _MULTI.get((bare,) + inner + (last_tok,))
            if canon is None:
                continue
            parts = canon.split(" ")
            if len(parts) == n:
                return n, [lead + prefix + parts[0]] + parts[1:-1] + [parts[-1] + trail]
            return n, [lead + prefix + canon + trail]
    return None


def _rewrites(tokens: Sequence[str]) -> list[tuple[int, int, list[str]]]:
    """Every replacement in `tokens`, left to right: (index, tokens consumed, new tokens)."""
    out: list[tuple[int, int, list[str]]] = []
    harbour: Optional[bool] = None  # computed on first need: most token lists hold no place noun
    i = 0
    while i < len(tokens):
        multi = _match_multi(tokens, i)
        if multi is not None:
            n, repl = multi
            out.append((i, n, repl))
            i += n
            continue
        lead, core, trail = _split(tokens[i])
        place_noun_ok = True
        if core and any(bare in _PLACE_NOUN_GARBLES for _p, bare in _prefix_splits(core)):
            if harbour is None:
                harbour = _names_harbour(tokens)
            place_noun_ok = not harbour
        canon = _lookup_single(core, place_noun_ok) if core else None
        if canon is not None:
            out.append((i, 1, [lead + canon + trail]))
        i += 1
    return out


def normalize_text(text: str) -> tuple[str, int]:
    """Apply the tables to whitespace tokens. Returns (text, replacements); unchanged text is
    returned as is."""
    tokens = (text or "").split()
    rewrites = _rewrites(tokens)
    if not rewrites:
        return text, 0
    out: list[str] = []
    pos = 0
    for i, n, repl in rewrites:
        out.extend(tokens[pos:i])
        out.extend(repl)
        pos = i + n
    out.extend(tokens[pos:])
    return " ".join(out), len(rewrites)


def normalize_word_list(words: list[dict]) -> tuple[list[dict], int]:
    """Apply the tables to Whisper word dicts. Returns (NEW dicts, replacements).

    Extra keys are kept. Every word a rewrite produces carries "term_normalized": True. Words
    merged into one take the first word's keys, its start, the last word's end and the LOWEST
    probability of the group (no key when none has one): a fixed spelling does not make the
    model's evidence stronger than its weakest piece.
    """
    rewrites = _rewrites([str(w.get("word", "")) for w in words])
    out: list[dict] = []
    pos = 0
    for i, n, repl in rewrites:
        out.extend(dict(w) for w in words[pos:i])
        group = words[i:i + n]
        if len(repl) == n:
            for w, new_word in zip(group, repl):
                out.append(dict(w, word=new_word, term_normalized=True))
        else:
            merged = dict(group[0], word=repl[0], term_normalized=True)
            if "end" in group[-1]:
                merged["end"] = group[-1]["end"]
            probs = [w["probability"] for w in group if w.get("probability") is not None]
            if probs:
                merged["probability"] = min(probs)
            else:
                merged.pop("probability", None)
            out.append(merged)
        pos = i + n
    out.extend(dict(w) for w in words[pos:])
    return out, len(rewrites)


def normalize_asr_segments(segments: list[dict]) -> tuple[list[dict], int]:
    """Normalize the transcriber's segments. Returns (new segments, replacements).

    A segment's text is rebuilt as " ".join(words) only when one of its words changed (the rule
    merge.py uses); a segment without words gets normalize_text on its text. Idempotent; the
    input is never mutated.
    """
    out: list[dict] = []
    total = 0
    for seg in segments:
        new = dict(seg)
        words = seg.get("words") or []
        if words:
            new_words, n = normalize_word_list(words)
            new["words"] = new_words
            if n:
                new["text"] = " ".join(w["word"] for w in new_words)
        else:
            text, n = normalize_text(seg.get("text") or "")
            if n:
                new["text"] = text
        total += n
        out.append(new)
    return out, total
