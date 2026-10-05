"""Removal of Whisper repetition artefacts from the transcriber's segments, before merge.

Repeats are common in real calls ("לא, לא, לא", a sentence restarted after a pause) and must stay.
What gives a decoder artefact away is the word alignment: a repeated copy the alignment found no
audio for has words of ~0 s (DTW works on 20 ms frames). So:

1. Echo/loop copies: in a run of consecutive repeats of a 1-12 word unit, the later copies that are
   squeezed (most of their words shorter than collapsed_word_sec) are removed — every such copy of
   a phrase, and of a single word only when at least min_collapsed_unigram_copies are squeezed
   (genuine fast runs of one word had at most 2). Copy 0 is never removed.
2. Runaway loops whose copies kept plausible timing: copies beyond a hard cap are removed.
3. Subtitle-credit hallucinations: a short segment that starts with a stock phrase is dropped.

Words and text stay consistent (a touched segment's text is " ".join(words), the rule merge.py
uses), word starts never decrease, and the input is never mutated. The returned stats carry
counts and times only — never transcript text, which may contain names.

Standard library only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

# Subtitle credits Whisper is known to hallucinate on non-speech. Insurance only — none occurred
# in the stored calls. Never add "תודה רבה": callers genuinely say it.
DEFAULT_STOCK_PHRASES: tuple[str, ...] = (
    "תודה שצפיתם",
    "תודה על הצפייה",
    "תודה רבה שצפיתם",
    "כתוביות",
    "תורגם על ידי",
    "תרגום וכתוביות",
    "הירשמו לערוץ",
    "subtitles by",
    "thanks for watching",
    "thank you for watching",
    "amara.org",
)

# Hebrew points and cantillation (niqqud), U+0591..U+05C7, are ignored when comparing words.
_NIQQUD = re.compile("[֑-ׇ]")
# Monitoring only: a word this long has usually absorbed a pause next to it.
_LONG_WORD_SEC = 2.0


@dataclass(frozen=True)
class CleanupConfig:
    max_ngram: int = 12  # longest repeated unit searched for, in words
    collapsed_word_sec: float = 0.03  # shorter = no audio behind the word (0-1 alignment frames)
    min_collapsed_unigram_copies: int = 3  # one-word runs: act only with this many squeezed copies
    max_unigram_copies: int = 10  # hard cap on copies of one word (0 = off); genuine max seen: 7
    max_phrase_copies: int = 5  # hard cap on copies of a 2+ word phrase (0 = off); genuine max: 3
    stock_phrases: tuple[str, ...] = DEFAULT_STOCK_PHRASES
    stock_max_words: int = 10  # a stock phrase counts only when it starts a segment this short

    @classmethod
    def from_settings(cls, settings: Any) -> "CleanupConfig":
        return cls(
            max_ngram=settings.asr_cleanup_max_ngram,
            collapsed_word_sec=settings.asr_cleanup_collapsed_word_sec,
            min_collapsed_unigram_copies=settings.asr_cleanup_min_collapsed_unigram_copies,
            max_unigram_copies=settings.asr_cleanup_max_unigram_copies,
            max_phrase_copies=settings.asr_cleanup_max_phrase_copies,
        )


def word_key(word: str) -> str:
    """Matching key: NFKC, no niqqud, no punctuation/symbols/spaces, lower case.
    "" means the word is punctuation only."""
    w = _NIQQUD.sub("", unicodedata.normalize("NFKC", word or ""))
    w = "".join(ch for ch in w if not unicodedata.category(ch).startswith(("P", "S", "Z", "C")))
    return w.lower()


def find_repeat_runs(keys: list[str], max_n: int) -> list[tuple[int, int, int]]:
    """Consecutive repeats in `keys`: (start, n, reps) with reps >= 2, non-overlapping, greedy
    left to right.

    At each position the unit length n (1..max_n) covering the most tokens (n * reps) wins; ties
    go to the smaller n, so "x x x x" is n=1 reps=4, not n=2 reps=2.
    """
    runs: list[tuple[int, int, int]] = []
    i, total = 0, len(keys)
    while i < total:
        best: Optional[tuple[int, int, int]] = None
        for n in range(1, max_n + 1):
            if i + 2 * n > total:
                break
            unit = keys[i:i + n]
            reps = 1
            while keys[i + reps * n:i + (reps + 1) * n] == unit:
                reps += 1
            if reps >= 2 and (best is None or n * reps > best[1] * best[2]):
                best = (i, n, reps)
        if best is not None:
            runs.append(best)
            i = best[0] + best[1] * best[2]
        else:
            i += 1
    return runs


def _is_squeezed(copy_words: list[tuple], min_sec: float) -> bool:
    """More than half of the copy's words have (almost) no duration."""
    short = sum(1 for (_, _, _, start, end) in copy_words if (end - start) < min_sec)
    return short * 2 > len(copy_words)


def clean_asr_segments(
    segments: list[dict], cfg: Optional[CleanupConfig] = None
) -> tuple[list[dict], dict]:
    """Remove repetition artefacts from [{start, end, text, words:[{start, end, word}]}].

    Returns (new segments, stats). Extra keys are kept; segments without words pass through
    (merge falls back to segment level for them). stats:
    {segments_in, segments_out, words_in, words_out, removed_words, runs_found, loops_found,
     removed_collapsed_copies, removed_capped_copies, stock_segments_dropped, segments_emptied,
     timestamps_fixed, collapsed_words, long_words,
     events: [{"t": start sec, "n": unit words, "reps": copies, "removed": copies removed}]}
    collapsed_words / long_words count the input words (monitoring only, no action).
    """
    cfg = cfg or CleanupConfig()
    stats: dict[str, Any] = {
        "segments_in": len(segments), "segments_out": 0,
        "words_in": 0, "words_out": 0, "removed_words": 0,
        "runs_found": 0, "loops_found": 0,
        "removed_collapsed_copies": 0, "removed_capped_copies": 0,
        "stock_segments_dropped": 0, "segments_emptied": 0, "timestamps_fixed": 0,
        "collapsed_words": 0, "long_words": 0,
        "events": [],
    }

    # Flatten the whole call, so echoes across segment boundaries are found. Punctuation-only
    # tokens ("-", "...") take no part in matching.
    flat: list[tuple[int, int, str, float, float]] = []  # (seg_idx, word_idx, key, start, end)
    for si, seg in enumerate(segments):
        for wi, w in enumerate(seg.get("words") or []):
            start, end = float(w.get("start") or 0.0), float(w.get("end") or 0.0)
            stats["words_in"] += 1
            if end - start < cfg.collapsed_word_sec:
                stats["collapsed_words"] += 1
            elif end - start > _LONG_WORD_SEC:
                stats["long_words"] += 1
            key = word_key(str(w.get("word", "")))
            if key:
                flat.append((si, wi, key, start, end))

    drop: set[tuple[int, int]] = set()  # (seg_idx, word_idx)
    for i, n, reps in find_repeat_runs([f[2] for f in flat], max(1, cfg.max_ngram)):
        stats["runs_found"] += 1
        if reps >= 3:
            stats["loops_found"] += 1
        copies = [flat[i + r * n:i + (r + 1) * n] for r in range(reps)]
        squeezed = [r for r in range(1, reps) if _is_squeezed(copies[r], cfg.collapsed_word_sec)]
        remove: set[int] = set()
        if n >= 2 or len(squeezed) >= cfg.min_collapsed_unigram_copies:
            remove.update(squeezed)
            stats["removed_collapsed_copies"] += len(squeezed)
        cap = cfg.max_unigram_copies if n == 1 else cfg.max_phrase_copies
        if cap:
            extra = [r for r in range(reps) if r not in remove][cap:]
            remove.update(extra)
            stats["removed_capped_copies"] += len(extra)
        if remove:
            for r in remove:
                drop.update((si, wi) for (si, wi, _, _, _) in copies[r])
            stats["events"].append(
                {"t": round(copies[0][0][3], 2), "n": n, "reps": reps, "removed": len(remove)}
            )

    stock = tuple(
        k for k in ("".join(word_key(t) for t in p.split()) for p in cfg.stock_phrases) if k
    )
    out: list[dict] = []
    prev_start: Optional[float] = None
    for si, seg in enumerate(segments):
        words = seg.get("words") or []
        toks = [k for k in (word_key(t) for t in (seg.get("text") or "").split()) if k]
        seg_key = "".join(toks)
        if stock and seg_key and len(toks) <= cfg.stock_max_words and seg_key.startswith(stock):
            stats["stock_segments_dropped"] += 1
            stats["removed_words"] += len(words)
            continue
        if not words:
            out.append(dict(seg))
            continue
        kept = [dict(w) for wi, w in enumerate(words) if (si, wi) not in drop]
        stats["removed_words"] += len(words) - len(kept)
        if not kept:
            stats["segments_emptied"] += 1
            continue
        # Time-order guard (never reorders): starts never decrease, no negative duration.
        fixed = 0
        for w in kept:
            if prev_start is not None and w["start"] < prev_start:
                w["start"] = prev_start
                fixed += 1
            if w["end"] < w["start"]:
                w["end"] = w["start"]
                fixed += 1
            prev_start = w["start"]
        stats["timestamps_fixed"] += fixed
        new = dict(seg)
        new["words"] = kept
        if len(kept) != len(words):
            # The same rule merge.py uses for display text, so text and words never disagree.
            new["text"] = " ".join(w["word"] for w in kept).strip()
            new["start"], new["end"] = kept[0]["start"], kept[-1]["end"]
        elif fixed:
            new["start"] = min(new.get("start", kept[0]["start"]), kept[0]["start"])
            new["end"] = max(new.get("end", kept[-1]["end"]), kept[-1]["end"])
        out.append(new)

    stats["segments_out"] = len(out)
    stats["words_out"] = sum(len(s.get("words") or []) for s in out)
    return out, stats
