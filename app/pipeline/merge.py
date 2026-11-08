"""Merge transcript + speaker diarization into a unified transcript with speaker references for each word/segment.

The approach: each word (or segment if there are no time stamps at the word level) goes to the speaker whose turns overlap it the most, summed over all of that speaker's turns. A word that falls in no turn follows the nearest word of its own ASR phrase.
With pyannote's exclusive diarization (one speaker at a time) the words are assigned on it, and short back-channel replies that it hands to the dominant speaker are restored from the regular diarization (R0).
Two conservative, punctuation-aware rules then fix the typical boundary errors: a 1-2 word island inside the other speaker's sentence (R1), and a sentence tail handed to the next speaker (R2).
Then we merge consecutive words of the same speaker into convenient display segments, breaking on speaker change or long gaps.

The thresholds were measured on the institute's calls; every looser value tried moved real short answers.
"""

from __future__ import annotations

import bisect
import logging
import re

log = logging.getLogger(__name__)

# Long gap of 1.2 seconds breaks a segment even if the speaker is the same
_GAP_SPLIT_SEC = 1.2
# Words shorter than 20 ms (alignment on fast or code-switched speech) are widened by +/-50 ms
_ZERO_WORD_SEC = 0.02
_ZERO_WORD_PAD = 0.05
# A word outside every turn, with no assigned neighbour in its ASR phrase: nearest turn within 1 s
_CONTEXT_MAX_DIST = 1.0
# R1 island: at most 2 words and 1.0 s, glued to both neighbours (gaps under 0.3 s)
_ISLAND_MAX_WORDS, _ISLAND_MAX_SEC, _ISLAND_MAX_GAP = 2, 1.0, 0.3
# R2 sentence tail: at most 2 words up to a sentence end, after a gap under 0.15 s
_TAIL_MAX_WORDS, _TAIL_MAX_GAP = 2, 0.15
# R0: a back-channel run of up to 3 words isolated by the regular diarization keeps its speaker,
# and so does a whole ASR phrase of up to 4 words isolated the same way ("כן, אני שמחה.")
_BACKCHANNEL_MAX_WORDS = 3
_PHRASE_MAX_WORDS = 4
_TIE_EPS = 1e-6
_DEFAULT_SPEAKER = "SPEAKER_00"
# Per-word keys carried from the ASR words to the merged words when present and not None (no rule
# reads them, so speaker labels never depend on them): the word probability (rounded to 3
# decimals), the decode window's "check against the audio" flag, and a per-word language.
CARRY_KEYS: tuple[str, ...] = ("probability", "window_flag", "lang")

_SENT_END = re.compile(r"[.?!…]['\"״)\]]*$")
_QUESTION_END = re.compile(r"\?['\"״)\]]*$")
_TOKEN = re.compile(r"[\w֐-׿']+")
# Short responses: legitimate turns of the other party that smoothing must never take away
_BACKCHANNEL = frozenset({
    "כן", "לא", "אהה", "אה", "אהא", "אהם", "הממ", "המ", "ממ", "מממ", "אוקיי", "אוקי", "נכון",
    "בסדר", "טוב", "יופי", "ברור", "נו", "וואו", "וואי", "באמת", "תודה", "רבה", "מעולה", "הבנתי",
    "בדיוק", "סבבה", "רגע", "אמן", "מה", "ok", "okay", "yes", "yeah", "no", "mm", "hmm", "right",
})


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_for_interval(start: float, end: float, turns: list[dict]) -> str | None:
    """The speaker with the largest overlap for the interval [start, end].

    If there is no overlap at all — choose the nearest turn in the center of time.
    A single-turn lookup; merge() itself sums the overlap per speaker (_TurnIndex).
    """
    if not turns:
        return None

    best_speaker = None
    best_overlap = 0.0
    for t in turns:
        ov = _overlap(start, end, t["start"], t["end"])
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = t["speaker"]

    if best_speaker is not None:
        return best_speaker

    # Without overlap: the nearest turn in the center of the interval
    mid = (start + end) / 2.0
    nearest = min(
        turns,
        key=lambda t: 0.0 if t["start"] <= mid <= t["end"] else min(abs(t["start"] - mid), abs(t["end"] - mid)),
    )
    return nearest["speaker"]


def _is_backchannel(text: str) -> bool:
    """1-3 tokens, all of them short responses ("כן.", "תודה רבה.", "אוקיי")."""
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    return 0 < len(tokens) <= _BACKCHANNEL_MAX_WORDS and all(t in _BACKCHANNEL for t in tokens)


def _ends_sentence(word: str) -> bool:
    return bool(_SENT_END.search(word.strip()))


class _TurnIndex:
    """Each speaker's turns sorted by start, so a word's overlaps are found by bisection."""

    def __init__(self, turns: list[dict]):
        by_speaker: dict[str, list[tuple[float, float]]] = {}
        for t in turns:
            if t["end"] > t["start"]:
                by_speaker.setdefault(t["speaker"], []).append((t["start"], t["end"]))
        self._turns = {spk: sorted(iv) for spk, iv in by_speaker.items()}
        self._starts = {spk: [b0 for b0, _ in iv] for spk, iv in self._turns.items()}
        self._max_len = {spk: max(b1 - b0 for b0, b1 in iv) for spk, iv in self._turns.items()}

    def overlaps(self, a0: float, a1: float) -> dict[str, tuple[float, float]]:
        """Per speaker: (overlap with [a0, a1] summed over its turns, start of its earliest such turn)."""
        out: dict[str, tuple[float, float]] = {}
        for spk, iv in self._turns.items():
            # Only turns starting before a1 can overlap; walking back, stop once even the longest
            # turn could no longer reach a0.
            j = bisect.bisect_left(self._starts[spk], a1) - 1
            reach = a0 - self._max_len[spk]
            total, first = 0.0, a1
            while j >= 0 and iv[j][0] > reach:
                ov = _overlap(a0, a1, iv[j][0], iv[j][1])
                if ov > 0:
                    total += ov
                    first = iv[j][0]
                j -= 1
            if total > 0:
                out[spk] = (total, first)
        return out

    def nearest(self, a0: float, a1: float) -> tuple[str | None, float]:
        """(speaker, distance) of the turn closest to [a0, a1]."""
        best: tuple[str | None, float] = (None, float("inf"))
        for spk, iv in self._turns.items():
            for b0, b1 in iv:
                d = max(0.0, b0 - a1, a0 - b1)
                if d < best[1]:
                    best = (spk, d)
        return best


def _flatten(asr_segments: list[dict]) -> list[dict]:
    """New word dicts in time order, each remembering its ASR segment (_seg).

    Falls to segment level if there are no words. Nothing is assumed about how many words a
    segment has: upstream clean-up may drop words, and term normalisation may join several
    into one word spanning their union.
    """
    words: list[dict] = []
    for si, seg in enumerate(asr_segments):
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                word = {"start": w["start"], "end": w["end"], "word": w["word"], "_seg": si}
                for key in CARRY_KEYS:
                    if w.get(key) is not None:
                        word[key] = w[key]
                words.append(word)
        else:
            # No time stamps at the word level — treat the segment as a single unit. It keeps the
            # segment's own window flag (and language, when a segment has one), so a flagged
            # decode window is still marked "check against the audio".
            unit = {"start": seg["start"], "end": seg["end"], "word": seg.get("text", ""), "_seg": si}
            for key in CARRY_KEYS[1:]:
                if seg.get(key) is not None:
                    unit[key] = seg[key]
            words.append(unit)
    # By start only: the sort is stable, so words that start together keep their transcript order
    # (the clean-up's time-order guard clamps a start to the previous one, leaving a shorter end).
    words.sort(key=lambda w: w["start"])
    return words


def _assign(words: list[dict], turns: list[dict]) -> int:
    """Set w["speaker"] on every word. Return how many words overlap no turn at all."""
    index = _TurnIndex(turns)
    prev = None
    for w in words:
        a0, a1 = w["start"], w["end"]
        if a1 - a0 < _ZERO_WORD_SEC:
            a0, a1 = a0 - _ZERO_WORD_PAD, a1 + _ZERO_WORD_PAD
        scores = index.overlaps(a0, a1)
        spk = None
        if scores:
            best = max(total for total, _ in scores.values())
            tied = [s for s, (total, _) in scores.items() if total >= best - _TIE_EPS]
            if len(tied) == 1:
                spk = tied[0]
            elif prev in tied:
                # Inside overlapping speech the ongoing speaker keeps the word
                spk = prev
            else:
                spk = min(tied, key=lambda s: scores[s][1])  # the turn that started first
        w["speaker"] = spk
        prev = spk or prev

    # Words outside every turn (whisper word edges stretch into pauses) follow their own ASR
    # phrase: the nearest word of the same segment that a turn placed (the previous one on a tie).
    outside = [i for i, w in enumerate(words) if w["speaker"] is None]
    for i in outside:
        w = words[i]
        found: tuple[float, str] | None = None
        for j in range(i - 1, -1, -1):
            if words[j]["_seg"] != w["_seg"]:
                break
            if not words[j].get("_context"):
                found = (w["start"] - words[j]["end"], words[j]["speaker"])
                break
        for j in range(i + 1, len(words)):
            if words[j]["_seg"] != w["_seg"]:
                break
            if words[j]["speaker"] is not None:
                gap = words[j]["start"] - w["end"]
                if found is None or gap < found[0]:
                    found = (gap, words[j]["speaker"])
                break
        if found is not None:
            w["speaker"] = found[1]
        else:
            spk, dist = index.nearest(w["start"], w["end"])
            if spk is not None and dist <= _CONTEXT_MAX_DIST:
                w["speaker"] = spk
            else:
                w["speaker"] = (words[i - 1]["speaker"] if i > 0 else None) or spk or _DEFAULT_SPEAKER
        w["_context"] = True
    return len(outside)


def _runs(words: list[dict]) -> list[list]:
    """[speaker, first index, last index] of each maximal run of one speaker."""
    runs: list[list] = []
    for i, w in enumerate(words):
        if runs and runs[-1][0] == w["speaker"]:
            runs[-1][2] = i
        else:
            runs.append([w["speaker"], i, i])
    return runs


def _text(words: list[dict], i0: int, i1: int) -> str:
    return " ".join(words[i]["word"] for i in range(i0, i1 + 1))


def _protected(words: list[dict], i0: int, i1: int) -> bool:
    return any(words[i].get("_protected") for i in range(i0, i1 + 1))


def _protect_backchannels(words: list[dict], regular_turns: list[dict]) -> int:
    """R0. The exclusive diarization can hand a short reply spoken over the other party to the
    dominant speaker. Where the regular (overlapping) diarization isolates, between two runs of
    one other speaker, a back-channel run or a whole short ASR phrase (whisper decoded it as one
    utterance, and the regular turns give all of it to one speaker), that run keeps the regular
    label and is protected from smoothing. Return how many words changed speaker."""
    ref = [{"start": w["start"], "end": w["end"], "word": w["word"], "_seg": w["_seg"]} for w in words]
    _assign(ref, regular_turns)
    runs = _runs(ref)
    seg_size: dict = {}
    for w in ref:
        seg_size[w["_seg"]] = seg_size.get(w["_seg"], 0) + 1
    restored = 0
    for k in range(1, len(runs) - 1):
        spk, i0, i1 = runs[k]
        n = i1 - i0 + 1
        if runs[k - 1][0] != runs[k + 1][0] or n > max(_BACKCHANNEL_MAX_WORDS, _PHRASE_MAX_WORDS):
            continue
        whole_phrase = ref[i0]["_seg"] == ref[i1]["_seg"] and seg_size[ref[i0]["_seg"]] == n
        if not ((n <= _BACKCHANNEL_MAX_WORDS and _is_backchannel(_text(ref, i0, i1)))
                or (n <= _PHRASE_MAX_WORDS and whole_phrase)):
            continue
        for i in range(i0, i1 + 1):
            if words[i]["speaker"] != spk:
                restored += 1
            words[i]["speaker"] = spk
            words[i]["_protected"] = True
    return restored


def _smooth(words: list[dict]) -> tuple[int, int]:
    """R1 then R2, each in one left-to-right pass. Return the words relabelled by each rule.

    R1 island: a run of B between two runs of A, of at most 2 words and 1.0 s, with gaps under
    0.3 s on both sides, goes to A, unless it is a back-channel, protected, a question (it ends in
    "?": a question is a turn of its own), or a standalone sentence (the word before it and its own
    last word both end a sentence).
    R2 sentence tail: at a switch A -> B inside one ASR phrase, after a gap under 0.15 s, where
    A's last word does not end a sentence and B reaches a sentence end within 2 words, those
    words go to A (whisper decoded one punctuated phrase: it belongs to the speaker who began
    it), unless they are a back-channel or protected.
    """
    r1 = 0
    runs = _runs(words)
    last_changed = -2
    for k in range(1, len(runs) - 1):
        if k - 1 == last_changed:
            # The left neighbour was just merged away, so the runs are stale: never chain relabels
            continue
        spk, i0, i1 = runs[k]
        host = runs[k - 1][0]
        if runs[k + 1][0] != host or spk == host:
            continue
        n = i1 - i0 + 1
        dur = words[i1]["end"] - words[i0]["start"]
        gap_before = words[i0]["start"] - words[i0 - 1]["end"]
        gap_after = words[i1 + 1]["start"] - words[i1]["end"]
        if (n > _ISLAND_MAX_WORDS or dur > _ISLAND_MAX_SEC
                or gap_before >= _ISLAND_MAX_GAP or gap_after >= _ISLAND_MAX_GAP):
            continue
        if _is_backchannel(_text(words, i0, i1)) or _protected(words, i0, i1):
            continue
        if _ends_sentence(words[i0 - 1]["word"]) and _ends_sentence(words[i1]["word"]):
            continue
        if _QUESTION_END.search(words[i1]["word"]):
            continue
        for i in range(i0, i1 + 1):
            words[i]["speaker"] = host
        r1 += n
        last_changed = k

    r2 = 0
    i = 1
    while i < len(words):
        prev, a, b = words[i - 1], words[i - 1]["speaker"], words[i]["speaker"]
        if a == b:
            i += 1
            continue
        k_end = None
        if (not _ends_sentence(prev["word"]) and words[i]["start"] - prev["end"] < _TAIL_MAX_GAP
                and words[i]["_seg"] == prev["_seg"]):
            for k in range(i, min(i + _TAIL_MAX_WORDS, len(words))):
                if words[k]["speaker"] != b or words[k]["_seg"] != prev["_seg"]:
                    break
                if _ends_sentence(words[k]["word"]):
                    k_end = k
                    break
        if (k_end is not None and not _is_backchannel(_text(words, i, k_end))
                and not _protected(words, i, k_end)):
            for k in range(i, k_end + 1):
                words[k]["speaker"] = a
            r2 += k_end - i + 1
            i = k_end + 1
            continue
        i += 1
    return r1, r2


def _build_segments(words: list[dict]) -> tuple[list[dict], list[str]]:
    """Display segments: a new one on a speaker change or a gap over 1.2 s. Word dicts are
    rebuilt as {start, end, word, speaker} plus the CARRY_KEYS the word has, without the
    internal keys."""
    segments: list[dict] = []
    cur: dict | None = None
    for w in words:
        word = {"start": w["start"], "end": w["end"], "word": w["word"], "speaker": w["speaker"]}
        if w.get("probability") is not None:
            word["probability"] = round(float(w["probability"]), 3)
        for key in CARRY_KEYS[1:]:
            if w.get(key) is not None:
                word[key] = w[key]
        new_block = (
            cur is None
            or word["speaker"] != cur["speaker"]
            or (word["start"] - cur["end"]) > _GAP_SPLIT_SEC
        )
        if new_block:
            if cur is not None:
                segments.append(cur)
            cur = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word["speaker"],
                "words": [word],
            }
        else:
            cur["end"] = max(cur["end"], word["end"])
            cur["words"].append(word)
    if cur is not None:
        segments.append(cur)

    # Build text for each segment + stable speaker order
    speaker_order: list[str] = []
    for seg in segments:
        seg["text"] = " ".join(w["word"] for w in seg["words"]).strip()
        if seg["speaker"] not in speaker_order:
            speaker_order.append(seg["speaker"])

    return segments, speaker_order


def merge(
    asr_segments: list[dict],
    turns: list[dict],
    exclusive_turns: list[dict] | None = None,
    *,
    smooth: bool = True,
    stats: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Return (segments, speaker_ids).

    segments — display segments: {start, end, speaker, text, words:[{start, end, word, speaker}]};
    a word also carries `probability` (3 decimals), `window_flag` and `lang` when its ASR word has
    them (CARRY_KEYS).
    speaker_ids — list of unique speaker IDs in the order of their appearance.

    turns — the regular diarization {start, end, speaker} (different speakers may overlap).
    exclusive_turns — pyannote's exclusive diarization of the same run (same labels). When given,
    words are assigned on it and back-channel replies are kept from `turns` (R0).
    smooth — relabel islands (R1) and sentence tails (R2).
    stats — when a dict, it is updated with counts only: {words, outside (words in no turn),
    exclusive, smooth, r0, r1, r2 (words relabelled by each rule)}.
    The inputs are not modified.
    """
    words = _flatten(asr_segments)
    outside = restored = r1 = r2 = 0
    if words:
        primary = exclusive_turns or turns
        if not primary:
            for w in words:
                w["speaker"] = _DEFAULT_SPEAKER
            outside = len(words)
        else:
            outside = _assign(words, primary)
            restored = _protect_backchannels(words, turns) if exclusive_turns and turns else 0
            r1, r2 = _smooth(words) if smooth else (0, 0)
            # Counts only — never transcript text
            log.debug(
                "merge: %d words, %d outside every turn, exclusive=%s, R0 %d, R1 %d, R2 %d words relabelled",
                len(words), outside, bool(exclusive_turns), restored, r1, r2,
            )
    if stats is not None:
        stats.update({"words": len(words), "outside": outside, "exclusive": bool(exclusive_turns),
                      "smooth": bool(smooth), "r0": restored, "r1": r1, "r2": r2})
    if not words:
        return [], []
    return _build_segments(words)
