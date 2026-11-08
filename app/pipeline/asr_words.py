"""Decoder word pieces -> display words (I1), and the per-window "check against the audio" flag (I2).

faster-whisper starts a new word at every punctuation-led token (its tokenizer splits on them), so
an inner apostrophe (geresh), hyphen, decimal point, thousands comma or a trailing "%" came out as
two words, and the transcript showed a space that the decoder never wrote ("ג 'יסטון", "3 .5",
"50 %"). The decoder's own segment text has the right spacing; the pieces carry it as `lead`:

- `lead_flags` marks, per raw faster-whisper word of one segment, whether it begins a new word:
  the raw word starts with whitespace, or the raw word before it was whitespace-only (the
  transcriber drops such a word, so its space belongs to the next one).
- `join_split_words` glues every piece with `lead=False` onto the word before it in the same
  segment. The transcriber never glues (its output, the raw sidecar, keeps the pieces); the worker
  glues in its post-processing, before the clean-up.

`window_flag` codes a decode window whose kept result failed a quality gate ("gate_failed") or was
accepted from sampling at a high temperature ("sampled_hi"). faster-whisper reports temperature 1.0
for a window where every temperature failed, whatever result it kept, so the gate is recomputed
from the window's compression ratio and average log-probability.

Standard library only; the runner keeps an identical copy (runner.asr_words) and its self-test
requires identical outputs.
"""

from __future__ import annotations

from typing import Optional, Sequence

WINDOW_FLAG_CODES: tuple[str, str] = ("gate_failed", "sampled_hi")
# A window accepted from sampling at this temperature or above is flagged
SAMPLED_HI_MIN_T: float = 0.6


def lead_flags(raw_words: Sequence[str]) -> list[Optional[bool]]:
    """One entry per raw faster-whisper word of ONE segment, in order.

    None for a word whose strip() is empty (the transcriber drops it); otherwise True iff the raw
    word starts with whitespace or the previous raw word was dropped as whitespace-only; else False.
    """
    out: list[Optional[bool]] = []
    prev_dropped = False
    for raw in raw_words:
        raw = raw or ""
        if not raw.strip():
            out.append(None)
            prev_dropped = True
            continue
        out.append(raw[:1].isspace() or prev_dropped)
        prev_dropped = False
    return out


def join_split_words(pieces: list[dict]) -> tuple[list[dict], int]:
    """Glue the pieces of ONE segment whose `lead` is False onto the word before them.

    A glued word keeps the first piece's keys and start; `word` is the pieces' strings joined with
    no separator, `end` is the last piece's end, `probability` the minimum over the pieces'
    non-null probabilities (no key when there is none), and `n_pieces` (>= 2) the number of
    pieces. A piece without a `lead` key counts as lead=True, so pre-Phase-A transcriber output and
    old sidecars never glue; the first piece always starts a word. Never glues across segments
    (the caller passes one segment), never mutates its input, idempotent.
    Returns (words, n_glued) with n_glued = len(pieces) - len(words).
    """
    words: list[dict] = []
    for p in pieces:
        if words and p.get("lead", True) is False:
            w = words[-1]
            w["word"] = str(w.get("word", "")) + str(p.get("word", ""))
            if "end" in p:
                w["end"] = p["end"]
            probs = [x for x in (w.get("probability"), p.get("probability")) if x is not None]
            if probs:
                w["probability"] = min(probs)
            else:
                w.pop("probability", None)
            w["n_pieces"] = int(w.get("n_pieces", 1)) + 1
        else:
            words.append(dict(p))
    return words, len(pieces) - len(words)


def join_split_segments(segments: list[dict]) -> tuple[list[dict], dict]:
    """join_split_words on each segment's words. Returns (new segment dicts, counts).

    `text`, `start`, `end` and every other segment key are kept; segments without words pass
    through (as copies). counts = {"pieces_in", "words_out", "glued"}: numbers only.
    """
    out: list[dict] = []
    stats = {"pieces_in": 0, "words_out": 0, "glued": 0}
    for seg in segments:
        new = dict(seg)
        words = seg.get("words") or []
        if words:
            glued, n = join_split_words(list(words))
            new["words"] = glued
            stats["pieces_in"] += len(words)
            stats["words_out"] += len(glued)
            stats["glued"] += n
        out.append(new)
    return out, stats


def window_flag(
    avg_logprob: Optional[float],
    compression_ratio: Optional[float],
    temperature: Optional[float],
    *,
    compression_ratio_threshold: Optional[float],
    log_prob_threshold: Optional[float],
) -> Optional[str]:
    """"gate_failed" if the window's kept result fails a configured gate (compression ratio above,
    or average log-probability below, its threshold: the strict inequalities faster-whisper uses;
    a None threshold or value disables that test), else "sampled_hi" if it was accepted at
    temperature >= SAMPLED_HI_MIN_T, else None."""
    failed = False
    if (compression_ratio_threshold is not None and compression_ratio is not None
            and compression_ratio > compression_ratio_threshold):
        failed = True
    if log_prob_threshold is not None and avg_logprob is not None and avg_logprob < log_prob_threshold:
        failed = True
    if failed:
        return "gate_failed"
    if temperature is not None and temperature >= SAMPLED_HI_MIN_T:
        return "sampled_hi"
    return None
