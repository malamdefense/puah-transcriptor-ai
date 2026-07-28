"""Merge transcript + speaker diarization into a unified transcript with speaker references for each word/segment.

The approach: for each word (or segment if there are no time stamps at the word level) we find the speaker turn with the largest temporal overlap. Then we merge consecutive words of the same speaker into convenient display segments, breaking on speaker change or long gaps.
"""

from __future__ import annotations

# Long gap of 1.2 seconds breaks a segment even if the speaker is the same
_GAP_SPLIT_SEC = 1.2


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_for_interval(start: float, end: float, turns: list[dict]) -> str | None:
    """The speaker with the largest overlap for the interval [start, end].

    If there is no overlap at all — choose the nearest turn in the center of time.
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


def _assign_words(asr_segments: list[dict], turns: list[dict]) -> list[dict]:
    """Flatten the segments into a list of words with speaker. Falls to segment level if there are no words."""
    words: list[dict] = []
    for seg in asr_segments:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                spk = _speaker_for_interval(w["start"], w["end"], turns)
                words.append(
                    {"start": w["start"], "end": w["end"], "word": w["word"], "speaker": spk}
                )
        else:
            # No time stamps at the word level — treat the segment as a single unit
            spk = _speaker_for_interval(seg["start"], seg["end"], turns)
            words.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "word": seg["text"],
                    "speaker": spk,
                }
            )
    return words


def merge(asr_segments: list[dict], turns: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (segments, speaker_ids).

    segments — display segments: {start, end, speaker, text, words:[...]}.
    speaker_ids — list of unique speaker IDs in the order of their appearance.
    """
    words = _assign_words(asr_segments, turns)
    if not words:
        return [], []

    # Fill missing speakers (if no turn is found) — assign to the previous/next speaker
    fallback = next((w["speaker"] for w in words if w["speaker"]), "SPEAKER_00")
    for w in words:
        if not w["speaker"]:
            w["speaker"] = fallback
        else:
            fallback = w["speaker"]

    segments: list[dict] = []
    cur: dict | None = None
    for w in words:
        new_block = (
            cur is None
            or w["speaker"] != cur["speaker"]
            or (w["start"] - cur["end"]) > _GAP_SPLIT_SEC
        )
        if new_block:
            if cur is not None:
                segments.append(cur)
            cur = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "words": [w],
            }
        else:
            cur["end"] = w["end"]
            cur["words"].append(w)
    if cur is not None:
        segments.append(cur)

    # Build text for each segment + stable speaker order
    speaker_order: list[str] = []
    for seg in segments:
        seg["text"] = " ".join(w["word"] for w in seg["words"]).strip()
        if seg["speaker"] not in speaker_order:
            speaker_order.append(seg["speaker"])

    return segments, speaker_order
