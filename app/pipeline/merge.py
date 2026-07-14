"""מיזוג תמלול + זיהוי דוברים לתמלול אחיד עם ייחוס דובר לכל מילה/קטע.

הגישה: לכל מילה (או קטע אם אין חותמות-זמן ברמת המילה) מוצאים את תור הדיבור
של הדובר עם החפיפה הזמנית הגדולה ביותר. לאחר מכן ממזגים מילים רצופות של אותו
דובר לקטעי תצוגה נוחים, תוך שבירה על מעבר דובר או על שתיקה ארוכה.
"""

from __future__ import annotations

# שתיקה ארוכה מ-1.2 שניות שוברת קטע גם אם הדובר זהה
_GAP_SPLIT_SEC = 1.2


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_for_interval(start: float, end: float, turns: list[dict]) -> str | None:
    """הדובר עם החפיפה הגדולה ביותר לקטע [start, end].

    אם אין חפיפה כלל — בוחר את התור הקרוב ביותר במרכז הזמן.
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

    # ללא חפיפה: התור הקרוב ביותר למרכז הקטע
    mid = (start + end) / 2.0
    nearest = min(
        turns,
        key=lambda t: 0.0 if t["start"] <= mid <= t["end"] else min(abs(t["start"] - mid), abs(t["end"] - mid)),
    )
    return nearest["speaker"]


def _assign_words(asr_segments: list[dict], turns: list[dict]) -> list[dict]:
    """משטח את הקטעים לרשימת מילים עם דובר. נופל לרמת קטע אם אין מילים."""
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
            # אין חותמות-זמן ברמת המילה — מתייחסים לקטע כיחידה אחת
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
    """מחזיר (segments, speaker_ids).

    segments — קטעי תצוגה: {start, end, speaker, text, words:[...]}.
    speaker_ids — רשימת מזהי הדוברים הייחודיים לפי סדר הופעתם.
    """
    words = _assign_words(asr_segments, turns)
    if not words:
        return [], []

    # מילוי דוברים חסרים (אם לא נמצא אף תור) — שיוך לשכן הקודם/הבא
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

    # בניית טקסט לכל קטע + סדר דוברים יציב
    speaker_order: list[str] = []
    for seg in segments:
        seg["text"] = " ".join(w["word"] for w in seg["words"]).strip()
        if seg["speaker"] not in speaker_order:
            speaker_order.append(seg["speaker"])

    return segments, speaker_order
