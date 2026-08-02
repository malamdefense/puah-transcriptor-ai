# -*- coding: utf-8 -*-
"""Success metrics (KPI) — based on 'Criteria-and-Success-Metrics.xlsx'.

Automatically calculates the measurable metrics, and explicitly marks the manual ones.
Used by scripts/eval_rag.py to build the metrics card in the dashboard.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .config import Settings

# Mapping the criteria from the Excel (single source for the dashboard)
CRITERIA = [
    {"key": "wer",          "name": "דיוק תמלול",   "target": "≥ 90% (WER ≤ 10%)",       "method": "השוואה לתמלולי ייחוס ידניים"},
    {"key": "coverage",     "name": "שלמות סיכום",  "target": "≥ 85% כיסוי נושאים",       "method": "LLM-judge מול סיכום הייחוס"},
    {"key": "pii",          "name": "יעילות PII",   "target": "0% דליפת מידע לענן",        "method": "ארכיטקטורה On-Prem + מצב offline"},
    {"key": "rtf",          "name": "זמן עיבוד",    "target": "≤ פי 3 מאורך השיחה",       "method": "זמן עיבוד ÷ אורך אודיו"},
    {"key": "satisfaction", "name": "שביעות רצון",  "target": "≥ 4/5 ממומחי תוכן",         "method": "סקר מובנה (ידני)"},
]


# ---------- 1. Transcription accuracy (WER) ----------

def _words(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def wer(reference: str, hypothesis: str) -> Optional[float]:
    """Word Error Rate — edit distance per word divided by the number of reference words. None if there is no reference."""
    r, h = _words(reference), _words(hypothesis)
    if not r:
        return None
    # Levenshtein on tokens (DP in one row)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(h)] / len(r)


def full_transcript_text(result: dict) -> str:
    return " ".join(seg.get("text", "") for seg in result.get("segments", []))


def find_reference_transcript(recording_stem: str, folder: str | Path) -> Optional[str]:
    """Finds a manual reference transcript: <stem>.gt.txt next to the recording, or references/<stem>.txt."""
    folder = Path(folder)
    for cand in (folder / f"{recording_stem}.gt.txt", folder / "references" / f"{recording_stem}.txt"):
        if cand.exists():
            return cand.read_text(encoding="utf-8")
    return None


# ---------- 2. Completeness of summary (topic coverage — LLM judge) ----------

def topic_coverage(settings: Settings, reference_summary: str, generated_summary: str) -> Optional[dict[str, Any]]:
    """Returns {coverage_pct, missing_topics} — how many of the essential topics in the reference summary are covered in the generated summary."""
    from .pipeline import summarize as sm

    if not (reference_summary or "").strip() or not (generated_summary or "").strip():
        return None
    messages = [
        {"role": "system", "content": "אתה בודק איכות מוקפד. החזר אך ורק JSON תקין, ללא טקסט נוסף."},
        {
            "role": "user",
            "content": (
                "להלן 'סיכום ייחוס' (אמת) ו'סיכום שנוצר' אוטומטית. קבע איזה אחוז מהנושאים "
                "המהותיים שבסיכום הייחוס מכוסים בסיכום שנוצר, ורשום נושאים מהותיים שחסרים.\n"
                'החזר: {"coverage_pct": <מספר 0-100>, "missing_topics": ["...", "..."]}\n\n'
                f"=== סיכום ייחוס ===\n{reference_summary}\n\n=== סיכום שנוצר ===\n{generated_summary}"
            ),
        },
    ]
    try:
        parsed = sm._parse_json(sm._chat(settings, messages))
    except Exception:  # noqa: BLE001
        return None
    if "coverage_pct" not in parsed:
        return None
    try:
        pct = max(0.0, min(100.0, float(parsed["coverage_pct"])))
    except (TypeError, ValueError):
        return None
    missing = parsed.get("missing_topics") or []
    return {"coverage_pct": pct, "missing_topics": [str(m) for m in missing if str(m).strip()]}


# ---------- 4. Processing time (RTF) ----------

def rtf(result: dict) -> Optional[float]:
    """Real-Time Factor = processing time / audio length. ≤ 3 = meets the target."""
    dur = result.get("duration_sec") or 0
    elapsed = result.get("elapsed_sec") or 0
    return (elapsed / dur) if (dur > 0 and elapsed > 0) else None


# ---------- 3. PII efficiency (architectural) ----------

def pii_egress_status(settings: Settings) -> dict[str, Any]:
    """PII egress to the cloud — currently always exists.

    Summary/feedback content goes to Gemini (after local PII enforcement — see llm_providers.BaseLLMProvider).
    Transcription and speaker diarization (Whisper/pyannote) remain local.
    """
    from .pipeline.pii import pii_available

    return {
        "no_cloud_egress": False,
        "pii_engine_local": pii_available(settings),
        "offline_capable": False,
        "note": (
            f"ספק LLM: Gemini ({settings.gemini_model}) — הספק המקומי הוסר. תוכן הסיכום/המשוב "
            "יוצא לענן של Google (לאחר הסרת PII מקומית חובה). התמלול וזיהוי הדוברים נשארים מקומיים."
        ),
    }
