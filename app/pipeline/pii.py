"""הסרת פרטים מזהים (PII) מתמלול, באמצעות HebSafeHarbor.

HebSafeHarbor דורש pydantic<2 / spaCy 3.2 שמתנגשים עם האפליקציה (pydantic v2),
לכן הוא מותקן בסביבה וירטואלית מבודדת (ברירת מחדל: .venv-pii) ונקרא כתת-תהליך
דרך app/pii_runner.py. כך אנו מקבלים את היכולת בלי לשבור את שאר המערכת.

deidentify_result מחזיר עותק חדש של התמלול שבו טקסט הקטעים והסיכום עברו ניקוי.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..config import Settings

# שורש הריפו: app/pipeline/pii.py -> pipeline -> app -> root
ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER = ROOT / "app" / "pii_runner.py"


class PIIError(RuntimeError):
    pass


def pii_python(settings: Settings) -> Path:
    venv = ROOT / settings.pii_venv
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pii_available(settings: Settings) -> bool:
    return pii_python(settings).exists() and RUNNER.exists()


def _run_pii(settings: Settings, texts: list[str]) -> list[str]:
    py = pii_python(settings)
    if not py.exists():
        raise PIIError(
            f"סביבת ה-PII לא נמצאה ({py}). הריצו: python rebuild.py (או הקימו את {settings.pii_venv} "
            "מ-requirements-pii.txt)."
        )
    try:
        proc = subprocess.run(
            [str(py), str(RUNNER)],
            input=json.dumps({"texts": texts}, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=settings.pii_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise PIIError(f"הסרת ה-PII חרגה מזמן הקצוב ({settings.pii_timeout_sec}s)") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-600:]
        raise PIIError(f"רץ ה-PII נכשל (קוד {proc.returncode}): {tail}")

    try:
        result = json.loads(proc.stdout)
        out = result["texts"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PIIError(f"פלט בלתי צפוי מרץ ה-PII: {exc}") from exc

    if len(out) != len(texts):
        raise PIIError("מספר הקטעים שחזרו מרץ ה-PII אינו תואם לקלט")
    return out


def deidentify_result(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    """מחזיר עותק חדש של התמלול שעבר הסרת PII (קטעים + סיכום).

    מבצע קריאת subprocess *אחת* לכל הטקסטים יחד (כדי לטעון את המודל פעם אחת בלבד).
    """
    segments = result.get("segments", [])
    seg_texts = [s.get("text", "") for s in segments]

    summ = result.get("summary") or {}
    sum_summary = [summ.get("summary", "")] if summ else []
    sum_keypts = list(summ.get("key_points", [])) if summ else []
    sum_actions = list(summ.get("action_items", [])) if summ else []
    sum_topics = list(summ.get("topics", [])) if summ else []
    sum_keywords = list(summ.get("keywords", [])) if summ else []

    has_client = bool((result.get("client_summary") or "").strip())
    has_feedback = bool((result.get("feedback") or "").strip())
    extra = []
    if has_client:
        extra.append(result["client_summary"])
    if has_feedback:
        extra.append(result["feedback"])

    flat = seg_texts + sum_summary + sum_keypts + sum_actions + sum_topics + sum_keywords + extra
    if not any(t and str(t).strip() for t in flat):
        raise PIIError("התמלול ריק — אין מה לנקות.")

    anon = _run_pii(settings, flat)

    new = copy.deepcopy(result)
    i = 0
    for seg in new.get("segments", []):
        seg["text"] = anon[i]
        i += 1
    if summ:
        ns = new["summary"]
        if sum_summary:
            ns["summary"] = anon[i]
            i += 1
        n = len(sum_keypts); ns["key_points"] = anon[i : i + n]; i += n
        n = len(sum_actions); ns["action_items"] = anon[i : i + n]; i += n
        n = len(sum_topics); ns["topics"] = anon[i : i + n]; i += n
        n = len(sum_keywords); ns["keywords"] = anon[i : i + n]; i += n
    if has_client:
        new["client_summary"] = anon[i]; i += 1
    if has_feedback:
        new["feedback"] = anon[i]; i += 1

    new["pii_deidentified"] = True
    return new
