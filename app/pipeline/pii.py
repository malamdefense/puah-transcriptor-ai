"""Deidentify PII from transcript, using HebSafeHarbor.

HebSafeHarbor requires pydantic<2 / spaCy 3.2 that conflicts with the application (pydantic v2),
therefore it is installed in a virtual environment (.venv-pii by default) and called as a subprocess
through app/pii_runner.py. This way we get the capability without breaking the rest of the system.

deidentify_result returns a new copy of the transcript with the segments and summary cleaned.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..config import Settings

# Root of the repository: app/pipeline/pii.py -> pipeline -> app -> root
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
            f"PII environment not found ({py}). Run: python rebuild.py (or create {settings.pii_venv} "
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
        raise PIIError(f"PII timed out ({settings.pii_timeout_sec}s)") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-600:]
        raise PIIError(f"PII failed (code {proc.returncode}): {tail}")

    try:
        result = json.loads(proc.stdout)
        out = result["texts"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PIIError(f"Unexpected output from PII: {exc}") from exc

    if len(out) != len(texts):
        raise PIIError("Number of segments returned from PII does not match the input")
    return out


def deidentify_result(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    """Returns a new copy of the transcript with the segments and summary cleaned.

    Performs a single subprocess call for all the texts together (to load the model once).
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
        raise PIIError("Transcript is empty — nothing to clean.")

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
