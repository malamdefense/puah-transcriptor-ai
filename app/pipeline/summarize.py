"""סיכום אוטומטי של שיחה באמצעות מודל שפה מקומי (ממשק תואם OpenAI).

עובד מול Ollama / vLLM / LM Studio — כל שרת שחושף /v1/chat/completions מקומית.
שום נתון אינו יוצא מהשרת. השיחות ארוכות לעיתים, לכן ממומשת סכימת map-reduce:
מסכמים כל קטע בנפרד ואז מאחדים לסיכום מובנה אחד.

הפלט: {summary, key_points, action_items, topics, keywords}.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ..config import Settings
from . import puah_prompts


class LLMError(RuntimeError):
    pass


# ---------- לקוח LLM (urllib, ללא תלות חיצונית) ----------

def _post_json(settings: Settings, url: str, payload: dict, auth: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.llm_timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise LLMError(f"שגיאת HTTP ממודל השפה ({exc.code}): {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(
            f"לא ניתן להגיע למודל השפה ב-{settings.llm_base_url} ({exc}). "
            "ודאו ש-Ollama/השרת המקומי פועל ושהמודל הורד (ollama pull)."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"שגיאה בקריאה למודל השפה: {exc}") from exc


def _ollama_native_url(settings: Settings) -> str:
    base = settings.llm_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base + "/api/chat"


def _chat(settings: Settings, messages: list[dict], temperature: float = 0.2) -> str:
    if settings.llm_provider == "ollama":
        # Ollama native: שולט ב-num_ctx (ה-/v1 מתעלם ממנו) ומכבה reasoning.
        url = _ollama_native_url(settings)
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": False,
            "think": settings.llm_think,
            "options": {
                "num_ctx": settings.llm_num_ctx,
                "temperature": temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }
        try:
            data = _post_json(settings, url, payload)
        except LLMError as exc:
            # מודלים ללא "חשיבה" עלולים לדחות את הפרמטר think — ננסה שוב בלעדיו.
            if "think" in str(exc).lower():
                payload.pop("think", None)
                data = _post_json(settings, url, payload)
            else:
                raise
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"תשובת Ollama בפורמט בלתי צפוי: {data}") from exc

    # ברירת מחדל: ממשק תואם OpenAI (vLLM / LM Studio)
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": settings.llm_max_tokens,
    }
    data = _post_json(settings, url, payload, auth=True)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"תשובת המודל בפורמט בלתי צפוי: {data}") from exc


def check_available(settings: Settings) -> bool:
    """בדיקה מהירה אם שרת המודל זמין."""
    try:
        url = settings.llm_base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {settings.llm_api_key}"})
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


# ---------- בניית טקסט התמלול ----------

def build_transcript_text(result: dict[str, Any]) -> str:
    speakers = result.get("speakers", {})
    lines = []
    for seg in result.get("segments", []):
        name = speakers.get(seg["speaker"], seg["speaker"])
        lines.append(f"{name}: {seg['text']}")
    return "\n".join(lines)


def _chunks_by_chars(text: str, max_chars: int) -> list[str]:
    lines = text.split("\n")
    chunks, cur, size = [], [], 0
    for ln in lines:
        if size + len(ln) > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


# ---------- ניתוח JSON עמיד ----------

def _parse_json(content: str) -> dict[str, Any]:
    """מחלץ אובייקט JSON מתוך תשובת המודל בצורה סלחנית."""
    text = content.strip()
    # הסרת גדרות קוד אם יש
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_summary(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(raw.get("summary", "")).strip(),
        "key_points": _as_str_list(raw.get("key_points")),
        "action_items": _as_str_list(raw.get("action_items")),
        "topics": _as_str_list(raw.get("topics")),
        "keywords": _as_str_list(raw.get("keywords")),
    }


# ---------- ולידציה של פלט + ניסיונות חוזרים ----------
# gemma נוטה לעיתים (כ-20% מהמקרים!) להתנוון ולפלוט טוקנים שמורים כמו <unused50> במקום
# תשובה. זה לא תלוי באורך הקלט ולא דטרמיניסטי — הרצה חוזרת של אותו פרומפט מצליחה.
# לכן: מוודאים שהפלט תקין, ואם לא — מנסים שוב עם טמפרטורה גבוהה יותר כדי לשבור את הניוון.

_JUNK_MARKERS = ("<unused", "<pad>", "<|", "�")


def _looks_valid(text: str, min_len: int = 60) -> bool:
    t = (text or "").strip()
    if len(t) < min_len:
        return False
    if any(m in t for m in _JUNK_MARKERS):
        return False
    if len(set(t)) < 12:  # פלט מנוון = מעט מאוד תווים ייחודיים
        return False
    return True


def _chat_checked(
    settings: Settings,
    messages: list[dict],
    attempts: int = 3,
    validator: Any = None,
) -> str:
    """קורא ל-LLM ומוודא שהפלט תקין; מנסה שוב (בטמפרטורה עולה) אם התקבל פלט מנוון."""
    check = validator or (lambda t: _looks_valid(t))
    last = ""
    for i in range(max(1, attempts)):
        out = _chat(settings, messages, temperature=0.2 + 0.25 * i)
        if check(out):
            return out
        last = out
    raise LLMError(
        f"המודל החזיר פלט לא תקין ב-{attempts} ניסיונות (ככל הנראה פלט מנוון). "
        f"תחילת הפלט: {last[:80]!r}"
    )


# ---------- פרומפטים (ניתנים לעריכה מהממשק דרך prompts_store) ----------

def _p(key: str) -> str:
    from .. import prompts_store  # ייבוא עצל למניעת מעגליות

    return prompts_store.get(key)


def _summarize_single(settings: Settings, transcript: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": puah_prompts.SUMMARY_SYSTEM},
        {"role": "user", "content": f"{_p('summary_instructions')}\n\n=== תמלול השיחה ===\n{transcript}"},
    ]
    # תקין = ניתן לפרסר ל-JSON עם תוכן
    content = _chat_checked(settings, messages, validator=lambda t: bool(_parse_json(t)))
    return _normalize_summary(_parse_json(content))


def _summarize_mapreduce(settings: Settings, transcript: str) -> dict[str, Any]:
    system = puah_prompts.SUMMARY_SYSTEM
    chunks = _chunks_by_chars(transcript, settings.summary_max_chars)
    partials = []
    for i, ch in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"זהו חלק {i} מתוך {len(chunks)} של תמלול שיחה. "
                    "כתוב סיכום נקודתי קצר בעברית של חלק זה (ללא JSON):\n\n" + ch
                ),
            },
        ]
        partials.append(_chat_checked(settings, messages, validator=lambda t: _looks_valid(t, 30)))

    combined = "\n\n".join(f"חלק {i}:\n{p}" for i, p in enumerate(partials, 1))
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{_p('summary_instructions')}\n\n"
                "להלן סיכומים חלקיים של חלקי השיחה. אחד אותם לסיכום כולל אחד:\n\n" + combined
            ),
        },
    ]
    content = _chat_checked(settings, messages, validator=lambda t: bool(_parse_json(t)))
    return _normalize_summary(_parse_json(content))


def generate_client_summary(settings: Settings, result: dict[str, Any]) -> str:
    """מפיק "סיכום לפונה/לקוח" לפי הפרומפט של פוע"ה (טקסט עברי רציף)."""
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("התמלול ריק — אין מה לסכם.")
    messages = [
        {"role": "system", "content": _p("puah_system")},
        {"role": "user", "content": _p("client_summary_task") + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat_checked(settings, messages, validator=lambda t: _looks_valid(t, 120)).strip()


def generate_feedback(settings: Settings, result: dict[str, Any]) -> str:
    """מפיק "משוב מקצועי ליועץ/לרב" לפי הפרומפט של פוע"ה (טקסט עברי)."""
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("התמלול ריק — אין מה לנתח.")
    messages = [
        {"role": "system", "content": _p("puah_system")},
        {"role": "user", "content": _p("feedback_task") + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat_checked(settings, messages, validator=lambda t: _looks_valid(t, 120)).strip()


def summarize(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    """מייצר סיכום מובנה לתמלול. מוסיף שדות model ו-generated_at_placeholder.

    מעלה LLMError אם שרת המודל לא זמין.
    """
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("התמלול ריק — אין מה לסכם.")

    if len(transcript) > settings.summary_max_chars:
        summary = _summarize_mapreduce(settings, transcript)
    else:
        summary = _summarize_single(settings, transcript)

    summary["model"] = settings.llm_model
    return summary
