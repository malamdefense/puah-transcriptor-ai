"""**Automatic conversation summarization using a local language model (OpenAI-compatible interface).**

Works with Ollama / vLLM / LM Studio — any server that exposes a local `/v1/chat/completions` endpoint.  
No data leaves the server.

Since conversations can be long, a **map-reduce strategy** is used:  
each segment is summarized individually and then combined into a single structured summary.

**Output:**

```json
{ "summary", "key_points", "action_items", "topics", "keywords" }
```
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ..config import Settings


class LLMError(RuntimeError):
    pass


# ---------- LLM client (urllib, no external dependencies) ----------

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
            f"Cannot reach the language model at {settings.llm_base_url} ({exc}). "
            "Ensure Ollama/local server is running and the model is downloaded (ollama pull)."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Error calling the language model: {exc}") from exc


def _ollama_native_url(settings: Settings) -> str:
    base = settings.llm_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base + "/api/chat"


def _chat(settings: Settings, messages: list[dict], temperature: float = 0.2) -> str:
    if settings.llm_provider == "ollama":
        # Ollama native: controls num_ctx (the /v1 ignores it) and disables reasoning.
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
            # Models without "thinking" may reject the think parameter — try again without it.
            if "think" in str(exc).lower():
                payload.pop("think", None)
                data = _post_json(settings, url, payload)
            else:
                raise
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response format: {data}") from exc

    # Default: OpenAI-compatible interface (vLLM / LM Studio)
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
        raise LLMError(f"Unexpected model response format: {data}") from exc


def check_available(settings: Settings) -> bool:
    """Quick check if the model server is available."""
    try:
        url = settings.llm_base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {settings.llm_api_key}"})
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


# ---------- Building the transcript text ----------

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


# ---------- Parsing JSON content ----------

def _parse_json(content: str) -> dict[str, Any]:
    """Extract JSON object from the model response gracefully."""
    text = content.strip()
    # Remove code blocks if present
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


# ---------- Prompts ----------

_SYSTEM = (
    "אתה עוזר מקצועי ודיסקרטי המסכם שיחות בעברית עבור מכון פוע\"ה. "
    "השיחות עשויות לכלול תוכן רגיש (רפואי, הלכתי, אישי) — שמור על ניסוח עובדתי ומכובד. "
    "החזר אך ורק JSON תקין, ללא טקסט נוסף וללא גדרות קוד."
)

_JSON_INSTRUCTIONS = (
    "סכם את השיחה והחזר JSON במבנה הבא בלבד:\n"
    "{\n"
    '  "summary": "פסקה תמציתית של עיקרי השיחה",\n'
    '  "key_points": ["נקודה עיקרית", "..."],\n'
    '  "action_items": ["משימה/המשך טיפול", "..."],\n'
    '  "topics": ["נושא", "..."],\n'
    '  "keywords": ["מילת מפתח", "..."]\n'
    "}\n"
    "כל הערכים בעברית. אם אין פריטים לשדה — החזר רשימה ריקה."
)


def _summarize_single(settings: Settings, transcript: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"{_JSON_INSTRUCTIONS}\n\n=== תמלול השיחה ===\n{transcript}"},
    ]
    content = _chat(settings, messages)
    return _normalize_summary(_parse_json(content))


def _summarize_mapreduce(settings: Settings, transcript: str) -> dict[str, Any]:
    chunks = _chunks_by_chars(transcript, settings.summary_max_chars)
    partials = []
    for i, ch in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": (
                    f"זהו חלק {i} מתוך {len(chunks)} של תמלול שיחה. "
                    "כתוב סיכום נקודתי קצר בעברית של חלק זה (ללא JSON):\n\n" + ch
                ),
            },
        ]
        partials.append(_chat(settings, messages))

    combined = "\n\n".join(f"חלק {i}:\n{p}" for i, p in enumerate(partials, 1))
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"{_JSON_INSTRUCTIONS}\n\n"
                "להלן סיכומים חלקיים של חלקי השיחה. אחד אותם לסיכום כולל אחד:\n\n" + combined
            ),
        },
    ]
    content = _chat(settings, messages)
    return _normalize_summary(_parse_json(content))


def generate_client_summary(settings: Settings, result: dict[str, Any]) -> str:
    """מפיק "סיכום לפונה/לקוח" לפי הפרומפט של פוע"ה (טקסט עברי רציף)."""
    from . import puah_prompts

    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("התמלול ריק — אין מה לסכם.")
    messages = [
        {"role": "system", "content": puah_prompts.SYSTEM},
        {"role": "user", "content": puah_prompts.CLIENT_SUMMARY_TASK + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat(settings, messages).strip()


def generate_feedback(settings: Settings, result: dict[str, Any]) -> str:
    """Produces "professional feedback to the consultant/rab" based on PUAH's prompt (Hebrew text)."""
    from . import puah_prompts

    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to analyze.")
    messages = [
        {"role": "system", "content": puah_prompts.SYSTEM},
        {"role": "user", "content": puah_prompts.FEEDBACK_TASK + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat(settings, messages).strip()


def summarize(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    """Creates a structured summary for the transcript. Adds model and generated_at_placeholder fields.

    Raises LLMError if the model server is not available.
    """
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to summarize.")

    if len(transcript) > settings.summary_max_chars:
        summary = _summarize_mapreduce(settings, transcript)
    else:
        summary = _summarize_single(settings, transcript)

    summary["model"] = settings.llm_model
    return summary
