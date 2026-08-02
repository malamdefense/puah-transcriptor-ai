"""Automatic summarization of a call using Google Gemini.

Gemini is the summarization engine. All text goes through local PII enforcement before going to the cloud (enforced in BaseLLMProvider — see llm_providers).
Long calls are often summarized using map-reduce: we summarize each segment separately and then combine them into a single structured summary.

The output: {summary, key_points, action_items, topics, keywords}.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Settings
from . import llm_providers, puah_prompts
from .llm_providers import LLMError  # Kept for compatibility: app.main / app.worker / tests import from here


def _chat(settings: Settings, messages: list[dict], temperature: float = 0.2) -> str:
    """The only entry point to the language model (Gemini, via the factory in llm_providers).

    Kept as an independent function (not a direct call to the provider in every call site) to maintain compatibility
    with existing tests that monkey-patch sm._chat.
    """
    return llm_providers.get_provider(settings).chat(messages, temperature)


def check_available(settings: Settings) -> bool:
    """Quick check if we can produce summaries.

    After removing the local provider, there is no local server to check against — the condition is the presence of a Gemini API key and a PII environment (required before going to the cloud). There is no actual network call, so the health check remains immediate and does not consume API quota.
    """
    from .pii import pii_available

    return bool(settings.gemini_api_key) and pii_available(settings)


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


# ---------- JSON parsing ----------

def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
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


# ---------- Validation of output + retries ----------
# Gemini tends to get stuck in loops (about 20% of the time!) and output junk tokens like <unused50> instead of a response. This is not dependent on the input length and is not deterministic — repeating the same prompt succeeds. Therefore: we validate the output, and if it is not valid — we try again with a higher temperature to break the loop.

_JUNK_MARKERS = ("<unused", "<pad>", "<|", "�")


def _looks_valid(text: str, min_len: int = 60) -> bool:
    t = (text or "").strip()
    if len(t) < min_len:
        return False
    if any(m in t for m in _JUNK_MARKERS):
        return False
    if len(set(t)) < 12:
        return False
    return True


def _chat_checked(
    settings: Settings,
    messages: list[dict],
    attempts: int = 3,
    validator: Any = None,
) -> str:
    """Call the LLM and validate the output; try again (with higher temperature) if the output is junk."""
    check = validator or (lambda t: _looks_valid(t))
    last = ""
    for i in range(max(1, attempts)):
        out = _chat(settings, messages, temperature=0.2 + 0.25 * i)
        if check(out):
            return out
        last = out
    raise LLMError(
        f"The model returned invalid output in {attempts} attempts (probably junk). "
        f"תחילת הפלט: {last[:80]!r}"
    )


# ---------- Prompts (editable via the prompts_store interface) ----------

def _p(key: str) -> str:
    from .. import prompts_store  # Lazy import to avoid circular imports

    return prompts_store.get(key)


def _summarize_single(settings: Settings, transcript: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": puah_prompts.SUMMARY_SYSTEM},
        {"role": "user", "content": f"{puah_prompts.SUMMARY_INSTRUCTIONS}\n\n=== תמלול השיחה ===\n{transcript}"},
    ]
    # Valid = can be parsed to JSON with content
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
                f"{puah_prompts.SUMMARY_INSTRUCTIONS}\n\n"
                "להלן סיכומים חלקיים של חלקי השיחה. אחד אותם לסיכום כולל אחד:\n\n" + combined
            ),
        },
    ]
    content = _chat_checked(settings, messages, validator=lambda t: bool(_parse_json(t)))
    return _normalize_summary(_parse_json(content))


def generate_client_summary(settings: Settings, result: dict[str, Any]) -> str:
    """Generate "Client Summary" based on Puah's prompts (continuous Hebrew text)."""
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to summarize.")
    messages = [
        {"role": "system", "content": _p("puah_system")},
        {"role": "user", "content": _p("client_summary_task") + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat_checked(settings, messages, validator=lambda t: _looks_valid(t, 120)).strip()


def generate_feedback(settings: Settings, result: dict[str, Any]) -> str:
    """Generate "Professional Feedback" based on Puah's prompts (Hebrew text)."""
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to analyze.")
    messages = [
        {"role": "system", "content": _p("puah_system")},
        {"role": "user", "content": _p("feedback_task") + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat_checked(settings, messages, validator=lambda t: _looks_valid(t, 120)).strip()


def summarize(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    """Create a structured summary for the transcript. Adds model and generated_at_placeholder fields.

    Raises LLMError if the model server is not available.
    """
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to summarize.")

    if len(transcript) > settings.summary_max_chars:
        summary = _summarize_mapreduce(settings, transcript)
    else:
        summary = _summarize_single(settings, transcript)

    summary["model"] = settings.gemini_model
    return summary
