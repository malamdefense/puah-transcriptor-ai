"""Generation of the two PUAH deliverables using Google Gemini.

Gemini is the only summarization engine — the local model provider was removed.
All text goes through local PII redaction before leaving to the cloud; that is enforced
structurally in BaseLLMProvider (see llm_providers), not here.

Produces:
  * generate_client_summary — the summary sent to the caller
  * generate_feedback       — the internal feedback for the consultant

The structured "automatic summary" that this module used to build with the model is
gone. The only parts of it still consumed downstream (topics + keywords, which feed the
dashboard) are now derived locally, without any AI call — see pipeline/keywords.py.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Settings
from . import llm_providers
from .llm_providers import LLMError  # Kept for compatibility: app.main / app.worker / tests import from here


def _chat(settings: Settings, messages: list[dict], temperature: float = 0.2) -> str:
    """The only entry point to the language model (Gemini, via the factory in llm_providers).

    Kept as an independent function (rather than calling the provider at every call
    site) so existing tests can monkey-patch sm._chat.
    """
    return llm_providers.get_provider(settings).chat(messages, temperature)


def check_available(settings: Settings) -> bool:
    """Quick check whether summaries can be produced at all.

    There is no local server to probe: the condition is a Gemini API key plus an
    installed PII environment (mandatory before anything leaves to the cloud). No
    network call is made, so the health check stays instant and consumes no API quota.
    """
    from .pii import pii_available

    return bool(settings.gemini_api_key) and pii_available(settings)


# ---------- Building the transcript text ----------

def build_transcript_text(result: dict[str, Any]) -> str:
    speakers = result.get("speakers", {})
    lines = []
    for seg in result.get("segments", []):
        name = speakers.get(seg["speaker"], seg["speaker"])
        lines.append(f"{name}: {seg['text']}")
    return "\n".join(lines)


# ---------- JSON parsing ----------
# Still used by eval_metrics.topic_coverage, which asks the model for a JSON verdict.

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


# ---------- Output validation + retries ----------
# Guards against degenerate model output: empty/truncated answers, or reserved junk
# tokens (<unused50>, <pad>) emitted instead of a real response. Not input-length
# dependent nor deterministic — re-running the same prompt usually succeeds. So we
# validate the output and retry with a rising temperature to break the degenerate state.

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
    """Call the LLM and validate the output; retry (with higher temperature) on junk."""
    check = validator or (lambda t: _looks_valid(t))
    last = ""
    for i in range(max(1, attempts)):
        out = _chat(settings, messages, temperature=0.2 + 0.25 * i)
        if check(out):
            return out
        last = out
    raise LLMError(
        f"The model returned invalid output in {attempts} attempts (probably degenerate). "
        f"Output starts with: {last[:80]!r}"
    )


# ---------- Prompts (editable via the prompts_store interface) ----------

def _p(key: str) -> str:
    from .. import prompts_store  # Lazy import to avoid circular imports

    return prompts_store.get(key)


def generate_client_summary(settings: Settings, result: dict[str, Any]) -> str:
    """Generate the "Client Summary" using PUAH's prompts (continuous Hebrew text)."""
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to summarize.")
    messages = [
        {"role": "system", "content": _p("puah_system")},
        {"role": "user", "content": _p("client_summary_task") + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat_checked(settings, messages, validator=lambda t: _looks_valid(t, 120)).strip()


def generate_feedback(settings: Settings, result: dict[str, Any]) -> str:
    """Generate the internal "Professional Feedback" using PUAH's prompts (Hebrew text)."""
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to analyze.")
    messages = [
        {"role": "system", "content": _p("puah_system")},
        {"role": "user", "content": _p("feedback_task") + "\n\n=== תמלול השיחה ===\n" + transcript},
    ]
    return _chat_checked(settings, messages, validator=lambda t: _looks_valid(t, 120)).strip()
