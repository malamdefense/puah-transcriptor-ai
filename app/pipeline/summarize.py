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
import logging
import re
from typing import Any

from ..config import Settings
from . import llm_providers, output_guard
from .pii_context import contextual_redact
from .llm_providers import LLMError  # Kept for compatibility: app.main / app.worker / tests import from here

log = logging.getLogger(__name__)


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
    """The transcript as the LLM sees it — only ever used to build a Gemini prompt.

    The conversation-level PII pass runs here, before the provider's own engine redaction
    (llm_providers.BaseLLMProvider): the engines receive the joined text, but a one-word answer
    on its own line or a number dictated across lines still defeats them. Raw names, phone
    digits and spelled-out e-mails are removed before anything is handed to the provider.
    """
    speakers = result.get("speakers", {})
    segs = result.get("segments", [])
    texts, _, _ = contextual_redact(
        [seg.get("text", "") for seg in segs],
        [seg.get("speaker", "") for seg in segs],
    )
    lines = []
    for seg, text in zip(segs, texts):
        name = speakers.get(seg["speaker"], seg["speaker"])
        lines.append(f"{name}: {text}")
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
    *,
    guard_tags: bool = False,
    template: str | None = None,
    report: dict | None = None,
) -> str:
    """Call the LLM and validate the output; retry (with higher temperature) on junk.

    guard_tags: an answer that is valid but still holds a redaction placeholder ("<ארגון_>") is
    retried with a repair turn — the answer plus a request to rewrite it without placeholders —
    instead of a blind re-roll, which reproduces a placeholder that comes from the prompt itself.
    Junk and placeholders share the same attempt budget. If every valid answer holds one, the
    answer with the fewest is sanitised deterministically (output_guard.sanitize_caller_text,
    with `template` — the raw task prompt — for restoring a copied template line) and returned:
    placeholders never raise, and neither does a repair turn that fails (the answer in hand is
    sanitised the same way). With settings.llm_tag_guard_enabled off the guard only observes:
    the first valid answer is returned as is, and the record and log still say what it holds.
    report, when given (with guard_tags), receives output_guard.check_record — labels only.
    """
    check = validator or (lambda t: _looks_valid(t))
    repair = guard_tags and settings.llm_tag_guard_enabled
    last = ""
    seen: list[str] = []  # labels found in any valid answer
    best: tuple[str, list[str]] | None = None  # the valid answer with the fewest placeholders
    convo = list(messages)
    n = used = max(1, attempts)
    for i in range(n):
        try:
            out = _chat(settings, convo, temperature=0.2 + 0.25 * i)
        except Exception as exc:  # noqa: BLE001
            if best is None:
                raise
            # A failed repair turn (quota, network, PII engines) must not lose the valid answer
            # in hand: it is sanitised below, as when the attempts run out. Type only in the log.
            log.warning("Repair turn failed (%s); sanitising the tagged answer instead", type(exc).__name__)
            used = i + 1
            break
        if not check(out):
            last = out
            continue
        labels = output_guard.find_tags(out) if guard_tags else []
        if not labels or not repair:
            if guard_tags:
                record = output_guard.check_record(
                    tags_found=seen + labels, sanitized=False, attempts=i + 1, residual=labels
                )
                if report is not None:
                    report.update(record)
                if labels:
                    log.warning("Model output holds redaction placeholders %s (guard off: kept as is)",
                                record["residual"])
                elif seen:
                    log.info("Redaction placeholders %s repaired by retry (attempt %d)",
                             record["tags_found"], i + 1)
            return out
        seen += labels
        if best is None or len(labels) < len(best[1]):
            best = (out, labels)
        convo =list(messages) + [{"role": "assistant", "content": out}, output_guard.repair_message(labels)]
    if best is not None:
        clean, _ = output_guard.sanitize_caller_text(best[0].strip(), template=template)
        residual = output_guard.find_tags(clean)
        record = output_guard.check_record(tags_found=seen, sanitized=True, attempts=used, residual=residual)
        if report is not None:
            report.update(record)
        log.warning("Model output kept redaction placeholders %s after %d attempts; sanitised%s",
                    record["tags_found"], used, " (RESIDUAL %s)" % record["residual"] if residual else "")
        return clean
    raise LLMError(
        f"The model returned invalid output in {attempts} attempts (probably degenerate). "
        f"Last output: {len(last or '')} characters"  # never quoted: model text can hold a name
    )


# ---------- Prompts (editable via the prompts_store interface) ----------

def _p(key: str) -> str:
    from .. import prompts_store  # Lazy import to avoid circular imports

    return prompts_store.get(key)


def _messages(task_key: str, transcript: str) -> list[dict]:
    """System + user messages for one deliverable.

    The fixed placeholder rule and the closing reminder are appended here, in code, so they hold
    whatever the editable prompts say (output_guard.PLACEHOLDER_RULE / REMINDER).
    """
    return [
        {"role": "system", "content": _p("puah_system") + "\n\n" + output_guard.PLACEHOLDER_RULE},
        {"role": "user", "content": _p(task_key) + "\n\n=== תמלול השיחה ===\n" + transcript
                                     + "\n=== סוף התמלול ===\n" + output_guard.REMINDER},
    ]


def generate_client_summary(settings: Settings, result: dict[str, Any], *, report: dict | None = None) -> str:
    """Generate the "Client Summary" using PUAH's prompts (continuous Hebrew text).

    It goes to the caller, so it is guarded against redaction placeholders (see _chat_checked);
    report, when given, receives the check record.
    """
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to summarize.")
    return _chat_checked(
        settings, _messages("client_summary_task", transcript), validator=lambda t: _looks_valid(t, 120),
        guard_tags=True, template=_p("client_summary_task"), report=report,
    ).strip()


def generate_feedback(settings: Settings, result: dict[str, Any], *, report: dict | None = None) -> str:
    """Generate the internal "Professional Feedback" using PUAH's prompts (Hebrew text).

    Guarded like the client summary: it is free when the output is clean, and the feedback lands
    in the Excel sheet that feeds the RAG corpus.
    """
    transcript = build_transcript_text(result)
    if not transcript.strip():
        raise LLMError("The transcript is empty — nothing to analyze.")
    return _chat_checked(
        settings, _messages("feedback_task", transcript), validator=lambda t: _looks_valid(t, 120),
        guard_tags=True, template=_p("feedback_task"), report=report,
    ).strip()
