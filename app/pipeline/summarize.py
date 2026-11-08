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

import contextvars
import json
import logging
import re
from typing import Any

from ..config import Settings
from . import booklet_rag, llm_providers, output_guard, reidentify
from .pii_context import contextual_redact
from .llm_providers import LLMError  # Kept for compatibility: app.main / app.worker / tests import from here

log = logging.getLogger(__name__)

_last_reid: contextvars.ContextVar[dict[str, dict] | None] = contextvars.ContextVar("puah_last_reid", default=None)


def _chat(settings: Settings, messages: list[dict], temperature: float = 0.2) -> str:
    """The only entry point to the language model (Gemini, via the factory in llm_providers).

    Kept as an independent function (rather than calling the provider at every call
    site) so existing tests can monkey-patch sm._chat.
    """
    return llm_providers.get_provider(settings).chat(messages, temperature)


def check_available(settings: Settings) -> bool:
    """Quick check whether summaries can be produced at all.

    Requires an installed PII environment (mandatory before any LLM call) plus either a
    Gemini API key (when llm_provider="gemini") or a reachable local Ollama server (when
    llm_provider="ollama").
    """
    from .pii import pii_available

    if getattr(settings, "llm_provider", "gemini") == "ollama":
        return llm_providers.ollama_reachable(settings) and pii_available(settings)
    return bool(settings.gemini_api_key) and pii_available(settings)


# ---------- Building the transcript text ----------

def build_transcript_text(result: dict[str, Any], vault: "reidentify.Vault | None" = None) -> str:
    """The transcript as the LLM sees it — only ever used to build a Gemini/Ollama prompt.

    The conversation-level PII pass runs here, before the provider's own engine redaction
    (llm_providers.BaseLLMProvider): the engines receive the joined text, but a one-word answer
    on its own line or a number dictated across lines still defeats them. Raw names, phone
    digits and spelled-out e-mails are removed before anything is handed to the provider.

    Speaker labels: a display name the user typed (a real name, say) used to reach the model
    unfiltered as the label of every line. With `vault=None`, only default labels ("דובר 1") and
    role words ("יועצת", "פונה") pass and anything else becomes "דובר N". With `vault` given,
    personal speaker names are stored in the in-memory `Vault` and replaced by numbered tags
    ("<שם_1>", "הרב <שם_1>", "יועצת <שם_1>") so they are restored locally in the deliverables.
    """
    raw_speakers = result.get("speakers", {}) or {}
    speakers = reidentify.safe_speaker_labels(raw_speakers, vault=vault)
    segs = result.get("segments", [])
    raw = [seg.get("text", "") for seg in segs]
    texts, _, _ = contextual_redact(raw, [seg.get("speaker", "") for seg in segs])
    texts = reidentify.redact_speaker_names_in_texts(texts, raw_speakers)
    if vault is not None:
        texts = [vault.number(r, t) for r, t in zip(raw, texts)]
    lines = []
    for seg, text in zip(segs, texts):
        name = speakers.get(seg["speaker"], "דובר")
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
    vault: "reidentify.Vault | None" = None,
    reid_out: dict | None = None,
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

    vault: re-identification. Each answer's numbered name/organisation tags are restored from the
    vault *before* the guard looks at it, so only what cannot be restored counts as a placeholder.
    The repair turn always carries the model's own tagged answer — never the restored text, which
    would send the values to the model. reid_out, when given, receives {"pseudo": the chosen
    answer as the model wrote it (tags, no values), "info": restore counts, "spans": positions of
    the restored values in the returned text}.
    """
    check = validator or (lambda t: _looks_valid(t))
    repair = guard_tags and settings.llm_tag_guard_enabled
    last = ""
    seen: list[str] = []  # labels found in any valid answer
    # the valid answer with the fewest placeholders: (as written, restored, labels, restore info)
    best: tuple[str, str, list[str], dict] | None = None
    convo = list(messages)
    n = used = max(1, attempts)

    def finish(pseudo: str, final: str, info: dict) -> str:
        if vault is not None and reid_out is not None:
            # positions in the stripped text: that is what the generate_* callers return
            reid_out.update(
                pseudo=vault.pseudonymize_speakers(pseudo.strip()),
                info=info,
                spans=vault.spans(final.strip()),
            )
        return final

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
        restored, info = vault.restore(out) if vault is not None else (out, {})
        labels = output_guard.find_tags(restored) if guard_tags else []
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
            return finish(out, restored, info)
        seen += labels
        if best is None or len(labels) < len(best[2]):
            best = (out, restored, labels, info)
        # `out`, never `restored`: the repair turn goes back to the model
        convo = list(messages) + [{"role": "assistant", "content": out},
                                  output_guard.repair_message(labels, reid=vault is not None)]
    if best is not None:
        clean, _ = output_guard.sanitize_caller_text(best[1].strip(), template=template)
        residual = output_guard.find_tags(clean)
        record = output_guard.check_record(tags_found=seen, sanitized=True, attempts=used, residual=residual)
        if report is not None:
            report.update(record)
        log.warning("Model output kept redaction placeholders %s after %d attempts; sanitised%s",
                    record["tags_found"], used, " (RESIDUAL %s)" % record["residual"] if residual else "")
        return finish(best[0], clean, best[3])
    raise LLMError(
        f"The model returned invalid output in {attempts} attempts (probably degenerate). "
        f"Last output: {len(last or '')} characters"  # never quoted: model text can hold a name
    )


# ---------- Prompts (editable via the prompts_store interface) ----------

def _p(key: str) -> str:
    from .. import prompts_store  # Lazy import to avoid circular imports

    return prompts_store.get(key)


def _messages(task_key: str, transcript: str, *, reid: bool = False, booklet_context: str = "") -> list[dict]:
    """System + user messages for one deliverable.

    The fixed placeholder rule and the closing reminder are appended here, in code, so they hold
    whatever the editable prompts say (output_guard.PLACEHOLDER_RULE / REMINDER, or their
    re-identification versions, which let the model keep numbered name/organisation tags).
    When `booklet_context` is non-empty, relevant definitions from 'תשעה ירחי מידע' are injected
    between the task instructions and `=== תמלול השיחה ===`.
    """
    rule = output_guard.PLACEHOLDER_RULE_REID if reid else output_guard.PLACEHOLDER_RULE
    reminder = output_guard.REMINDER_REID if reid else output_guard.REMINDER
    ctx_block = ("\n\n" + booklet_context.strip()) if booklet_context and booklet_context.strip() else ""
    return [
        {"role": "system", "content": _p("puah_system") + "\n\n" + rule},
        {"role": "user", "content": _p(task_key) + ctx_block + "\n\n=== תמלול השיחה ===\n" + transcript
                                     + "\n=== סוף התמלול ===\n" + reminder},
    ]


def _reid_vault(settings: Settings, deliverable: str) -> "reidentify.Vault | None":
    """A fresh vault for one deliverable, or None when re-identification is off for it."""
    if not settings.pii_reidentify_enabled:
        return None
    if deliverable == "client_summary" and not settings.pii_reidentify_client_summary:
        return None
    if deliverable == "feedback" and not settings.pii_reidentify_feedback:
        return None
    cats = settings.reidentify_category_list
    return reidentify.Vault(cats) if cats else None


def _generate(settings: Settings, result: dict[str, Any], *, task_key: str, deliverable: str,
              empty_msg: str, report: dict | None, reid: dict | None) -> str:
    store = dict(_last_reid.get() or {})
    store.pop(deliverable, None)
    _last_reid.set(store)

    vault = _reid_vault(settings, deliverable)
    booklet_ctx = booklet_rag.retrieve_booklet_context(settings, result)
    # The vault is current for the whole generation: the provider numbers the engines' tags into it
    with reidentify.use_vault(vault):
        transcript = build_transcript_text(result, vault)
        if not transcript.strip():
            raise LLMError(empty_msg)
        out: dict = {}
        text = _chat_checked(
            settings, _messages(task_key, transcript, reid=vault is not None, booklet_context=booklet_ctx),
            validator=lambda t: _looks_valid(t, 120),
            guard_tags=True, template=_p(task_key), report=report, vault=vault, reid_out=out,
        ).strip()
    entry: dict = {}
    if vault is not None:
        # labels, counts and positions only — plus the answer as the model wrote it (tags, no
        # values), which the de-identified copy and the Excel export use instead of the text.
        entry = {
            "pseudo": out.get("pseudo"),
            "record": reidentify.reid_record(out.get("info") or {}, out.get("spans")),
        }
    store = dict(_last_reid.get() or {})
    if entry:
        store[deliverable] = entry
    else:
        store.pop(deliverable, None)
    _last_reid.set(store)
    if reid is not None:
        reid.clear()
        reid.update(entry)
    return text


def store_deliverable(
    result: dict[str, Any], key: str, text: str, check: dict | None, reid: dict | None = None
) -> None:
    """Put a generated deliverable into `result` with its records.

    key: "client_summary" / "feedback". Next to the text: `{key}_check` (placeholder check) and,
    when it was re-identified, `{key}_pseudo` (the answer with tags, no values — what the
    de-identified copy and the Excel export use) and `{key}_reid` (counts and positions only).
    Stale records of a previous generation are removed. When `reid` is omitted, consumes the last
    re-identification record produced by `_generate` for `key` in this context.
    """
    store = dict(_last_reid.get() or {})
    popped = store.pop(key, None)
    _last_reid.set(store)
    effective_reid = popped if reid is None else reid

    result[key] = text
    if check:
        result[f"{key}_check"] = check
    else:
        result.pop(f"{key}_check", None)
    if effective_reid and effective_reid.get("record") is not None:
        result[f"{key}_pseudo"] = effective_reid.get("pseudo") or ""
        result[f"{key}_reid"] = effective_reid["record"]
    else:
        result.pop(f"{key}_pseudo", None)
        result.pop(f"{key}_reid", None)


def generate_client_summary(settings: Settings, result: dict[str, Any], *, report: dict | None = None,
                            reid: dict | None = None) -> str:
    """Generate the "Client Summary" using PUAH's prompts (continuous Hebrew text).

    It goes to the caller, so it is guarded against redaction placeholders (see _chat_checked);
    report, when given, receives the check record. With re-identification on, the names and
    organisations the model referred to by their numbered tags are put back (locally); reid, when
    given, receives {"pseudo": the tagged answer, "record": reidentify.reid_record}.
    """
    return _generate(settings, result, task_key="client_summary_task", deliverable="client_summary",
                     empty_msg="The transcript is empty — nothing to summarize.", report=report, reid=reid)


def generate_feedback(settings: Settings, result: dict[str, Any], *, report: dict | None = None,
                      reid: dict | None = None) -> str:
    """Generate the internal "Professional Feedback" using PUAH's prompts (Hebrew text).

    Guarded like the client summary: it is free when the output is clean, and the feedback lands
    in the Excel sheet that feeds the RAG corpus (the sheet always takes the de-identified version).
    """
    return _generate(settings, result, task_key="feedback_task", deliverable="feedback",
                     empty_msg="The transcript is empty — nothing to analyze.", report=report, reid=reid)
