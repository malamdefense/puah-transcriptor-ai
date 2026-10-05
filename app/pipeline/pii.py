"""Redact PII (Personally Identifiable Information) from the transcript — two chained engines.

1. HebSafeHarbor (app/pii_runner.py, .venv-pii) — Hebrew NER (names, places, organisations),
   Israeli ID, Hebrew dates, cities/countries, medical lexicons. Requires pydantic<2 / spaCy 3.2 /
   presidio 2.2.31, which conflict with the application, so it lives in its own venv.
2. Presidio, current release (app/presidio_runner.py, .venv-presidio) — runs on HebSafeHarbor's
   output and adds checksum/pattern recognizers: Israeli ID, Israeli phone formats, IBAN, credit
   cards, e-mails, URLs, IPs, crypto wallets, numeric dates, bank accounts, license plates,
   passports. Needs numpy<2.5 / spaCy>=3.4 — incompatible with both other venvs, hence a third one.

Both are called as subprocesses with the same JSON protocol ({"texts": [...]} in / out). Each
pass is one subprocess call for all texts together (the engine is loaded once per pass).

Fail-closed: if an enabled engine is missing, PIIError is raised — nothing half-redacted is
returned (the Gemini provider relies on this before sending content to the cloud).

deidentify_result returns a new copy of the transcript where the segment texts and summary have
been redacted.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from .pii_context import contextual_redact, redact_freeform, restore_safe_terms, unnest_tags

log = logging.getLogger(__name__)

# Root of the repo: app/pipeline/pii.py -> pipeline -> app -> root
ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER = ROOT / "app" / "pii_runner.py"
PRESIDIO_RUNNER = ROOT / "app" / "presidio_runner.py"


class PIIError(RuntimeError):
    pass


def _venv_python(venv: str) -> Path:
    return ROOT / venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pii_python(settings: Settings) -> Path:
    """Interpreter of the HebSafeHarbor environment (.venv-pii)."""
    return _venv_python(settings.pii_venv)


def presidio_python(settings: Settings) -> Path:
    """Interpreter of the Presidio environment (.venv-presidio)."""
    return _venv_python(settings.pii_presidio_venv)


# A probe only *imports* the engine — no model is loaded — but that still pulls spaCy and
# Presidio, so it is not instant. The result is cached: availability is asked for on every
# status call and before every job, and the environments do not change under a running process.
_PROBE_TTL_SEC = 300.0
_PROBE_TIMEOUT_SEC = 120.0

_probe_cache: dict[tuple[str, str], tuple[float, tuple[bool, str]]] = {}


def _probe_engine(py: Path, runner: Path, label: str, *, refresh: bool = False) -> tuple[bool, str]:
    """Run `<venv python> <runner> --probe`: it imports its engine, loads no model, exits 0 or 3.

    Returns (loads, reason). An engine whose files are on disk but whose native dependencies are
    refused — a blocked DLL, a broken venv, an ABI mismatch — fails here, which is the whole
    point: presence on disk never proved the engine could run.
    """
    key = (str(py), str(runner))
    now = time.monotonic()
    if not refresh:
        cached = _probe_cache.get(key)
        if cached and now - cached[0] < _PROBE_TTL_SEC:
            return cached[1]

    try:
        proc = subprocess.run(
            [str(py), str(runner), "--probe"],
            input="", capture_output=True, text=True, encoding="utf-8",
            timeout=_PROBE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        result = (False, f"{label} did not answer a probe within {_PROBE_TIMEOUT_SEC:g}s")
    except OSError as exc:
        result = (False, f"{label} could not be started ({py}): {exc}")
    else:
        if proc.returncode == 0:
            result = (True, "")
        else:
            # Both runners print one summary line before giving up, and presidio_runner precedes
            # it with a stub note per blocked spaCy extension — a raw tail would be all noise.
            lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
            why = lines[-1].strip()[:400] if lines else f"no output (code {proc.returncode})"
            result = (False, f"{label} is installed but does not load: {why}")

    _probe_cache[key] = (now, result)
    return result


def _engine_usable(
    label: str, py: Path, runner: Path, *, refresh: bool = False
) -> tuple[bool, str]:
    """Installed *and* importable. The file check comes first — it is free, and skipping the
    probe for an engine that is plainly absent keeps the error specific."""
    if not py.exists():
        return False, f"{label}: environment not found ({py})"
    if not runner.exists():
        return False, f"{label}: runner script not found ({runner})"
    return _probe_engine(py, runner, label, refresh=refresh)


def _engine_specs(settings: Settings) -> list[tuple[str, Path, Path]]:
    return [
        ("hebsafeharbor", pii_python(settings), RUNNER),
        ("presidio", presidio_python(settings), PRESIDIO_RUNNER),
    ]


def pii_engine_status(settings: Settings, *, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Per-engine diagnostics: is it on disk, does it import, and if not — why.

    Probes both engines regardless of which are enabled; it is the explicit "tell me everything"
    view. pii_available() is the one that short-circuits.
    """
    status: dict[str, dict[str, Any]] = {}
    for label, py, runner in _engine_specs(settings):
        installed = py.exists() and runner.exists()
        usable, reason = _engine_usable(label, py, runner, refresh=refresh)
        status[label] = {"installed": installed, "loads": usable, "error": reason or None}
    return status


def pii_engines(settings: Settings) -> dict[str, bool]:
    """Whether each engine is *usable* (installed and importable) — for diagnostics / dashboards.

    This used to report mere presence on disk, which reported a dead chain as healthy.
    """
    return {k: bool(v["loads"]) for k, v in pii_engine_status(settings).items()}


def pii_available(settings: Settings, *, refresh: bool = False) -> bool:
    """True when every *enabled* engine is installed **and actually imports**.

    Gates the Gemini path (summarize.check_available) and the status endpoint, so it must not
    answer True for an engine that cannot run: that is a false green on the one control that
    keeps identifying details from leaving the building. Checks HebSafeHarbor (always required)
    before Presidio, so a plainly missing first engine costs no probe.
    """
    ok, _ = _engine_usable("hebsafeharbor", pii_python(settings), RUNNER, refresh=refresh)
    if not ok:
        return False
    if settings.pii_presidio_enabled:
        ok, _ = _engine_usable("presidio", presidio_python(settings), PRESIDIO_RUNNER, refresh=refresh)
        if not ok:
            return False
    return True


def _run_stage(py: Path, runner: Path, payload: dict[str, Any], timeout: float, label: str) -> dict[str, Any]:
    """Run one engine as a subprocess; returns its parsed JSON output (with a 'texts' list)."""
    try:
        proc = subprocess.run(
            [str(py), str(runner)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise PIIError(f"{label} timed out ({timeout}s)") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-600:]
        raise PIIError(f"{label} failed (code {proc.returncode}): {tail}")

    try:
        result = json.loads(proc.stdout)
        out = result["texts"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PIIError(f"Unexpected output from {label}: {exc}") from exc

    if not isinstance(out, list) or len(out) != len(payload["texts"]):
        raise PIIError(f"Number of segments returned from {label} does not match the input")
    return result


def _run_pii(settings: Settings, texts: list[str]) -> list[str]:
    """Run the full chain: HebSafeHarbor, then (if enabled) Presidio on its output.

    Every enabled engine is checked up-front: HebSafeHarbor loads a transformer model and costs
    tens of seconds, so a fail-closed run must not pay for it only to discover afterwards that
    Presidio is missing and throw the redacted text away.
    """
    py = pii_python(settings)
    if not py.exists():
        raise PIIError(
            f"PII environment not found ({py}). Run: python rebuild.py (or create {settings.pii_venv} "
            "מ-requirements-pii.txt)."
        )

    ppy = None
    if settings.pii_presidio_enabled:
        ppy = presidio_python(settings)
        if not ppy.exists():
            raise PIIError(
                f"Presidio environment not found ({ppy}). Run: python rebuild.py (or create "
                f"{settings.pii_presidio_venv} from requirements-presidio.txt), or set "
                "PUAH_PII_PRESIDIO_ENABLED=false."
            )

    out = _run_stage(py, RUNNER, {"texts": texts}, settings.pii_timeout_sec, "PII (HebSafeHarbor)")["texts"]

    if ppy is not None:
        res = _run_stage(
            ppy,
            PRESIDIO_RUNNER,
            {"texts": out, "score_threshold": settings.pii_presidio_score_threshold},
            settings.pii_timeout_sec,
            "PII (Presidio)",
        )
        out = res["texts"]
        counts = res.get("entities") or {}
        if counts:
            log.info("Presidio redacted additional entities: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # Undo the engines' over-redaction on a closed list (drug/device names, function words,
    # health funds, countries) — otherwise Gemini summarises a contraception call without the
    # contraceptive. Fail closed: if this step breaks, keep the engines' output as it was.
    out = [unnest_tags(t) if isinstance(t, str) else t for t in out]
    try:
        out, restored = restore_safe_terms(texts, out)
        if restored:
            log.info("Restored %d non-identifying words the engines had tagged", restored)
    except Exception as exc:  # noqa: BLE001
        log.warning("Restoring non-identifying words failed (engine output kept): %s", exc)
    return out


def pii_title(job_id: str) -> str:
    """Title of a de-identified export. Never the upload's file name: that often carries a
    person's name and the call's timestamp ("<name>record-<epoch>.mp3")."""
    return f"תמלול ללא פרטים מזהים — {job_id[:8]}"


def pii_download_name(job_id: str, ext: str) -> str:
    """File name for downloading a de-identified export (same reason as pii_title)."""
    return f"transcript-{job_id[:8]}.pii.{ext}"


def redact_texts(settings: Settings, texts: list[str]) -> list[str]:
    """Redact PII from a list of raw strings (e.g. LLM messages before going to a third-party cloud provider).

    Public wrapper around _run_pii — the entry point of GeminiModelProvider.
    """
    return _run_pii(settings, texts)


def deidentify_result(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    """Return a new copy of the transcript where the segment texts and summary have been redacted.

    Performs a single subprocess call per engine for all the texts together (to load each model once only),
    then a conversation-level pass (pii_context) for what the engines cannot see one segment at a time:
    one-word answers, numbers dictated across turns, spelled-out e-mails, names that recur.
    The raw per-word timings (segments[].words) are dropped: they hold every original word, and nothing
    that reads the de-identified copy uses them.
    """
    def as_text(value: Any) -> str:
        return value if isinstance(value, str) else ("" if value is None else str(value))

    def as_list(value: Any) -> list[str]:
        return [as_text(v) for v in value] if isinstance(value, (list, tuple)) else []

    segments = result.get("segments") or []
    seg_texts = [as_text(s.get("text")) for s in segments]

    # A summary of an unexpected shape is dropped from the copy below, never passed through raw.
    summ = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    sum_summary = [as_text(summ.get("summary"))] if summ else []
    sum_keypts = as_list(summ.get("key_points"))
    sum_actions = as_list(summ.get("action_items"))
    sum_topics = as_list(summ.get("topics"))
    sum_keywords = as_list(summ.get("keywords"))

    client_text = as_text(result.get("client_summary"))
    feedback_text = as_text(result.get("feedback"))
    has_client = bool(client_text.strip())
    has_feedback = bool(feedback_text.strip())
    extra = []
    if has_client:
        extra.append(client_text)
    if has_feedback:
        extra.append(feedback_text)

    flat = seg_texts + sum_summary + sum_keypts + sum_actions + sum_topics + sum_keywords + extra
    if not any(t and str(t).strip() for t in flat):
        raise PIIError("The transcript is empty — nothing to clean.")

    anon = _run_pii(settings, flat)

    # Conversation-level pass over the engine output, and the same terms over every summary
    # field. Fail closed: anything that goes wrong here raises PIIError, so a half-applied pass
    # is never written out as "de-identified".
    try:
        ctx_texts, terms, counts = contextual_redact(
            anon[: len(segments)],
            [s.get("speaker", "") for s in segments],
            originals=seg_texts,
        )

        def scrub(text: Any) -> str:
            return redact_freeform(as_text(text), terms)

        new = copy.deepcopy(result)
        for seg, text in zip(new.get("segments") or [], ctx_texts):
            seg["text"] = text
            seg.pop("words", None)
        i = len(segments)
        if summ:
            ns = new["summary"]
            if sum_summary:
                ns["summary"] = scrub(anon[i])
                i += 1
            n = len(sum_keypts); ns["key_points"] = [scrub(t) for t in anon[i : i + n]]; i += n
            n = len(sum_actions); ns["action_items"] = [scrub(t) for t in anon[i : i + n]]; i += n
            n = len(sum_topics); ns["topics"] = [scrub(t) for t in anon[i : i + n]]; i += n
            n = len(sum_keywords); ns["keywords"] = [scrub(t) for t in anon[i : i + n]]; i += n
        elif "summary" in new:
            new["summary"] = {}
        if has_client:
            new["client_summary"] = scrub(anon[i]); i += 1
        if has_feedback:
            new["feedback"] = scrub(anon[i]); i += 1
    except PIIError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PIIError(f"Conversation-level PII pass failed: {exc}") from exc
    log.info("Conversation-level PII pass: %s", ", ".join(f"{k}={v}" for k, v in counts.items()))

    new["pii_deidentified"] = True
    new["pii_context"] = counts  # counts only — never the terms themselves
    new["pii_engines"] = ["hebsafeharbor"] + (["presidio"] if settings.pii_presidio_enabled else [])
    return new
