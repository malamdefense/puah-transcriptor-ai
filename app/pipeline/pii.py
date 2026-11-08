"""Redact PII (Personally Identifiable Information) from the transcript — two chained engines.

1. HebSafeHarbor (app/pii_runner.py, .venv-pii) — Hebrew NER (names, places, organisations),
   Israeli ID, Hebrew dates, cities/countries, medical lexicons. Requires pydantic<2 / spaCy 3.2 /
   presidio 2.2.31, which conflict with the application, so it lives in its own venv.
2. Presidio, current release (app/presidio_runner.py, .venv-presidio) — runs on HebSafeHarbor's
   output and adds checksum/pattern recognizers: Israeli ID, Israeli phone formats, IBAN, credit
   cards, e-mails, URLs, IPs, crypto wallets, numeric dates, bank accounts, license plates,
   passports. Needs numpy<2.5 / spaCy>=3.4 — incompatible with both other venvs, hence a third one.

Both run as persistent `--serve` subprocesses speaking JSON-lines over stdin/stdout so the heavy
models (482 MB RoBERTa transformer + Presidio analyzers) load ONCE per worker lifetime instead of
4–8 times per call. In-memory section caching further ensures that the call transcript and static
prompts are redacted only once across client summary, consultant feedback, repair turns, and
before/after `.pii.json` exports.

Fail-closed: if an enabled engine is missing or fails, PIIError is raised — nothing half-redacted
is returned (the LLM provider relies on this before sending content to the cloud).

deidentify_result returns a new copy of the transcript where the segment texts and summary have
been redacted.
"""

from __future__ import annotations

import atexit
import collections
import copy
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from . import reidentify
from .pii_context import contextual_redact, redact_freeform, restore_safe_terms, unnest_tags

log = logging.getLogger(__name__)

# Root of the repo: app/pipeline/pii.py -> pipeline -> app -> root
ROOT = Path(__file__).resolve().parent.parent.parent
RUNNER = ROOT / "app" / "pii_runner.py"
PRESIDIO_RUNNER = ROOT / "app" / "presidio_runner.py"

_ORIG_SUBPROCESS_RUN = subprocess.run


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
    """Per-engine diagnostics: is it on disk, does it import, and if not — why."""
    status: dict[str, dict[str, Any]] = {}
    for label, py, runner in _engine_specs(settings):
        installed = py.exists() and runner.exists()
        usable, reason = _engine_usable(label, py, runner, refresh=refresh)
        status[label] = {"installed": installed, "loads": usable, "error": reason or None}
    return status


def pii_engines(settings: Settings) -> dict[str, bool]:
    """Whether each engine is *usable* (installed and importable) — for diagnostics / dashboards."""
    return {k: bool(v["loads"]) for k, v in pii_engine_status(settings).items()}


def pii_available(settings: Settings, *, refresh: bool = False) -> bool:
    """True when every *enabled* engine is installed **and actually imports**."""
    ok, _ = _engine_usable("hebsafeharbor", pii_python(settings), RUNNER, refresh=refresh)
    if not ok:
        return False
    if settings.pii_presidio_enabled:
        ok, _ = _engine_usable("presidio", presidio_python(settings), PRESIDIO_RUNNER, refresh=refresh)
        if not ok:
            return False
    return True


# ---------- Persistent `--serve` daemon management ----------

class _EngineDaemon:
    """Persistent `--serve` subprocess for one PII engine (HebSafeHarbor or Presidio).

    Loads the heavy NLP models once on first use (or warmup) and processes subsequent batches
    over stdin/stdout JSON-lines in ~0.2–0.5s instead of 15–30s of cold-start import/model load.
    """

    def __init__(self, py: Path, runner: Path, label: str, *, idle_timeout_sec: float | None = None):
        self.py = py
        self.runner = runner
        self.label = label
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._out_q: queue.Queue[bytes | None] = queue.Queue()
        self._err_lines: collections.deque[str] = collections.deque(maxlen=50)
        self._idle_timer: threading.Timer | None = None
        self._idle_gen: int = 0
        self._idle_override_sec: float | None = idle_timeout_sec
        self.start_count: int = 0

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _cancel_idle_locked(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()
        self._idle_gen += 1

    def _arm_idle_locked(self, timeout_sec: float) -> None:
        self._cancel_idle_locked()
        if timeout_sec <= 0 or not self.is_alive():
            return
        gen = self._idle_gen
        timer = threading.Timer(timeout_sec, self._close_if_idle, args=(gen,))
        timer.daemon = True
        timer.start()
        self._idle_timer = timer

    def _close_if_idle(self, gen: int) -> None:
        with self._lock:
            if self._idle_gen != gen or not self.is_alive():
                return
            self._idle_timer = None
            log.info("%s persistent daemon idle timeout reached — shutting down to free memory", self.label)
            self._shutdown_locked()

    def _kill_locked(self) -> None:
        self._cancel_idle_locked()
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass

    def _shutdown_locked(self) -> None:
        self._cancel_idle_locked()
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None and proc.stdin is not None:
            try:
                proc.stdin.write(b'{"cmd": "shutdown"}\n')
                proc.stdin.flush()
                proc.stdin.close()
                proc.wait(timeout=1.5)
                self._proc = None
                return
            except Exception:  # noqa: BLE001
                pass
        self._kill_locked()

    def close(self) -> None:
        with self._lock:
            self._shutdown_locked()

    def _stderr_tail(self) -> str:
        return "\n".join(self._err_lines).strip()[-600:]

    def _start_locked(self, timeout: float) -> None:
        self._kill_locked()
        self._out_q = queue.Queue()
        self._err_lines.clear()
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                [str(self.py), str(self.runner), "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise PIIError(f"{self.label} could not be started ({self.py}): {exc}") from exc

        self._proc = proc
        self.start_count += 1
        out_q = self._out_q
        err_lines = self._err_lines

        def _read_stdout(pipe) -> None:
            try:
                for line in iter(pipe.readline, b""):
                    out_q.put(line)
            except Exception:  # noqa: BLE001
                pass
            finally:
                out_q.put(None)

        def _read_stderr(pipe) -> None:
            try:
                for line in iter(pipe.readline, b""):
                    err_lines.append(line.decode("utf-8", errors="replace").rstrip("\r\n"))
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_read_stdout, args=(proc.stdout,), daemon=True).start()
        threading.Thread(target=_read_stderr, args=(proc.stderr,), daemon=True).start()

        try:
            raw = out_q.get(timeout=timeout)
        except queue.Empty as exc:
            self._kill_locked()
            raise PIIError(f"{self.label} timed out during startup ({timeout}s)") from exc

        if raw is None:
            rc = proc.poll()
            tail = self._stderr_tail()
            self._kill_locked()
            raise PIIError(f"{self.label} failed (code {rc}): {tail}")

        try:
            handshake = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._kill_locked()
            raise PIIError(f"Unexpected startup output from {self.label}: {exc}") from exc

        if not isinstance(handshake, dict) or not handshake.get("ready"):
            self._kill_locked()
            raise PIIError(f"{self.label} did not report ready: {handshake}")

        log.info("%s persistent daemon started in %.2fs (pid=%s)", self.label, time.monotonic() - t0, proc.pid)

    def _exchange_locked(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None
        line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

        try:
            raw = self._out_q.get(timeout=timeout)
        except queue.Empty as exc:
            self._kill_locked()
            raise PIIError(f"{self.label} timed out ({timeout}s)") from exc

        if raw is None:
            rc = self._proc.poll() if self._proc else None
            tail = self._stderr_tail()
            self._kill_locked()
            raise PIIError(f"{self.label} exited unexpectedly (code {rc}): {tail}")

        try:
            result = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._kill_locked()
            raise PIIError(f"Unexpected output from {self.label}: {exc}") from exc

        if not isinstance(result, dict):
            self._kill_locked()
            raise PIIError(f"Unexpected output from {self.label}: not a JSON object")

        if "error" in result:
            raise PIIError(f"{self.label} failed: {result['error']}")

        out = result.get("texts")
        if not isinstance(out, list) or len(out) != len(payload["texts"]):
            raise PIIError(f"Number of segments returned from {self.label} does not match the input")
        return result

    def refresh_idle_timer(self, timeout_sec: float) -> None:
        with self._lock:
            if not self.is_alive():
                self._cancel_idle_locked()
            elif timeout_sec > 0:
                self._arm_idle_locked(timeout_sec)
            else:
                self._cancel_idle_locked()

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        with self._lock:
            self._cancel_idle_locked()
            try:
                was_warm = self.is_alive()
                if not was_warm:
                    self._start_locked(timeout)
                try:
                    return self._exchange_locked(payload, timeout)
                except (BrokenPipeError, OSError, PIIError) as exc:
                    # If a pre-existing idle daemon died between calls, restart once and retry;
                    # never retry a timeout or a failure on a freshly started daemon.
                    if not was_warm or "timed out" in str(exc):
                        raise
                    log.warning("%s daemon connection lost (%s); restarting once", self.label, exc)
                    self._start_locked(timeout)
                    return self._exchange_locked(payload, timeout)
            finally:
                idle_sec = self._idle_override_sec if self._idle_override_sec is not None else get_pii_daemon_idle_timeout()
                if self.is_alive() and idle_sec > 0:
                    self._arm_idle_locked(idle_sec)


_daemons_lock = threading.Lock()
_daemons: dict[tuple[str, str], _EngineDaemon] = {}
_daemon_idle_timeout_sec: float = 0.0


def get_pii_daemon_idle_timeout() -> float:
    return _daemon_idle_timeout_sec


def set_pii_daemon_idle_timeout(seconds: float) -> None:
    """Configure auto-shutdown after inactivity (0 = stay warm for the lifetime of the process)."""
    global _daemon_idle_timeout_sec
    val = max(0.0, float(seconds))
    _daemon_idle_timeout_sec = val
    with _daemons_lock:
        daemons = list(_daemons.values())
    for d in daemons:
        if d._idle_override_sec is None:
            d.refresh_idle_timer(val)


def _get_daemon(py: Path, runner: Path, label: str) -> _EngineDaemon:
    key = (str(py), str(runner))
    with _daemons_lock:
        daemon = _daemons.get(key)
        if daemon is None:
            daemon = _EngineDaemon(py, runner, label)
            _daemons[key] = daemon
        return daemon


def shutdown_pii_daemons() -> None:
    """Terminate all persistent PII engine subprocesses."""
    with _daemons_lock:
        daemons = list(_daemons.values())
        _daemons.clear()
    for d in daemons:
        d.close()


atexit.register(shutdown_pii_daemons)


def warmup_pii_daemons(settings: Settings) -> None:
    """Pre-load the enabled PII engine daemons at worker startup so the first call has zero cold-start."""
    py = pii_python(settings)
    if py.exists() and RUNNER.exists():
        try:
            _get_daemon(py, RUNNER, "PII (HebSafeHarbor)").request({"texts": []}, settings.pii_timeout_sec)
        except Exception as exc:  # noqa: BLE001
            log.warning("HebSafeHarbor daemon warmup skipped: %s", exc)
    if settings.pii_presidio_enabled:
        ppy = presidio_python(settings)
        if ppy.exists() and PRESIDIO_RUNNER.exists():
            try:
                _get_daemon(ppy, PRESIDIO_RUNNER, "PII (Presidio)").request(
                    {"texts": [], "score_threshold": settings.pii_presidio_score_threshold},
                    settings.pii_timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Presidio daemon warmup skipped: %s", exc)


def _run_stage_oneshot(py: Path, runner: Path, payload: dict[str, Any], timeout: float, label: str) -> dict[str, Any]:
    """Fallback one-shot subprocess execution (used only if subprocess.run is mocked in tests)."""
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


def _run_stage(py: Path, runner: Path, payload: dict[str, Any], timeout: float, label: str) -> dict[str, Any]:
    """Run one engine stage via the persistent `--serve` daemon (or oneshot if subprocess.run is mocked)."""
    if subprocess.run is not _ORIG_SUBPROCESS_RUN:
        return _run_stage_oneshot(py, runner, payload, timeout, label)
    return _get_daemon(py, runner, label).request(payload, timeout)


_DEFAULT_RUN_STAGE = _run_stage


# ---------- Single-pass section cache & static booklet shielding ----------
# During one call, the worker used to redact the same transcript segments in deidentify_result
# (before summaries), again inside generate_client_summary (+ up to 2 repair turns), again inside
# generate_feedback (+ up to 2 repair turns), and again in deidentify_result (after summaries).
# Splitting composite LLM prompts on the fixed transcript/booklet delimiters and caching redacted
# sections in memory ensures the transcript is redacted by the NLP engines only ONCE per call, and
# static training-booklet excerpts are never mutated or numbered into the caller's Vault.

_BOOKLET_START = "=== מידע מקצועי רלוונטי מחוברת ההדרכה (תשעה ירחי מידע — מכון פוע\"ה) ==="
_BOOKLET_END = "=== סוף מידע מחוברת ההדרכה ==="
_SPLIT_DELIMS_RE = re.compile(
    r"(\n=== תמלול השיחה ===\n|\n=== סוף התמלול ===\n"
    r"|\n?" + re.escape(_BOOKLET_START) + r"\n[\s\S]*?\n" + re.escape(_BOOKLET_END) + r"\n?)"
)

_cache_lock = threading.Lock()
_redact_cache: collections.OrderedDict[tuple[bool, float, str], str] = collections.OrderedDict()
_MAX_CACHE_ENTRIES = 4096


def clear_redact_cache() -> None:
    with _cache_lock:
        _redact_cache.clear()


def _run_pii_raw(settings: Settings, py: Path, ppy: Path | None, texts: list[str]) -> list[str]:
    """Execute the HebSafeHarbor -> Presidio -> restore_safe_terms chain on `texts`."""
    if not texts:
        return []
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
            log.info(
                "Presidio redacted additional entities: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            )

    out = [unnest_tags(t) if isinstance(t, str) else t for t in out]
    try:
        out, restored = restore_safe_terms(texts, out)
        if restored:
            log.info("Restored %d non-identifying words the engines had tagged", restored)
    except Exception as exc:  # noqa: BLE001
        log.warning("Restoring non-identifying words failed (engine output kept): %s", exc)
    return out


def _is_passthrough_delim(part: str) -> bool:
    """True for fixed prompt delimiters and static training-booklet blocks (never contain caller PII)."""
    if part in ("\n=== תמלול השיחה ===\n", "\n=== סוף התמלול ===\n"):
        return True
    stripped = part.strip()
    return stripped.startswith(_BOOKLET_START) and stripped.endswith(_BOOKLET_END)


def _run_pii(settings: Settings, texts: list[str]) -> list[str]:
    """Run the full chain: HebSafeHarbor, then (if enabled) Presidio on its output.

    Every enabled engine is checked up-front so a fail-closed run never proceeds if an engine
    environment is missing. When the real `_run_stage` is active, composite prompts are split on
    transcript/booklet delimiters and deduplicated through the in-memory section cache so each
    transcript is redacted only once across summary, feedback, repair turns, and `.pii.json` export.
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

    # If a unit test monkeypatched _run_stage, bypass section splitting and caching so the test
    # observes the exact 1:1 stage arguments it expects.
    if _run_stage is not _DEFAULT_RUN_STAGE:
        return _run_pii_raw(settings, py, ppy, texts)

    pres_flag = bool(settings.pii_presidio_enabled)
    thresh = float(settings.pii_presidio_score_threshold)

    # Split each text into (part, is_passthrough) pieces and collect unique uncached pieces
    structured: list[list[tuple[str, bool]]] = []
    missing_list: list[str] = []
    missing_seen: set[str] = set()

    with _cache_lock:
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                structured.append([(t if isinstance(t, str) else "", True)])
                continue
            parts = _SPLIT_DELIMS_RE.split(t)
            pieces: list[tuple[str, bool]] = []
            for p in parts:
                if not p:
                    continue
                if _is_passthrough_delim(p) or not p.strip():
                    pieces.append((p, True))
                else:
                    pieces.append((p, False))
                    ckey = (pres_flag, thresh, p)
                    if ckey in _redact_cache:
                        _redact_cache.move_to_end(ckey)
                    elif p not in missing_seen:
                        missing_seen.add(p)
                        missing_list.append(p)
            structured.append(pieces)

    if missing_list:
        redacted_missing = _run_pii_raw(settings, py, ppy, missing_list)
        with _cache_lock:
            for orig_p, red_p in zip(missing_list, redacted_missing):
                ckey = (pres_flag, thresh, orig_p)
                _redact_cache[ckey] = red_p
                _redact_cache.move_to_end(ckey)
                while len(_redact_cache) > _MAX_CACHE_ENTRIES:
                    _redact_cache.popitem(last=False)

    out: list[str] = []
    with _cache_lock:
        for pieces in structured:
            rebuilt = []
            for p, passthrough in pieces:
                if passthrough:
                    rebuilt.append(p)
                else:
                    rebuilt.append(_redact_cache[(pres_flag, thresh, p)])
            out.append("".join(rebuilt))
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

    Public wrapper around _run_pii — the entry point of BaseLLMProvider.
    """
    return _run_pii(settings, texts)


def deidentify_result(settings: Settings, result: dict[str, Any]) -> dict[str, Any]:
    """Return a new copy of the transcript where the segment texts and summary have been redacted.

    Performs a single batch call per engine for all uncached texts together, then a
    conversation-level pass (pii_context) for what the engines cannot see one segment at a time:
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

    # When a deliverable was re-identified, its `*_pseudo` field holds the model's tagged text
    # (before real names/organisations were restored locally). Using that here guarantees the
    # de-identified copy never carries a restored name even if the NER engines miss it in prose.
    raw_client = result.get("client_summary_pseudo") or result.get("client_summary")
    raw_feedback = result.get("feedback_pseudo") or result.get("feedback")
    client_text = reidentify.to_generic(as_text(raw_client))
    feedback_text = reidentify.to_generic(as_text(raw_feedback))
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
            return reidentify.to_generic(redact_freeform(as_text(text), terms))

        new = copy.deepcopy(result)
        if "speakers" in new:
            new["speakers"] = reidentify.safe_speaker_labels(new.get("speakers") or {})
        for k in ("client_summary_pseudo", "feedback_pseudo", "client_summary_reid", "feedback_reid"):
            new.pop(k, None)
        for seg, text in zip(new.get("segments") or [], ctx_texts):
            seg["text"] = reidentify.to_generic(text)
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
