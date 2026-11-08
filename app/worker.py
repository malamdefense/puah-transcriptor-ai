"""Worker process: takes tasks from the queue, runs the pipeline and saves the results.

Runs as a separate process from the API so that GPU models remain in memory and the API remains available.
Runs one task at a time (the GPU is a serial resource) — prevents OOM.
"""

from __future__ import annotations

import json
import logging
import signal
import time
import traceback
from pathlib import Path

from . import secrets_store
from .config import Settings, get_settings
from .db import (
    claim_next_job,
    get_conn,
    index_job_segments,
    init_db,
    reset_stuck_jobs,
    save_result,
    update_job,
)
from .pipeline import asr_config, audio_quality
from .pipeline.asr_cleanup import CleanupConfig, clean_asr_segments
from .pipeline.asr_terms import normalize_asr_segments
from .pipeline.asr_words import join_split_segments
from .pipeline.audio import extract_audio, get_duration
from .pipeline.diarize import DiarizationResult, Diarizer
from .pipeline.export_docx import build_docx
from .pipeline.language import annotate_segments, detect_languages
from .pipeline.merge import merge
from .pipeline.pii import PIIError, deidentify_result, pii_title, shutdown_pii_daemons, warmup_pii_daemons
from .pipeline.keywords import build_summary as build_local_summary
from .pipeline.summarize import (
    generate_client_summary,
    generate_feedback,
    store_deliverable,
)
from .pipeline.transcribe import Transcriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s %(message)s",
)
log = logging.getLogger("puah.worker")

_running = True


def _stop(*_args):
    global _running
    _running = False
    log.info("Received stop signal — finishing the current task...")


# Distribution of overall progress between pipeline stages
_STAGE_WEIGHTS = {
    "extracting_audio": (0.0, 5.0),
    "diarizing": (5.0, 28.0),
    "transcribing": (28.0, 88.0),
    "merging": (88.0, 92.0),
    "summarizing": (92.0, 100.0),
}


def _stage_progress(stage: str, inner_pct: float) -> float:
    lo, hi = _STAGE_WEIGHTS[stage]
    return lo + (hi - lo) * (max(0.0, min(100.0, inner_pct)) / 100.0)


def _evict_ollama_llm_from_vram(settings: Settings) -> None:
    """Unload the local Ollama LLM from GPU VRAM before diarization/ASR so PyAnnote and
    CTranslate2 never contend with a 14+ GB LLM on a 16 GB GPU. Best-effort."""
    if getattr(settings, "llm_provider", "gemini") != "ollama":
        return
    model = (getattr(settings, "llm_model", None) or "").strip()
    if not model:
        return
    try:
        import urllib.request
        base = (getattr(settings, "llm_base_url", "") or "http://localhost:11434").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        req = urllib.request.Request(
            f"{base}/api/generate",
            data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:  # noqa: BLE001
        pass


def process_job(job: dict, transcriber: Transcriber, diarizer: Diarizer) -> None:
    settings = get_settings()
    # The worker is a separate process with its own settings cache (get_settings is
    # lru_cached), so a key entered from the UI (data/secrets.json) would not be picked
    # up here until a restart. Re-applying before each job costs one small JSON read —
    # negligible compared to the job duration.
    secrets_store.apply_overrides(settings, settings.data_dir)
    job_id = job["id"]
    uploaded = settings.uploads_dir / f"{job_id}.mp4"
    audio_path = settings.audio_dir / f"{job_id}.wav"

    last_update = [0.0]

    def report(stage: str, inner_pct: float, force: bool = False) -> None:
        # Limit DB update rate to ~once per second to avoid cluttering
        now = time.time()
        if not force and (now - last_update[0]) < 1.0:
            return
        last_update[0] = now
        update_job(job_id, stage=stage, progress=round(_stage_progress(stage, inner_pct), 1))

    # 1) Extract audio
    log.info("Task %s: Extracting audio", job_id)
    update_job(job_id, stage="extracting_audio", progress=0)
    extract_audio(uploaded, audio_path)
    duration = get_duration(audio_path) or get_duration(uploaded)
    update_job(job_id, duration_sec=duration)
    report("extracting_audio", 100.0, force=True)

    # 2) Speaker diarization — before the transcription, since the recording-quality check needs
    # each speaker's turns. Always on the original audio.
    _evict_ollama_llm_from_vram(settings)
    log.info("Task %s: Speaker diarization", job_id)
    update_job(job_id, stage="diarizing")
    hint = job.get("num_speakers") or settings.default_num_speakers
    diar = _diarize(diarizer, audio_path, hint)
    report("diarizing", 100.0, force=True)

    # 3) Recording-quality check (gated fixes go to a copy only the ASR hears), then transcription
    log.info("Task %s: Transcription (%.0f seconds audio)", job_id, duration)
    update_job(job_id, stage="transcribing")
    asr_in = _audio_check(settings, job_id, audio_path, diar)
    try:
        asr_segments = transcriber.transcribe(
            asr_in.path if asr_in else audio_path,
            duration_sec=duration,
            progress_cb=lambda pct: report("transcribing", pct),
            vad_threshold=asr_in.vad_threshold if asr_in else None,
        )
    finally:
        if asr_in is not None and asr_in.processed:
            asr_in.path.unlink(missing_ok=True)
    report("transcribing", 100.0, force=True)

    # 3b) Word pieces glued, repetition clean-up, then domain-term spelling, on the raw ASR words —
    # before the merge: the clean-up reads the alignment's per-word durations, and the merge must
    # see the final words.
    asr_raw = asr_segments  # the post-processing returns new lists and never mutates its input
    asr_segments, asr_post = _postprocess_asr(settings, job_id, asr_segments)

    # 4) Merge (no fix moves the time axis: the original diarization lines up with the words)
    log.info("Task %s: Merge", job_id)
    update_job(job_id, stage="merging")
    _save_merge_inputs(settings, job_id, asr_raw, diar, hint)
    merge_stats: dict = {}
    segments, speaker_ids = _merge(settings, job_id, asr_segments, diar, merge_stats)
    asr_post["merge"] = merge_stats  # counts only
    asr_post["windows"] = _window_counts(asr_raw)
    provenance = _asr_provenance(settings, job_id, transcriber)

    # Language from the transcript's script, per segment and for the call (decoding stays
    # settings.language: the model's own language-ID always answers "he")
    lang = _detect_language(settings, job_id, segments)

    # Display names for speakers: "Speaker 1", "Speaker 2", ...
    speakers = {sid: f"דובר {i}" for i, sid in enumerate(speaker_ids, start=1)}

    result = {
        "job_id": job_id,
        "language": lang["language"] if lang else job.get("language", settings.language),
        "duration_sec": duration,
        "num_speakers": len(speaker_ids),
        "speakers": speakers,
        "segments": segments,
        # The model that decoded the job, by repo id — never a local path
        "asr_model": (provenance.get("asr_config") or {}).get("model_repo_id") or _asr_model_name(settings),
        "diarization_model": settings.diarization_model,
        "summary": None,
        "client_summary": None,
        "feedback": None,
    }
    if lang:
        result["languages"] = lang["languages"]
        result["language_shares"] = lang["language_shares"]
        update_job(job_id, language=lang["language"])
    if asr_post:
        result["asr_postprocess"] = asr_post  # counts and times only, never text
    result.update(provenance)  # asr_config, confidence_display: ids, settings and hashes only
    if asr_in is not None:
        quality = _audio_report(settings, job_id, asr_in, asr_segments, diar)
        if quality is not None:
            result["audio_quality"] = quality  # numbers, codes and fixed messages only
    save_result(job_id, result)

    # Full index for full search
    index_job_segments(job_id, result)

    # 5) Word export + PII de-identification of the transcript (best-effort).
    # Runs before the summaries so that the redacted transcript exists first (the local
    # keywords are drawn from it); the summaries are produced immediately after.
    pii_copy = None
    if segments:
        _export_word(settings, job_id, job, result)
        log.info("Task %s: PII removal", job_id)
        pii_copy = _export_pii(settings, job_id, job, result)

    # 6) Summaries — generated automatically right after the transcript PII step.
    # Each product is best-effort on its own: a failure must not fail the job nor discard
    # the transcript that was already saved.
    if getattr(settings, "llm_provider", "gemini") == "ollama":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    produced = False
    if segments:
        update_job(job_id, stage="summarizing", progress=_stage_progress("summarizing", 10))

        # Topics + keywords are extracted locally (no AI call): instant, offline and
        # free. They populate the same `summary` object the dashboard aggregates over,
        # so /api/stats keeps working exactly as before. The narrative "automatic
        # summary" is no longer produced — the client summary and consultant feedback
        # below are the deliverables.
        try:
            # From the de-identified copy: keywords feed the dashboard across all recordings
            # and must not carry a name (build_summary falls back to the context pass).
            result["summary"] = build_local_summary(pii_copy or result)
            save_result(job_id, result)
            produced = True
            log.info(
                "Task %s: Topics/keywords extracted locally (%d topics, %d keywords)",
                job_id, len(result["summary"]["topics"]), len(result["summary"]["keywords"]),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: Local topic/keyword extraction failed (ignored) — %s", job_id, exc)

    # PUAH products: client summary + consultant feedback (each best-effort separately).
    # Each text is stored with its placeholder-check record (labels only, never text; summarize
    # logs the outcome when a placeholder was repaired or removed).
    if settings.client_summary_enabled and segments:
        try:
            check: dict = {}
            text = generate_client_summary(settings, result, report=check)
            store_deliverable(result, "client_summary", text, check)
            save_result(job_id, result)
            produced = True
            log.info("Task %s: Client summary generated", job_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: Skipped client summary — %s", job_id, exc)
    if settings.feedback_enabled and segments:
        try:
            check = {}
            text = generate_feedback(settings, result, report=check)
            store_deliverable(result, "feedback", text, check)
            save_result(job_id, result)
            produced = True
            log.info("Task %s: Feedback generated", job_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: Skipped feedback — %s", job_id, exc)
    _evict_ollama_llm_from_vram(settings)

    # 7) Refresh the de-identified copy so it also covers the generated summaries.
    # Skipped when nothing was produced, to avoid a redundant PII pass.
    if produced:
        _export_pii(settings, job_id, job, result)

    update_job(
        job_id,
        status="done",
        stage="summarizing",
        progress=100.0,
        num_speakers=len(speaker_ids),
        finished_at=time.time(),
        error=None,
    )
    log.info("Task %s: Completed (%d speakers, %d segments)", job_id, len(speaker_ids), len(segments))


def _postprocess_asr(settings: Settings, job_id: str, asr_segments: list[dict]) -> tuple[list[dict], dict]:
    """Word pieces glued (I1), repetition clean-up, then domain-term spelling — each best-effort,
    keeping the input on failure.

    Returns (segments, audit) where audit holds counts and times only. Failures log the exception
    type alone, so nothing from the call can reach the log.
    """
    audit: dict = {}
    if settings.asr_join_split_words:
        try:
            asr_segments, gstats = join_split_segments(asr_segments)
            audit["glue"] = gstats
            log.info("Task %s: ASR word pieces glued: %d of %d pieces joined their word",
                     job_id, gstats["glued"], gstats["pieces_in"])
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: ASR word gluing skipped (%s)", job_id, type(exc).__name__)
    if settings.asr_cleanup_enabled:
        try:
            asr_segments, cstats = clean_asr_segments(asr_segments, CleanupConfig.from_settings(settings))
            audit["cleanup"] = cstats
            log.info(
                "Task %s: ASR cleanup removed %d/%d words (%d echo/loop copies, %d capped, %d stock segments)",
                job_id, cstats["removed_words"], cstats["words_in"], cstats["removed_collapsed_copies"],
                cstats["removed_capped_copies"], cstats["stock_segments_dropped"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: ASR cleanup skipped (%s)", job_id, type(exc).__name__)
    if settings.asr_term_normalization_enabled:
        try:
            asr_segments, n_terms = normalize_asr_segments(asr_segments)
            audit["terms_normalized"] = n_terms
            log.info("Task %s: ASR term normalization fixed %d term(s)", job_id, n_terms)
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: ASR term normalization skipped (%s)", job_id, type(exc).__name__)
    return asr_segments, audit


def _audio_check(settings: Settings, job_id: str, audio_path: Path, diar: DiarizationResult):
    """Pre-ASR recording-quality check — best-effort: on any failure the original is transcribed at
    the configured VAD threshold and the job carries no audio_quality."""
    if not settings.audio_check_enabled:
        return None
    try:
        return audio_quality.prepare_asr_input(audio_path, diar, settings, settings.audio_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: Audio check skipped (%s) — transcribing the original", job_id, type(exc).__name__)
        return None


def _audio_report(settings: Settings, job_id: str, asr_in, asr_segments: list[dict], diar: DiarizationResult):
    """The complete recording-quality report (per-speaker word confidence and flags) — best-effort."""
    try:
        return audio_quality.finalize_report(asr_in.report, asr_segments, diar, settings)
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: Audio quality report skipped (%s)", job_id, type(exc).__name__)
        return None


def _diarize(diarizer: Diarizer, audio_path: Path, hint: int | None) -> DiarizationResult:
    """Both diarization views in one pipeline call; diarizers exposing only diarize() (test fakes)
    give the regular turns alone, so the merge assigns words on them."""
    diarize_full = getattr(diarizer, "diarize_full", None)
    if diarize_full is not None:
        return diarize_full(audio_path, num_speakers=hint)
    return DiarizationResult(turns=diarizer.diarize(audio_path, num_speakers=hint), exclusive_turns=None)


def _merge(settings: Settings, job_id: str, asr_segments: list[dict], diar: DiarizationResult,
           stats: dict | None = None):
    """Word-to-speaker merge. The exclusive-turn assignment and the smoothing rules are an
    improvement, not a requirement: if they fail, the job still gets the plain overlap assignment.

    `stats` (a dict, filled in place) receives the merge's counts (words, outside, exclusive,
    smooth, r0/r1/r2 relabels) plus rules_skipped; logged at INFO when merge_log_counts."""
    stats = {} if stats is None else stats
    try:
        out = merge(
            asr_segments,
            diar.turns,
            exclusive_turns=diar.exclusive_turns if settings.merge_use_exclusive else None,
            smooth=settings.merge_smoothing,
            stats=stats,
        )
        stats["rules_skipped"] = False
    except Exception as exc:  # noqa: BLE001
        if not (settings.merge_use_exclusive or settings.merge_smoothing):
            raise
        log.warning("Task %s: Merge rules skipped (%s) — plain assignment on the regular turns",
                    job_id, type(exc).__name__)
        stats.clear()  # the counts of the failed attempt do not describe the result
        out = merge(asr_segments, diar.turns, exclusive_turns=None, smooth=False, stats=stats)
        stats["rules_skipped"] = True
    if settings.merge_log_counts:
        # Counts only — never transcript text
        log.info("Task %s: Merge %d words, %d outside every turn, relabelled R0 %d, R1 %d, R2 %d "
                 "(exclusive=%s, smoothing=%s, rules skipped=%s)",
                 job_id, stats.get("words", 0), stats.get("outside", 0), stats.get("r0", 0),
                 stats.get("r1", 0), stats.get("r2", 0), stats.get("exclusive"), stats.get("smooth"),
                 stats["rules_skipped"])
    return out


def _window_counts(asr_raw: list[dict]) -> dict:
    """Decode windows of the transcriber output (distinct `seek`), how many were re-decoded by
    temperature fallback, and how many carry each window flag. Counts only."""
    windows: dict = {}
    for seg in asr_raw:
        seek = seg.get("seek")
        if seek is not None and seek not in windows:
            windows[seek] = seg
    first = list(windows.values())
    return {
        "n": len(first),
        "fallback": sum(1 for s in first if (s.get("temperature") or 0.0) > 0.0),
        "gate_failed": sum(1 for s in first if s.get("window_flag") == "gate_failed"),
        "sampled_hi": sum(1 for s in first if s.get("window_flag") == "sampled_hi"),
    }


def _asr_model_name(settings: Settings) -> str:
    """The configured model's repo id, or "local" for a model directory (never a path)."""
    return asr_config.model_identity(settings.asr_model).get("repo_id") or "local"


def _asr_provenance(settings: Settings, job_id: str, transcriber) -> dict:
    """{"asr_config": ..., "confidence_display": ...} for the result: ids, settings and hashes,
    never a path or text. Best-effort: provenance never fails a job."""
    try:
        identity_fn = getattr(transcriber, "loaded_identity", None)
        if identity_fn is not None:
            ident = identity_fn()
        else:  # a transcriber without the method (tests): what was requested
            ident = {"device": getattr(transcriber, "device", None),
                     "compute_type": getattr(transcriber, "compute_type", None),
                     **asr_config.model_identity(settings.asr_model)}
        decode = getattr(transcriber, "last_decode", None) or {}
        out = {"asr_config": asr_config.build_asr_config(
            settings,
            model_repo_id=ident.get("repo_id"),
            model_commit=ident.get("commit"),
            device=ident.get("device"),
            compute_type=ident.get("compute_type"),
            condition_on_previous_text=decode.get("condition_on_previous_text"),
        )}
        if settings.confidence_display_enabled:
            out["confidence_display"] = asr_config.build_confidence_display(
                settings, model_commit=ident.get("commit"))
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: ASR provenance not recorded (%s)", job_id, type(exc).__name__)
        return {}


def _save_merge_inputs(
    settings: Settings, job_id: str, asr_raw: list[dict], diar: DiarizationResult, hint: int | None
) -> None:
    """Keep the inputs of the post-processing and merge next to the result ({job_id}.diar.json).

    Transcription is not reproducible run to run, so without the raw transcriber output and the
    diarization an improved clean-up, term table or merge could never be re-applied to an existing
    job. Written before the merge, so it also exists when the merge is what failed. Best-effort.
    Same sensitivity as {job_id}.json; deleted with the job, never exported.
    """
    try:
        (settings.results_dir / f"{job_id}.diar.json").write_text(
            json.dumps(
                {
                    "asr_segments_raw": asr_raw,
                    "turns": diar.turns,
                    "exclusive_turns": diar.exclusive_turns,
                    "params": {
                        "asr_join_split_words": settings.asr_join_split_words,
                        "asr_cleanup_enabled": settings.asr_cleanup_enabled,
                        "asr_term_normalization_enabled": settings.asr_term_normalization_enabled,
                        "num_speakers_hint": hint,
                        "min_speakers": settings.diarization_min_speakers,
                        "max_speakers": settings.diarization_max_speakers,
                        "min_cluster_size": settings.diarization_min_cluster_size,
                        "diarization_model": settings.diarization_model,
                        "merge_use_exclusive": settings.merge_use_exclusive,
                        "merge_smoothing": settings.merge_smoothing,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: Merge inputs not saved (%s)", job_id, type(exc).__name__)


def _detect_language(settings: Settings, job_id: str, segments: list[dict]) -> dict | None:
    """Script-based language of the call, annotating each segment in place — best-effort."""
    if not settings.language_detection:
        return None
    try:
        langs = settings.asr_language_list
        lang = detect_languages(segments, langs, default=langs[0])
        annotate_segments(segments, langs, lang["language"])
        log.info("Task %s: Language %s (shares %s)", job_id, lang["languages"], lang["language_shares"])
        return lang
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: Language detection failed (ignored) — %s", job_id, type(exc).__name__)
        return None


def _export_word(settings: Settings, job_id: str, job: dict, result: dict) -> None:
    """Word export of the original transcript — best-effort, never fails the task."""
    if not settings.word_export_enabled:
        return
    try:
        build_docx(result, settings.results_dir / f"{job_id}.docx",
                   title=job.get("original_name") or job_id)
        log.info("Task %s: Word created", job_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: Word export failed (ignored) — %s", job_id, exc)


def _export_pii(settings: Settings, job_id: str, job: dict, result: dict) -> dict | None:
    """Write the de-identified copy (.pii.json / .pii.docx) — best-effort; returns it, or None.

    Called twice per job: once right after the transcript is ready (so the redacted
    transcript exists as early as possible, and the local keywords are drawn from it), and
    once after the summaries are generated so the exported copy also covers them.
    deidentify_result handles all fields in a single subprocess call, so each pass is one PII
    invocation. Note that the Gemini summaries are NOT produced from this copy: they get the raw
    result, protected by the context pass in summarize.build_transcript_text and the engine pass
    inside the provider.

    The Word title is neutral on purpose: the upload's file name often carries a person's name
    and the call's timestamp ("<name>record-<epoch>.mp3").
    """
    if not settings.pii_enabled:
        return None
    try:
        pii_result = deidentify_result(settings, result)
        (settings.results_dir / f"{job_id}.pii.json").write_text(
            json.dumps(pii_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if settings.word_export_enabled:
            build_docx(pii_result, settings.results_dir / f"{job_id}.pii.docx",
                       title=pii_title(job_id), deidentified=True)
        log.info("Task %s: PII version saved", job_id)
        return pii_result
    except PIIError as exc:
        log.warning("Task %s: PII skipped — %s", job_id, exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: PII failed (ignored) — %s", job_id, exc)
    return None


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    settings = get_settings()
    init_db()
    n = reset_stuck_jobs()
    if n:
        log.info("Oops %d tasks got stuck in processing", n)

    transcriber = Transcriber(settings)
    diarizer = Diarizer(settings)

    if settings.preload_models:
        log.info("Preloading models (device=%s, compute=%s)...", transcriber.device, transcriber.compute_type)
        try:
            _evict_ollama_llm_from_vram(settings)
            transcriber.load()
            diarizer.load()
            warmup_pii_daemons(settings)
            log.info("Models loaded successfully.")
        except Exception as exc:
            log.warning("Preloading failed (will continue lazily when needed): %s", exc)

    log.info("Worker is running. Waiting for tasks...")
    while _running:
        job = None
        try:
            job = claim_next_job()
        except Exception as exc:
            log.error("Error in task claim: %s", exc)
            time.sleep(settings.poll_interval_sec)
            continue

        if job is None:
            time.sleep(settings.poll_interval_sec)
            continue

        try:
            process_job(job, transcriber, diarizer)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("Task %s failed: %s\n%s", job["id"], exc, tb)
            update_job(
                job["id"],
                status="error",
                error=f"{exc}",
                finished_at=time.time(),
            )

    # Clean shutdown of the DB connection and PII daemons
    shutdown_pii_daemons()
    try:
        get_conn().close()
    except Exception:
        pass
    log.info("Worker stopped.")


if __name__ == "__main__":
    main()
