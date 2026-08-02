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
from .pipeline.audio import extract_audio, get_duration
from .pipeline.diarize import Diarizer
from .pipeline.export_docx import build_docx
from .pipeline.merge import merge
from .pipeline.pii import PIIError, deidentify_result
from .pipeline.keywords import build_summary as build_local_summary
from .pipeline.summarize import (
    generate_client_summary,
    generate_feedback,
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
    "transcribing": (5.0, 65.0),
    "diarizing": (65.0, 88.0),
    "merging": (88.0, 92.0),
    "summarizing": (92.0, 100.0),
}


def _stage_progress(stage: str, inner_pct: float) -> float:
    lo, hi = _STAGE_WEIGHTS[stage]
    return lo + (hi - lo) * (max(0.0, min(100.0, inner_pct)) / 100.0)


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

    # 2) Transcription
    log.info("Task %s: Transcription (%.0f seconds audio)", job_id, duration)
    update_job(job_id, stage="transcribing")
    asr_segments = transcriber.transcribe(
        audio_path,
        duration_sec=duration,
        progress_cb=lambda pct: report("transcribing", pct),
    )
    report("transcribing", 100.0, force=True)

    # 3) Speaker diarization
    log.info("Task %s: Speaker diarization", job_id)
    update_job(job_id, stage="diarizing")
    turns = diarizer.diarize(audio_path, num_speakers=job.get("num_speakers"))
    report("diarizing", 100.0, force=True)

    # 4) Merge
    log.info("Task %s: Merge", job_id)
    update_job(job_id, stage="merging")
    segments, speaker_ids = merge(asr_segments, turns)

    # Display names for speakers: "Speaker 1", "Speaker 2", ...
    speakers = {sid: f"דובר {i}" for i, sid in enumerate(speaker_ids, start=1)}

    result = {
        "job_id": job_id,
        "language": job.get("language", settings.language),
        "duration_sec": duration,
        "num_speakers": len(speaker_ids),
        "speakers": speakers,
        "segments": segments,
        "asr_model": settings.asr_model,
        "diarization_model": settings.diarization_model,
        "summary": None,
        "client_summary": None,
        "feedback": None,
    }
    save_result(job_id, result)

    # Full index for full search
    index_job_segments(job_id, result)

    # 5) Word export + PII de-identification of the transcript (best-effort).
    # Runs before the summaries so that the redacted transcript exists first, and the
    # summaries are produced immediately after it — no manual trigger needed.
    if segments:
        _export_word(settings, job_id, job, result)
        log.info("Task %s: PII removal", job_id)
        _export_pii(settings, job_id, job, result)

    # 6) Summaries — generated automatically right after the transcript PII step.
    # Each product is best-effort on its own: a failure must not fail the job nor discard
    # the transcript that was already saved.
    produced = False
    if segments:
        update_job(job_id, stage="summarizing", progress=_stage_progress("summarizing", 10))

        # Topics + keywords are extracted locally (no AI call): instant, offline and
        # free. They populate the same `summary` object the dashboard aggregates over,
        # so /api/stats keeps working exactly as before. The narrative "automatic
        # summary" is no longer produced — the client summary and consultant feedback
        # below are the deliverables.
        try:
            result["summary"] = build_local_summary(result)
            save_result(job_id, result)
            produced = True
            log.info(
                "Task %s: Topics/keywords extracted locally (%d topics, %d keywords)",
                job_id, len(result["summary"]["topics"]), len(result["summary"]["keywords"]),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: Local topic/keyword extraction failed (ignored) — %s", job_id, exc)

    # PUAH products: client summary + consultant feedback (each best-effort separately)
    if settings.client_summary_enabled and segments:
        try:
            result["client_summary"] = generate_client_summary(settings, result)
            save_result(job_id, result)
            produced = True
            log.info("Task %s: Client summary generated", job_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: Skipped client summary — %s", job_id, exc)
    if settings.feedback_enabled and segments:
        try:
            result["feedback"] = generate_feedback(settings, result)
            save_result(job_id, result)
            produced = True
            log.info("Task %s: Feedback generated", job_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("Task %s: Skipped feedback — %s", job_id, exc)

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


def _export_pii(settings: Settings, job_id: str, job: dict, result: dict) -> None:
    """Write the de-identified copy (.pii.json / .pii.docx) — best-effort.

    Called twice per job: once right after the transcript is ready (so the redacted
    transcript exists as early as possible, and the summaries that follow are produced
    against an already-redacted job), and once after the summaries are generated so the
    exported copy also covers them. deidentify_result handles all fields in a single
    subprocess call, so each pass is one PII invocation.
    """
    if not settings.pii_enabled:
        return
    try:
        pii_result = deidentify_result(settings, result)
        (settings.results_dir / f"{job_id}.pii.json").write_text(
            json.dumps(pii_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if settings.word_export_enabled:
            build_docx(pii_result, settings.results_dir / f"{job_id}.pii.docx",
                       title=job.get("original_name") or job_id, deidentified=True)
        log.info("Task %s: PII version saved", job_id)
    except PIIError as exc:
        log.warning("Task %s: PII skipped — %s", job_id, exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("Task %s: PII failed (ignored) — %s", job_id, exc)


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
            transcriber.load()
            diarizer.load()
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

    # Clean shutdown of the DB connection
    try:
        get_conn().close()
    except Exception:
        pass
    log.info("Worker stopped.")


if __name__ == "__main__":
    main()
