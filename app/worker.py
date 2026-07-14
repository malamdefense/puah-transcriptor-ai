"""תהליך ה-worker: צורך משימות מהתור, מריץ את צינור העיבוד ושומר תוצאות.

רץ כתהליך נפרד מ-API כדי שמודלי ה-GPU יישארו תושבים בזיכרון וכדי ש-API יישאר זמין.
מעבד משימה אחת בכל רגע (ה-GPU הוא משאב סדרתי) — מונע OOM.
"""

from __future__ import annotations

import json
import logging
import signal
import time
import traceback
from pathlib import Path

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
from .pipeline.summarize import (
    LLMError,
    generate_client_summary,
    generate_feedback,
    summarize,
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
    log.info("מקבל אות עצירה — מסיים לאחר המשימה הנוכחית...")


# חלוקת אחוז ההתקדמות הכולל בין שלבי הצינור
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
    job_id = job["id"]
    uploaded = settings.uploads_dir / f"{job_id}.mp4"
    audio_path = settings.audio_dir / f"{job_id}.wav"

    last_update = [0.0]

    def report(stage: str, inner_pct: float, force: bool = False) -> None:
        # מגביל קצב עדכוני DB ל-~פעם בשנייה כדי לא להציף
        now = time.time()
        if not force and (now - last_update[0]) < 1.0:
            return
        last_update[0] = now
        update_job(job_id, stage=stage, progress=round(_stage_progress(stage, inner_pct), 1))

    # 1) חילוץ אודיו
    log.info("משימה %s: חילוץ אודיו", job_id)
    update_job(job_id, stage="extracting_audio", progress=0)
    extract_audio(uploaded, audio_path)
    duration = get_duration(audio_path) or get_duration(uploaded)
    update_job(job_id, duration_sec=duration)
    report("extracting_audio", 100.0, force=True)

    # 2) תמלול
    log.info("משימה %s: תמלול (%.0f שניות אודיו)", job_id, duration)
    update_job(job_id, stage="transcribing")
    asr_segments = transcriber.transcribe(
        audio_path,
        duration_sec=duration,
        progress_cb=lambda pct: report("transcribing", pct),
    )
    report("transcribing", 100.0, force=True)

    # 3) זיהוי דוברים
    log.info("משימה %s: זיהוי דוברים", job_id)
    update_job(job_id, stage="diarizing")
    turns = diarizer.diarize(audio_path, num_speakers=job.get("num_speakers"))
    report("diarizing", 100.0, force=True)

    # 4) מיזוג
    log.info("משימה %s: מיזוג", job_id)
    update_job(job_id, stage="merging")
    segments, speaker_ids = merge(asr_segments, turns)

    # שמות תצוגה דיפולטיביים לדוברים: "דובר 1", "דובר 2", ...
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

    # אינדוקס לחיפוש מלא
    index_job_segments(job_id, result)

    # 5) סיכום אוטומטי (best-effort — לא מפיל את המשימה אם המודל לא זמין)
    if settings.summary_enabled and segments:
        log.info("משימה %s: סיכום", job_id)
        update_job(job_id, stage="summarizing", progress=_stage_progress("summarizing", 10))
        try:
            result["summary"] = summarize(settings, result)
            save_result(job_id, result)
            log.info("משימה %s: סיכום הופק", job_id)
        except LLMError as exc:
            log.warning("משימה %s: דילוג על סיכום — %s", job_id, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("משימה %s: שגיאה בסיכום (מתעלם) — %s", job_id, exc)

        # תוצרי פוע"ה: סיכום ללקוח + משוב ליועץ (כל אחד best-effort בנפרד)
        if settings.client_summary_enabled:
            try:
                result["client_summary"] = generate_client_summary(settings, result)
                save_result(job_id, result)
                log.info("משימה %s: סיכום ללקוח הופק", job_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("משימה %s: דילוג על סיכום ללקוח — %s", job_id, exc)
        if settings.feedback_enabled:
            try:
                result["feedback"] = generate_feedback(settings, result)
                save_result(job_id, result)
                log.info("משימה %s: משוב ליועץ הופק", job_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("משימה %s: דילוג על משוב — %s", job_id, exc)

    # 6) ייצוא Word + הסרת PII (best-effort — לא מפילים את המשימה)
    if segments:
        _finalize_outputs(settings, job_id, job, result)

    update_job(
        job_id,
        status="done",
        stage="summarizing",
        progress=100.0,
        num_speakers=len(speaker_ids),
        finished_at=time.time(),
        error=None,
    )
    log.info("משימה %s: הסתיימה (%d דוברים, %d קטעים)", job_id, len(speaker_ids), len(segments))


def _finalize_outputs(settings: Settings, job_id: str, job: dict, result: dict) -> None:
    """ייצוא Word והסרת PII — שניהם best-effort, לא מפילים את המשימה."""
    title = job.get("original_name") or job_id
    results_dir = settings.results_dir

    # ייצוא Word של התמלול המקורי
    if settings.word_export_enabled:
        try:
            build_docx(result, results_dir / f"{job_id}.docx", title=title)
            log.info("משימה %s: נוצר Word", job_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("משימה %s: ייצוא Word נכשל (מתעלם) — %s", job_id, exc)

    # הסרת PII: שמירת JSON + Word מנוקים, באותו שם בתוספת .pii
    if settings.pii_enabled:
        log.info("משימה %s: הסרת PII", job_id)
        try:
            pii_result = deidentify_result(settings, result)
            (results_dir / f"{job_id}.pii.json").write_text(
                json.dumps(pii_result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if settings.word_export_enabled:
                build_docx(
                    pii_result, results_dir / f"{job_id}.pii.docx", title=title, deidentified=True
                )
            log.info("משימה %s: גרסת PII נשמרה", job_id)
        except PIIError as exc:
            log.warning("משימה %s: הסרת PII דולגה — %s", job_id, exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("משימה %s: הסרת PII נכשלה (מתעלם) — %s", job_id, exc)


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    settings = get_settings()
    init_db()
    n = reset_stuck_jobs()
    if n:
        log.info("אופסו %d משימות שנתקעו ב-processing", n)

    transcriber = Transcriber(settings)
    diarizer = Diarizer(settings)

    if settings.preload_models:
        log.info("טוען מודלים מראש (device=%s, compute=%s)...", transcriber.device, transcriber.compute_type)
        try:
            transcriber.load()
            diarizer.load()
            log.info("המודלים נטענו בהצלחה.")
        except Exception as exc:  # noqa: BLE001
            log.warning("טעינה מוקדמת נכשלה (ימשיך עצלן בעת הצורך): %s", exc)

    log.info("ה-worker פעיל. ממתין למשימות...")
    while _running:
        job = None
        try:
            job = claim_next_job()
        except Exception as exc:  # noqa: BLE001
            log.error("שגיאה בתפיסת משימה: %s", exc)
            time.sleep(settings.poll_interval_sec)
            continue

        if job is None:
            time.sleep(settings.poll_interval_sec)
            continue

        try:
            process_job(job, transcriber, diarizer)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("משימה %s נכשלה: %s\n%s", job["id"], exc, tb)
            update_job(
                job["id"],
                status="error",
                error=f"{exc}",
                finished_at=time.time(),
            )

    # סגירה נקייה של חיבור ה-DB של ה-thread
    try:
        get_conn().close()
    except Exception:
        pass
    log.info("ה-worker נעצר.")


if __name__ == "__main__":
    main()
