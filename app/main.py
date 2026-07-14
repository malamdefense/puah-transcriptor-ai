"""שרת ה-API + הגשת ממשק ה-Web.

ה-API קל-משקל: מקבל העלאות, מנהל את התור (SQLite), ומגיש את התוצאות.
העיבוד הכבד מתבצע בתהליך ה-worker הנפרד (app/worker.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote

from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, prompts_store
from .config import get_settings
from .db import (
    compute_stats,
    create_job,
    delete_job,
    get_job,
    index_job_segments,
    init_db,
    list_jobs,
    load_result,
    result_path,
    save_result,
    search_segments,
)
from .pipeline.export import EXPORTERS
from .pipeline.export_docx import build_docx_bytes
from .pipeline.summarize import (
    LLMError,
    check_available,
    generate_client_summary,
    generate_feedback,
    summarize,
)
from .schemas import (
    JobSummary,
    SearchMatch,
    SearchResponse,
    SpeakerRename,
    TranscriptResult,
)

settings = get_settings()

app = FastAPI(title="מערכת תמלול פוע\"ה", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
ALLOWED_EXT = {".mp4", ".m4a", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".aac", ".avi"}

# יצירת הסכימה בעליית התהליך (idempotent) — לא תלוי באירוע lifespan
init_db()


# ---------- API ----------

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "asr_model": settings.asr_model,
        "summary_enabled": settings.summary_enabled,
        "llm_model": settings.llm_model,
    }


@app.post("/api/jobs", response_model=JobSummary, status_code=201)
async def upload_job(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(default=None),
    language: str = Form(default="he"),
) -> JobSummary:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"סוג קובץ לא נתמך: {ext or '(ללא סיומת)'}. נתמכים: {', '.join(sorted(ALLOWED_EXT))}",
        )

    if num_speakers is not None and num_speakers <= 0:
        num_speakers = None

    job_id = create_job(original_name=file.filename or "recording", num_speakers=num_speakers, language=language)

    # שמירת הקובץ המקורי (תמיד בסיומת .mp4 לצורך אחידות — ה-ffmpeg מזהה לפי תוכן)
    dest = settings.uploads_dir / f"{job_id}.mp4"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    delete_job(job_id)
                    raise HTTPException(
                        status_code=413,
                        detail=f"הקובץ גדול מהמותר ({settings.max_upload_mb}MB).",
                    )
                out.write(chunk)
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        delete_job(job_id)
        raise HTTPException(status_code=400, detail="התקבל קובץ ריק.")

    return JobSummary.from_row(get_job(job_id))


@app.get("/api/jobs", response_model=list[JobSummary])
def jobs() -> list[JobSummary]:
    return [JobSummary.from_row(r) for r in list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobSummary)
def job_status(job_id: str) -> JobSummary:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="משימה לא נמצאה")
    return JobSummary.from_row(row)


@app.get("/api/jobs/{job_id}/result", response_model=TranscriptResult)
def job_result(job_id: str) -> TranscriptResult:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="משימה לא נמצאה")
    if row["status"] != "done":
        raise HTTPException(status_code=409, detail=f"המשימה במצב '{row['status']}' — אין תוצאה עדיין")
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="קובץ התוצאה חסר")
    return TranscriptResult(**result)


@app.put("/api/jobs/{job_id}/speakers", response_model=TranscriptResult)
def rename_speakers(job_id: str, payload: SpeakerRename) -> TranscriptResult:
    """עדכון שמות תצוגה לדוברים (למשל 'דובר 1' -> 'יועץ')."""
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="תוצאה לא נמצאה")
    for sid, name in payload.speakers.items():
        if sid in result["speakers"] and name.strip():
            result["speakers"][sid] = name.strip()
    save_result(job_id, result)
    index_job_segments(job_id, result)  # רענון אינדקס החיפוש עם השמות המעודכנים
    return TranscriptResult(**result)


@app.post("/api/jobs/{job_id}/summarize", response_model=TranscriptResult)
def summarize_job(job_id: str) -> TranscriptResult:
    """יצירה/רענון של הסיכום, סיכום ללקוח ומשוב ליועץ באמצעות מודל השפה המקומי."""
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="תוצאה לא נמצאה")
    try:
        result["summary"] = summarize(settings, result)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # תוצרי פוע"ה — best-effort (אם אחד נכשל, עדיין מחזירים את השאר)
    if settings.client_summary_enabled:
        try:
            result["client_summary"] = generate_client_summary(settings, result)
        except LLMError:
            pass
    if settings.feedback_enabled:
        try:
            result["feedback"] = generate_feedback(settings, result)
        except LLMError:
            pass
    save_result(job_id, result)
    return TranscriptResult(**result)


@app.get("/api/export/feedback.xlsx")
def export_feedback_xlsx() -> Response:
    """ייצוא Excel מצרפי: לכל הקלטה — שם, סיכום ללקוח, משוב (כמו משוב_שיחות.xlsx)."""
    import io

    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "משוב שיחות"
    ws.sheet_view.rightToLeft = True
    ws.append(["name of the record", "סיכום ללקוח", "משוב"])
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 70
    ws.column_dimensions["C"].width = 70

    rows = 0
    for job in list_jobs():
        if job["status"] != "done":
            continue
        res = load_result(job["id"])
        if not res:
            continue
        cs = res.get("client_summary")
        fb = res.get("feedback")
        if not cs and not fb:
            continue
        name = (job.get("original_name") or job["id"]).rsplit(".", 1)[0]
        ws.append([name, cs or "", fb or ""])
        rows += 1
        for cell in ws[ws.max_row]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    buf = io.BytesIO()
    wb.save(buf)
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=_content_disposition("משוב_שיחות.xlsx"),
    )


@app.get("/api/llm/status")
def llm_status() -> dict:
    """בדיקת זמינות שרת מודל השפה (קריאה קצרה)."""
    if not settings.summary_enabled:
        return {"enabled": False, "available": False, "model": settings.llm_model}
    return {
        "enabled": True,
        "available": check_available(settings),
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }


# ---------- עריכת פרומפטים (סיכומים) ----------

@app.get("/api/prompts")
def get_prompts() -> dict:
    """מחזיר את כל הפרומפטים הניתנים לעריכה (ערך פעיל + ברירת מחדל)."""
    return {"prompts": prompts_store.all_prompts()}


@app.put("/api/prompts")
def update_prompts(payload: dict) -> dict:
    """שמירת עריכות פרומפטים. גוף: {key: value, ...}."""
    updates = payload.get("prompts", payload)
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="פורמט לא תקין")
    prompts_store.save({k: v for k, v in updates.items() if isinstance(v, str)})
    return {"prompts": prompts_store.all_prompts()}


@app.post("/api/prompts/reset")
def reset_prompts(key: str | None = None) -> dict:
    """שחזור לברירת מחדל — פרומפט בודד (key=...) או הכול."""
    prompts_store.reset(key)
    return {"prompts": prompts_store.all_prompts()}


# ---------- לוגואים (מיתוג) ----------

_BRAND_SIDES = {"right", "left"}
_IMG_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
}


def _brand_dir() -> Path:
    d = settings.data_dir / "branding"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _brand_file(side: str) -> Path | None:
    for f in _brand_dir().glob(f"{side}.*"):
        return f
    return None


@app.get("/api/branding")
def branding_status() -> dict:
    """אילו לוגואים מוגדרים."""
    return {"right": _brand_file("right") is not None, "left": _brand_file("left") is not None}


@app.get("/api/branding/{side}")
def get_logo(side: str) -> FileResponse:
    if side not in _BRAND_SIDES:
        raise HTTPException(status_code=404, detail="צד לא תקין")
    f = _brand_file(side)
    if not f:
        raise HTTPException(status_code=404, detail="לוגו לא הוגדר")
    return FileResponse(f, media_type=_IMG_MIME.get(f.suffix.lower(), "application/octet-stream"))


@app.post("/api/branding/{side}")
async def set_logo(side: str, file: UploadFile = File(...)) -> dict:
    if side not in _BRAND_SIDES:
        raise HTTPException(status_code=404, detail="צד לא תקין")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _IMG_MIME:
        raise HTTPException(status_code=400, detail=f"סוג תמונה לא נתמך: {ext or '?'}")
    for old in _brand_dir().glob(f"{side}.*"):
        old.unlink(missing_ok=True)
    dest = _brand_dir() / f"{side}{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"ok": True, "side": side}


@app.delete("/api/branding/{side}", status_code=204)
def delete_logo(side: str) -> Response:
    if side not in _BRAND_SIDES:
        raise HTTPException(status_code=404, detail="צד לא תקין")
    for f in _brand_dir().glob(f"{side}.*"):
        f.unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/stats")
def stats() -> dict:
    """סטטיסטיקות לדשבורד: אגרגציות מה-DB + מילות מפתח וכיסוי סיכום/PII מקובצי התוצאה."""
    from collections import Counter

    s = compute_stats()
    keywords: Counter = Counter()
    topics: Counter = Counter()
    with_summary = 0
    with_pii = 0
    for job in list_jobs():
        if job["status"] != "done":
            continue
        if (settings.results_dir / f"{job['id']}.pii.json").exists():
            with_pii += 1
        res = load_result(job["id"])
        summ = (res or {}).get("summary") or {}
        if summ.get("summary"):
            with_summary += 1
        for k in summ.get("keywords", []) or []:
            if str(k).strip():
                keywords[str(k).strip()] += 1
        for t in summ.get("topics", []) or []:
            if str(t).strip():
                topics[str(t).strip()] += 1

    s["with_summary"] = with_summary
    s["with_pii"] = with_pii
    s["top_keywords"] = [{"term": t, "count": c} for t, c in keywords.most_common(15)]
    s["top_topics"] = [{"term": t, "count": c} for t, c in topics.most_common(10)]
    return s


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = "", limit: int = 100) -> SearchResponse:
    """חיפוש מלא בכל התמלולים. מחזיר קטעים תואמים עם חותמת זמן ודובר."""
    q = q.strip()
    if not q:
        return SearchResponse(query=q, count=0, matches=[])
    rows = search_segments(q, limit=min(max(limit, 1), 500))
    matches = [SearchMatch(**r) for r in rows]
    return SearchResponse(query=q, count=len(matches), matches=matches)


_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _download_name(job_id: str, suffix: str, ext: str) -> str:
    row = get_job(job_id)
    base = (row.get("original_name") if row else None) or f"transcript-{job_id}"
    base = base.rsplit(".", 1)[0]  # הסרת סיומת המקור (mp4 וכו')
    return f"{base}{suffix}.{ext}"


def _content_disposition(filename: str) -> dict[str, str]:
    """כותרת Content-Disposition בטוחה לשמות עבריים (RFC 5987 + fallback ASCII)."""
    base, _, ext = filename.rpartition(".")
    base = base or filename
    ext = ext or "dat"
    ascii_base = base.encode("ascii", "ignore").decode().strip().strip(".")
    fallback = f"{ascii_base}.{ext}" if ascii_base else f"transcript.{ext}"
    return {"Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"}


@app.get("/api/jobs/{job_id}/export")
def export(job_id: str, format: str = "txt") -> Response:
    fmt = format.lower()
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="תוצאה לא נמצאה")

    if fmt == "docx":
        content = build_docx_bytes(result, title=_download_name(job_id, "", "").rstrip("."))
        return Response(
            content=content,
            media_type=_DOCX_MIME,
            headers=_content_disposition(_download_name(job_id, "", "docx")),
        )

    if fmt not in EXPORTERS:
        raise HTTPException(status_code=400, detail=f"פורמט לא נתמך: {fmt}")
    media_type, fn = EXPORTERS[fmt]
    return Response(
        content=fn(result),
        media_type=media_type,
        headers=_content_disposition(_download_name(job_id, "", fmt)),
    )


@app.get("/api/jobs/{job_id}/pii")
def export_pii(job_id: str, format: str = "json") -> Response:
    """הורדת הגרסה שעברה הסרת PII (json/docx)."""
    fmt = format.lower()
    pii_json = settings.results_dir / f"{job_id}.pii.json"
    if not pii_json.exists():
        raise HTTPException(status_code=404, detail="גרסת PII לא נמצאה (אולי הסרת ה-PII דולגה)")

    if fmt == "json":
        return FileResponse(
            pii_json,
            media_type="application/json",
            filename=_download_name(job_id, ".pii", "json"),
        )
    if fmt == "docx":
        pii_docx = settings.results_dir / f"{job_id}.pii.docx"
        if pii_docx.exists():
            return FileResponse(pii_docx, media_type=_DOCX_MIME, filename=_download_name(job_id, ".pii", "docx"))
        # נבנה מה-JSON אם קובץ ה-docx חסר
        import json as _json

        content = build_docx_bytes(
            _json.loads(pii_json.read_text(encoding="utf-8")),
            title=_download_name(job_id, ".pii", "").rstrip("."),
            deidentified=True,
        )
        return Response(
            content=content,
            media_type=_DOCX_MIME,
            headers=_content_disposition(_download_name(job_id, ".pii", "docx")),
        )
    raise HTTPException(status_code=400, detail=f"פורמט לא נתמך: {fmt}")


@app.get("/api/jobs/{job_id}/media")
def media(job_id: str) -> FileResponse:
    """הגשת קובץ המקור לצורך נגן בממשק (תומך ב-range requests)."""
    path = settings.uploads_dir / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="קובץ מקור לא נמצא")
    return FileResponse(path, media_type="video/mp4")


@app.delete("/api/jobs/{job_id}", status_code=204)
def remove_job(job_id: str) -> Response:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="משימה לא נמצאה")
    (settings.uploads_dir / f"{job_id}.mp4").unlink(missing_ok=True)
    (settings.audio_dir / f"{job_id}.wav").unlink(missing_ok=True)
    result_path(job_id).unlink(missing_ok=True)
    for extra in (f"{job_id}.docx", f"{job_id}.pii.json", f"{job_id}.pii.docx"):
        (settings.results_dir / extra).unlink(missing_ok=True)
    delete_job(job_id)
    return Response(status_code=204)


# ---------- הגשת ממשק ה-Web ----------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
