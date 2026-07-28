"""API server + Web interface.

The API is lightweight: it accepts uploads, manages the queue (SQLite), and serves the results.
Heavy processing is performed in the separate worker process (app/worker.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote

from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, prompts_store, secrets_store
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

app = FastAPI(title="Puah Transcription System", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
ALLOWED_EXT = {".mp4", ".m4a", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".aac", ".avi"}

# Create the index on startup (idempotent) — not dependent on the lifespan event
init_db()


# ---------- API ----------

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "asr_model": settings.asr_model,
        "summary_enabled": settings.summary_enabled,
        "llm_model": settings.gemini_model,
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
            detail=f"Unsupported file type: {ext or '(no extension)'}. Supported: {', '.join(sorted(ALLOWED_EXT))}",
        )

    if num_speakers is not None and num_speakers <= 0:
        num_speakers = None

    job_id = create_job(original_name=file.filename or "recording", num_speakers=num_speakers, language=language)

    # Save the original file (always in .mp4 format for uniformity — ffmpeg identifies by content)
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
        raise HTTPException(status_code=400, detail="Empty file received.")

    return JobSummary.from_row(get_job(job_id))


@app.get("/api/jobs", response_model=list[JobSummary])
def jobs() -> list[JobSummary]:
    return [JobSummary.from_row(r) for r in list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobSummary)
def job_status(job_id: str) -> JobSummary:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobSummary.from_row(row)


@app.get("/api/jobs/{job_id}/result", response_model=TranscriptResult)
def job_result(job_id: str) -> TranscriptResult:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job in status '{row['status']}' — no result yet")
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result file missing")
    return TranscriptResult(**result)


@app.put("/api/jobs/{job_id}/speakers", response_model=TranscriptResult)
def rename_speakers(job_id: str, payload: SpeakerRename) -> TranscriptResult:
    """Update speaker display names (e.g. 'Speaker 1' -> 'Consultant')."""
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    for sid, name in payload.speakers.items():
        if sid in result["speakers"] and name.strip():
            result["speakers"][sid] = name.strip()
    save_result(job_id, result)
    index_job_segments(job_id, result)  # Re-index the search with the updated names
    return TranscriptResult(**result)


@app.post("/api/jobs/{job_id}/summarize", response_model=TranscriptResult)
def summarize_job(job_id: str) -> TranscriptResult:
    """Create/update the summary, client summary, and feedback using Gemini."""
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    try:
        result["summary"] = summarize(settings, result)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # Puah's outputs — best-effort (if one fails, still return the others)
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
    """Export Excel with attachments: for every recording — name, client summary, feedback (like Feedback_Calls.xlsx)."""
    import io

    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Feedback Calls"
    ws.sheet_view.rightToLeft = True
    ws.append(["name of the record", "Client Summary", "Feedback"])
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
        headers=_content_disposition("Feedback_Calls.xlsx"),
    )


@app.get("/api/llm/status")
def llm_status() -> dict:
    """Availability of the summary engine (Gemini). See summarize.check_available — no network call."""
    if not settings.summary_enabled:
        return {"enabled": False, "available": False, "model": settings.gemini_model}
    return {
        "enabled": True,
        "available": check_available(settings),
        "model": settings.gemini_model,
        "provider": "gemini",
    }


# ---------- Editing prompts (summaries) ----------

@app.get("/api/prompts")
def get_prompts() -> dict:
    """Returns all prompts that can be edited (active value + default)."""
    return {"prompts": prompts_store.all_prompts()}


@app.put("/api/prompts")
def update_prompts(payload: dict) -> dict:
    """Save prompt edits. Body: {key: value, ...}."""
    updates = payload.get("prompts", payload)
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Invalid format")
    prompts_store.save({k: v for k, v in updates.items() if isinstance(v, str)})
    return {"prompts": prompts_store.all_prompts()}


@app.post("/api/prompts/reset")
def reset_prompts(key: str | None = None) -> dict:
    """Restore to default — single prompt (key=...) or all."""
    prompts_store.reset(key)
    return {"prompts": prompts_store.all_prompts()}


# ---------- API secrets (overrides .env) ----------

@app.get("/api/secrets")
def get_secrets() -> dict:
    """Status of API secrets — only stored, never the full value."""
    return {"secrets": secrets_store.status(settings)}


@app.put("/api/secrets/{key}")
def update_secret(key: str, payload: dict) -> dict:
    """Save a secret entered from the interface. Overrides the value in .env at runtime."""
    value = payload.get("value", payload.get("api_key", ""))
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="Invalid format")
    try:
        secrets_store.save(settings, key, value)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown key: {key}") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"secrets": secrets_store.status(settings)}


@app.delete("/api/secrets/{key}")
def delete_secret(key: str) -> dict:
    """Delete the secret entered from the interface — returns to the value in .env (if any)."""
    try:
        secrets_store.clear(settings, key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown key: {key}") from None
    return {"secrets": secrets_store.status(settings)}


# ---------- Logos (branding) ----------

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
    """Which logos are defined."""
    return {"right": _brand_file("right") is not None, "left": _brand_file("left") is not None}


@app.get("/api/branding/{side}")
def get_logo(side: str) -> FileResponse:
    if side not in _BRAND_SIDES:
        raise HTTPException(status_code=404, detail="Invalid side")
    f = _brand_file(side)
    if not f:
        raise HTTPException(status_code=404, detail="Logo not defined")
    return FileResponse(f, media_type=_IMG_MIME.get(f.suffix.lower(), "application/octet-stream"))


@app.post("/api/branding/{side}")
async def set_logo(side: str, file: UploadFile = File(...)) -> dict:
    if side not in _BRAND_SIDES:
        raise HTTPException(status_code=404, detail="Invalid side")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _IMG_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {ext or '?'}")
    for old in _brand_dir().glob(f"{side}.*"):
        old.unlink(missing_ok=True)
    dest = _brand_dir() / f"{side}{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"ok": True, "side": side}


@app.delete("/api/branding/{side}", status_code=204)
def delete_logo(side: str) -> Response:
    if side not in _BRAND_SIDES:
        raise HTTPException(status_code=404, detail="Invalid side")
    for f in _brand_dir().glob(f"{side}.*"):
        f.unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/stats")
def stats() -> dict:
    """Dashboard statistics: aggregations from the DB + keywords and summary/PII coverage from the result files."""
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
        # "There is a summary" = the summary object contains some content. (summary field was removed from the default,
        # but remains in backward compatibility with jobs that were created before the change.)
        if any(summ.get(f) for f in ("summary", "key_points", "action_items", "topics", "keywords")):
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
    """Full search across all transcripts. Returns matching segments with timestamp and speaker."""
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
    base = base.rsplit(".", 1)[0]  # Remove the original extension (mp4, etc.)
    return f"{base}{suffix}.{ext}"


def _content_disposition(filename: str) -> dict[str, str]:
    """Safe Content-Disposition header for Hebrew names (RFC 5987 + fallback ASCII)."""
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
        raise HTTPException(status_code=404, detail="Result not found")

    if fmt == "docx":
        content = build_docx_bytes(result, title=_download_name(job_id, "", "").rstrip("."))
        return Response(
            content=content,
            media_type=_DOCX_MIME,
            headers=_content_disposition(_download_name(job_id, "", "docx")),
        )

    if fmt not in EXPORTERS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")
    media_type, fn = EXPORTERS[fmt]
    return Response(
        content=fn(result),
        media_type=media_type,
        headers=_content_disposition(_download_name(job_id, "", fmt)),
    )


@app.get("/api/jobs/{job_id}/pii")
def export_pii(job_id: str, format: str = "json") -> Response:
    """Download the PII-removed version (json/docx)."""
    fmt = format.lower()
    pii_json = settings.results_dir / f"{job_id}.pii.json"
    if not pii_json.exists():
        raise HTTPException(status_code=404, detail="PII-removed version not found (maybe PII removal was skipped)")

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
        # Build from JSON if the docx file is missing
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
    raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")


@app.get("/api/jobs/{job_id}/media")
def media(job_id: str) -> FileResponse:
    """Download the original file for playback (supports range requests)."""
    path = settings.uploads_dir / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original file not found")
    return FileResponse(path, media_type="video/mp4")


@app.delete("/api/jobs/{job_id}", status_code=204)
def remove_job(job_id: str) -> Response:
    row = get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    (settings.uploads_dir / f"{job_id}.mp4").unlink(missing_ok=True)
    (settings.audio_dir / f"{job_id}.wav").unlink(missing_ok=True)
    result_path(job_id).unlink(missing_ok=True)
    for extra in (f"{job_id}.docx", f"{job_id}.pii.json", f"{job_id}.pii.docx"):
        (settings.results_dir / extra).unlink(missing_ok=True)
    delete_job(job_id)
    return Response(status_code=204)


# ---------- Web interface ----------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
