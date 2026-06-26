"""**API server + Web interface.**

The API is lightweight: accepts uploads, manages the queue (SQLite), and serves the results.
Heavy processing is done in the separate worker process (app/worker.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote

from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
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

# Create the schema on startup (idempotent) — not dependent on lifespan event
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
            detail=f"Unsupported file type: {ext or '(no extension)'}. Supported: {', '.join(sorted(ALLOWED_EXT))}",
        )

    if num_speakers is not None and num_speakers <= 0:
        num_speakers = None

    job_id = create_job(original_name=file.filename or "recording", num_speakers=num_speakers, language=language)

    # Save the original file (always in .mp4 for consistency — ffmpeg identifies by content)
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
                        detail=f"File too large ({settings.max_upload_mb}MB).",
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
        raise HTTPException(status_code=409, detail=f"Job in '{row['status']}' state — no result yet")
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result file missing")
    return TranscriptResult(**result)


@app.put("/api/jobs/{job_id}/speakers", response_model=TranscriptResult)
def rename_speakers(job_id: str, payload: SpeakerRename) -> TranscriptResult:
    """Update speaker display names (e.g. 'Speaker 1' -> 'Advisor')."""
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
    """Create/update the summary, client summary, and advisor feedback using the local language model."""
    result = load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    try:
        result["summary"] = summarize(settings, result)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # Puah-specific outputs — best-effort (if one fails, still return the rest)
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
    """Export Excel attachment: for each recording — name, client summary, feedback (like feedback_calls.xlsx)."""
    import io

    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Feedback calls"
    ws.sheet_view.rightToLeft = True
    ws.append(["name of the record", "Client summary", "Feedback"])
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
        headers=_content_disposition("feedback_calls.xlsx"),
    )


@app.get("/api/llm/status")
def llm_status() -> dict:
    """Check language model server availability (short call)."""
    if not settings.summary_enabled:
        return {"enabled": False, "available": False, "model": settings.llm_model}
    return {
        "enabled": True,
        "available": check_available(settings),
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
    }


@app.get("/api/stats")
def stats() -> dict:
    """Dashboard stats: counts from DB + keywords/topics from merged result files."""
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
    """Full search across all recordings. Returns matching segments with timestamp and speaker."""
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
    """Content-Disposition header safe for Hebrew names (RFC 5987 + fallback ASCII)."""
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
        raise HTTPException(status_code=400, detail=f"Format not supported: {fmt}")
    media_type, fn = EXPORTERS[fmt]
    return Response(
        content=fn(result),
        media_type=media_type,
        headers=_content_disposition(_download_name(job_id, "", fmt)),
    )


@app.get("/api/jobs/{job_id}/pii")
def export_pii(job_id: str, format: str = "json") -> Response:
    """Download the PII-redacted version (json/docx)."""
    fmt = format.lower()
    pii_json = settings.results_dir / f"{job_id}.pii.json"
    if not pii_json.exists():
        raise HTTPException(status_code=404, detail="PII version not found (maybe PII removal was skipped)")

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
    raise HTTPException(status_code=400, detail=f"Format not supported: {fmt}")


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
