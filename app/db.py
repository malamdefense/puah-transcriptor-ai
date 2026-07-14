"""שכבת גישה למסד הנתונים (SQLite).

SQLite נבחר במכוון: פריסת On-Prem על שרת בודד, ללא צורך בשירות broker חיצוני.
מצב WAL מאפשר קריאה (ה-API) במקביל לכתיבה (ה-worker).
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .config import get_settings

_local = threading.local()


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    status        TEXT NOT NULL,        -- queued | processing | done | error
    stage         TEXT,                 -- extracting_audio | transcribing | diarizing | merging
    progress      REAL NOT NULL DEFAULT 0,
    num_speakers  INTEGER,              -- רמז למספר דוברים (NULL = אוטומטי)
    language      TEXT NOT NULL DEFAULT 'he',
    duration_sec  REAL,
    error         TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs (status, created_at);

-- אינדקס חיפוש מלא על קטעי התמלול.
-- tokenize=trigram מאפשר חיפוש תת-מחרוזת — קריטי לעברית, שבה תחיליות (ה/ו/ב/ל/מ/ש)
-- נצמדות למילה: חיפוש "ייעוץ" מוצא גם "הייעוץ", "פוריות" מוצא "לפוריות" וכו'.
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    job_id UNINDEXED,
    seg_index UNINDEXED,
    start UNINDEXED,
    speaker,
    text,
    tokenize='trigram case_sensitive 0'
);
"""


def _connect() -> sqlite3.Connection:
    settings = get_settings()
    conn = sqlite3.connect(settings.db_path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def get_conn() -> sqlite3.Connection:
    """חיבור פר-thread (sqlite אינו מאפשר שיתוף חיבור בין threads)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)


def create_job(original_name: str, num_speakers: Optional[int], language: str) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    get_conn().execute(
        """INSERT INTO jobs (id, original_name, status, progress, num_speakers,
                             language, created_at, updated_at)
           VALUES (?, ?, 'queued', 0, ?, ?, ?, ?)""",
        (job_id, original_name, num_speakers, language, now, now),
    )
    return job_id


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    row = get_conn().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 200) -> list[dict[str, Any]]:
    rows = get_conn().execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    get_conn().execute(
        f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id)
    )


def claim_next_job() -> Optional[dict[str, Any]]:
    """תופס באופן אטומי את המשימה הממתינה הוותיקה ביותר ומסמן אותה כ-processing.

    BEGIN IMMEDIATE נועל את הכתיבה כך ששני workers לא יתפסו את אותה משימה.
    """
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' "
            "ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT;")
            return None
        now = time.time()
        conn.execute(
            "UPDATE jobs SET status='processing', stage='extracting_audio', "
            "progress=0, started_at=?, updated_at=? WHERE id=?",
            (now, now, row["id"]),
        )
        conn.execute("COMMIT;")
        job = dict(row)
        job.update(status="processing", stage="extracting_audio", started_at=now)
        return job
    except Exception:
        conn.execute("ROLLBACK;")
        raise


def reset_stuck_jobs() -> int:
    """מחזיר ל-queued משימות שנתקעו ב-processing (לדוגמה לאחר קריסת worker)."""
    cur = get_conn().execute(
        "UPDATE jobs SET status='queued', stage=NULL, progress=0, started_at=NULL "
        "WHERE status='processing'"
    )
    return cur.rowcount


def delete_job(job_id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.execute("DELETE FROM segments_fts WHERE job_id = ?", (job_id,))


# --- חיפוש מלא (FTS5) ---

def index_job_segments(job_id: str, result: dict[str, Any]) -> None:
    """מאנדקס מחדש את קטעי התמלול של משימה לחיפוש (שומר את שם התצוגה של הדובר)."""
    conn = get_conn()
    conn.execute("DELETE FROM segments_fts WHERE job_id = ?", (job_id,))
    speakers = result.get("speakers", {})
    rows = [
        (
            job_id,
            i,
            float(seg.get("start", 0.0)),
            speakers.get(seg["speaker"], seg["speaker"]),
            seg.get("text", ""),
        )
        for i, seg in enumerate(result.get("segments", []))
    ]
    if rows:
        conn.executemany(
            "INSERT INTO segments_fts (job_id, seg_index, start, speaker, text) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def _fts_match(query: str) -> Optional[str]:
    """בונה שאילתת FTS5 (trigram): כל מילה באורך >=3 כתת-מחרוזת, בחיבור AND.

    מחזיר None אם אין מילה באורך 3+ (trigram דורש לפחות 3 תווים) — אז נופלים ל-LIKE.
    """
    tokens = [t for t in re.findall(r"\w+", query, flags=re.UNICODE) if len(t) >= 3]
    if not tokens:
        return None
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


_SEARCH_SELECT = (
    "SELECT s.job_id AS job_id, s.seg_index AS seg_index, s.start AS start, "
    "s.speaker AS speaker, s.text AS text, j.original_name AS original_name "
    "FROM segments_fts s JOIN jobs j ON j.id = s.job_id "
)


def search_segments(query: str, limit: int = 100) -> list[dict[str, Any]]:
    q = query.strip()
    if not q:
        return []
    match = _fts_match(q)
    conn = get_conn()
    if match is not None:
        rows = conn.execute(
            _SEARCH_SELECT + "WHERE segments_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    else:
        # שאילתה קצרה מ-3 תווים — חיפוש תת-מחרוזת ישיר
        like = f"%{q}%"
        rows = conn.execute(
            _SEARCH_SELECT + "WHERE s.text LIKE ? OR s.speaker LIKE ? ORDER BY s.start LIMIT ?",
            (like, like, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --- סטטיסטיקות לדשבורד ---

def compute_stats() -> dict[str, Any]:
    """אגרגציות מהירות מטבלת ה-jobs ומאינדקס החיפוש (ללא קריאת קובצי תוצאה)."""
    conn = get_conn()
    by_status = {
        r["status"]: r["c"]
        for r in conn.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status").fetchall()
    }
    total = sum(by_status.values())
    audio_seconds = conn.execute(
        "SELECT COALESCE(SUM(duration_sec), 0) s FROM jobs WHERE status = 'done'"
    ).fetchone()["s"]
    segments = conn.execute("SELECT COUNT(*) c FROM segments_fts").fetchone()["c"]

    speaker_dist = {
        str(r["n"]): r["c"]
        for r in conn.execute(
            "SELECT num_speakers n, COUNT(*) c FROM jobs "
            "WHERE status = 'done' AND num_speakers IS NOT NULL GROUP BY num_speakers ORDER BY num_speakers"
        ).fetchall()
    }
    avg = conn.execute(
        "SELECT AVG(finished_at - started_at) p, AVG(duration_sec) d FROM jobs "
        "WHERE status = 'done' AND finished_at IS NOT NULL AND started_at IS NOT NULL"
    ).fetchone()
    perday = conn.execute(
        "SELECT DATE(created_at, 'unixepoch', 'localtime') d, COUNT(*) c FROM jobs "
        "GROUP BY d ORDER BY d DESC LIMIT 14"
    ).fetchall()
    per_day = [{"date": r["d"], "count": r["c"]} for r in reversed(perday)]

    return {
        "total": total,
        "by_status": by_status,
        "audio_seconds": round(audio_seconds or 0.0, 1),
        "segments": segments,
        "speaker_distribution": speaker_dist,
        "avg_processing_sec": round(avg["p"] or 0.0, 1),
        "avg_duration_sec": round(avg["d"] or 0.0, 1),
        "per_day": per_day,
    }


# --- אחסון תוצאות (קובץ JSON לכל משימה) ---

def result_path(job_id: str) -> Path:
    return get_settings().results_dir / f"{job_id}.json"


def save_result(job_id: str, result: dict[str, Any]) -> None:
    path = result_path(job_id)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def load_result(job_id: str) -> Optional[dict[str, Any]]:
    path = result_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
