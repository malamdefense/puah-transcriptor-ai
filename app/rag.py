"""RAG סמנטי — אינדוקס "סיכום ללקוח" ו"משוב" מקובץ האקסל, וחיפוש דמיון סמנטי.

ה-embeddings מחושבים מקומית דרך Ollama (ברירת מחדל bge-m3 — רב-לשוני עם עברית טובה),
והאינדקס נשמר ב-SQLite (data/rag.db). שום נתון אינו יוצא מהשרת.

שימוש:
    build_from_excel("משוב_שיחות.xlsx")          # בונה את האינדקס מכל שורה
    search("טקסט סיכום שנוצר", kind="summary")   # מחזיר את הרשומות הדומות ביותר
"""

from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import Settings, get_settings


class RagError(RuntimeError):
    pass


KINDS = ("summary", "feedback")

SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_entries (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,     -- summary | feedback
    recording TEXT NOT NULL,     -- שם ההקלטה (עמודה A באקסל)
    text      TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rag_kind ON rag_entries (kind);
"""


def _db(settings: Optional[Settings] = None) -> sqlite3.Connection:
    s = settings or get_settings()
    path = s.data_dir / "rag.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _ollama_root(settings: Settings) -> str:
    base = settings.llm_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def embed_texts(settings: Settings, texts: list[str]) -> list[list[float]]:
    """מחשב embeddings לרשימת טקסטים דרך Ollama (/api/embed)."""
    if not texts:
        return []
    url = _ollama_root(settings) + "/api/embed"
    body = json.dumps({"model": settings.embed_model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.llm_timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RagError(
            f"לא ניתן להגיע ל-Ollama ב-{url} ({exc}). ודאו שהוא פועל ושהמודל "
            f"'{settings.embed_model}' הורד:  ollama pull {settings.embed_model}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RagError(f"שגיאה בחישוב embeddings: {exc}") from exc

    embs = data.get("embeddings")
    if not isinstance(embs, list) or len(embs) != len(texts):
        raise RagError(f"תשובת embeddings בלתי צפויה: {str(data)[:200]}")
    return embs


def _unit(vec: Any) -> np.ndarray:
    """מנרמל לווקטור יחידה (כדי שמכפלה סקלרית = cosine similarity)."""
    a = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n else a


# ---------- בניית האינדקס ----------

def read_excel_rows(xlsx_path: str | Path) -> list[dict[str, str]]:
    """קורא את האקסל: עמודה A=שם הקלטה, B=סיכום ללקוח, C=משוב."""
    import openpyxl

    ws = openpyxl.load_workbook(xlsx_path, data_only=True).worksheets[0]
    rows: list[dict[str, str]] = []
    for r in range(2, ws.max_row + 1):
        name = ws.cell(r, 1).value
        if not name or not str(name).strip():
            continue
        rows.append(
            {
                "recording": str(name).strip(),
                "summary": str(ws.cell(r, 2).value or "").strip(),
                "feedback": str(ws.cell(r, 3).value or "").strip(),
            }
        )
    return rows


def build_from_excel(xlsx_path: str | Path, settings: Optional[Settings] = None) -> dict[str, Any]:
    """בונה מחדש את האינדקס הסמנטי מכל שורה באקסל (2 רשומות לשורה: סיכום + משוב)."""
    s = settings or get_settings()
    rows = read_excel_rows(xlsx_path)
    if not rows:
        raise RagError("לא נמצאו שורות באקסל")

    items: list[tuple[str, str, str]] = []  # (kind, recording, text)
    for row in rows:
        for kind in KINDS:
            if row[kind]:
                items.append((kind, row["recording"], row[kind]))

    vecs = embed_texts(s, [t for _, _, t in items])

    conn = _db(s)
    conn.execute("DELETE FROM rag_entries")
    for (kind, rec, text), v in zip(items, vecs):
        a = _unit(v)
        conn.execute(
            "INSERT INTO rag_entries (kind, recording, text, dim, embedding) VALUES (?,?,?,?,?)",
            (kind, rec, text, int(a.shape[0]), a.tobytes()),
        )
    return {
        "rows": len(rows),
        "entries": len(items),
        "recordings": len({r["recording"] for r in rows}),
        "model": s.embed_model,
        "dim": int(len(vecs[0])) if vecs else 0,
    }


# ---------- חיפוש ----------

def search(
    text: str, kind: str, top_k: int = 3, settings: Optional[Settings] = None
) -> list[dict[str, Any]]:
    """מחזיר את top_k הרשומות הדומות ביותר סמנטית (cosine), מסוג kind."""
    if kind not in KINDS:
        raise RagError(f"kind לא תקין: {kind}")
    s = settings or get_settings()
    conn = _db(s)
    rows = conn.execute(
        "SELECT recording, text, embedding FROM rag_entries WHERE kind = ?", (kind,)
    ).fetchall()
    if not rows:
        raise RagError(f"האינדקס ריק עבור kind={kind} — הריצו build_from_excel קודם.")

    q = _unit(embed_texts(s, [text])[0])
    mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = mat @ q  # כל הווקטורים מנורמלים -> מכפלה סקלרית = cosine
    order = np.argsort(-sims)[: max(1, top_k)]
    return [
        {"recording": rows[i]["recording"], "text": rows[i]["text"], "score": float(sims[i])}
        for i in order
    ]


def stats(settings: Optional[Settings] = None) -> dict[str, Any]:
    conn = _db(settings)
    out = {
        r["kind"]: {"entries": r["c"], "recordings": r["n"]}
        for r in conn.execute(
            "SELECT kind, COUNT(*) c, COUNT(DISTINCT recording) n FROM rag_entries GROUP BY kind"
        ).fetchall()
    }
    return out
