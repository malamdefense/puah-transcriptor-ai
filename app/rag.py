"""Semantic RAG — indexes the "client summary" and "feedback" columns from the Excel
sheet and performs semantic similarity search over them.

Vector store: **ChromaDB** in embedded mode (PersistentClient -> data/chroma/).
Stays fully On-Prem — no external server, no telemetry (explicitly disabled), no data
leaves the machine. Embeddings are computed locally via Ollama (bge-m3 by default) and
Chroma indexes them (HNSW, cosine distance) — scales to thousands of transcripts.

To use a separate Chroma server instead: set PUAH_CHROMA_HOST (an HttpClient is used).

Usage:
    build_from_excel("משוב_שיחות.xlsx")           # build the index from every row
    search("generated summary text", kind="summary")  # most semantically similar rows
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .config import Settings, get_settings

# Must run before importing chromadb — disables telemetry (Chroma reports to PostHog by default!).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")


class RagError(RuntimeError):
    pass


KINDS = ("summary", "feedback")
_COLLECTION = "puah_rag"


# ---------- Chroma client (embedded / server) ----------

def _client(settings: Settings):
    import chromadb  # lazy import
    from chromadb.config import Settings as ChromaSettings

    chroma_conf = ChromaSettings(anonymized_telemetry=False)
    host = getattr(settings, "chroma_host", None)
    if host:
        return chromadb.HttpClient(host=host, port=settings.chroma_port, settings=chroma_conf)
    path = settings.data_dir / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=chroma_conf)


def _collection(settings: Settings):
    """Single collection; summary/feedback are separated via metadata. Cosine distance."""
    return _client(settings).get_or_create_collection(
        name=_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


# ---------- Embedding computation via Ollama ----------

def _ollama_root(settings: Settings) -> str:
    base = settings.llm_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def embed_texts(settings: Settings, texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts via Ollama (/api/embed)."""
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
    return [[float(x) for x in v] for v in embs]


# ---------- Index building ----------

def read_excel_rows(xlsx_path: str | Path) -> list[dict[str, str]]:
    """Read the Excel sheet: column A=recording name, B=client summary, C=feedback."""
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
    """Rebuild the semantic index from every Excel row (2 entries per row: summary + feedback)."""
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

    # Reset the collection and rebuild it
    client = _client(s)
    try:
        client.delete_collection(_COLLECTION)
    except Exception:  # noqa: BLE001 — collection does not exist yet
        pass
    col = client.get_or_create_collection(name=_COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"{kind}:{i}" for i, (kind, _, _) in enumerate(items)],
        embeddings=vecs,
        documents=[t for _, _, t in items],
        metadatas=[{"kind": k, "recording": r} for k, r, _ in items],
    )
    return {
        "rows": len(rows),
        "entries": len(items),
        "recordings": len({r["recording"] for r in rows}),
        "model": s.embed_model,
        "dim": len(vecs[0]) if vecs else 0,
        "store": "chromadb",
    }


# ---------- Search ----------

def search(
    text: str, kind: str, top_k: int = 3, settings: Optional[Settings] = None
) -> list[dict[str, Any]]:
    """Return the top_k semantically closest entries (cosine) of the given kind."""
    if kind not in KINDS:
        raise RagError(f"kind לא תקין: {kind}")
    s = settings or get_settings()
    col = _collection(s)
    if col.count() == 0:
        raise RagError("האינדקס ריק — הריצו build_from_excel קודם.")

    q = embed_texts(s, [text])[0]
    res = col.query(
        query_embeddings=[q],
        n_results=max(1, top_k),
        where={"kind": kind},
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict[str, Any]] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {"recording": meta["recording"], "text": doc, "score": 1.0 - float(dist)}  # cosine
        )
    return out


def stats(settings: Optional[Settings] = None) -> dict[str, Any]:
    s = settings or get_settings()
    col = _collection(s)
    out: dict[str, Any] = {}
    for kind in KINDS:
        got = col.get(where={"kind": kind}, include=["metadatas"])
        ids = got.get("ids") or []
        if ids:
            recs = {m["recording"] for m in (got.get("metadatas") or [])}
            out[kind] = {"entries": len(ids), "recordings": len(recs)}
    return out
