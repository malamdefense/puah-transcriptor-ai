"""Hybrid SQLite Retrieval (FTS5 trigram + pre-computed bge-m3 vectors + RRF) for the training booklet.

Connects 'תשעה ירחי מידע.docx' (Machon Puah's professional medical-halachic training manual) to
the summarization and feedback pipeline using `data/puah.db`:
  1. `booklet_chunks` table in `puah.db`: stores each semantic section (`id`, `section`, `title`,
     `terms`, `text`, and `embedding` BLOB of 1024-dim float16 `bge-m3` vectors).
  2. `booklet_fts` virtual table in `puah.db`: `USING fts5(..., tokenize='trigram case_sensitive 0')`,
     handling Hebrew attached prefixes (`ו/ה/ב/כ/ל/מ/ש`) natively via 3-gram substring matching.
  3. Reciprocal Rank Fusion (RRF): combines the FTS5 trigram BM25 ranking and the `bge-m3` cosine
     similarity ranking in < 1 ms, gated so retrieval only triggers when domain terms from the
     booklet actually appear in the call transcript.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings, get_settings
from ..db import get_conn, init_db

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_JSON_PATH = Path(__file__).resolve().parent / "booklet_chunks.json"
VECTORS_BIN_PATH = Path(__file__).resolve().parent / "booklet_vectors.f16"

BOOKLET_HEADER = "=== מידע מקצועי רלוונטי מחוברת ההדרכה (תשעה ירחי מידע — מכון פוע\"ה) ==="
BOOKLET_FOOTER = "=== סוף מידע מחוברת ההדרכה ==="
BOOKLET_INSTRUCTION = (
    "היעזר במידע המקצועי שלהלן אך ורק לצורך דיוק בטרמינולוגיה, באיות המונחים ובהבנת ההקשר "
    "ההלכתי-רפואי של הדברים שנאמרו בשיחה. אין להוסיף לסיכום עובדות מהחוברת שלא עלו בשיחה עצמה."
)

_seed_lock = threading.Lock()
_seeded_dbs: dict[tuple[str, int], int] = {}
_chunks_cache: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
_vec_cache: dict[tuple[str, int], tuple[list[int], np.ndarray]] = {}
_vocab_cache: list[tuple[str, str]] | None = None


def _db_cache_key(conn: sqlite3.Connection) -> tuple[str, int] | None:
    """Return a unique cache key `(db_file, id(conn))` for a SQLite connection without disk stat."""
    db_list = conn.execute("PRAGMA database_list;").fetchall()
    db_file = str(db_list[0][2]) if db_list else ""
    if not db_file or db_file == ":memory:":
        return None
    return (db_file, id(conn))

# Generic words that appear in booklet headings but are too common on their own to trigger RAG
_GENERIC_STOPWORDS = frozenset({
    "מבוא", "הקדמה", "רקע", "כללי", "סיכום", "בדיקה", "בדיקות", "סוגי", "תהליך", "מערכת",
    "אמצעי", "אמצעים", "שונות", "נוספות", "הערות", "טיפול", "טיפולים", "בעיות", "שאלות",
    "המשך", "חלק", "פרק", "אישה", "האישה", "גבר", "הגבר", "זוג", "הזוג", "שלב", "שלבים",
})


def _load_bundle() -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """Load pre-computed chunks and float16 bge-m3 vectors from disk."""
    if not CHUNKS_JSON_PATH.exists():
        return [], None
    meta = json.loads(CHUNKS_JSON_PATH.read_text(encoding="utf-8"))
    chunks = meta.get("chunks") or []
    dim = int(meta.get("dim") or 1024)
    vecs: np.ndarray | None = None
    if VECTORS_BIN_PATH.exists() and chunks:
        raw = np.frombuffer(VECTORS_BIN_PATH.read_bytes(), dtype=np.float16)
        if raw.size == len(chunks) * dim:
            vecs = raw.reshape(len(chunks), dim)
    return chunks, vecs


def ensure_booklet_index(conn: sqlite3.Connection | None = None) -> int:
    """Ensure `booklet_chunks` and `booklet_fts` exist in `puah.db` and are populated.

    Returns the number of indexed booklet chunks in SQLite. Fast (< 0.01ms when already seeded).
    """
    if conn is None:
        init_db()
        conn = get_conn()

    cid_key = _db_cache_key(conn)
    if cid_key is not None:
        cached_n = _seeded_dbs.get(cid_key)
        if cached_n:
            return cached_n

    with _seed_lock:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS booklet_chunks (
                id          INTEGER PRIMARY KEY,
                section     TEXT NOT NULL,
                title       TEXT NOT NULL,
                terms       TEXT NOT NULL,
                text        TEXT NOT NULL,
                embedding   BLOB
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS booklet_fts USING fts5(
                title,
                section,
                terms,
                text,
                tokenize='trigram case_sensitive 0'
            );
            """
        )
        row = conn.execute("SELECT COUNT(*) FROM booklet_chunks;").fetchone()
        existing = int(row[0]) if row else 0
        if existing > 0:
            if cid_key is not None:
                _seeded_dbs[cid_key] = existing
            return existing

        chunks, vecs = _load_bundle()
        if not chunks:
            return 0

        conn.execute("BEGIN IMMEDIATE;")
        try:
            for idx, c in enumerate(chunks):
                cid = int(c["id"])
                emb_blob = vecs[idx].tobytes() if vecs is not None and idx < len(vecs) else None
                conn.execute(
                    """INSERT OR REPLACE INTO booklet_chunks (id, section, title, terms, text, embedding)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (cid, c["section"], c["title"], c["terms"], c["text"], emb_blob),
                )
                conn.execute(
                    """INSERT INTO booklet_fts (rowid, title, section, terms, text)
                       VALUES (?, ?, ?, ?, ?)""",
                    (cid, c["title"], c["section"], c["terms"], c["text"]),
                )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

        if cid_key is not None:
            _seeded_dbs[cid_key] = len(chunks)
            _vec_cache.pop(cid_key, None)
            _chunks_cache.pop(cid_key, None)
        return len(chunks)


def _get_domain_vocabulary(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Build a list of (search_stem, canonical_term) from the booklet titles, terms, and core glossary."""
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache

    rows = conn.execute("SELECT title, terms FROM booklet_chunks;").fetchall()
    candidates: set[str] = set()
    for r in rows:
        for raw_field in (r["title"], r["terms"]):
            for part in re.split(r"[,،/|\(\)\[\]–—\-:]", raw_field or ""):
                clean = re.sub(r"\s+", " ", part).strip(" .\"'״׳")
                if not clean or len(clean) < 3:
                    continue
                if clean in _GENERIC_STOPWORDS:
                    continue
                if clean.startswith("המשך "):
                    continue
                candidates.add(clean)
                # Also add individual meaningful Hebrew/English words (>=4 chars) from multi-word titles
                for w in clean.split():
                    w_clean = w.strip(" .\"'״׳")
                    if len(w_clean) >= 4 and w_clean not in _GENERIC_STOPWORDS:
                        candidates.add(w_clean)

    # Core halachic & fertility counseling terms that frequently appear in calls
    core_terms = (
        "שבעה נקיים", "ז' נקיים", "הפסק טהרה", "מוך דחוק", "בדיקה פנימית", "בדיקת רופא",
        "חופת נידה", "דם חימוד", "מורה הוראה", "כתמים", "הכתמות", "דימום", "וסת", "מחזור",
        "ביוץ", "מקווה", "טבילה", "בלנית", "שקיפות עורפית", "חלבון עוברי", "תבחין משולש",
        "תבחין מרובע", "סקירת מערכות", "העמסת סוכר", "סוכרת הריון", "סיסי שליה", "דיקור מי שפיר",
        "מי שפיר", "שליה", "שליית פתח", "היפרדות שליה", "רעלת הריון", "הריון חוץ רחמי",
        "הפלה", "גרידה", "ציטוטק", "היסטרוסקופיה", "לפרוסקופיה", "צילום רחם", "מעקב זקיקים",
        "זקיקים", "שחלות", "חצוצרות", "רירית הרחם", "צוואר הרחם", "אנדומטריוזיס", "שחלות פוליציסטיות",
        "גלולות", "סרזט", "מירנה", "קיילינה", "ג'יידס", "נוברינג", "אברה", "התקן", "התקן תוך רחמי",
        "פרוגסטרון", "אסטרוגן", "דופסטון", "פרימולוט", "אוטרוגסטן", "אנדומטרין", "איקקלומין",
        "לטרוזול", "גונאל", "אוביטרל", "דקפפטיל", "צטרוטייד", "מנופור", "פרגובריס",
        "הזרעה", "הזרעה תוך רחמית", "הפריה", "הפריה חוץ גופית", "שאיבת ביציות", "החזרת עוברים",
        "הקפאת ביציות", "תרומת ביצית", "תרומת זרע", "פונדקאות", "אבחון גנטי", "סקר גנטי",
        "דור ישרים", "קריוטיפ", "שבב גנטי", "אקסום", "טריזומיה", "תסמונת דאון",
        "בדיקת זרע", "תאי זרע", "וריקוצלה", "אזוספרמיה", "השבחת זרע", "הוצאת זרע",
        "זירוז לידה", "ניתוח קיסרי", "מצג עכוז", "היפוך", "משכב לידה", "ירידת מים", "הנקה",
        "גיל המעבר", "מנופאוזה", "פאפ", "קולפוסקופיה", "ספקולום", "פיפל",
        "ivf", "iui", "icsi", "pgd", "pgt", "nipt", "cvs", "hsg", "iud", "cmv", "gbs", "fsh", "lh", "hcg",
    )
    for ct in core_terms:
        candidates.add(ct)

    vocab: list[tuple[str, str]] = []
    for term in sorted(candidates, key=len, reverse=True):
        # Strip leading ה from Hebrew terms >= 5 chars so "ההכתמות" / "בהכתמות" matches stem "כתמות"
        stem = term.lower()
        if len(stem) >= 5 and stem.startswith("ה") and not stem.startswith("הי") and " " not in stem:
            stem = stem[1:]
        if len(stem) >= 3:
            vocab.append((stem, term))

    _vocab_cache = vocab
    return vocab


def extract_matched_terms(text: str, conn: sqlite3.Connection | None = None) -> list[str]:
    """Return canonical domain terms from the booklet vocabulary that appear in `text`."""
    if not text or not text.strip():
        return []
    if conn is None:
        conn = get_conn()
    ensure_booklet_index(conn)
    vocab = _get_domain_vocabulary(conn)
    low = text.lower()
    matched: list[str] = []
    seen: set[str] = set()
    for stem, canon in vocab:
        if canon.lower() in seen:
            continue
        if stem in low:
            # For short ASCII acronyms (<=4 chars like iui, ivf, hsg, cmv), require word boundary
            if stem.isascii() and len(stem) <= 4:
                if not re.search(rf"(?<![a-z0-9]){re.escape(stem)}(?![a-z0-9])", low):
                    continue
            seen.add(canon.lower())
            matched.append(canon)
    return matched


def _fts_trigram_search(conn: sqlite3.Connection, terms: list[str], limit: int = 15) -> list[tuple[int, float]]:
    """Run SQLite FTS5 trigram BM25 search over `booklet_fts` for the given terms."""
    # Split phrases into distinct words >= 3 chars so FTS5 trigram intersects 2-4 trigrams per word
    # (< 0.5ms) and matches inflected Hebrew words even when prefixes are attached to individual words.
    clauses: list[str] = []
    seen_toks: set[str] = set()
    for t in terms[:12]:
        clean = re.sub(r'["\'\(\)\[\]\*\^\-\+:]', " ", t).strip()
        for w in clean.split():
            w_low = w.lower()
            if len(w_low) >= 3 and w_low not in seen_toks and w_low not in _GENERIC_STOPWORDS:
                seen_toks.add(w_low)
                clauses.append(f'"{w}"')
                if len(clauses) >= 10:
                    break
        if len(clauses) >= 10:
            break
    if not clauses:
        return []

    match_expr = " OR ".join(clauses)
    try:
        rows = conn.execute(
            """
            SELECT rowid, bm25(booklet_fts, 5.0, 2.0, 4.0, 1.0) AS score
            FROM booklet_fts
            WHERE booklet_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (match_expr, limit),
        ).fetchall()
        # bm25() in SQLite FTS5 returns negative values (more negative = better match)
        return [(int(r["rowid"]), float(-r["score"])) for r in rows]
    except sqlite3.OperationalError as exc:
        log.debug("FTS5 trigram query fallback (%s)", exc)
        return []


def _load_db_chunks(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    """Load and cache `booklet_chunks` rows (with pre-lowercased fields for <0.1ms matching)."""
    cid_key = _db_cache_key(conn)
    if cid_key is not None and cid_key in _chunks_cache:
        return _chunks_cache[cid_key]

    rows = conn.execute("SELECT id, section, title, terms, text FROM booklet_chunks;").fetchall()
    by_id: dict[int, dict[str, Any]] = {}
    for r in rows:
        d = dict(r)
        d["_title_low"] = f"{d.get('title', '')} {d.get('terms', '')}".lower()
        d["_text_low"] = str(d.get("text", "")).lower()
        by_id[int(r["id"])] = d
    if by_id and cid_key is not None:
        _chunks_cache[cid_key] = by_id
    return by_id


def _load_db_vectors(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray] | None:
    """Load the pre-computed float16 embeddings from `booklet_chunks` in `puah.db` (cached per DB)."""
    cid_key = _db_cache_key(conn)
    if cid_key is not None and cid_key in _vec_cache:
        return _vec_cache[cid_key]

    rows = conn.execute(
        "SELECT id, embedding FROM booklet_chunks WHERE embedding IS NOT NULL ORDER BY id ASC;"
    ).fetchall()
    if not rows:
        return None

    ids: list[int] = []
    vecs: list[np.ndarray] = []
    for r in rows:
        blob = r["embedding"]
        if not blob:
            continue
        v = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
        ids.append(int(r["id"]))
        vecs.append(v)
    if not vecs:
        return None

    mat = np.vstack(vecs)
    if cid_key is not None:
        _vec_cache[cid_key] = (ids, mat)
    return ids, mat


def _build_precomputed_query_vector(
    matched_terms: list[str],
    fts_ranked: list[tuple[int, float]],
    by_id: dict[int, dict[str, Any]],
    ids: list[int],
    mat: np.ndarray,
) -> np.ndarray | None:
    """Construct a 1024-dim bge-m3 query vector in <0.2ms from pre-computed vectors in `puah.db`.

    Weights each chunk's pre-computed `bge-m3` embedding by term specificity (title/terms hits
    + FTS5 trigram seed ranks) and L2-normalizes the resulting vector so cosine similarity
    `mat @ q_vec` surfaces semantically related chunks across the booklet without network I/O.
    """
    if not ids or mat.size == 0:
        return None
    id_to_row = {cid: idx for idx, cid in enumerate(ids)}
    weights = np.zeros(len(ids), dtype=np.float32)

    for mt in matched_terms[:16]:
        mt_low = mt.lower()
        spec = max(1.0, len(mt_low) / 4.0)
        for cid, chunk in by_id.items():
            row_idx = id_to_row.get(cid)
            if row_idx is None:
                continue
            title_terms = chunk.get("_title_low") or f"{chunk.get('title', '')} {chunk.get('terms', '')}".lower()
            if mt_low in title_terms:
                weights[row_idx] += 3.0 * spec
            elif mt_low in (chunk.get("_text_low") or chunk.get("text", "").lower()):
                weights[row_idx] += 1.0 * spec

    for rank, (cid, _bm25) in enumerate(fts_ranked[:8], start=1):
        row_idx = id_to_row.get(cid)
        if row_idx is not None:
            weights[row_idx] += 2.0 / (rank + 1)

    if not np.any(weights > 0):
        return None

    q_vec = weights @ mat
    norm = float(np.linalg.norm(q_vec))
    return (q_vec / norm).astype(np.float32) if norm > 0 else None


def _embed_query_ollama(query_text: str, settings: Settings, timeout: float = 2.5) -> np.ndarray | None:
    """Embed `query_text` using local Ollama `bge-m3` when live embedding is requested."""
    if not getattr(settings, "booklet_rag_vector_enabled", True):
        return None
    if not getattr(settings, "booklet_rag_live_embed", False):
        return None
    base = (settings.llm_base_url or "http://localhost:11434/v1").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    url = f"{base}/api/embed"
    payload = json.dumps({"model": settings.embed_model or "bge-m3", "input": [query_text[:1500]]}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        embs = data.get("embeddings") or []
        if not embs:
            return None
        v = np.asarray(embs[0], dtype=np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v
    except Exception:  # noqa: BLE001 — offline / unit tests degrade gracefully
        return None


def reciprocal_rank_fusion(
    fts_ranked: list[tuple[int, float]],
    vec_ranked: list[tuple[int, float]],
    title_bonuses: dict[int, float] | None = None,
    *,
    k: int = 60,
    w_fts: float = 1.2,
    w_vec: float = 1.0,
) -> list[tuple[int, float]]:
    """Fuse FTS5 trigram BM25 ranks and bge-m3 vector cosine ranks via Reciprocal Rank Fusion (RRF)."""
    scores: dict[int, float] = {}
    for rank, (doc_id, _raw) in enumerate(fts_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_fts / (k + rank)
    for rank, (doc_id, cos_sim) in enumerate(vec_ranked, start=1):
        if cos_sim >= 0.38:  # filter out weak semantic matches
            scores[doc_id] = scores.get(doc_id, 0.0) + w_vec / (k + rank)
    if title_bonuses:
        for doc_id, bonus in title_bonuses.items():
            if doc_id in scores or bonus >= 0.015:
                scores[doc_id] = scores.get(doc_id, 0.0) + bonus
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def search_booklet(
    query_text: str,
    settings: Settings | None = None,
    *,
    top_k: int | None = None,
    conn: sqlite3.Connection | None = None,
    embed_fn: Any = None,
) -> list[dict[str, Any]]:
    """Hybrid search over `booklet_chunks` + `booklet_fts` in `puah.db` using FTS5 trigram + bge-m3 + RRF.

    Returns up to `top_k` chunk dicts: `[{"id", "section", "title", "terms", "text", "score", "matched_terms"}]`.
    Returns `[]` when the query contains no relevant domain terms (relevance gating).
    """
    settings = settings or get_settings()
    if not getattr(settings, "booklet_rag_enabled", True):
        return []
    k_out = top_k or int(getattr(settings, "booklet_rag_top_k", 3) or 3)

    if conn is None:
        init_db()
        conn = get_conn()
    n_chunks = ensure_booklet_index(conn)
    if n_chunks == 0:
        return []

    matched_terms = extract_matched_terms(query_text, conn)
    if not matched_terms:
        return []

    # 1. Lexical leg: SQLite FTS5 trigram BM25
    fts_ranked = _fts_trigram_search(conn, matched_terms, limit=20)

    # 2. Direct title/term hit bonus
    by_id = _load_db_chunks(conn)
    title_bonuses: dict[int, float] = {}
    for cid, c in by_id.items():
        title_low = c.get("_title_low") or f"{c['title']} {c['terms']}".lower()
        hits = sum(1 for mt in matched_terms if mt.lower() in title_low)
        if hits:
            title_bonuses[cid] = min(0.03, 0.012 * hits)

    # 3. Vector leg: bge-m3 cosine similarity over pre-computed embeddings in puah.db
    vec_ranked: list[tuple[int, float]] = []
    if getattr(settings, "booklet_rag_vector_enabled", True):
        vec_data = _load_db_vectors(conn)
        if vec_data is not None:
            ids, mat = vec_data
            focused_query = "מונחים בשיחה: " + ", ".join(matched_terms[:10]) + "\n" + query_text[:1000]
            if callable(embed_fn):
                q_vec = embed_fn(focused_query)
            else:
                q_vec = _embed_query_ollama(focused_query, settings)
                if q_vec is None:
                    q_vec = _build_precomputed_query_vector(matched_terms, fts_ranked, by_id, ids, mat)
            if q_vec is not None and q_vec.shape[0] == mat.shape[1]:
                sims = mat @ q_vec
                top_idx = np.argsort(-sims)[:20]
                vec_ranked = [(ids[int(i)], float(sims[int(i)])) for i in top_idx if float(sims[int(i)]) > 0.35]

    # 4. Reciprocal Rank Fusion (RRF)
    fused = reciprocal_rank_fusion(fts_ranked, vec_ranked, title_bonuses)
    results: list[dict[str, Any]] = []
    for cid, score in fused[:k_out]:
        chunk = by_id.get(cid)
        if not chunk:
            continue
        item = {k: v for k, v in chunk.items() if not k.startswith("_")}
        item["score"] = round(score, 5)
        item["matched_terms"] = matched_terms[:10]
        results.append(item)
    return results


def retrieve_booklet_context(settings: Settings, result: dict[str, Any], *, max_chars: int = 1800) -> str:
    """Build the formatted reference block from 'תשעה ירחי מידע' for a call transcript, or '' if none applies."""
    if not getattr(settings, "booklet_rag_enabled", True):
        return ""
    try:
        segs = result.get("segments") or []
        seg_text = " ".join(str(s.get("text") or "") for s in segs if s.get("text"))
        summ = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        kw_text = " ".join(str(k) for k in (summ.get("keywords") or []) + (summ.get("topics") or []))
        query = f"{kw_text}\n{seg_text}".strip()
        if not query:
            return ""

        chunks = search_booklet(query, settings)
        if not chunks:
            return ""

        bullets: list[str] = []
        used_chars = 0
        for c in chunks:
            header = f"• [{c['section']} — {c['title']}]"
            body = c["text"].strip()
            entry = f"{header}: {body}"
            if bullets and used_chars + len(entry) > max_chars:
                break
            bullets.append(entry)
            used_chars += len(entry) + 1

        if not bullets:
            return ""

        log.info(
            "Booklet RAG retrieved %d chunk(s) for matched terms %s",
            len(bullets),
            chunks[0].get("matched_terms", [])[:6],
        )
        return (
            f"{BOOKLET_HEADER}\n"
            f"{BOOKLET_INSTRUCTION}\n"
            + "\n".join(bullets)
            + f"\n{BOOKLET_FOOTER}"
        )
    except Exception as exc:  # noqa: BLE001 — fail-safe: never fail summarization if RAG has an error
        log.warning("Booklet RAG retrieval skipped (%s)", type(exc).__name__)
        return ""
