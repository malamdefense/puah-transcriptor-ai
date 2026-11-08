#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract and index the Machon Puah training booklet ('תשעה ירחי מידע.docx').

Uses pure stdlib (zipfile + xml.etree.ElementTree) to read the .docx without python-docx/lxml
(which Windows Application Control may block), groups paragraphs into semantic sections/chunks,
computes normalized bge-m3 embeddings via local Ollama, saves the portable bundle under
`app/pipeline/booklet_chunks.json` + `app/pipeline/booklet_vectors.f16`, and seeds the SQLite
`booklet_chunks` + `booklet_fts` (trigram) tables in `data/puah.db`.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOCX_PATH = ROOT / "תשעה ירחי מידע.docx"
CHUNKS_JSON_PATH = ROOT / "app" / "pipeline" / "booklet_chunks.json"
VECTORS_BIN_PATH = ROOT / "app" / "pipeline" / "booklet_vectors.f16"

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_docx_paragraphs(docx_path: Path) -> list[tuple[str, str]]:
    """Return list of (style_val, text) for every non-empty paragraph in word/document.xml."""
    with zipfile.ZipFile(docx_path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    body = root.find(f"{W_NS}body")
    if body is None:
        return []

    out: list[tuple[str, str]] = []
    for elem in body:
        if elem.tag == f"{W_NS}p":
            style_el = elem.find(f"./{W_NS}pPr/{W_NS}pStyle")
            style = style_el.attrib.get(f"{W_NS}val", "") if style_el is not None else ""
            texts = [t.text for t in elem.iter(f"{W_NS}t") if t.text]
            line = re.sub(r"\s+", " ", "".join(texts)).strip()
            if line:
                out.append((style, line))
        elif elem.tag == f"{W_NS}tbl":
            # Extract table rows as structured lines
            rows_txt = []
            for row in elem.iter(f"{W_NS}tr"):
                cells = []
                for cell in row.iter(f"{W_NS}tc"):
                    ct = " ".join(
                        re.sub(r"\s+", " ", "".join(t.text for t in p.iter(f"{W_NS}t") if t.text)).strip()
                        for p in cell.iter(f"{W_NS}p")
                    ).strip()
                    if ct:
                        cells.append(ct)
                if cells:
                    rows_txt.append(" | ".join(cells))
            if rows_txt:
                out.append(("Table", "\n".join(rows_txt)))
    return out


def _is_heading(style: str, text: str) -> bool:
    if "\n" in text:  # Table
        return False
    if style.lower().startswith("heading") or style in ("1", "2", "3", "4", "Title", "Subtitle"):
        return True
    # Short standalone title lines without sentence-ending period (unless abbreviation like I.V.F.)
    words = text.split()
    if len(words) <= 9 and len(text) <= 75:
        if text.startswith("הלכתית:") or text.startswith("חשוב ") or text.startswith("לדוגמה:"):
            return False
        if not text.endswith(".") or re.search(r"[A-Za-z]\.$", text):
            return True
    return False


_PAREN_EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\./ ]{1,35}[A-Za-z0-9]")


def _extract_terms(section: str, title: str, text: str) -> list[str]:
    terms: list[str] = []
    for src in (title, section):
        clean = re.sub(r"[\(\)\[\]–—\-:]", " ", src)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean and len(clean) >= 2:
            terms.append(clean)
    for m in _PAREN_EN_RE.finditer(f"{title} {text}"):
        tok = m.group(0).strip(" .-")
        if len(tok) >= 2:
            terms.append(tok)
    # Deduplicate preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:12]


def build_chunks(paragraphs: list[tuple[str, str]]) -> list[dict]:
    """Skip TOC and chunk the body into ~300-850 char semantic sections with title & section."""
    # Find the second occurrence of 'מבוא – מבנה אנטומי ותהליך פיזיולוגי' (start of body after TOC)
    start_idx = 0
    intro_hits = [
        i for i, (_, t) in enumerate(paragraphs)
        if "מבוא" in t and "מבנה אנטומי" in t
    ]
    if len(intro_hits) >= 2:
        start_idx = intro_hits[1]
    elif intro_hits:
        start_idx = intro_hits[0]

    body = paragraphs[start_idx:]
    raw_sections: list[tuple[str, str, list[str]]] = []
    curr_major = "מבוא – מבנה אנטומי ותהליך פיזיולוגי"
    curr_title = curr_major
    curr_paras: list[str] = []

    for style, text in body:
        if _is_heading(style, text):
            if curr_paras:
                raw_sections.append((curr_major, curr_title, curr_paras))
                curr_paras = []
            else:
                # Two consecutive headings: first is major section, second is sub-section title
                curr_major = curr_title
            curr_title = text
        else:
            curr_paras.append(text)
    if curr_paras:
        raw_sections.append((curr_major, curr_title, curr_paras))

    chunks: list[dict] = []
    MAX_CHARS = 850
    MIN_CHARS = 160

    for major, title, paras in raw_sections:
        buf: list[str] = []
        buf_len = 0
        part_no = 1
        for p in paras:
            # Always keep 'הלכתית:' attached to the preceding paragraph if reasonable
            is_halachic = p.startswith("הלכתית")
            if buf and (buf_len + len(p) > MAX_CHARS) and not (is_halachic and buf_len < 1100):
                chunk_text = "\n".join(buf).strip()
                if len(chunk_text) >= 30:
                    ctitle = title if part_no == 1 else f"{title} (המשך {part_no})"
                    chunks.append({
                        "id": len(chunks) + 1,
                        "section": major,
                        "title": ctitle,
                        "terms": ", ".join(_extract_terms(major, title, chunk_text)),
                        "text": chunk_text,
                    })
                    part_no += 1
                buf = [p]
                buf_len = len(p)
            else:
                buf.append(p)
                buf_len += len(p) + 1
        if buf:
            chunk_text = "\n".join(buf).strip()
            if chunks and len(chunk_text) < MIN_CHARS and chunks[-1]["title"].startswith(title):
                chunks[-1]["text"] += "\n" + chunk_text
                chunks[-1]["terms"] = ", ".join(
                    _extract_terms(major, title, chunks[-1]["text"])
                )
            elif len(chunk_text) >= 20:
                ctitle = title if part_no == 1 else f"{title} (המשך {part_no})"
                chunks.append({
                    "id": len(chunks) + 1,
                    "section": major,
                    "title": ctitle,
                    "terms": ", ".join(_extract_terms(major, title, chunk_text)),
                    "text": chunk_text,
                })

    # Re-number sequentially
    for i, c in enumerate(chunks, start=1):
        c["id"] = i
    return chunks


def embed_texts_ollama(texts: list[str], base_url: str = "http://localhost:11434", model: str = "bge-m3") -> np.ndarray:
    """Compute L2-normalized embeddings via local Ollama /api/embed (or /api/embeddings)."""
    vecs: list[list[float]] = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/embed",
            data=json.dumps({"model": model, "input": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        embs = data.get("embeddings") or []
        if len(embs) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(embs)}")
        vecs.extend(embs)

    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    arr = arr / norms
    return arr.astype(np.float16)


def main() -> int:
    if not DOCX_PATH.exists():
        print(f"Booklet not found: {DOCX_PATH}", file=sys.stderr)
        return 1

    paras = extract_docx_paragraphs(DOCX_PATH)
    print(f"Extracted {len(paras)} paragraphs/tables from {DOCX_PATH.name}")
    chunks = build_chunks(paras)
    print(f"Built {len(chunks)} semantic chunks (avg {sum(len(c['text']) for c in chunks)//max(1,len(chunks))} chars)")

    embed_inputs = [
        f"{c['section']} — {c['title']}\nמונחים: {c['terms']}\n{c['text']}"
        for c in chunks
    ]
    print("Computing bge-m3 embeddings via local Ollama...")
    vecs_f16 = embed_texts_ollama(embed_inputs)
    print(f"Computed embeddings matrix: shape={vecs_f16.shape}, dtype={vecs_f16.dtype} ({vecs_f16.nbytes / 1024:.1f} KB)")

    CHUNKS_JSON_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "source": DOCX_PATH.name,
                "embedding_model": "bge-m3",
                "dim": int(vecs_f16.shape[1]),
                "count": len(chunks),
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    VECTORS_BIN_PATH.write_bytes(vecs_f16.tobytes())
    print(f"Saved {CHUNKS_JSON_PATH} and {VECTORS_BIN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
