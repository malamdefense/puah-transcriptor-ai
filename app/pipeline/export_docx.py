"""Export the transcript to a Word document (.docx) with right-to-left (Hebrew) layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

from .export import _fmt_clock, _speaker_name


def _rtl(paragraph) -> None:
    """Converts a paragraph to RTL and aligns it to the right (suitable for Hebrew in Word)."""
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    bidi.set(qn("w:val"), "1")
    pPr.append(bidi)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def _rtl_run(run) -> None:
    rPr = run._r.get_or_add_rPr()
    rtl = OxmlElement("w:rtl")
    rtl.set(qn("w:val"), "1")
    rPr.append(rtl)


def _add(doc, text: str = "", bold: bool = False, size: int | None = None):
    p = doc.add_paragraph()
    _rtl(p)
    if text:
        run = p.add_run(text)
        run.bold = bold
        _rtl_run(run)
        if size:
            run.font.size = Pt(size)
    return p


def _bullets(doc, items: list[str]) -> None:
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        _rtl(p)
        run = p.add_run(str(it))
        _rtl_run(run)


def _build_document(result: dict[str, Any], title: str, deidentified: bool) -> "Document":
    doc = Document()

    heading = doc.add_heading(title or "Transcript", level=0)
    _rtl(heading)

    if deidentified:
        note = _add(doc, "⚠ This document has been deidentified (PII)", bold=True)
        note.runs[0].font.size = Pt(11)

    dur = _fmt_clock(result.get("duration_sec", 0.0))
    n_spk = result.get("num_speakers", len(result.get("speakers", {})))
    _add(doc, f"Duration: {dur}  ·  Speakers: {n_spk}  ·  Language: {result.get('language', 'he')}", size=10)

    # Summary
    summ = result.get("summary")
    if summ and (summ.get("summary") or summ.get("key_points")):
        _add(doc, "Automatic summary", bold=True, size=14)
        if summ.get("summary"):
            _add(doc, summ["summary"])
        if summ.get("key_points"):
            _add(doc, "Key points:", bold=True)
            _bullets(doc, summ["key_points"])
        if summ.get("action_items"):
            _add(doc, "Action items and next steps:", bold=True)
            _bullets(doc, summ["action_items"])
        if summ.get("topics"):
            _add(doc, "Topics: " + " · ".join(summ["topics"]))
        if summ.get("keywords"):
            _add(doc, "Keywords: " + " · ".join(summ["keywords"]))

    # Client summary (internal)
    cs = result.get("client_summary")
    if cs:
        _add(doc, "Client summary", bold=True, size=14)
        for para in str(cs).split("\n"):
            if para.strip():
                _add(doc, para)

    # Feedback to consultant (internal)
    fb = result.get("feedback")
    if fb:
        _add(doc, "Feedback to consultant (internal)", bold=True, size=14)
        for para in str(fb).split("\n"):
            if para.strip():
                _add(doc, para)

    # Transcript
    _add(doc, "Transcript", bold=True, size=14)
    for seg in result.get("segments", []):
        name = _speaker_name(result, seg["speaker"])
        p = doc.add_paragraph()
        _rtl(p)
        prefix = p.add_run(f"[{_fmt_clock(seg['start'])}] {name}: ")
        prefix.bold = True
        _rtl_run(prefix)
        body = p.add_run(seg.get("text", ""))
        _rtl_run(body)

    return doc


def build_docx(
    result: dict[str, Any],
    path: str | Path,
    title: str,
    deidentified: bool = False,
) -> Path:
    """Builds a Word document from the transcript and saves it to path. Returns the path."""
    doc = _build_document(result, title, deidentified)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


def build_docx_bytes(result: dict[str, Any], title: str, deidentified: bool = False) -> bytes:
    """Builds a Word document and returns its bytes (for direct download)."""
    import io

    doc = _build_document(result, title, deidentified)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
