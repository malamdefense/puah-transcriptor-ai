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
from .language import LANGUAGE_NAMES_HE
from .output_guard import sanitize_caller_text


def _rtl(paragraph) -> None:
    """Make a paragraph RTL and align it to the right (valid for Hebrew in Word)."""
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


def _language_label(result: dict[str, Any]) -> str:
    """Hebrew names of the call's languages ("אנגלית + עברית"); old results have only `language`."""
    codes = result.get("languages") or [result.get("language", "he")]
    return " + ".join(LANGUAGE_NAMES_HE.get(c, c) for c in codes)


def _build_document(
    result: dict[str, Any], title: str, deidentified: bool, templates: dict[str, str] | None = None
) -> "Document":
    doc = Document()

    heading = doc.add_heading(title or "תמלול", level=0)
    _rtl(heading)

    if deidentified:
        note = _add(doc, "⚠ מסמך זה עבר הסרת פרטים מזהים (PII)", bold=True)
        note.runs[0].font.size = Pt(11)

    dur = _fmt_clock(result.get("duration_sec", 0.0))
    n_spk = result.get("num_speakers", len(result.get("speakers", {})))
    _add(doc, f"משך: {dur}  ·  דוברים: {n_spk}  ·  שפה: {_language_label(result)}", size=10)

    # Structured summary
    summ = result.get("summary")
    if summ and (summ.get("summary") or summ.get("key_points")):
        _add(doc, "סיכום אוטומטי", bold=True, size=14)
        if summ.get("summary"):
            _add(doc, summ["summary"])
        if summ.get("key_points"):
            _add(doc, "נקודות עיקריות:", bold=True)
            _bullets(doc, summ["key_points"])
        if summ.get("action_items"):
            _add(doc, "פעולות והמשך טיפול:", bold=True)
            _bullets(doc, summ["action_items"])
        if summ.get("topics"):
            _add(doc, "נושאים: " + " · ".join(summ["topics"]))
        if summ.get("keywords"):
            _add(doc, "מילות מפתח: " + " · ".join(summ["keywords"]))

    # Client summary (PUAH product). The original-transcript document must never carry a
    # redaction placeholder (results from before the guard): a no-op for clean text. The
    # de-identified document keeps its tags — they are the point of that copy.
    cs = result.get("client_summary")
    fb = result.get("feedback")
    if not deidentified:
        # With the raw task prompts, a copied template line comes back verbatim (same as the Excel)
        tpl = templates or {}
        cs = sanitize_caller_text(str(cs), template=tpl.get("client_summary"))[0] if cs else cs
        fb = sanitize_caller_text(str(fb), template=tpl.get("feedback"))[0] if fb else fb
    if cs:
        _add(doc, "סיכום ללקוח", bold=True, size=14)
        for para in str(cs).split("\n"):
            if para.strip():
                _add(doc, para)

    # Consultant feedback (internal)
    if fb:
        _add(doc, "משוב ליועץ (פנימי)", bold=True, size=14)
        for para in str(fb).split("\n"):
            if para.strip():
                _add(doc, para)

    # Transcript
    _add(doc, "תמלול", bold=True, size=14)
    for seg in result.get("segments", []):
        name = _speaker_name(result, seg["speaker"])
        p = doc.add_paragraph()
        # English segments (language detected from the script) read left to right
        ltr = seg.get("language") == "en"
        if ltr:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        else:
            _rtl(p)
        prefix = p.add_run(f"[{_fmt_clock(seg['start'])}] {name}: ")
        prefix.bold = True
        if ltr:
            prefix.font.cs_bold = True  # Word bolds a Hebrew speaker name via bCs, not b
        else:
            _rtl_run(prefix)
        body = p.add_run(seg.get("text", ""))
        if not ltr:
            _rtl_run(body)

    return doc


def build_docx(
    result: dict[str, Any],
    path: str | Path,
    title: str,
    deidentified: bool = False,
    templates: dict[str, str] | None = None,
) -> Path:
    """Build a Word document from the transcript and save it to path. Return the path.

    templates — the raw task prompts ({"client_summary": ..., "feedback": ...}) used to restore a
    template line if a stored text still holds a redaction placeholder.
    """
    doc = _build_document(result, title, deidentified, templates)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path


def build_docx_bytes(
    result: dict[str, Any], title: str, deidentified: bool = False, templates: dict[str, str] | None = None
) -> bytes:
    """Build a Word document and return its content as bytes (for direct download)."""
    import io

    doc = _build_document(result, title, deidentified, templates)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
