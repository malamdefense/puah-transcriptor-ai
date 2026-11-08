"""Export the transcript to a Word document (.docx) with right-to-left (Hebrew) layout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# python-docx needs lxml's compiled etree. An application-control policy (Windows Smart App
# Control) can refuse to load it; then only the Word export is unavailable, not the whole API and
# worker that import this module.
try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    _DOCX_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover - environment-specific
    _DOCX_IMPORT_ERROR = _exc

from .export import _fmt_clock, _speaker_name
from .language import LANGUAGE_NAMES_HE
from .output_guard import sanitize_caller_text


class WordExportUnavailable(RuntimeError):
    """python-docx (or lxml under it) could not be imported on this machine."""


def word_export_available() -> bool:
    return _DOCX_IMPORT_ERROR is None


def _require_docx() -> None:
    if _DOCX_IMPORT_ERROR is not None:
        raise WordExportUnavailable(f"python-docx could not be loaded: {_DOCX_IMPORT_ERROR}")


def _rtl(paragraph) -> None:
    """Make a paragraph RTL and align it to the right (valid for Hebrew in Word).

    Word reads jc in a right-to-left paragraph relative to its direction: "left" is the start of
    the line — the right edge — and "right" its end, the left edge. jc="right" left every Hebrew
    line of the export flush left."""
    pPr = paragraph._p.get_or_add_pPr()
    bidi = OxmlElement("w:bidi")
    bidi.set(qn("w:val"), "1")
    pPr.append(bidi)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT


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


# Recording-quality check (result["audio_quality"], written by the worker): numbers and fixed
# Hebrew messages, never transcript text. A message names its speaker as "{speaker}", replaced
# here by the display name — the same names the transcript below prints, so the de-identified
# document gets the section too without exposing anything new.
_AQ_TITLE = "בדיקת איכות הקלטה"
_AQ_FOOTNOTE = "הבדיקה אוטומטית; ההקלטה המקורית לא שונתה."
_AQ_SEVERITY = {"severe": (0, "חמור"), "warn": (1, "אזהרה"), "info": (2, "לידיעה")}
_AQ_FIX_LABEL = "טופל אוטומטית"
_AQ_ANY_SPEAKER = "אחד הדוברים"
_AQ_SPEAKER_ID = re.compile(r"^SPEAKER_\d+$")


def _aq_message(entry: dict[str, Any], names: dict[str, Any]) -> str:
    """The entry's message with "{speaker}" -> the display name(s) of the speaker(s) it is about:
    its `speaker` (flags), else the speaker keys of its `detail` (a speaker_gain fix)."""
    spk = entry.get("speaker")
    if spk:
        ids = [str(spk)]
    else:
        detail = entry.get("detail")
        ids = [str(k) for k in detail if _AQ_SPEAKER_ID.match(str(k)) or k in names] \
            if isinstance(detail, dict) else []
    who = ", ".join(str(names.get(i) or "").strip() or i for i in ids) if ids else _AQ_ANY_SPEAKER
    return entry["message_he"].replace("{speaker}", who)


def audio_quality_lines(result: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, message) lines of the recording-quality check, for the export.

    Empty for a result without a check (results from before it) and for a clean check (no warn or
    severe flag, nothing fixed). Otherwise every flag, most severe first, then the applied fixes
    (each fix once, though the worker lists it under both "fixes" and "flags").
    """
    aq = result.get("audio_quality")
    if not isinstance(aq, dict):
        return []
    names = result.get("speakers") if isinstance(result.get("speakers"), dict) else {}

    def entries(key: str) -> list[dict[str, Any]]:
        items = aq.get(key)
        if not isinstance(items, list):
            return []
        return [e for e in items
                if isinstance(e, dict) and isinstance(e.get("message_he"), str) and e["message_he"].strip()]

    flags, fixes = entries("flags"), entries("fixes")
    # The worker also lists every applied fix as an info flag of the same code (for a speaker_gain
    # fix that flag names the lifted speaker, the fix itself does not). One line per fix: the twin
    # flag is folded into it and words the fix line when it names a speaker.
    twins: set[int] = set()
    fix_entries = []
    for fx in fixes:
        code = fx.get("code")
        j = next((i for i, f in enumerate(flags)
                  if i not in twins and isinstance(code, str) and f.get("code") == code), None)
        if j is not None:
            twins.add(j)
        fix_entries.append(flags[j] if j is not None and flags[j].get("speaker") else fx)
    flags = [f for i, f in enumerate(flags) if i not in twins]
    sev = [s if isinstance(s, str) and s in _AQ_SEVERITY else "info" for s in (e.get("severity") for e in flags)]
    order = sorted(range(len(flags)), key=lambda i: (_AQ_SEVERITY[sev[i]][0], i))
    if not fixes and not any(s in ("warn", "severe") for s in sev):
        return []
    lines = [(_AQ_SEVERITY[sev[i]][1], _aq_message(flags[i], names)) for i in order]
    lines += [(_AQ_FIX_LABEL, _aq_message(e, names)) for e in fix_entries]
    return lines


def _audio_quality_section(doc, result: dict[str, Any]) -> None:
    lines = audio_quality_lines(result)
    if not lines:
        return
    _add(doc, _AQ_TITLE, bold=True, size=14)
    for label, message in lines:
        p = doc.add_paragraph(style="List Bullet")
        _rtl(p)
        head = p.add_run(f"{label}: ")
        head.bold = True
        head.font.cs_bold = True  # an RTL run is bolded by bCs in Word
        _rtl_run(head)
        _rtl_run(p.add_run(message))
    _add(doc, _AQ_FOOTNOTE, size=9)


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

    # Recording-quality check: only when it found something to warn about or fixed something
    _audio_quality_section(doc, result)

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
    _require_docx()
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

    _require_docx()
    doc = _build_document(result, title, deidentified, templates)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
