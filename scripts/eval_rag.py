#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semantic (RAG) evaluation of the summaries and feedback the system produces.

What it does:
  1. Builds a semantic RAG index from every row of "משוב_שיחות.xlsx"
     (column A=recording, B=client summary, C=feedback).
  2. Runs the recordings in the folder through the *same* system pipeline:
     transcription (ivrit.ai) -> diarization (pyannote) -> merge -> client summary + feedback.
  3. For each generated summary — retrieves the semantically closest entry from the RAG
     index and checks whether it belongs to the same recording it was generated from.
     The same is done for the feedback.
  4. Produces a report: HTML + Excel + JSON under data/eval/.

Usage:
    .venv\\Scripts\\python.exe scripts/eval_rag.py                 # all recordings in the folder
    .venv\\Scripts\\python.exe scripts/eval_rag.py --only-gt       # only those with ground truth
    .venv\\Scripts\\python.exe scripts/eval_rag.py --limit 5

Each recording's result is cached (data/eval/results/), so re-runs are fast.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app import eval_metrics as em  # noqa: E402
from app import rag  # noqa: E402
from app.config import Settings  # noqa: E402
from app.pipeline.audio import extract_audio, get_duration  # noqa: E402
from app.pipeline.diarize import Diarizer  # noqa: E402
from app.pipeline.merge import merge  # noqa: E402
from app.pipeline.summarize import generate_client_summary, generate_feedback  # noqa: E402
from app.pipeline.transcribe import Transcriber  # noqa: E402

AUDIO_EXT = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".mkv", ".webm", ".aac"}
DEFAULT_FOLDER = "פרויקט AI סיכומים ומשוב יעוץ"
DEFAULT_XLSX = "משוב_שיחות.xlsx"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------- Running a recording through the system pipeline ----------

def _ok(text: str | None) -> bool:
    """Is the output valid? (not empty, no degenerate junk tokens, not too short)"""
    t = (text or "").strip()
    return len(t) >= 120 and "<unused" not in t


def process_recording(
    path: Path, transcriber, diarizer, settings, out_dir: Path, force_regen: bool = False
) -> dict:
    """Run a recording through the system pipeline; returns a result with
    client_summary + feedback (cached).

    force_regen=True: regenerate the summary/feedback from the cached transcript even
    when it is valid — used to compare LLM providers (or thinking modes) on exactly the
    same transcript, without re-transcribing (expensive, and irrelevant to comparing
    summary/feedback quality).
    """
    cache = out_dir / "results" / f"{path.stem}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if not force_regen and _ok(data.get("client_summary")) and _ok(data.get("feedback")):
            log(f"  [מטמון] {path.name}")
            return data
        # The cached transcript is fine — regenerate summary/feedback without re-transcribing
        if data.get("segments"):
            reason = "השוואת ספקי LLM" if force_regen else "הפלט הקודם היה פגום"
            log(f"  [מטמון] תמלול קיים · מפיק מחדש סיכום/משוב ({reason})")
            # Measure the LLM time on its own. The original elapsed_sec is the
            # transcription time (kept from the first run) and therefore does not
            # reflect the cost of producing the summary/feedback — which is exactly
            # the difference that matters when comparing providers/thinking modes.
            t_llm = time.time()
            data["client_summary"] = generate_client_summary(settings, data)
            data["feedback"] = generate_feedback(settings, data)
            data["llm_elapsed_sec"] = round(time.time() - t_llm, 1)
            data["llm_model"] = settings.gemini_model
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"  ✓ הופק מחדש ({data['llm_elapsed_sec']}s להפקה)")
            return data

    t0 = time.time()
    wav = out_dir / "audio" / f"{path.stem}.wav"
    extract_audio(path, wav)
    duration = get_duration(wav)

    log(f"  תמלול… ({duration:.0f}s אודיו)")
    asr = transcriber.transcribe(wav, duration)

    log("  זיהוי דוברים…")
    turns = diarizer.diarize(wav, num_speakers=None)

    segments, speaker_ids = merge(asr, turns)
    speakers = {sid: f"דובר {i}" for i, sid in enumerate(speaker_ids, start=1)}
    result = {
        "job_id": path.stem,
        "recording": path.stem,
        "language": settings.language,
        "duration_sec": duration,
        "num_speakers": len(speaker_ids),
        "speakers": speakers,
        "segments": segments,
        "asr_model": settings.asr_model,
        "diarization_model": settings.diarization_model,
    }

    log("  סיכום ללקוח…")
    result["client_summary"] = generate_client_summary(settings, result)
    log("  משוב…")
    result["feedback"] = generate_feedback(settings, result)
    result["elapsed_sec"] = round(time.time() - t0, 1)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    wav.unlink(missing_ok=True)  # the temporary WAV is no longer needed
    log(f"  ✓ הושלם ב-{result['elapsed_sec']}s ({len(segments)} קטעים, {len(speaker_ids)} דוברים)")
    return result


# ---------- Reports ----------

def _kind_he(kind: str) -> str:
    return "סיכום ללקוח" if kind == "summary" else "משוב"


def compute_kpis(results, gt_rows, settings, folder) -> dict:
    """Compute the success metrics (KPI) defined in 'קרטריוני-ומדדי-ההצלחה.xlsx'."""
    import statistics

    kpi: dict = {}

    # 4) Processing time (RTF <= 3)
    rtfs = [v for r in results if (v := em.rtf(r)) is not None]
    if rtfs:
        mean_rtf = statistics.mean(rtfs)
        kpi["rtf"] = {"measured": f"×{mean_rtf:.1f}", "verdict": "pass" if mean_rtf <= 3 else "fail",
                      "detail": f"ממוצע {len(rtfs)} הקלטות (טווח ×{min(rtfs):.1f}–×{max(rtfs):.1f})"}
    else:
        kpi["rtf"] = {"measured": "—", "verdict": "pending", "detail": "אין נתוני זמן"}

    # 1) Transcription accuracy (WER) — requires manual reference transcripts
    wers = []
    for r in results:
        ref = em.find_reference_transcript(r["recording"], folder)
        if ref and (w := em.wer(ref, em.full_transcript_text(r))) is not None:
            wers.append(w)
    if wers:
        mean_wer = statistics.mean(wers)
        kpi["wer"] = {"measured": f"{1 - mean_wer:.0%} (WER {mean_wer:.0%})",
                      "verdict": "pass" if (1 - mean_wer) >= 0.90 else "fail",
                      "detail": f"על {len(wers)} תמלולי ייחוס"}
    else:
        kpi["wer"] = {"measured": "ממתין", "verdict": "pending",
                      "detail": "הוסיפו תמלולי ייחוס: <שם>.gt.txt לצד ההקלטות (או references/<שם>.txt)"}

    # 2) Summary completeness (topic coverage — LLM judge).
    # Since the local provider was removed, the judging is done by Gemini — i.e. the
    # model grades its own output. This is a known methodological limitation (possible
    # upward bias) and is stated explicitly in the report.
    covs = []
    for r in results:
        rec = r["recording"]
        if rec in gt_rows and (r.get("client_summary") or "").strip():
            c = em.topic_coverage(settings, gt_rows[rec]["summary"], r["client_summary"])
            if c:
                covs.append(c["coverage_pct"])
    if covs:
        mean_cov = statistics.mean(covs)
        kpi["coverage"] = {"measured": f"{mean_cov:.0f}%", "verdict": "pass" if mean_cov >= 85 else "fail",
                           "detail": f"LLM-judge ({settings.gemini_model}) על {len(covs)} סיכומים מול "
                                     "ground truth · שימו לב: המודל שופט את עצמו (הטיה אפשרית)"}
    else:
        kpi["coverage"] = {"measured": "—", "verdict": "pending", "detail": "נדרש LLM פעיל + ground truth"}

    # 3) PII efficiency (0% egress — architectural)
    p = em.pii_egress_status(settings)
    kpi["pii"] = {
        "measured": "0% egress" if p["no_cloud_egress"] else "יציאה לענן (Gemini, אחרי הסרת PII)",
        "verdict": "pass" if p["no_cloud_egress"] else "fail",
        "detail": p["note"] + (" · מנוע PII מקומי מותקן" if p["pii_engine_local"] else " · .venv-pii לא מותקן"),
    }

    # 5) Satisfaction (manual)
    kpi["satisfaction"] = {"measured": "—", "verdict": "manual",
                           "detail": "סקר מובנה למומחי התוכן בסיום (אין אוטומציה)"}
    return kpi


def build_report(settings: Settings, results: list[dict], gt_rows: dict, folder: Path, rag_stats: dict) -> dict:
    """Centralize RAG retrieval/scoring and KPI+RAG metric computation into one
    uniform dict (the report).

    Split out of main() so the exact same metrics can be computed outside a full run —
    for example when comparing providers, over results that are already cached.
    """
    log("\nשולף מה-RAG ומנקד…")
    rows = []
    for res in results:
        rec = res["recording"]
        row = {"recording": rec, "has_gt": rec in gt_rows}
        for kind, field in (("summary", "client_summary"), ("feedback", "feedback")):
            gen = (res.get(field) or "").strip()
            entry = {"generated": gen, "top": [], "hit": False, "rank": None}
            if gen:
                top = rag.search(gen, kind, top_k=3, settings=settings)
                entry["top"] = top
                if row["has_gt"]:
                    entry["hit"] = bool(top) and top[0]["recording"] == rec
                    full = rag.search(gen, kind, top_k=999, settings=settings)
                    for idx, t in enumerate(full, 1):
                        if t["recording"] == rec:
                            entry["rank"] = idx
                            break
            row[kind] = entry
        rows.append(row)

    metrics = {}
    for kind in ("summary", "feedback"):
        gt = [r for r in rows if r["has_gt"] and r[kind]["top"]]
        nogt = [r for r in rows if not r["has_gt"] and r[kind]["top"]]
        hits = [r for r in gt if r[kind]["hit"]]
        metrics[kind] = {
            "gt_count": len(gt),
            "top1_acc": (len(hits) / len(gt)) if gt else 0.0,
            "mrr": (sum(1 / r[kind]["rank"] for r in gt if r[kind]["rank"]) / len(gt)) if gt else 0.0,
            "mean_hit_score": (sum(r[kind]["top"][0]["score"] for r in hits) / len(hits)) if hits else 0.0,
            "mean_nogt_score": (sum(r[kind]["top"][0]["score"] for r in nogt) / len(nogt)) if nogt else 0.0,
        }

    log("\nמחשב מדדי הצלחה (KPI)…")
    kpi = compute_kpis(results, gt_rows, settings, folder)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "embed_model": settings.embed_model,
        "llm_provider": "gemini",
        "llm_model": settings.gemini_model,
        "processed": len(results),
        "rag": {"entries": sum(v["entries"] for v in rag_stats.values()),
                "recordings": max((v["recordings"] for v in rag_stats.values()), default=0)},
        "kpi": kpi,
        "metrics": metrics,
        "rows": rows,
    }


def write_html(report: dict, path: Path) -> None:
    e = html.escape

    def rows_html(kind: str) -> str:
        out = []
        for r in report["rows"]:
            m = r[kind]
            gt = r["has_gt"]
            if not gt:
                verdict, cls = "אין ground truth", "na"
            elif m["hit"]:
                verdict, cls = "✓ התאמה נכונה", "hit"
            else:
                verdict, cls = "✗ התאמה שגויה", "miss"
            top = m["top"]
            tops = "".join(
                f"<li><b>{e(t['recording'])}</b> — {t['score']:.3f}</li>" for t in top
            )
            out.append(
                f"""<tr class="{cls}">
                  <td class="rec">{e(r['recording'])}</td>
                  <td class="verdict">{verdict}</td>
                  <td>{e(top[0]['recording']) if top else '—'}</td>
                  <td class="score">{top[0]['score']:.3f}</td>
                  <td>{('#' + str(m['rank'])) if m.get('rank') else '—'}</td>
                  <td><ol class="tops">{tops}</ol></td>
                  <td class="gen">{e((m['generated'] or '')[:300])}…</td>
                </tr>"""
            )
        return "\n".join(out)

    def block(kind: str) -> str:
        s = report["metrics"][kind]
        return f"""
        <h2>{_kind_he(kind)}</h2>
        <div class="metrics">
          <div class="m"><b>{s['top1_acc']:.0%}</b><span>דיוק Top-1 (על {s['gt_count']} עם ground truth)</span></div>
          <div class="m"><b>{s['mrr']:.2f}</b><span>MRR</span></div>
          <div class="m"><b>{s['mean_hit_score']:.3f}</b><span>ציון ממוצע — התאמה נכונה</span></div>
          <div class="m"><b>{s['mean_nogt_score']:.3f}</b><span>ציון ממוצע — ללא ground truth (בקרה שלילית)</span></div>
        </div>
        <table>
          <thead><tr>
            <th>הקלטה (מקור)</th><th>תוצאה</th><th>נשלף מה-RAG (Top-1)</th><th>ציון</th>
            <th>דירוג הנכון</th><th>Top-3</th><th>הטקסט שנוצר</th>
          </tr></thead>
          <tbody>{rows_html(kind)}</tbody>
        </table>"""

    _V = {"pass": ("✓ עומד ביעד", "kpi-pass"), "fail": ("✗ לא עומד", "kpi-fail"),
          "pending": ("⧗ ממתין לנתונים", "kpi-pend"), "manual": ("✋ מדידה ידנית", "kpi-man")}

    def kpi_block() -> str:
        kpi = report.get("kpi", {})
        rows = []
        for crit in em.CRITERIA:
            k = kpi.get(crit["key"], {})
            label, cls = _V.get(k.get("verdict", "pending"), _V["pending"])
            rows.append(
                f"""<tr class="{cls}">
                  <td class="rec">{e(crit['name'])}</td>
                  <td>{e(crit['target'])}</td>
                  <td class="score">{e(k.get('measured', '—'))}</td>
                  <td class="verdict">{label}</td>
                  <td class="gen">{e(k.get('detail', ''))}</td></tr>"""
            )
        return f"""
        <h2>🎯 מדדי הצלחה (KPI)</h2>
        <table><thead><tr><th>קריטריון</th><th>יעד</th><th>נמדד</th><th>סטטוס</th><th>שיטת מדידה / הערה</th></tr></thead>
          <tbody>{''.join(rows)}</tbody></table>"""

    doc = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<title>דוח הערכה סמנטית — מכון פוע"ה</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;background:#0e1526;color:#e8edf7;margin:0;padding:28px;}}
 h1{{margin:0 0 6px}} .sub{{color:#93a1c0;margin:0 0 22px}}
 h2{{margin:30px 0 10px;border-bottom:1px solid #24304d;padding-bottom:6px}}
 .metrics{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px}}
 .m{{background:#16203a;border:1px solid #24304d;border-radius:10px;padding:10px 16px;min-width:150px}}
 .m b{{display:block;font-size:22px}} .m span{{font-size:12px;color:#93a1c0}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th,td{{border:1px solid #24304d;padding:7px 9px;text-align:right;vertical-align:top}}
 th{{background:#16203a;color:#93a1c0;font-weight:600}}
 tr.hit td.verdict{{color:#43c08a;font-weight:700}}
 tr.miss td.verdict{{color:#e5645b;font-weight:700}}
 tr.na td.verdict{{color:#93a1c0}}
 tr.na{{opacity:.72}}
 .rec{{font-weight:600;white-space:nowrap}} .score{{font-variant-numeric:tabular-nums}}
 .gen{{color:#b9c4da;max-width:380px}} ol.tops{{margin:0;padding-inline-start:18px}}
 .note{{background:#16203a;border:1px solid #24304d;border-radius:10px;padding:12px 16px;margin-top:26px;color:#93a1c0;font-size:13px}}
 tr.kpi-pass td.verdict{{color:#43c08a;font-weight:700}}
 tr.kpi-fail td.verdict{{color:#e5645b;font-weight:700}}
 tr.kpi-pend td.verdict{{color:#e0b341;font-weight:700}}
 tr.kpi-man td.verdict{{color:#93a1c0;font-weight:700}}
</style></head><body>
<h1>דוח הערכה — מכון פוע"ה</h1>
<p class="sub">{report['generated_at']} · מודל embedding: {e(report['embed_model'])} ·
   {report['processed']} הקלטות עובדו · RAG: {report['rag']['entries']} רשומות מ-{report['rag']['recordings']} הקלטות</p>
{kpi_block()}
{block('summary')}
{block('feedback')}
<div class="note">
<b>איך לקרוא:</b> לכל הקלטה, המערכת הפיקה סיכום/משוב. הטקסט שנוצר נשלח כשאילתה ל-RAG שנבנה
מהאקסל, והשורה הדומה ביותר סמנטית נשלפה. "התאמה נכונה" = הרשומה שנשלפה שייכת לאותה הקלטה.
הקלטות ללא ground truth (לא מופיעות באקסל) משמשות <b>בקרה שלילית</b> — הציון שלהן אמור להיות
נמוך משמעותית, כי אין להן התאמה אמיתית באינדקס.
</div>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_xlsx(report: dict, path: Path) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    for i, kind in enumerate(("summary", "feedback")):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.title = _kind_he(kind)
        ws.sheet_view.rightToLeft = True
        ws.append(["הקלטה (מקור)", "יש ground truth", "נשלף מה-RAG (Top-1)", "ציון", "התאמה נכונה?", "דירוג הנכון", "הטקסט שנוצר"])
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in report["rows"]:
            m = r[kind]
            top = m["top"][0] if m["top"] else {"recording": "", "score": 0}
            ws.append([
                r["recording"], "כן" if r["has_gt"] else "לא",
                top["recording"], round(top["score"], 4),
                ("כן" if m["hit"] else "לא") if r["has_gt"] else "—",
                m.get("rank") or "—",
                m["generated"] or "",
            ])
        for col, w in zip("ABCDEFG", (34, 14, 34, 9, 13, 12, 90)):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")

    # KPI sheet
    kpi = report.get("kpi", {})
    ws = wb.create_sheet("מדדי הצלחה", 0)
    ws.sheet_view.rightToLeft = True
    _VX = {"pass": "עומד ביעד", "fail": "לא עומד", "pending": "ממתין לנתונים", "manual": "מדידה ידנית"}
    ws.append(["קריטריון", "יעד (KPI)", "נמדד", "סטטוס", "שיטת מדידה / הערה"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for crit in em.CRITERIA:
        k = kpi.get(crit["key"], {})
        ws.append([crit["name"], crit["target"], k.get("measured", "—"),
                   _VX.get(k.get("verdict", "pending"), "—"), k.get("detail", "")])
    for col, w in zip("ABCDE", (22, 24, 18, 16, 70)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(path)


# ---------- Provider comparison (same metrics, computed per provider, side by side) ----------

def _report_label(report: dict) -> str:
    # Historical reports (from before the local provider was removed) still carry
    # llm_provider="local" and the thinking flag — the label keeps reflecting them
    # correctly so they remain comparable.
    provider = "מקומי" if report.get("llm_provider") == "local" else "חיצוני"
    if report.get("local_model_think"):
        provider += "-חשיבה"
    return f"{provider} ({report.get('llm_model', '?')})"


def _gen_time(report: dict, results_dir: Path | None) -> tuple[str, str]:
    """Time to produce the summary+feedback (LLM only) per recording — mean, and how
    it was measured.

    Preferred source: llm_elapsed_sec, actually measured (newer runs). When absent
    (data collected before that measurement existed) it is derived from the deltas
    between result-file write times, which are written one per recording in sequence.
    That is an estimate, and is labelled as such rather than presented as a direct
    measurement.
    """
    import statistics

    vals = [r["llm_elapsed_sec"] for r in report.get("rows", []) if r.get("llm_elapsed_sec")]
    if vals:
        return f"{statistics.mean(vals):.0f}s", "נמדד ישירות (llm_elapsed_sec)"

    if results_dir and results_dir.exists():
        times = sorted(f.stat().st_mtime for f in results_dir.glob("*.json"))
        if len(times) >= 2:
            deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)]
            mean = statistics.mean(deltas)
            total_min = (times[-1] - times[0]) / 60
            return f"~{mean:.0f}s", f"אומדן מזמני כתיבת הקבצים · סה\"כ {total_min:.1f} דק' ל-{len(times)} הקלטות"
    return "—", "לא נמדד"


def write_comparison_html(
    reports: list[tuple[str, dict]], path: Path, results_dirs: list[Path] | None = None
) -> None:
    """Produce a single report showing the *same* KPI and RAG metrics
    (from write_html/compute_kpis), computed separately per provider (report) and
    placed side by side — so there is no need to compare separate report.html files
    by hand."""
    e = html.escape
    labels = [lbl for lbl, _ in reports]
    results_dirs = results_dirs or [None] * len(reports)

    def kpi_block() -> str:
        _V = {"pass": ("✓ עומד", "kpi-pass"), "fail": ("✗ לא עומד", "kpi-fail"),
              "pending": ("⧗ ממתין", "kpi-pend"), "manual": ("✋ ידני", "kpi-man")}
        head = "".join(f"<th>{e(lbl)} — נמדד</th><th>{e(lbl)} — סטטוס</th>" for lbl in labels)
        rows = []
        for crit in em.CRITERIA:
            cells = []
            for _, rep in reports:
                k = rep.get("kpi", {}).get(crit["key"], {})
                label, cls = _V.get(k.get("verdict", "pending"), _V["pending"])
                cells.append(f'<td class="score">{e(k.get("measured", "—"))}</td><td class="verdict {cls}">{label}</td>')
            rows.append(f"<tr><td class='rec'>{e(crit['name'])}</td><td>{e(crit['target'])}</td>{''.join(cells)}</tr>")

        # Generation-time row — not part of CRITERIA (not an official target in the
        # Excel), but it is the main practical difference between the modes, so it is
        # surfaced explicitly rather than left hidden.
        gen_cells, gen_notes = [], []
        for (lbl, rep), rdir in zip(reports, results_dirs):
            measured, how = _gen_time(rep, rdir)
            gen_cells.append(f'<td class="score">{e(measured)}</td><td class="verdict">{e(how)}</td>')
            gen_notes.append(f"{lbl}: {measured}")
        rows.append(
            "<tr><td class='rec'>זמן הפקת סיכום+משוב</td>"
            "<td>לכל הקלטה (LLM בלבד)</td>" + "".join(gen_cells) + "</tr>"
        )

        return f"""
        <h2>🎯 מדדי הצלחה (KPI) — השוואה</h2>
        <table><thead><tr><th>קריטריון</th><th>יעד</th>{head}</tr></thead>
          <tbody>{''.join(rows)}</tbody></table>
        <div class="note"><b>הערה על זמן העיבוד:</b> מדד "זמן עיבוד" (RTF) שבטבלה מחושב מזמן
        <i>התמלול</i> השמור מהריצה הראשונה, ולכן הוא זהה בכל העמודות ואינו מבחין בין הספקים.
        השורה "זמן הפקת סיכום+משוב" היא ההבדל המעשי בפועל: {e(' · '.join(gen_notes))}.</div>"""

    def rag_metrics_block(kind: str) -> str:
        cards = []
        for lbl, rep in reports:
            s = rep["metrics"][kind]
            cards.append(f"""
              <div class="m"><b>{s['top1_acc']:.0%}</b><span>{e(lbl)} · דיוק Top-1 ({s['gt_count']} עם GT)</span></div>
              <div class="m"><b>{s['mrr']:.2f}</b><span>{e(lbl)} · MRR</span></div>
              <div class="m"><b>{s['mean_hit_score']:.3f}</b><span>{e(lbl)} · ציון התאמה נכונה</span></div>""")
        return f"""
        <h2>{_kind_he(kind)} — מדדי RAG</h2>
        <div class="metrics">{''.join(cards)}</div>"""

    def recordings_block() -> str:
        all_recs = sorted({r["recording"] for _, rep in reports for r in rep["rows"]})
        by_report = [{r["recording"]: r for r in rep["rows"]} for _, rep in reports]
        head = "".join(
            f"<th>{e(lbl)} — סיכום</th><th>{e(lbl)} — משוב</th>" for lbl in labels
        )
        out = []
        for rec in all_recs:
            cells = []
            for idx in by_report:
                row = idx.get(rec)
                if not row:
                    cells.append("<td>—</td><td>—</td>")
                    continue
                for kind in ("summary", "feedback"):
                    m = row[kind]
                    if not row["has_gt"]:
                        cells.append("<td class='na'>—</td>")
                    elif not m["top"]:
                        cells.append("<td class='na'>אין פלט</td>")
                    else:
                        verdict = "✓" if m["hit"] else "✗"
                        cls = "hit" if m["hit"] else "miss"
                        cells.append(f"<td class='{cls}'>{verdict} ({m['top'][0]['score']:.3f})</td>")
            out.append(f"<tr><td class='rec'>{e(rec)}</td>{''.join(cells)}</tr>")
        return f"""
        <h2>השוואה פר-הקלטה (התאמה ל-RAG)</h2>
        <table><thead><tr><th>הקלטה</th>{head}</tr></thead>
          <tbody>{''.join(out)}</tbody></table>
        <div class="note">הטקסט המלא שנוצר לכל הקלטה נמצא בדוח הפרטני של אותו ספק
        (report.html בתיקיית ה-out שלו). טבלה זו מציגה רק התאמה/אי-התאמה וציון RAG.</div>"""

    counts = " · ".join(f"{e(lbl)}: {rep['processed']} הקלטות" for lbl, rep in reports)
    doc = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<title>דוח השוואה — מקומי לעומת חיצוני — מכון פוע"ה</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;background:#0e1526;color:#e8edf7;margin:0;padding:28px;}}
 h1{{margin:0 0 6px}} .sub{{color:#93a1c0;margin:0 0 22px}}
 h2{{margin:30px 0 10px;border-bottom:1px solid #24304d;padding-bottom:6px}}
 .metrics{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px}}
 .m{{background:#16203a;border:1px solid #24304d;border-radius:10px;padding:10px 16px;min-width:170px}}
 .m b{{display:block;font-size:22px}} .m span{{font-size:12px;color:#93a1c0}}
 table{{width:100%;border-collapse:collapse;font-size:13px}}
 th,td{{border:1px solid #24304d;padding:7px 9px;text-align:right;vertical-align:top}}
 th{{background:#16203a;color:#93a1c0;font-weight:600}}
 .rec{{font-weight:600;white-space:nowrap}} .score{{font-variant-numeric:tabular-nums}}
 td.hit{{color:#43c08a;font-weight:700}} td.miss{{color:#e5645b;font-weight:700}} td.na{{color:#93a1c0}}
 .note{{background:#16203a;border:1px solid #24304d;border-radius:10px;padding:12px 16px;margin-top:26px;color:#93a1c0;font-size:13px}}
 .verdict.kpi-pass{{color:#43c08a;font-weight:700}} .verdict.kpi-fail{{color:#e5645b;font-weight:700}}
 .verdict.kpi-pend{{color:#e0b341;font-weight:700}} .verdict.kpi-man{{color:#93a1c0;font-weight:700}}
</style></head><body>
<h1>דוח השוואה — מודל מקומי לעומת מודל חיצוני</h1>
<p class="sub">{time.strftime("%Y-%m-%d %H:%M")} · {counts}</p>
<div class="note"><b>שימו לב:</b> מספר ההקלטות עשוי להיות שונה בין הספקים (למשל אם ריצה חיצונית
נבלמה על ידי מכסת API) — זה בא לידי ביטוי במפורש במספרי ה-n שבכל טבלה, ואינו מוסתר.</div>
{kpi_block()}
{rag_metrics_block('summary')}
{rag_metrics_block('feedback')}
{recordings_block()}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_comparison_xlsx(reports: list[tuple[str, dict]], path: Path) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "מדדי הצלחה"
    ws.sheet_view.rightToLeft = True
    _VX = {"pass": "עומד ביעד", "fail": "לא עומד", "pending": "ממתין לנתונים", "manual": "מדידה ידנית"}
    header = ["קריטריון", "יעד (KPI)"]
    for lbl, _ in reports:
        header += [f"{lbl} — נמדד", f"{lbl} — סטטוס"]
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True)
    for crit in em.CRITERIA:
        row = [crit["name"], crit["target"]]
        for _, rep in reports:
            k = rep.get("kpi", {}).get(crit["key"], {})
            row += [k.get("measured", "—"), _VX.get(k.get("verdict", "pending"), "—")]
        ws.append(row)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")

    ws2 = wb.create_sheet("פר-הקלטה (RAG)")
    ws2.sheet_view.rightToLeft = True
    header2 = ["הקלטה"]
    for lbl, _ in reports:
        header2 += [f"{lbl} — סיכום", f"{lbl} — משוב"]
    ws2.append(header2)
    for c in ws2[1]:
        c.font = Font(bold=True)
    all_recs = sorted({r["recording"] for _, rep in reports for r in rep["rows"]})
    by_report = [{r["recording"]: r for r in rep["rows"]} for _, rep in reports]
    for rec in all_recs:
        row = [rec]
        for idx in by_report:
            r = idx.get(rec)
            if not r:
                row += ["—", "—"]
                continue
            for kind in ("summary", "feedback"):
                m = r[kind]
                if not r["has_gt"] or not m["top"]:
                    row.append("—")
                else:
                    row.append(f"{'✓' if m['hit'] else '✗'} ({m['top'][0]['score']:.3f})")
        ws2.append(row)
    wb.save(path)


# ---------- Main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="הערכה סמנטית של סיכומים ומשובים מול RAG של האקסל")
    ap.add_argument("--folder", default=DEFAULT_FOLDER, help="תיקיית ההקלטות")
    ap.add_argument("--xlsx", default=DEFAULT_XLSX, help="קובץ ה-ground truth")
    ap.add_argument("--limit", type=int, default=0, help="הגבלת מספר הקלטות (0 = הכול)")
    ap.add_argument("--only-gt", action="store_true", help="רק הקלטות שיש להן שורה באקסל")
    ap.add_argument("--rebuild-rag", action="store_true", help="בניית האינדקס מחדש")
    ap.add_argument("--out", default="data/eval", help="תיקיית הפלט")
    ap.add_argument(
        "--regen-summaries",
        action="store_true",
        help="הפקה מחדש של סיכום/משוב מהתמלול הקיים במטמון, גם אם תקין (למשל: השוואת ספקי LLM)",
    )
    ap.add_argument(
        "--compare",
        nargs="+",
        metavar="DIR",
        help="דילוג על עיבוד — טוען report.json קיימים מהתיקיות שסופקו (למשל data/eval "
             "data/eval_gemini) ומפיק דוח השוואה משותף עם אותם מדדים בדיוק, זה לצד זה. "
             "לא מבצע קריאות LLM/עיבוד חדשות — משתמש בנתונים שכבר חושבו.",
    )
    args = ap.parse_args()

    if args.compare:
        out_dir = ROOT / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        reports, results_dirs = [], []
        for d in args.compare:
            rep_path = ROOT / d / "report.json"
            if not rep_path.exists():
                log(f"✗ לא נמצא {rep_path} — הריצו קודם: python scripts/eval_rag.py --out {d} [...]")
                return 1
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            reports.append((_report_label(rep), rep))
            results_dirs.append(ROOT / d / "results")
        write_comparison_html(reports, out_dir / "report.html", results_dirs)
        try:
            write_comparison_xlsx(reports, out_dir / "report.xlsx")
        except PermissionError:
            log("  ⚠ report.xlsx פתוח בתוכנה אחרת — נשמרו HTML/JSON בלבד")
        (out_dir / "report.json").write_text(
            json.dumps([{"label": lbl, "report": rep} for lbl, rep in reports], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log("\nדוח השוואה:")
        for lbl, rep in reports:
            log(f"  {lbl}: {rep['processed']} הקלטות")
        log(f"\nנשמר: {out_dir/'report.html'} · report.xlsx · report.json")
        return 0

    settings = Settings()
    out_dir = ROOT / args.out
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    # 1) RAG
    if args.rebuild_rag or not rag.stats(settings):
        log("בונה RAG מהאקסל…")
        info = rag.build_from_excel(ROOT / args.xlsx, settings)
        log(f"  ✓ {info['entries']} רשומות מ-{info['recordings']} הקלטות (dim={info['dim']}, {info['model']})")
    gt_rows = {r["recording"]: r for r in rag.read_excel_rows(ROOT / args.xlsx)}
    rag_stats = rag.stats(settings)
    log(f"RAG: {rag_stats}")

    # 2) Recordings
    folder = ROOT / args.folder
    recs = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if args.only_gt:
        recs = [p for p in recs if p.stem in gt_rows]
    if args.limit:
        recs = recs[: args.limit]
    log(f"\nהקלטות לעיבוד: {len(recs)} (מתוכן {sum(1 for p in recs if p.stem in gt_rows)} עם ground truth)\n")

    # 3) Processing
    transcriber, diarizer = Transcriber(settings), Diarizer(settings)
    results = []
    for i, p in enumerate(recs, 1):
        log(f"[{i}/{len(recs)}] {p.name}")
        try:
            results.append(
                process_recording(
                    p, transcriber, diarizer, settings, out_dir, force_regen=args.regen_summaries
                )
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  ✗ נכשל: {exc}")

    # 4-5) RAG retrieval + scoring + KPI metrics
    report = build_report(settings, results, gt_rows, folder, rag_stats)
    kpi, metrics = report["kpi"], report["metrics"]

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, out_dir / "report.html")
    write_xlsx(report, out_dir / "report.xlsx")

    # 6) Console summary
    _VC = {"pass": "✓ עומד", "fail": "✗ לא עומד", "pending": "⧗ ממתין", "manual": "✋ ידני"}
    log("\n" + "=" * 72)
    log("מדדי הצלחה (KPI):")
    for crit in em.CRITERIA:
        k = kpi.get(crit["key"], {})
        log(f"  {crit['name']:<12} | יעד {crit['target']:<22} | נמדד {str(k.get('measured','—')):<16} | "
            f"{_VC.get(k.get('verdict','pending'),'?')}")
    log("=" * 72)
    log("הערכה סמנטית (RAG):")
    for kind in ("summary", "feedback"):
        m = metrics[kind]
        log(f"{_kind_he(kind):<12} | Top-1: {m['top1_acc']:.0%} ({m['gt_count']} עם GT) | MRR {m['mrr']:.2f} | "
            f"ציון התאמה {m['mean_hit_score']:.3f} מול בקרה {m['mean_nogt_score']:.3f}")
    log("=" * 72)
    for r in report["rows"]:
        if not r["has_gt"]:
            continue
        s, f = r["summary"], r["feedback"]
        log(f"{r['recording']:<34} סיכום:{'✓' if s['hit'] else '✗'} ({s['top'][0]['score']:.3f})  "
            f"משוב:{'✓' if f['hit'] else '✗'} ({f['top'][0]['score']:.3f})")
    log(f"\nדוחות נשמרו: {out_dir/'report.html'} · report.xlsx · report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
