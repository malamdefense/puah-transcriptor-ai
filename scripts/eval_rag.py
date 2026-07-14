#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""הערכה סמנטית (RAG) של הסיכומים והמשובים שהמערכת מפיקה.

מה זה עושה:
  1. בונה RAG סמנטי מכל שורה ב-"משוב_שיחות.xlsx" (עמודה A=הקלטה, B=סיכום ללקוח, C=משוב).
  2. מריץ את ההקלטות שבתיקייה דרך *אותו צינור* של המערכת:
     תמלול (ivrit.ai) → זיהוי דוברים (pyannote) → מיזוג → סיכום ללקוח + משוב (Ollama).
  3. לכל סיכום שנוצר — שולף מה-RAG את הרשומה הדומה ביותר סמנטית, ובודק האם היא מקושרת
     לאותה הקלטה שממנה נוצר הסיכום. ואותו דבר עבור המשוב.
  4. מפיק דוח: HTML + Excel + JSON תחת data/eval/.

הרצה:
    .venv\\Scripts\\python.exe scripts/eval_rag.py                 # כל ההקלטות בתיקייה
    .venv\\Scripts\\python.exe scripts/eval_rag.py --only-gt       # רק אלו שיש להן ground truth
    .venv\\Scripts\\python.exe scripts/eval_rag.py --limit 5

התוצאות של כל הקלטה נשמרות במטמון (data/eval/results/), כך שהרצה חוזרת מהירה.
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


# ---------- הרצת הקלטה בצינור המערכת ----------

def _ok(text: str | None) -> bool:
    """פלט תקין? (לא ריק, לא טוקנים מנוונים של gemma, לא קצר מדי)"""
    t = (text or "").strip()
    return len(t) >= 120 and "<unused" not in t


def process_recording(path: Path, transcriber, diarizer, settings, out_dir: Path) -> dict:
    """מריץ הקלטה דרך צינור המערכת ומחזיר result עם client_summary + feedback (עם מטמון)."""
    cache = out_dir / "results" / f"{path.stem}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if _ok(data.get("client_summary")) and _ok(data.get("feedback")):
            log(f"  [מטמון] {path.name}")
            return data
        # התמלול במטמון תקין, אך הסיכום/המשוב היו פגומים — מפיקים מחדש בלי לתמלל שוב
        if data.get("segments"):
            log("  [מטמון] תמלול קיים · מפיק מחדש סיכום/משוב (הפלט הקודם היה פגום)")
            data["client_summary"] = generate_client_summary(settings, data)
            data["feedback"] = generate_feedback(settings, data)
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            log("  ✓ הופק מחדש")
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
    wav.unlink(missing_ok=True)  # ה-WAV הזמני כבר לא נחוץ
    log(f"  ✓ הושלם ב-{result['elapsed_sec']}s ({len(segments)} קטעים, {len(speaker_ids)} דוברים)")
    return result


# ---------- דוחות ----------

def _kind_he(kind: str) -> str:
    return "סיכום ללקוח" if kind == "summary" else "משוב"


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
</style></head><body>
<h1>דוח הערכה סמנטית (RAG)</h1>
<p class="sub">מכון פוע"ה · {report['generated_at']} · מודל embedding: {e(report['embed_model'])} ·
   {report['processed']} הקלטות עובדו · RAG: {report['rag']['entries']} רשומות מ-{report['rag']['recordings']} הקלטות</p>
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
    wb.save(path)


# ---------- ראשי ----------

def main() -> int:
    ap = argparse.ArgumentParser(description="הערכה סמנטית של סיכומים ומשובים מול RAG של האקסל")
    ap.add_argument("--folder", default=DEFAULT_FOLDER, help="תיקיית ההקלטות")
    ap.add_argument("--xlsx", default=DEFAULT_XLSX, help="קובץ ה-ground truth")
    ap.add_argument("--limit", type=int, default=0, help="הגבלת מספר הקלטות (0 = הכול)")
    ap.add_argument("--only-gt", action="store_true", help="רק הקלטות שיש להן שורה באקסל")
    ap.add_argument("--rebuild-rag", action="store_true", help="בניית האינדקס מחדש")
    ap.add_argument("--out", default="data/eval", help="תיקיית הפלט")
    args = ap.parse_args()

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

    # 2) הקלטות
    folder = ROOT / args.folder
    recs = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if args.only_gt:
        recs = [p for p in recs if p.stem in gt_rows]
    if args.limit:
        recs = recs[: args.limit]
    log(f"\nהקלטות לעיבוד: {len(recs)} (מתוכן {sum(1 for p in recs if p.stem in gt_rows)} עם ground truth)\n")

    # 3) עיבוד
    transcriber, diarizer = Transcriber(settings), Diarizer(settings)
    results = []
    for i, p in enumerate(recs, 1):
        log(f"[{i}/{len(recs)}] {p.name}")
        try:
            results.append(process_recording(p, transcriber, diarizer, settings, out_dir))
        except Exception as exc:  # noqa: BLE001
            log(f"  ✗ נכשל: {exc}")

    # 4) שליפה מה-RAG + ניקוד
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

    # 5) מדדים
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

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "embed_model": settings.embed_model,
        "processed": len(results),
        "rag": {"entries": sum(v["entries"] for v in rag_stats.values()),
                "recordings": max((v["recordings"] for v in rag_stats.values()), default=0)},
        "metrics": metrics,
        "rows": rows,
    }

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, out_dir / "report.html")
    write_xlsx(report, out_dir / "report.xlsx")

    # 6) סיכום לקונסולה
    log("\n" + "=" * 72)
    for kind in ("summary", "feedback"):
        m = metrics[kind]
        log(f"{_kind_he(kind):<12} | Top-1: {m['top1_acc']:.0%} ({m['gt_count']} עם GT) | MRR {m['mrr']:.2f} | "
            f"ציון התאמה {m['mean_hit_score']:.3f} מול בקרה {m['mean_nogt_score']:.3f}")
    log("=" * 72)
    for r in rows:
        if not r["has_gt"]:
            continue
        s, f = r["summary"], r["feedback"]
        log(f"{r['recording']:<34} סיכום:{'✓' if s['hit'] else '✗'} ({s['top'][0]['score']:.3f})  "
            f"משוב:{'✓' if f['hit'] else '✗'} ({f['top'][0]['score']:.3f})")
    log(f"\nדוחות נשמרו: {out_dir/'report.html'} · report.xlsx · report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
