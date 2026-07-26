#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rebuild.py — הפעלה בלחיצה אחת של כל מערכת התמלול וזיהוי הדוברים של מכון פוע"ה.

מה הסקריפט עושה (לפי הסדר):
  1. יוצר סביבת Python וירטואלית (.venv) אם אינה קיימת.
  2. מתקין את התלויות (torch+CUDA ואז requirements) — רק בפעם הראשונה, או עם --reinstall.
  3. יוצר קובץ .env מתוך .env.example אם חסר, ומזהיר אם חסר טוקן HuggingFace.
  4. בודק ש-ffmpeg זמין, וש-Ollama (מודל הסיכום) זמין — אזהרות בלבד, לא חוסם.
  5. מפעיל את תהליך ה-worker (עיבוד על ה-GPU) ואת שרת ה-API, ופותח דפדפן.

הרצה:
    python rebuild.py

דגלים שימושיים:
    --reinstall      התקנת התלויות מחדש (למשל לאחר שינוי requirements)
    --skip-install   דילוג מוחלט על בדיקת/התקנת התלויות
    --cpu            התקנת torch בגרסת CPU (ללא CUDA)
    --no-venv        שימוש ב-Python הנוכחי במקום ב-venv
    --no-browser     לא לפתוח דפדפן אוטומטית
    --host / --port  ברירת מחדל 0.0.0.0:8000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# הדפסת עברית תקינה גם בקונסולת Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MARKER = VENV / ".puah_deps_ok"  # סימון שהתלויות הותקנו בהצלחה
PII_VENV = ROOT / ".venv-pii"
PII_MARKER = PII_VENV / ".puah_pii_ok"
# cu128 = CUDA 12.8 — תומך ב-Ampere/Ada/Hopper/Blackwell. ל-GPU ישן יותר אפשר --cuda-index cu124.
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

_ICONS = {"info": "•", "ok": "✓", "warn": "⚠", "err": "✗", "step": "»"}


def log(msg: str, kind: str = "info") -> None:
    print(f" {_ICONS.get(kind, '•')} {msg}", flush=True)


# ---------- סביבה ותלויות ----------

def venv_python(venv: Path = VENV) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_venv() -> Path:
    py = venv_python()
    if py.exists():
        return py
    log("יוצר סביבת Python וירטואלית (.venv)…", "step")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    return py


def install_deps(py: Path, cpu: bool, cuda_index: str = CUDA_INDEX) -> None:
    log("מתקין תלויות — בפעם הראשונה זה עשוי לקחת כמה דקות (הורדת torch גדולה)…", "step")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    # קודם requirements (pyannote מושך torchcodec + torch CPU כתלות), ואז כופים
    # torch בגרסת CUDA *בסוף* — אחרת ה-CPU torch גובר ומבטל את ה-GPU.
    log("מתקין את התלויות (requirements.txt)…")
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    if cpu:
        log("מצב CPU — מדלג על התקנת torch ל-CUDA.")
    else:
        log(f"מתקין torch + torchaudio ל-CUDA (אחרון, גובר על CPU): {cuda_index}")
        subprocess.check_call([str(py), "-m", "pip", "uninstall", "-y", "torch", "torchaudio"])
        subprocess.check_call(
            [str(py), "-m", "pip", "install", "torch", "torchaudio", "--index-url", cuda_index]
        )
    MARKER.write_text("ok", encoding="utf-8")
    log("כל התלויות הותקנו.", "ok")


def _find_pii_base_python(explicit: str | None) -> tuple[list[str] | None, str | None]:
    """מאתר Python 3.8–3.10 (נדרש ל-spaCy 3.2 של HebSafeHarbor). מחזיר (cmd, version)."""
    candidates: list[list[str]] = []
    if explicit:
        candidates.append([explicit])
    for v in ("3.10", "3.9", "3.8"):
        candidates.append(["py", f"-{v}"])      # py launcher (Windows)
        candidates.append([f"python{v}"])       # שם מפורש (Linux/macOS)
    candidates.append([sys.executable])  # אם המפרש הנוכחי מתאים
    probe = "import sys;print('%d.%d'%sys.version_info[:2])"
    for cmd in candidates:
        try:
            out = subprocess.run(cmd + ["-c", probe], capture_output=True, text=True, timeout=15)
        except Exception:
            continue
        if out.returncode != 0:
            continue
        try:
            major, minor = map(int, out.stdout.strip().split("."))
        except ValueError:
            continue
        if major == 3 and 8 <= minor <= 10:
            return cmd, f"{major}.{minor}"
    return None, None


def ensure_pii_venv(reinstall: bool = False, pii_python: str | None = None) -> None:
    """מקים סביבה וירטואלית מבודדת ל-HebSafeHarbor (.venv-pii) — נדרש להסרת PII.

    מבודד כי HebSafeHarbor דורש pydantic<2 / spaCy 3.2 שמתנגשים עם האפליקציה,
    ו-spaCy 3.2 דורש Python 3.8–3.10 (לא 3.11+).
    """
    req = ROOT / "requirements-pii.txt"
    if not req.exists():
        log("requirements-pii.txt חסר — מדלג על הקמת סביבת ה-PII.", "warn")
        return
    if PII_MARKER.exists() and not reinstall:
        log("סביבת ה-PII (.venv-pii) כבר מותקנת — מדלג.", "ok")
        return

    py = PII_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        base, ver = _find_pii_base_python(pii_python)
        if not base:
            log("לא נמצא Python 3.8–3.10 (נדרש ל-HebSafeHarbor/spaCy 3.2) — הסרת PII תדולג.", "warn")
            log("התקינו:  winget install Python.Python.3.10  ואז הריצו שוב (או: --pii-python <path>).", "info")
            return
        log(f"בונה את סביבת ה-PII (.venv-pii) עם Python {ver}…", "step")
        subprocess.check_call(base + ["-m", "venv", str(PII_VENV)])

    log("מתקין HebSafeHarbor + מודל NER עברי (he_ner_news_trf) — הורדה גדולה, כמה דקות…", "step")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(req)])
    PII_MARKER.write_text("ok", encoding="utf-8")
    log("סביבת ה-PII מוכנה.", "ok")


# ---------- קובץ הגדרות ----------

def _env_value(key: str, default: str | None = None) -> str | None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(key + "=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return os.environ.get(key, default)


def _has_real_hf_token() -> bool:
    val = _env_value("PUAH_HF_TOKEN", "") or ""
    return val.startswith("hf_") and "xxxx" not in val


def ensure_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        shutil.copyfile(ROOT / ".env.example", env)
        log(".env נוצר מתוך .env.example.", "ok")
    if not _has_real_hf_token():
        log("PUAH_HF_TOKEN לא מוגדר ב-.env — זיהוי הדוברים לא יעבוד עד שתגדירו טוקן HuggingFace.", "warn")


# ---------- בדיקות תנאים מקדימים ----------

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        log("ffmpeg נמצא.", "ok")
    else:
        log("ffmpeg לא נמצא ב-PATH — התמלול ייכשל. התקינו: winget install Gyan.FFmpeg", "warn")


def check_ollama() -> None:
    if (_env_value("PUAH_SUMMARY_ENABLED", "true") or "").lower() in ("0", "false", "no"):
        return
    base = (_env_value("PUAH_LLM_BASE_URL", "http://localhost:11434/v1") or "").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/models", timeout=3):
            log("שרת מודל השפה זמין — סיכום אוטומטי פעיל.", "ok")
    except Exception:
        model = _env_value("PUAH_LLM_MODEL", "") or ""
        log("שרת מודל השפה לא זמין — הסיכום יידלג (התמלול עצמו יעבוד). להפעלת סיכום: התקינו Ollama והריצו:", "warn")
        if model:
            log(f"    ollama pull {model}", "info")


# ---------- הפעלת השירותים ----------

def _stream(proc: subprocess.Popen, tag: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{tag}] {line.rstrip()}", flush=True)


def wait_for_api(port: int, timeout: float = 120.0) -> bool:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def launch(py: Path, host: str, port: int, open_browser: bool, offline: bool = False) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # אזהרת symlinks קוסמטית ב-Windows
    if offline:
        # שימוש אך ורק במטמון המקומי — אפס קריאות רשת בזמן ריצה (air-gapped).
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        log("מצב OFFLINE: שימוש במטמון מקומי בלבד (אין קריאות רשת בזמן ריצה).", "ok")

    popen_kw = dict(
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    log("מפעיל את תהליך ה-worker (טעינת מודלים על ה-GPU עשויה לקחת רגע)…", "step")
    worker = subprocess.Popen([str(py), "-m", "app.worker"], **popen_kw)
    log("מפעיל את שרת ה-API…", "step")
    api = subprocess.Popen(
        [str(py), "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)],
        **popen_kw,
    )

    threading.Thread(target=_stream, args=(worker, "worker"), daemon=True).start()
    threading.Thread(target=_stream, args=(api, "api"), daemon=True).start()

    url = f"http://localhost:{port}"
    if wait_for_api(port):
        log(f"המערכת פעילה ומוכנה לשימוש:  {url}", "ok")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
    else:
        log("שרת ה-API לא הגיב בזמן — עיינו בלוג שלמעלה לאיתור התקלה.", "warn")

    log("לעצירת המערכת לחצו Ctrl+C.", "info")
    rc = 0
    try:
        while True:
            if worker.poll() is not None:
                log("תהליך ה-worker הסתיים באופן בלתי צפוי — עוצר.", "err")
                rc = worker.returncode or 1
                break
            if api.poll() is not None:
                log("שרת ה-API הסתיים באופן בלתי צפוי — עוצר.", "err")
                rc = api.returncode or 1
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("מתקבל Ctrl+C — עוצר את השירותים…", "step")
    finally:
        for p in (api, worker):
            if p.poll() is None:
                p.terminate()
        for p in (api, worker):
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
    log("המערכת הופסקה.", "ok")
    return rc


# ---------- ראשי ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="הפעלה בלחיצה אחת של מערכת התמלול של פוע\"ה")
    p.add_argument("--reinstall", action="store_true", help="התקנת התלויות מחדש")
    p.add_argument("--skip-install", action="store_true", help="דילוג על בדיקת/התקנת תלויות")
    p.add_argument("--cpu", action="store_true", help="התקנת torch בגרסת CPU (ללא CUDA)")
    p.add_argument("--cuda-index", default=CUDA_INDEX, help="index-url של torch ל-CUDA (ברירת מחדל cu128)")
    p.add_argument("--skip-pii", action="store_true", help="דילוג על הקמת סביבת ה-PII (.venv-pii)")
    p.add_argument("--pii-python", default=None, help="נתיב ל-Python 3.8–3.10 לבניית .venv-pii")
    p.add_argument("--no-venv", action="store_true", help="שימוש ב-Python הנוכחי במקום ב-venv")
    p.add_argument("--no-browser", action="store_true", help="לא לפתוח דפדפן אוטומטית")
    p.add_argument(
        "--install-only",
        action="store_true",
        help="התקנה בלבד (venv + תלויות + .venv-pii) ויציאה — לשימוש ב-startup-script של שרת",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="מצב מנותק-רשת: מדלג על התקנות ומכריח שימוש במטמון מקומי בלבד "
        "(HF_HUB_OFFLINE). השתמשו רק לאחר שכל המודלים כבר ירדו פעם אחת.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    log('=== מערכת תמלול וזיהוי דוברים — מכון פוע"ה ===', "step")

    py = Path(sys.executable) if args.no_venv else ensure_venv()

    if args.skip_install or args.offline:
        log("מדלג על התקנת תלויות.", "info")
    elif args.reinstall or not MARKER.exists():
        try:
            install_deps(py, cpu=args.cpu, cuda_index=args.cuda_index)
        except subprocess.CalledProcessError as exc:
            log(f"התקנת התלויות נכשלה (קוד {exc.returncode}). בדקו את הפלט שלמעלה.", "err")
            return 1
    else:
        log("התלויות כבר מותקנות — מדלג (להתקנה מחדש: --reinstall).", "ok")

    ensure_env()
    check_ffmpeg()
    check_ollama()

    # סביבת PII מבודדת (best-effort — כשל לא חוסם את ההפעלה)
    if args.skip_pii or args.offline:
        log("מדלג על הקמת סביבת ה-PII.", "info")
    else:
        try:
            ensure_pii_venv(reinstall=args.reinstall, pii_python=args.pii_python)
        except subprocess.CalledProcessError as exc:
            log(f"הקמת סביבת ה-PII נכשלה (קוד {exc.returncode}) — הסרת PII תדולג. אפשר להקים ידנית בהמשך.", "warn")

    if args.install_only:
        log("ההתקנה הושלמה (--install-only) — לא מפעיל שרתים.", "ok")
        return 0

    return launch(
        py, host=args.host, port=args.port, open_browser=not args.no_browser, offline=args.offline
    )


if __name__ == "__main__":
    sys.exit(main())
