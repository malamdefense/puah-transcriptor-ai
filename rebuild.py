#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rebuild.py — One-click activation of the entire Pua Institute transcription and speaker identification system.
What the script does (in order):
  1. Creates a virtual Python environment (.venv) if it doesn't exist.
  2. Installs the dependencies (torch+CUDA and then requirements) — only the first time, or with --reinstall.
  3. Creates a .env file from .env.example if missing, and warns if HuggingFace token is missing.
  4. Checks if ffmpeg is available, and if Ollama (the summary model) is available — warnings only, not blocking.
  5. Launches the worker process (processing on the GPU) and the API server, and opens the browser.

Run:
    python rebuild.py

דגלים שימושיים:
    --reinstall      reinstall dependencies (e.g. after changing requirements)
    --skip-install   skip completely the dependency check/installation
    --cpu            install torch in CPU-only mode (no CUDA)
    --no-venv        use the current Python instead of creating a venv
    --no-browser     don't open the browser automatically
    --host / --port  default 0.0.0.0:8000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

    # Proper UTF-8 encoding in the console on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MARKER = VENV / ".puah_deps_ok"  # Marker that the dependencies were installed successfully
PII_VENV = ROOT / ".venv-pii"
PII_MARKER = PII_VENV / ".puah_pii_ok"
# Second PII engine (Presidio, current release) — its own venv: needs numpy<2.5 / spaCy>=3.4,
# which fits neither .venv (numpy 2.5 + CUDA torch) nor .venv-pii (HebSafeHarbor pins Presidio 2.2.31).
PII_RUNNER = ROOT / "app" / "pii_runner.py"
PRESIDIO_RUNNER = ROOT / "app" / "presidio_runner.py"

# Samples the verification redacts. Each carries more than one independent signal, so a single
# recogniser going quiet does not fail a healthy install:
#   - HebSafeHarbor: a personal name and a city, both plain Hebrew NER.
#   - Presidio: an Israeli ID that passes the national checksum (123456782 does; a random
#     9-digit number does not, and would silently never match) plus an e-mail address, which
#     is the most deterministic built-in recogniser there is.
PII_SAMPLE_HEB = "\u05e9\u05de\u05d9 \u05d9\u05d5\u05e1\u05d9 \u05db\u05d4\u05df \u05d5\u05d0\u05e0\u05d9 \u05d2\u05e8 \u05d1\u05ea\u05dc \u05d0\u05d1\u05d9\u05d1."
PII_SAMPLE_PRESIDIO = (
    "\u05ea\u05e2\u05d5\u05d3\u05ea \u05d4\u05d6\u05d4\u05d5\u05ea \u05e9\u05dc\u05d9 \u05d4\u05d9\u05d0 123456782 "
    "\u05d5\u05d4\u05de\u05d9\u05d9\u05dc \u05e9\u05dc\u05d9 yossi@example.com."
)
VERIFY_TIMEOUT_HEB = 900.0       # loads a transformer on first use
VERIFY_TIMEOUT_PRESIDIO = 300.0
PROBE_TIMEOUT = 120.0

PRESIDIO_VENV = ROOT / ".venv-presidio"
PRESIDIO_MARKER = PRESIDIO_VENV / ".puah_presidio_ok"
# cu128 = CUDA 12.8 — supports Ampere/Ada/Hopper/Blackwell. For older GPUs, use --cuda-index cu124.
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
    log("Creating a virtual Python environment (.venv)…", "step")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    return py


def install_deps(py: Path, cpu: bool, cuda_index: str = CUDA_INDEX) -> None:
    log("Installing dependencies — this may take a few minutes (large download of torch)…", "step")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    # First requirements (pyannote pulls torchcodec + torch CPU dependency), then installs
    # torch in CUDA *last* — otherwise the CPU torch dominates and cancels the GPU.
    log("Installing dependencies (requirements.txt)…")
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    if cpu:
        log("CPU mode — skipping installation of torch for CUDA.")
    else:
        log(f"Installing torch + torchaudio for CUDA (last, dominates CPU): {cuda_index}")
        subprocess.check_call([str(py), "-m", "pip", "uninstall", "-y", "torch", "torchaudio"])
        subprocess.check_call(
            [str(py), "-m", "pip", "install", "torch", "torchaudio", "--index-url", cuda_index]
        )
    MARKER.write_text("ok", encoding="utf-8")
    log("All dependencies installed.", "ok")


def _run_engine(py: Path, runner: Path, payload: dict, timeout: float) -> tuple[bool, str, list]:
    """Call a PII runner over its normal JSON protocol. Returns (ok, reason, texts)."""
    try:
        proc = subprocess.run(
            [str(py), str(runner)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"no answer within {timeout:g}s", []
    except OSError as exc:
        return False, f"could not be started ({py}): {exc}", []

    if proc.returncode != 0:
        # Both runners print one summary line before giving up; presidio_runner precedes it with
        # a note per stubbed spaCy extension, so the last line is the informative one.
        lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
        return False, (lines[-1].strip()[:300] if lines else f"exit code {proc.returncode}"), []

    try:
        texts = json.loads(proc.stdout)["texts"]
    except Exception as exc:  # noqa: BLE001
        return False, f"unexpected output: {exc}", []
    return True, "", texts


def verify_engine(py: Path, runner: Path, label: str, sample: str, timeout: float) -> bool:
    """Prove the engine really removes identifying details — do not take pip's word for it.

    pip returning 0 means the wheels landed. It says nothing about an NER model that failed to
    download, a half-built venv, or a native extension the OS refuses to load. Redacting one
    Hebrew sentence exercises the import, the model load and the recognisers in one go, and is
    the only evidence worth writing a "this environment is ready" marker on.
    """
    log(f"Verifying {label} on a Hebrew sample…", "step")
    ok, reason, texts = _run_engine(py, runner, {"texts": [sample]}, timeout)
    if not ok:
        log(f"{label} does not work: {reason}", "err")
        return False
    if not texts or texts[0] == sample:
        log(f"{label} ran but removed nothing — its model or recognisers are probably missing.", "err")
        log(f"    in:  {sample}", "info")
        log(f"    out: {texts[0] if texts else '(no output)'}", "info")
        return False
    log(f"{label} verified — {texts[0]}", "ok")
    return True


def probe_engine(py: Path, runner: Path, label: str) -> bool:
    """Cheap import check for an environment that is already marked as installed.

    A marker records that the engine worked when it was built; it cannot record that the engine
    still works. An environment can be killed later without anyone touching it — a Windows
    Application Control policy refusing a DLL did exactly that here, and the stale marker meant
    every later run reported "already installed" while nothing could be redacted.
    """
    if not py.exists():
        log(f"{label}: interpreter missing ({py}).", "warn")
        return False
    try:
        proc = subprocess.run(
            [str(py), str(runner), "--probe"],
            input="", capture_output=True, text=True, encoding="utf-8", timeout=PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"{label}: could not be checked ({exc}).", "warn")
        return False
    if proc.returncode == 0:
        return True
    lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
    log(f"{label} is installed but no longer loads: "
        f"{lines[-1].strip()[:300] if lines else f'exit code {proc.returncode}'}", "err")
    return False


def _find_pii_base_python(explicit: str | None) -> tuple[list[str] | None, str | None]:
    """Finds Python 3.8–3.10 (required for spaCy 3.2 of HebSafeHarbor). Returns (cmd, version)."""
    candidates: list[list[str]] = []
    if explicit:
        candidates.append([explicit])
    for v in ("3.10", "3.9", "3.8"):
        candidates.append(["py", f"-{v}"])      # py launcher (Windows)
        candidates.append([f"python{v}"])       # explicit name (Linux/macOS)
    candidates.append([sys.executable])  # if the current interpreter matches
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
    """Creates a virtual environment isolated for HebSafeHarbor (.venv-pii) — required for PII removal.

    Isolated because HebSafeHarbor requires pydantic<2 / spaCy 3.2 that conflicts with the main application,
    and spaCy 3.2 requires Python 3.8–3.10 (not 3.11+).
    """
    req = ROOT / "requirements-pii.txt"
    if not req.exists():
        log("requirements-pii.txt is missing — skipping creation of the PII environment.", "warn")
        return
    py = venv_python(PII_VENV)
    if PII_MARKER.exists() and not reinstall:
        if probe_engine(py, PII_RUNNER, "HebSafeHarbor"):
            log("The PII environment (.venv-pii) is already installed — skipping.", "ok")
            return
        # Deliberately neither rebuilt nor silenced: the usual cause is external (a blocked DLL,
        # a moved interpreter) and a multi-gigabyte reinstall would not fix it. Warn on every
        # run until someone deals with it — the point is that this stops going unnoticed.
        log("PII removal is NOT working. Summaries will be refused (fail-closed) until it is fixed.", "err")
        log("If you believe the environment is merely corrupt, rebuild it with --reinstall.", "info")
        return
    if not py.exists():
        base, ver = _find_pii_base_python(pii_python)
        if not base:
            log("Python 3.8–3.10 not found (required for HebSafeHarbor/spaCy 3.2) — PII removal will be skipped.", "warn")
            log("Install:  winget install Python.Python.3.10  and then run again (or: --pii-python <path>).", "info")
            return
        log(f"Building the PII environment (.venv-pii) with Python {ver}…", "step")
        subprocess.check_call(base + ["-m", "venv", str(PII_VENV)])

    log("Installing HebSafeHarbor + Hebrew NER model (he_ner_news_trf) — large download, may take a few minutes…", "step")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(req)])

    if not verify_engine(py, PII_RUNNER, "HebSafeHarbor", PII_SAMPLE_HEB, VERIFY_TIMEOUT_HEB):
        PII_MARKER.unlink(missing_ok=True)
        log("The PII environment was installed but does not redact — not marking it ready.", "err")
        return
    PII_MARKER.write_text("ok", encoding="utf-8")
    log("The PII environment is ready.", "ok")


def _find_presidio_base_python(explicit: str | None) -> tuple[list[str] | None, str | None]:
    """Finds Python 3.10–3.14 (supported by the current Presidio release). Returns (cmd, version).

    Prefers the application's own .venv interpreter (same base Python, no extra install needed).
    """
    candidates: list[list[str]] = []
    if explicit:
        candidates.append([explicit])
    app_py = venv_python()
    if app_py.exists():
        candidates.append([str(app_py)])
    candidates.append([sys.executable])
    for v in ("3.13", "3.12", "3.11", "3.10", "3.14"):
        candidates.append(["py", f"-{v}"])      # py launcher (Windows)
        candidates.append([f"python{v}"])       # explicit name (Linux/macOS)
    candidates.append(["python3"])
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
        if major == 3 and 10 <= minor <= 14:
            return cmd, f"{major}.{minor}"
    return None, None


def ensure_presidio_venv(reinstall: bool = False, presidio_python: str | None = None) -> None:
    """Creates the isolated environment for the second PII engine, Presidio (.venv-presidio).

    The worker chains it after HebSafeHarbor (app/presidio_runner.py as a subprocess). Pure pip
    install, no model download — works air-gapped once the wheels are cached.
    """
    req = ROOT / "requirements-presidio.txt"
    if not req.exists():
        log("requirements-presidio.txt is missing — skipping creation of the Presidio environment.", "warn")
        return
    py = venv_python(PRESIDIO_VENV)
    if PRESIDIO_MARKER.exists() and not reinstall:
        if probe_engine(py, PRESIDIO_RUNNER, "Presidio"):
            log("The Presidio environment (.venv-presidio) is already installed — skipping.", "ok")
            return
        log("The second PII engine is NOT working. PII removal fails closed while it is enabled.", "err")
        log("Rebuild with --reinstall, or set PUAH_PII_PRESIDIO_ENABLED=false to run on HebSafeHarbor alone.", "info")
        return
    if not py.exists():
        base, ver = _find_presidio_base_python(presidio_python)
        if not base:
            log("Python 3.10–3.14 not found — the Presidio PII engine will be skipped (or: --presidio-python <path>).", "warn")
            return
        log(f"Building the Presidio environment (.venv-presidio) with Python {ver}…", "step")
        subprocess.check_call(base + ["-m", "venv", str(PRESIDIO_VENV)])

    log("Installing Presidio (analyzer + anonymizer)…", "step")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(req)])

    if not verify_engine(py, PRESIDIO_RUNNER, "Presidio", PII_SAMPLE_PRESIDIO, VERIFY_TIMEOUT_PRESIDIO):
        PRESIDIO_MARKER.unlink(missing_ok=True)
        log("Presidio was installed but does not redact — not marking it ready.", "err")
        return
    PRESIDIO_MARKER.write_text("ok", encoding="utf-8")
    log("The Presidio environment is ready.", "ok")


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
        log(".env created from .env.example.", "ok")
    if not _has_real_hf_token():
        log("PUAH_HF_TOKEN is not set in .env — speaker identification will not work until you set a HuggingFace token.", "warn")


# ---------- בדיקות תנאים מקדימים ----------

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        log("ffmpeg found.", "ok")
    else:
        log("ffmpeg not found in PATH — transcription will fail. Install: winget install Gyan.FFmpeg", "warn")


def check_ollama() -> None:
    """Checks both conditions for summary: Gemini API key (cloud) and Ollama (embeddings only).

    The local LLM provider was removed — summary is generated by Gemini. Ollama is still required for the embedding model
    (bge-m3) that feeds the semantic search and evaluation.
    """
    if (_env_value("PUAH_SUMMARY_ENABLED", "true") or "").lower() in ("0", "false", "no"):
        return

    # 1) מנוע הסיכום — Gemini
    if _env_value("PUHA_GEMINI_API_KEY", ""):
        log("Gemini API key is set — automatic summary is active.", "ok")
    else:
        log("PUHA_GEMINI_API_KEY is missing — summary will be skipped (transcription itself will still work).", "warn")
        log("Set an API key in .env or in the settings (⚙) in the interface.", "info")

    # 2) Ollama — required for embeddings (RAG/evaluation) only
    base = (_env_value("PUAH_LLM_BASE_URL", "http://localhost:11434/v1") or "").rstrip("/")
    embed = _env_value("PUAH_EMBED_MODEL", "bge-m3") or "bge-m3"
    try:
        with urllib.request.urlopen(base + "/models", timeout=3):
            log(f"Ollama is available — embeddings ({embed}) are active.", "ok")
    except Exception:
        log("Ollama is not available — semantic search/evaluation will be skipped. Install: ollama pull {embed}", "warn")
        log(f"    ollama pull {embed}", "info")


# ---------- Launching the services ----------

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
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if offline:
        # Use only the local cache — zero network calls at runtime (air-gapped).
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        log("Offline mode: using only the local cache (no network calls at runtime).", "ok")

    popen_kw = dict(
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    log("Launching the worker process (loading models on the GPU may take a few minutes)…", "step")
    worker = subprocess.Popen([str(py), "-m", "app.worker"], **popen_kw)
    log("Launching the API server…", "step")
    api = subprocess.Popen(
        [str(py), "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)],
        **popen_kw,
    )

    threading.Thread(target=_stream, args=(worker, "worker"), daemon=True).start()
    threading.Thread(target=_stream, args=(api, "api"), daemon=True).start()

    url = f"http://localhost:{port}"
    if wait_for_api(port):
        log(f"The system is active and ready for use:  {url}", "ok")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
    else:
        log("The API server did not respond in time — check the logs above for the error.", "warn")

    log("To stop the system, press Ctrl+C.", "info")
    rc = 0
    try:
        while True:
            if worker.poll() is not None:
                log("The worker process unexpectedly finished — stopping.", "err")
                rc = worker.returncode or 1
                break
            if api.poll() is not None:
                log("The API server unexpectedly finished — stopping.", "err")
                rc = api.returncode or 1
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("Received Ctrl+C — stopping the services…", "step")
    finally:
        for p in (api, worker):
            if p.poll() is None:
                p.terminate()
        for p in (api, worker):
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
    log("The system has stopped.", "ok")
    return rc


# ---------- ראשי ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click activation of the Pua Institute transcription and speaker identification system")
    p.add_argument("--reinstall", action="store_true", help="reinstall dependencies")
    p.add_argument("--skip-install", action="store_true", help="skip completely the dependency check/installation")
    p.add_argument("--cpu", action="store_true", help="install torch in CPU-only mode (no CUDA)")
    p.add_argument("--cuda-index", default=CUDA_INDEX, help="index-url of torch for CUDA (default cu128)")
    p.add_argument("--skip-pii", action="store_true", help="skip creation of the PII environment (.venv-pii)")
    p.add_argument("--pii-python", default=None, help="path to Python 3.8–3.10 for building .venv-pii")
    p.add_argument("--skip-presidio", action="store_true", help="skip creation of the Presidio PII environment (.venv-presidio)")
    p.add_argument("--presidio-python", default=None, help="path to Python 3.10–3.14 for building .venv-presidio")
    p.add_argument("--no-venv", action="store_true", help="use the current Python instead of creating a venv")
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    p.add_argument(
        "--install-only",
        action="store_true",
        help="installation only (venv + dependencies + .venv-pii + .venv-presidio) and exit — for use with the server's startup script",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="offline mode: skip installation and enforce using only the local cache "
        "(HF_HUB_OFFLINE). Use only after all models have been downloaded once.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    log('=== Pua Institute transcription and speaker identification system ===', "step")

    py = Path(sys.executable) if args.no_venv else ensure_venv()

    if args.skip_install or args.offline:
        log("Skipping dependency installation.", "info")
    elif args.reinstall or not MARKER.exists():
        try:
            install_deps(py, cpu=args.cpu, cuda_index=args.cuda_index)
        except subprocess.CalledProcessError as exc:
            log(f"Dependency installation failed (code {exc.returncode}). Check the output above.", "err")
            return 1
    else:
        log("Dependencies already installed — skipping (for reinstall: --reinstall).", "ok")

    ensure_env()
    check_ffmpeg()
    check_ollama()

    # Isolated PII environment (best-effort — failure does not block the launch)
    if args.skip_pii or args.offline:
        log("Skipping creation of the PII environment.", "info")
    else:
        try:
            ensure_pii_venv(reinstall=args.reinstall, pii_python=args.pii_python)
        except subprocess.CalledProcessError as exc:
            log(f"Creation of the PII environment failed (code {exc.returncode}) — PII removal will be skipped. You can create it manually later.", "warn")

    # Second PII engine — Presidio (best-effort as well)
    if args.skip_presidio or args.offline:
        log("Skipping creation of the Presidio environment.", "info")
    else:
        try:
            ensure_presidio_venv(reinstall=args.reinstall, presidio_python=args.presidio_python)
        except subprocess.CalledProcessError as exc:
            log(f"Creation of the Presidio environment failed (code {exc.returncode}) — PII removal will fail until it exists (or set PUAH_PII_PRESIDIO_ENABLED=false).", "warn")

    if args.install_only:
        log("Installation completed (--install-only) — not launching the servers.", "ok")
        return 0

    return launch(
        py, host=args.host, port=args.port, open_browser=not args.no_browser, offline=args.offline
    )


if __name__ == "__main__":
    sys.exit(main())
