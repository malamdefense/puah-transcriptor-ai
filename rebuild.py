#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rebuild.py — One-click run of the entire Puah transcription system.

What it does (in order):
  1. Creates a Python virtual environment (.venv) if it doesn't exist.
  2. Installs dependencies (torch+CUDA then requirements) — only on first run, or with --reinstall.
  3. Creates a .env file from .env.example if missing, and warns if HuggingFace token is missing.
  4. Checks if ffmpeg is available, and if Ollama (summary model) is available — warnings only, not blocking.
  5. Launches the worker process (processing on GPU) and the API server, and opens the browser.

Run:
    python rebuild.py

Useful flags:
    --reinstall      reinstall dependencies (e.g. after changing requirements)
    --skip-install   skip absolute installation of dependencies
    --cpu            install torch in CPU-only mode (no CUDA)
    --no-venv        use the current Python instead of venv
    --no-browser     don't open the browser automatically
    --host / --port  default 0.0.0.0:8000
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

# Print valid UTF-8 in console on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MARKER = VENV / ".puah_deps_ok"  # Marker that dependencies were installed successfully
PII_VENV = ROOT / ".venv-pii"
PII_MARKER = PII_VENV / ".puah_pii_ok"
# cu128 = CUDA 12.8 — supports Ampere/Ada/Hopper/Blackwell. For older GPUs: --cuda-index cu124.
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

_ICONS = {"info": "•", "ok": "✓", "warn": "⚠", "err": "✗", "step": "»"}


def log(msg: str, kind: str = "info") -> None:
    print(f" {_ICONS.get(kind, '•')} {msg}", flush=True)


# ---------- Environment and dependencies ----------

def venv_python(venv: Path = VENV) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_venv() -> Path:
    py = venv_python()
    if py.exists():
        return py
    log("Creating Python virtual environment (.venv)…", "step")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    return py


def install_deps(py: Path, cpu: bool, cuda_index: str = CUDA_INDEX) -> None:
    log("Installing dependencies — this may take a few minutes (large torch download)…", "step")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])

    # First requirements (pyannote pulls torchcodec + torch CPU dependency), then push torch to CUDA *last*
    # so the CPU torch can upgrade — otherwise the CPU torch will take over and cancel the GPU.
    log("Installing dependencies (requirements.txt)…")
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    if cpu:
        log("CPU mode — skipping CUDA torch installation.")
    else:
        log(f"Installing torch + torchaudio for CUDA (last, takes over CPU): {cuda_index}")
        subprocess.check_call([str(py), "-m", "pip", "uninstall", "-y", "torch", "torchaudio"])
        subprocess.check_call(
            [str(py), "-m", "pip", "install", "torch", "torchaudio", "--index-url", cuda_index]
        )
    MARKER.write_text("ok", encoding="utf-8")
    log("All dependencies installed.", "ok")


def _find_pii_base_python(explicit: str | None) -> tuple[list[str] | None, str | None]:
    """Finds Python 3.8–3.10 (required for spaCy 3.2 of HebSafeHarbor). Returns (cmd, version)."""
    candidates: list[list[str]] = []
    if explicit:
        candidates.append([explicit])
    for v in ("3.10", "3.9", "3.8"):
        candidates.append(["py", f"-{v}"])  # py launcher only on Windows
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
    """Creates an isolated Python 3.10 venv for HebSafeHarbor (.venv-pii) — required for PII removal.

    Isolated because HebSafeHarbor requires pydantic<2 / spaCy 3.2 that conflicts with the main app,
    and spaCy 3.2 requires Python 3.8–3.10 (not 3.11+).
    """
    req = ROOT / "requirements-pii.txt"
    if not req.exists():
        log("requirements-pii.txt missing — skipping PII venv creation.", "warn")
        return
    if PII_MARKER.exists() and not reinstall:
        log("PII venv (.venv-pii) already exists — skipping.", "ok")
        return

    py = PII_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        base, ver = _find_pii_base_python(pii_python)
        if not base:
            log("Python 3.8–3.10 not found (required for HebSafeHarbor/spaCy 3.2) — PII removal will be skipped.", "warn")
            log("Install: winget install Python.Python.3.10 then run again (or: --pii-python <path>).", "warn")
            return
        log(f"Building PII venv (.venv-pii) with Python {ver}…", "step")
        subprocess.check_call(base + ["-m", "venv", str(PII_VENV)])

    log("Installing HebSafeHarbor + Hebrew NER model (he_ner_news_trf) — large download, may take a few minutes…", "step")
    subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(req)])
    PII_MARKER.write_text("ok", encoding="utf-8")
    log("PII venv ready.", "ok")


# ---------- Configuration file ----------

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
        log("PUAH_HF_TOKEN not set in .env — speaker detection will fail until you set a HuggingFace token.", "warn")


# ---------- Pre-flight checks ----------

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        log("ffmpeg found.", "ok")
    else:
        log("ffmpeg not found in PATH — transcription will fail. Install: winget install Gyan.FFmpeg", "warn")


def check_ollama() -> None:
    if (_env_value("PUAH_SUMMARY_ENABLED", "true") or "").lower() in ("0", "false", "no"):
        return
    base = (_env_value("PUAH_LLM_BASE_URL", "http://localhost:11434/v1") or "").rstrip("/")
    try:
        with urllib.request.urlopen(base + "/models", timeout=3):
            log("Language model server available — automatic summary is active.", "ok")
    except Exception:
        model = _env_value("PUAH_LLM_MODEL", "") or ""
        log("Language model server not available — summary will be skipped (transcription will still work). Install Ollama and run:", "warn")
        if model:
            log(f"    ollama pull {model}", "info")


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
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Symlink warning on Windows
    if offline:
        # Use only local cache — no network calls during runtime (air-gapped).
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        log("Offline mode: using local cache only (no network calls during runtime).", "ok")

    popen_kw = dict(
        cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    log("Launching worker process (model download on GPU may take a few minutes)…", "step")
    worker = subprocess.Popen([str(py), "-m", "app.worker"], **popen_kw)
    log("Launching API server…", "step")
    api = subprocess.Popen(
        [str(py), "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)],
        **popen_kw,
    )

    threading.Thread(target=_stream, args=(worker, "worker"), daemon=True).start()
    threading.Thread(target=_stream, args=(api, "api"), daemon=True).start()

    url = f"http://localhost:{port}"
    if wait_for_api(port):
        log(f"System ready for use: {url}", "ok")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
    else:
        log("API server didn't respond in time — check the logs above for the error.", "warn")

    log("To stop the system, press Ctrl+C.", "info")
    rc = 0
    try:
        while True:
            if worker.poll() is not None:
                log("Worker process unexpectedly finished — stopping.", "err")
                rc = worker.returncode or 1
                break
            if api.poll() is not None:
                log("API server unexpectedly finished — stopping.", "err")
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
    log("System stopped.", "ok")
    return rc


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-click run of the Puah transcription system.")
    p.add_argument("--reinstall", action="store_true", help="Reinstall dependencies")
    p.add_argument("--skip-install", action="store_true", help="Skip absolute installation of dependencies")
    p.add_argument("--cpu", action="store_true", help="Install torch in CPU-only mode (no CUDA)")
    p.add_argument("--cuda-index", default=CUDA_INDEX, help="index-url of torch for CUDA (default cu128)")
    p.add_argument("--skip-pii", action="store_true", help="Skip PII venv creation (.venv-pii)")
    p.add_argument("--pii-python", default=None, help="Path to Python 3.8–3.10 for PII venv")
    p.add_argument("--no-venv", action="store_true", help="Use the current Python instead of venv")
    p.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: skip installation and force using local cache only "
        "(HF_HUB_OFFLINE). Use only after all models have been downloaded once.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    log('=== Puah transcription system — Puah ===', "step")

    py = Path(sys.executable) if args.no_venv else ensure_venv()

    if args.skip_install or args.offline:
        log("Skipping dependency installation.", "info")
    elif args.reinstall or not MARKER.exists():
        try:
            install_deps(py, cpu=args.cpu, cuda_index=args.cuda_index)
        except subprocess.CalledProcessError as exc:
            log(f"Dependency installation failed (code {exc.returncode}). Check the logs above.", "err")
            return 1
    else:
        log("Dependencies already installed — skipping (for reinstall: --reinstall).", "ok")

    ensure_env()
    check_ffmpeg()
    check_ollama()

    # Isolated PII venv (best-effort — failure doesn't block the launch)
    if args.skip_pii or args.offline:
        log("Skipping PII venv creation.", "info")
    else:
        try:
            ensure_pii_venv(reinstall=args.reinstall, pii_python=args.pii_python)
        except subprocess.CalledProcessError as exc:
            log(f"PII venv creation failed (code {exc.returncode}) — PII removal will be skipped. Can be created manually later.", "warn")

    return launch(
        py, host=args.host, port=args.port, open_browser=not args.no_browser, offline=args.offline
    )


if __name__ == "__main__":
    sys.exit(main())
