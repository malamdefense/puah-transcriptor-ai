# Production Deployment Guide — Puah Institute Transcription System

This guide describes **On-Prem** deployment of the system on a production server with GPU.  
All processing is local; conversation content never leaves the server (see [README → Air-Gapped](README.md)).

> **The recommended path is native deployment** (not Docker) — it has been fully tested end-to-end and includes all capabilities  
> (transcription, diarization, summarization, client summary, feedback, and PII removal).  
> Docker deployment is provided at the end as a **partial alternative** (see limitations there).

---

## 1. Architecture Overview

The system consists of three local processes sharing the `data/` directory:

| Process | Role | Resource |
|--------|------|---------|
| **API + Web** (`uvicorn app.main:app`) | Uploads, UI, exports, search | Light CPU |
| **Worker** (`python -m app.worker`) | Transcription → diarization → summary → Word → PII | **GPU** (serialized) |
| **Ollama** (`ollama serve`) | LLM for summary/feedback | GPU |

Additionally:
- Isolated `.venv-pii` (Python 3.10) for HebSafeHarbor (called as subprocess)

> **Only ONE worker allowed.**  
> The GPU is a serialized resource — do not run multiple workers (OOM / conflicts).  
> Concurrent uploads are queued and processed sequentially.

---

## 2. Server Requirements

### Hardware
- **NVIDIA GPU** with ≥12GB VRAM (recommended 16GB+)  
  (running Whisper + pyannote + 12B LLM concurrently)
- **Blackwell GPUs (RTX PRO / RTX 50xx)** require **CUDA 12.8 (`cu128`)**
- 16GB+ RAM
- 50GB+ disk (models + uploads + transcripts)
- Encrypted disk recommended

### OS
- Windows Server 2019+ / Windows 11  
- OR Linux (Ubuntu 22.04+)

### Software
- NVIDIA drivers (`nvidia-smi` works)
- **Python 3.13** (main app) + **Python 3.10** (PII env)
- `ffmpeg` in PATH
- **Ollama**

---

## 3. Installation

```powershell
# 1. Clone/copy project (e.g. C:\puah or /opt/puah)

# 2. Setup
python rebuild.py
# For non-Blackwell GPUs:
# python rebuild.py --cuda-index https://download.pytorch.org/whl/cu124

# 3. Pull LLM model
ollama pull gemma4:12b

# 4. Configure .env (see section 4)

# 5. Approve pyannote terms (once):
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://huggingface.co/pyannote/segmentation-3.0
# https://huggingface.co/pyannote/speaker-diarization-community-1
````

> On first job execution, models (\~GBs) will be downloaded and cached.  
> It is recommended to run a full test recording before going live.

***

## 4. Production Configuration (`.env`)

```ini
PUAH_HF_TOKEN=hf_xxxxxxxx
PUAH_LLM_MODEL=gemma4:12b
PUAH_LLM_NUM_CTX=16384
PUAH_CORS_ORIGINS=["https://puah-transcribe.internal"]
PUAH_MAX_UPLOAD_MB=2048
PUAH_DATA_DIR=data
```

> Protect `.env` (contains HF token)  
> Linux: `chmod 600 .env`  
> Windows: restrict to service user

***

## 5. Run as a Service

### Linux — systemd

```bash
sudo cp deploy/systemd/puah-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl enable --now puah-worker puah-api
journalctl -u puah-worker -f
```

***

### Windows — NSSM

```powershell
nssm install PuahWorker "C:\puah\.venv\Scripts\python.exe" "-m app.worker"
nssm set PuahWorker AppDirectory "C:\puah"

nssm install PuahApi "C:\puah\.venv\Scripts\python.exe" "-m uvicorn app.main:app --host 127.0.0.1 --port 8000"

nssm start PuahWorker
nssm start PuahApi
```

Ollama runs as a service automatically — verify via `services.msc`.

***

## 6. Reverse Proxy + HTTPS

* Bind API to `127.0.0.1:8000`
* Expose via nginx / IIS with TLS

Key nginx settings:

* `client_max_body_size 2048m;`
* `proxy_read_timeout 3600s;`

***

## 7. Security

* **No built-in authentication**
  * Restrict to internal network/VPN
  * Add proxy authentication (Basic / SSO / mTLS)
* Bind to localhost only
* Encrypt disk (`data/`)
* Restrict access to `.env` and `data/`
* Use offline mode: `python rebuild.py --offline`

***

## 8. Backup & Recovery

Backup everything under `data/`:

```bash
sqlite3 data/puah.db ".backup '/backup/puah-$(date +%F).db'"
rsync -a data/results data/uploads /backup/puah-data/
```

Recommended:

* Daily backups
* Encrypted storage
* Off-server copy

***

## 9. Operations & Maintenance

* **Logs**
  * Linux: `journalctl -u puah-worker -f`
  * Windows: NSSM logs

* **Disk monitoring**
  * `uploads/`, `results/`, HuggingFace cache, `.ollama`

* **Ollama health**
  ```bash
  curl localhost:11434/api/tags
  ```

* **Upgrade**
  ```bash
  python rebuild.py --reinstall
  sudo systemctl restart puah-worker puah-api
  ```

* **Health endpoints**
  * `/api/health`
  * `/api/llm/status`

***

## 10. Docker Deployment (Partial Alternative)

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull gemma4:12b
```

> ⚠️ Limitation:  
> PII removal (HebSafeHarbor) is **NOT included** (requires Python 3.10 side environment).

***

## 11. Go-Live Checklist

* [ ] `nvidia-smi` works
* [ ] Python 3.13 + 3.10 + ffmpeg + Ollama installed
* [ ] `python rebuild.py` completed successfully
* [ ] LLM model pulled
* [ ] `.env` configured and secured
* [ ] HuggingFace approvals completed
* [ ] Test recording succeeded end-to-end
* [ ] Services configured and auto-starting
* [ ] Reverse proxy + TLS configured
* [ ] Access restricted (no public exposure)
* [ ] Daily backup implemented and tested

```