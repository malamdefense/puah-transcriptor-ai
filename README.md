# Hebrew Transcription & Speaker Diarization System — Puah Institute

An end-to-end ecosystem for ingesting recordings (mp4 and more), performing **Hebrew transcription**, and identifying **speakers (diarization)** within the transcript.

The entire system runs **On-Prem** no recordings or transcripts leave the organization, ensuring full compliance with strict privacy and sensitivity requirements.



![System Architector](diagram.png)

---

## What the System Does

1. Accepts recordings (`mp4`, `mov`, `m4a`, `mp3`, `wav`, `mkv`, `webm`, etc.) via a Web interface.  
2. Extracts audio using `ffmpeg`.  
3. Transcribes Hebrew using **ivrit.ai** (Whisper-based, optimized for Hebrew) via `faster-whisper`.  
4. Performs **speaker diarization** using `pyannote.audio`.  
5. Merges outputs - assigning each segment to a speaker based on time overlap.  
6. Generates an **automatic summary** (highlights, actions, topics, keywords) using a **local LLM** (Ollama / vLLM / LM Studio).  
7. Enables viewing, playback sync, **speaker renaming**, and export to TXT / SRT / VTT / JSON / Word.  
8. Provides **full-text search** (Hebrew-aware) and in-transcript highlighting.  
9. Generates a **Word document** and a **PII-redacted version** using HebSafeHarbor.  
10. Produces:
    - **Client Summary**
    - **Advisor/Rabbi Feedback**
    Both exportable to Excel.

---

## Architecture

| Component | Description |
|----------|-------------|
| `app/main.py` | FastAPI server (uploads, API, UI, exports) |
| `app/worker.py` | GPU worker processing queue |
| `app/pipeline/` | Audio → Transcription → Diarization → Merge → Summary → Export → PII |
| `app/db.py` | SQLite queue + FTS5 Hebrew search |
| `web/` | RTL Hebrew UI (no build step) |

- API and worker run as separate processes  
- Shared `data/` directory  
- **Single GPU worker** to prevent memory overflow  

---

## Prerequisites

- NVIDIA **GPU** (recommended)
- **Python 3.10–3.12**
- `ffmpeg` in PATH
- HuggingFace Token (for diarization models)

---

## HuggingFace Setup (Required)

Approve models:

- https://huggingface.co/pyannote/speaker-diarization-3.1  
- https://huggingface.co/pyannote/segmentation-3.0  

Then create a token:
https://huggingface.co/settings/tokens  

Add to `.env`:
```

PUAH\_HF\_TOKEN=your\_token

````

---

## Local LLM (Optional but Recommended)

Default:

```bash
ollama pull gemma4:12b
````

| Model       | Benefit                | Env Value    |
| ----------- | ---------------------- | ------------ |
| Gemma 4 12B | Reliable, multilingual | `gemma4:12b` |
| DictaLM 3.0 | Best Hebrew quality    | `dictalm3`   |

***

## Quick Start ⭐

```bash
python rebuild.py
```

This will:

* Create virtual environments
* Install dependencies
* Validate ffmpeg & Ollama
* Launch API + worker
* Open browser at `http://localhost:8000`

***

## Manual Setup (Windows)

```powershell
.\scripts\setup.ps1
.\scripts\run-worker.ps1
.\scripts\run-api.ps1
```

***

## Air-Gapped Deployment 🔒

* No runtime internet usage
* All inference runs locally
* No telemetry

Offline mode:

```bash
python rebuild.py --offline
```

***

## Docker Deployment

```bash
cp .env.example .env
docker compose up --build -d
```

***

## Configuration (`.env`)

| Variable               | Description         |
| ---------------------- | ------------------- |
| `PUAH_HF_TOKEN`        | HuggingFace token   |
| `PUAH_ASR_MODEL`       | Transcription model |
| `PUAH_DEVICE`          | `cuda` / `cpu`      |
| `PUAH_LLM_MODEL`       | Summarization model |
| `PUAH_LLM_BASE_URL`    | LLM endpoint        |
| `PUAH_LLM_NUM_CTX`     | Context window      |
| `PUAH_SUMMARY_ENABLED` | Enable summaries    |
| `PUAH_PII_ENABLED`     | Enable PII removal  |
| `PUAH_DATA_DIR`        | Storage directory   |

***

## API

| Method | Endpoint                    | Description        |
| ------ | --------------------------- | ------------------ |
| POST   | `/api/jobs`                 | Upload recording   |
| GET    | `/api/jobs`                 | List jobs          |
| GET    | `/api/jobs/{id}`            | Job status         |
| GET    | `/api/jobs/{id}/result`     | Full transcript    |
| PUT    | `/api/jobs/{id}/speakers`   | Rename speakers    |
| POST   | `/api/jobs/{id}/summarize`  | Generate summary   |
| GET    | `/api/search?q=`            | Search transcripts |
| GET    | `/api/llm/status`           | LLM status         |
| GET    | `/api/export/feedback.xlsx` | Export Excel       |
| DELETE | `/api/jobs/{id}`            | Delete job         |

Example:

```bash
curl -F "file=@meeting.mp4" -F "num_speakers=2" http://localhost:8000/api/jobs
```

***

## Testing

```bash
python tests/test_merge.py
python tests/test_search_summary.py
python tests/test_export_pii.py
python tests/test_api.py
```

***

## Accuracy Tips

* Use `whisper-large-v3-ct2` for max accuracy
* Provide number of speakers when possible
* Phone audio is more challenging

***

## Troubleshooting

* **ffmpeg missing** → add to PATH
* **pyannote errors** → verify HF token + approvals
* **CUDA issues** → match `torch` to GPU
* **stuck jobs** → restart worker

***

## Privacy

* All processing is **local**
* Data stored only under `data/`
* No external calls during inference
* Recommended:
  * Disk encryption
  * Retention policies

***

## Puah-Specific Outputs

Each conversation produces:

### Client Summary

* Personal, structured (70–200 words)
* Includes halachic guidance
* Includes medical disclaimer

### Advisor Feedback

* Score (1–5)
* Strengths
* Improvements
* Safety note

***

## Word & PII Export

Generated automatically:

* `*.docx` (full transcript)
* `*.pii.docx` / `*.pii.json` (redacted)

***

## Notes

* PII removal runs in isolated venv (`.venv-pii`)
* Requires Python 3.8–3.10 for compatibility

***
