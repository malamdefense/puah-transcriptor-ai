# Hebrew transcription and speaker identification system — Puha Institute

Eco System for receiving recordings (mp4 and more), **Hebrew transcription** and **speaker identification** (diarization)
within the transcription. **Transcription and speaker identification run On-Prem** — no recording leaves the organization.
**Summary and feedback are produced in Google Gemini (cloud)** After de-identification locally — the local model provider
has been removed from the system (see "Summary Engine" below).


## What the system does

1. Receives an `mp4` recording (and also `mov`, `m4a`, `mp3`, `wav`, `mkv`, `webm` and more) via a Web interface.
2. Extracts audio with `ffmpeg`.
3. Transcribes into Hebrew with the **ivrit.ai** model (based on Whisper, one of the best in Hebrew) via `faster-whisper`.
4. Identifies **who spoke when** with `pyannote.audio`.
5. Merges: attributes each word/segment to the speaker based on time overlap, and produces a readable transcript with speaker labels.
6. **Automatically summarizes** the conversation (summary, main points, actions and follow-up, topics and keywords)
Using **Google Gemini** (cloud) — the content undergoes local PII removal before sending.
7. Allows viewing, synchronized listening, **changing speaker names** (e.g. "Speaker 1" → "Advisor"), and exporting to TXT / SRT / VTT / JSON / **Word**.
8. **Full search in all transcriptions** (including handling Hebrew prefixes) + **Search and highlight within transcription** + clickable keywords.
9. **Produces a Word document** for each transcription, and a **de-identified version (PII)** — saved to a separate file (`*.pii`).
10. **Produces a "Summary for the Client" and "Feedback for the Advisor/Rabbi"** for each call according to the designated prompt of the Po'a (family purity, fertility, halacha).
11. **Settings screen** ⚙ — Editing the System Prompts of the summaries directly from the interface (with restoration to default),
and adding/replacing **logos on the right and left** in the title. The connection indicator is only displayed when there is a malfunction.

The API and the worker are two separate processes that share the `data/` folder and the DB.
The worker processes **one task at a time** (the GPU is a serial resource) to avoid memory overflow.

## Prerequisites

- **NVIDIA GPU** + latest drivers (recommended; also works on CPU, but slow).
- **Python 3.10–3.12** (recommended; 3.13 may require dependencies adjustments).
- **ffmpeg** in PATH. On Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg`.
- **HuggingFace token** with acceptance of the terms of use of the recognition models (see below).

### Recognition model approval (one-time)

`pyannote` is a "gated" model. You must log in with your HuggingFace account and accept the terms on both pages:

1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0

Then create a read token at https://huggingface.co/settings/tokens and put it in `.env` as `PUAH_HF_TOKEN`.

Output: `data/eval/report.html` (readable report), `report.xlsx`, and `report.json`. Transcripts are cached —
Fast replay.
## Quick Start — One Command ⭐

```bash
python rebuild.py
```

That's it. The script does everything automatically:
1. Creates a `.venv` environment and installs all dependencies (torch+CUDA and requirements) — **first time only**.
2. Creates `.env` from the example if missing, and checks for ffmpeg and Ollama (warnings if missing).
3. Starts the worker and API server, and opens a browser at **http://localhost:8000**.

On subsequent runs it skips the installation and boots up in seconds. To stop: `Ctrl+C`.

**Before the first run** Recommended: Install ffmpeg, set `PUAH_HF_TOKEN` in `.env` (for speaker detection),
Set `PUHA_GEMINI_API_KEY` (summarization engine), and run `ollama pull bge-m3` (embeddings).
The script will warn about each of them if missing — but the system will boot anyway.

Flags: `--reinstall` · `--cpu` (CPU torch) · `--skip-install` · `--offline` (network-disconnected) · `--no-browser` · `--port 8000`

> On the first run, the worker will download the models (several GB). This happens once and is cached.