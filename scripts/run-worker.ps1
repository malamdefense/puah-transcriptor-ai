# הרצת תהליך ה-worker (מעבד את התור על ה-GPU). השאירו פעיל ברקע.
$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }
if (-not $env:PUAH_LLM_PROVIDER) { $env:PUAH_LLM_PROVIDER = "ollama" }
if (-not $env:PUAH_LLM_MODEL)    { $env:PUAH_LLM_MODEL    = "muse-glimmer" }
python -m app.worker
