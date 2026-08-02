# הרצת תהליך ה-worker (מעבד את התור על ה-GPU). השאירו פעיל ברקע.
$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }
python -m app.worker
