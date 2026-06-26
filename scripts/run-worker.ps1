# Run the worker process (processes jobs on the GPU). Keep it running in the background.
$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }
python -m app.worker
