# Run the API + Web interface. Open in browser: http://localhost:8000
$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
