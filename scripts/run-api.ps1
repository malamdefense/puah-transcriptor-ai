# הרצת שרת ה-API + ממשק ה-Web.  פתחו בדפדפן: http://localhost:8000
$ErrorActionPreference = "Stop"
if (Test-Path ".venv\Scripts\Activate.ps1") { & .\.venv\Scripts\Activate.ps1 }
if (-not $env:PUAH_LLM_PROVIDER) { $env:PUAH_LLM_PROVIDER = "ollama" }
if (-not $env:PUAH_LLM_MODEL)    { $env:PUAH_LLM_MODEL    = "muse-glimmer" }
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
