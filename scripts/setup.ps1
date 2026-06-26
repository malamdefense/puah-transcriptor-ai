# Create a local development environment in Windows (PowerShell).
# Run:  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "== Create Python virtual environment ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

Write-Host "== Install dependencies ==" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "== Install torch with CUDA 12.8 (latest — beats torch CPU that pyannote pulls) ==" -ForegroundColor Cyan
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env file — edit it and add PUAH_HF_TOKEN." -ForegroundColor Yellow
}

Write-Host "`nReady. Verify ffmpeg is installed (ffmpeg -version)." -ForegroundColor Green
Write-Host "Run: one window -> .\scripts\run-worker.ps1 ; second window -> .\scripts\run-api.ps1" -ForegroundColor Green
