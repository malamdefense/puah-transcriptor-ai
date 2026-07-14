# הקמת סביבת פיתוח מקומית ב-Windows (PowerShell).
# הרצה:  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "== הקמת סביבת Python וירטואלית ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

Write-Host "== התקנת התלויות ==" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "== התקנת torch עם CUDA 12.8 (אחרון — גובר על torch CPU ש-pyannote מושך) ==" -ForegroundColor Cyan
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "נוצר קובץ .env — ערכו אותו והכניסו PUAH_HF_TOKEN." -ForegroundColor Yellow
}

Write-Host "`nהוכן. ודאו ש-ffmpeg מותקן (ffmpeg -version)." -ForegroundColor Green
Write-Host "הרצה: חלון אחד -> .\scripts\run-worker.ps1 ; חלון שני -> .\scripts\run-api.ps1" -ForegroundColor Green
