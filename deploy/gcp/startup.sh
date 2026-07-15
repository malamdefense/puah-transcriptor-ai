#!/usr/bin/env bash
# startup-script ל-VM של GCP — מריץ פעם אחת בעלייה הראשונה של המכונה.
# מתקין: דרייברי NVIDIA, ffmpeg, Ollama, את האפליקציה, ואת השירותים.
#
# מטא-דאטה שהסקריפט קורא (מוגדרת ב-gcloud compute instances create):
#   app-bucket   — באקט GCS שממנו נמשך קוד האפליקציה   (חובה)
#   hf-secret    — שם הסוד ב-Secret Manager עם טוקן ה-HF (חובה)
#   llm-model    — מודל הסיכום (ברירת מחדל gemma4:12b)
#   embed-model  — מודל ה-embedding (ברירת מחדל bge-m3)
#
# לוגים:  sudo journalctl -u google-startup-scripts -f

set -euo pipefail
exec > >(tee -a /var/log/puah-startup.log) 2>&1
echo "=== Puah startup $(date -Is) ==="

APP_DIR=/opt/puah
meta() { curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || echo "${2:-}"; }

APP_BUCKET="$(meta app-bucket)"
HF_SECRET="$(meta hf-secret)"
LLM_MODEL="$(meta llm-model gemma4:12b)"
EMBED_MODEL="$(meta embed-model bge-m3)"
[ -n "$APP_BUCKET" ] || { echo "חסר metadata app-bucket"; exit 1; }

# ---------- 1. חבילות בסיס ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ffmpeg libsndfile1 curl ca-certificates \
    zstd            # נדרש למתקין של Ollama

# ---------- 2. דרייברי NVIDIA (המתקין הרשמי של GCP) ----------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- מתקין דרייברי NVIDIA ---"
  curl -fsSL -o /tmp/install_gpu_driver.py \
    https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
  python3 /tmp/install_gpu_driver.py --force
fi
nvidia-smi || { echo "התקנת ה-GPU נכשלה"; exit 1; }

# ---------- 3. Ollama (מותקן כשירות systemd) ----------
if ! command -v ollama >/dev/null 2>&1; then
  echo "--- מתקין Ollama ---"
  curl -fsSL https://ollama.com/install.sh | sh
fi
systemctl enable --now ollama
until curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 2; done

# ---------- 4. קוד האפליקציה מ-GCS ----------
echo "--- מוריד את האפליקציה מ-gs://${APP_BUCKET} ---"
id -u puah >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin puah
mkdir -p "$APP_DIR"
gcloud storage rsync -r "gs://${APP_BUCKET}/puah" "$APP_DIR" --delete-unmatched-destination-objects
chown -R puah:puah "$APP_DIR"

# ---------- 5. קובץ .env (טוקן HF מ-Secret Manager) ----------
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi
HF_TOKEN="$(gcloud secrets versions access latest --secret="$HF_SECRET")"
sed -i "s|^PUAH_HF_TOKEN=.*|PUAH_HF_TOKEN=${HF_TOKEN}|"        "$APP_DIR/.env"
sed -i "s|^PUAH_LLM_MODEL=.*|PUAH_LLM_MODEL=${LLM_MODEL}|"     "$APP_DIR/.env"
sed -i "s|^PUAH_EMBED_MODEL=.*|PUAH_EMBED_MODEL=${EMBED_MODEL}|" "$APP_DIR/.env"
chown puah:puah "$APP_DIR/.env" && chmod 600 "$APP_DIR/.env"

# ---------- 6. התקנת התלויות (venv ראשי + .venv-pii) ----------
# Ubuntu 22.04 מגיע עם Python 3.10 — מתאים גם לאפליקציה וגם ל-HebSafeHarbor.
echo "--- מתקין תלויות (עשוי לקחת מספר דקות) ---"
cd "$APP_DIR"
sudo -u puah python3 rebuild.py --install-only

# ---------- 7. משיכת מודלי ה-LLM ----------
echo "--- מושך מודלים: ${LLM_MODEL} + ${EMBED_MODEL} ---"
ollama pull "$LLM_MODEL"
ollama pull "$EMBED_MODEL"

# ---------- 8. שירותי systemd ----------
install -m 644 "$APP_DIR/deploy/systemd/puah-worker.service" /etc/systemd/system/
install -m 644 "$APP_DIR/deploy/systemd/puah-api.service"   /etc/systemd/system/
# ה-API חייב להאזין ל-0.0.0.0 כדי שה-Load Balancer יגיע אליו (ה-firewall חוסם כל השאר).
sed -i 's|--host 127.0.0.1|--host 0.0.0.0|' /etc/systemd/system/puah-api.service
systemctl daemon-reload
systemctl enable --now puah-worker puah-api

echo "=== ✓ Puah מוכן. בדיקה: curl -s localhost:8000/api/health ==="
