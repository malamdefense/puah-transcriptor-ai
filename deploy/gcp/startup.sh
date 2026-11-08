#!/usr/bin/env bash
# startup-script ל-VM של GCP — מריץ פעם אחת בעלייה הראשונה של המכונה.
# מתקין: דרייברי NVIDIA, ffmpeg, Ollama, את האפליקציה, ואת השירותים.
#
# מטא-דאטה שהסקריפט קורא (מוגדרת ב-gcloud compute instances create):
#   app-bucket    — באקט GCS שממנו נמשך קוד האפליקציה      (חובה)
#   hf-secret     — שם הסוד ב-Secret Manager עם טוקן ה-HF    (חובה)
#   gemini-secret — שם הסוד ב-Secret Manager עם מפתח Gemini  (אופציונלי; ניתן להזין גם ב-⚙ בממשק)
#   embed-model   — מודל ה-embedding (ברירת מחדל bge-m3)
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
GEMINI_SECRET="$(meta gemini-secret)"
EMBED_MODEL="$(meta embed-model bge-m3)"
[ -n "$APP_BUCKET" ] || { echo "חסר metadata app-bucket"; exit 1; }

# ---------- 1. חבילות בסיס ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ffmpeg libsndfile1 curl ca-certificates \
    zstd            # נדרש למתקין של Ollama

# ---------- 2. דרייברי NVIDIA (550+ — Ollama דורש 550, לא ה-535 של מתקין GCP) ----------
# מתקינים ישירות nvidia-driver-550-server: אחרת Ollama נופל ל-CPU ("driver too old").
if ! command -v nvidia-smi >/dev/null 2>&1 || \
   [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | cut -d. -f1)" -lt 550 ] 2>/dev/null; then
  echo "--- מתקין דרייבר NVIDIA 550 ---"
  apt-get install -y nvidia-driver-550-server || {
    # נפילה חזרה למתקין של GCP אם החבילה לא זמינה
    curl -fsSL -o /tmp/install_gpu_driver.py \
      https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
    python3 /tmp/install_gpu_driver.py --force
  }
  # startup-script רץ כ-root בעליית המכונה; טוענים את המודול בלי reboot מלא
  modprobe nvidia nvidia_uvm 2>/dev/null || true
fi
nvidia-smi || { echo "התקנת ה-GPU נכשלה — ודא reboot"; exit 1; }

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

# ---------- 5. קובץ .env (טוקן HF ומפתח Gemini אופציונלי מ-Secret Manager) ----------
if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi
HF_TOKEN="$(gcloud secrets versions access latest --secret="$HF_SECRET" | tr -d '\r\n')"
GEMINI_KEY=""
if [ -n "$GEMINI_SECRET" ]; then
  GEMINI_KEY="$(gcloud secrets versions access latest --secret="$GEMINI_SECRET" 2>/dev/null | tr -d '\r\n' || true)"
  [ -n "$GEMINI_KEY" ] || echo "⚠ gemini-secret='${GEMINI_SECRET}' לא נקרא מ-Secret Manager — ניתן להזין PUHA_GEMINI_API_KEY ב-.env או ב-⚙ בממשק."
fi

python3 - "$APP_DIR/.env" "$HF_TOKEN" "$EMBED_MODEL" "$GEMINI_KEY" <<'PY'
import pathlib, sys

env_path = pathlib.Path(sys.argv[1])
hf_token, embed_model, gemini_key = sys.argv[2], sys.argv[3], sys.argv[4]
lines = env_path.read_text(encoding="utf-8").splitlines()
updates = {
    "PUAH_HF_TOKEN": hf_token,
    "PUAH_EMBED_MODEL": embed_model,
}
if gemini_key:
    updates["PUHA_GEMINI_API_KEY"] = gemini_key

seen = set()
out = []
warn_ollama = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("PUAH_LLM_PROVIDER=") and not stripped.startswith("#"):
        if stripped.split("=", 1)[1].strip().lower() == "ollama":
            warn_ollama = True
    replaced = False
    for k, v in updates.items():
        if stripped.startswith(k + "="):
            out.append(f"{k}={v}")
            seen.add(k)
            replaced = True
            break
    if not replaced:
        out.append(line)

for k, v in updates.items():
    if k not in seen:
        out.append(f"{k}={v}")

env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
if warn_ollama:
    print(
        "⚠ אזהרה: בקובץ .env מוגדר PUAH_LLM_PROVIDER=ollama (כמו בפיתוח מקומי), "
        "אך startup.sh מושך רק את מודל ה-embedding ולא מודל שפה מקומי. לשימוש ב-Gemini מחק/עדכן ל-PUAH_LLM_PROVIDER=gemini.",
        file=sys.stderr,
    )
PY
chown puah:puah "$APP_DIR/.env" && chmod 600 "$APP_DIR/.env"
if ! grep -qE '^PUHA_GEMINI_API_KEY=.+' "$APP_DIR/.env"; then
  echo "⚠ הערה: PUHA_GEMINI_API_KEY אינו מוגדר ב-.env (ניתן להגדיר דרך מטא-דאטה gemini-secret או במסך הגדרות ⚙ בממשק)."
fi

# ---------- 6. התקנת התלויות (venv ראשי + .venv-pii + .venv-presidio) ----------
# Ubuntu 22.04 מגיע עם Python 3.10 — מתאים גם לאפליקציה וגם ל-HebSafeHarbor.
echo "--- מתקין תלויות (עשוי לקחת מספר דקות) ---"
cd "$APP_DIR"
sudo -u puah python3 rebuild.py --install-only

# ---------- 7. משיכת מודל ה-embedding ----------
# הסיכום מופק ב-Gemini (ענן); Ollama משמש ל-embeddings בלבד.
echo "--- מושך מודל embedding: ${EMBED_MODEL} ---"
ollama pull "$EMBED_MODEL"

# ---------- 8. שירותי systemd ----------
install -m 644 "$APP_DIR/deploy/systemd/puah-worker.service" /etc/systemd/system/
install -m 644 "$APP_DIR/deploy/systemd/puah-api.service"   /etc/systemd/system/
# ה-API חייב להאזין ל-0.0.0.0 כדי שה-Load Balancer יגיע אליו (ה-firewall חוסם כל השאר).
sed -i 's|--host 127.0.0.1|--host 0.0.0.0|' /etc/systemd/system/puah-api.service
systemctl daemon-reload
systemctl enable --now puah-worker puah-api

echo "=== ✓ Puah מוכן. בדיקה: curl -s localhost:8000/api/health ==="
