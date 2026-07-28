#!/usr/bin/env bash
# =============================================================================
#  Puah — התקנה מלאה על VM של GCP (מריצים *על ה-VM*, לא ב-Cloud Shell)
# -----------------------------------------------------------------------------
#  אידמפוטנטי ומתאושש: אם נדרש reboot (התקנת דרייבר) — הרץ שוב אחריו, והוא
#  ימשיך מהנקודה הנכונה. כל שלב בודק את עצמו ומדלג אם כבר בוצע.
#
#  שימוש:
#     cd ~/<תיקיית-הפרויקט>
#     bash deploy/gcp/02-provision.sh
#
#  אם הקבצים הועברו מ-Windows וה-bash מתלונן על $'\r' — הרץ פעם אחת:
#     sed -i 's/\r$//' deploy/gcp/*.sh
# =============================================================================
set -Eeuo pipefail

# ---------- הגדרות (ניתן לדרוס עם משתני סביבה) ----------
# ספק ה-LLM המקומי הוסר — הסיכום מופק ב-Gemini (ענן). Ollama נדרש ל-embeddings בלבד.
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"   # L4=Ada → cu128
SETUP_PII="${SETUP_PII:-1}"          # 0 = דלג על הסרת PII
SETUP_SERVICES="${SETUP_SERVICES:-1}" # 0 = לא להתקין systemd (הרצה ידנית)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_USER="$(id -un)"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

# ---------- תשתית לוגים ----------
c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_step=$'\e[36m'; c_off=$'\e[0m'
say()  { echo "${c_step}» $*${c_off}"; }
ok()   { echo "${c_ok}✓ $*${c_off}"; }
warn() { echo "${c_warn}⚠ $*${c_off}"; }
die()  { echo "${c_err}✗ $*${c_off}" >&2; exit 1; }
hr()   { echo "────────────────────────────────────────────────────────────"; }

trap 'die "כשל בשורה $LINENO. עצור, קרא את השגיאה למעלה, ותקן."' ERR

# =============================================================================
say "מערכת תמלול פוע\"ה — התקנה על GCP VM"
echo "  פרויקט: $PROJECT_DIR"
echo "  משתמש : $RUN_USER   |  סיכום: Gemini (ענן)   |  embed: $EMBED_MODEL"
hr

# ---------- שלב 0: שמירות (guard rails) ----------
# 0.1 לא Cloud Shell — Cloud Shell הוא ארעי וללא GPU.
if [[ "${CLOUD_SHELL:-}" == "true" || "$(hostname)" == cs-* ]]; then
  die "אתה ב-Cloud Shell! הסקריפט הזה רץ *על ה-VM*. התחבר עם: gcloud compute ssh <vm> --zone=<zone>"
fi
# 0.2 systemd קיים (מוודא שזו מכונה אמיתית).
[[ -d /run/systemd/system ]] || die "systemd אינו פעיל — זו כנראה אינה VM אמיתית."
# 0.3 קוד הפרויקט שלם.
[[ -f "$PROJECT_DIR/rebuild.py" && -d "$PROJECT_DIR/app" ]] || die "קוד הפרויקט לא נמצא ב-$PROJECT_DIR"
ok "סביבה תקינה (VM אמיתית, קוד קיים)"

# ---------- שלב 1: חבילות מערכת ----------
say "שלב 1/9 — חבילות מערכת"
NEED_PKGS=()
for pkg in python3-venv python3-pip ffmpeg libsndfile1 zstd curl ca-certificates; do
  dpkg -s "$pkg" >/dev/null 2>&1 || NEED_PKGS+=("$pkg")
done
if ((${#NEED_PKGS[@]})); then
  echo "  מתקין: ${NEED_PKGS[*]}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${NEED_PKGS[@]}"
fi
ok "חבילות מערכת מותקנות (כולל zstd הנדרש ל-Ollama)"

# ---------- שלב 2: דרייבר NVIDIA (עם התאוששות מ-reboot) ----------
say "שלב 2/9 — דרייבר NVIDIA / GPU"
if nvidia-smi -L >/dev/null 2>&1; then
  ok "GPU פעיל: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  # אולי הדרייבר מותקן אך המודול לא נטען (לפני reboot)
  sudo modprobe nvidia nvidia_uvm 2>/dev/null || true
  if nvidia-smi -L >/dev/null 2>&1; then
    ok "מודול ה-GPU נטען: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  else
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      warn "הדרייבר לא מותקן — מתקין (המתקין הרשמי של GCP, 5–10 דק')…"
      curl -fsSL -o /tmp/install_gpu_driver.py \
        https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
      sudo python3 /tmp/install_gpu_driver.py || warn "המתקין החזיר שגיאה — ייתכן שהושלם בכל זאת"
      sudo modprobe nvidia nvidia_uvm 2>/dev/null || true
    fi
    if ! nvidia-smi -L >/dev/null 2>&1; then
      hr
      warn "הדרייבר הותקן אך דורש REBOOT כדי להיטען לקרנל."
      echo "  הרץ:   ${c_step}sudo reboot${c_off}"
      echo "  ואז התחבר מחדש (${c_step}gcloud compute ssh <vm> --zone=<zone>${c_off}) והרץ שוב את הסקריפט הזה."
      echo "  (הסקריפט אידמפוטנטי — ימשיך מכאן.)"
      hr
      exit 10
    fi
    ok "GPU פעיל לאחר טעינת מודול"
  fi
fi

# 2b: גרסת הדרייבר — Ollama דורש 550+ (המתקין של GCP מתקין 535, ואז Ollama נופל ל-CPU!)
MIN_DRV="${MIN_DRIVER:-550}"
DRV_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)"
if [[ "${DRV_VER:-0}" -lt "$MIN_DRV" ]]; then
  warn "דרייבר NVIDIA $DRV_VER ישן מדי ל-Ollama (נדרש ${MIN_DRV}+). משדרג ל-nvidia-driver-${MIN_DRV}-server…"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "nvidia-driver-${MIN_DRV}-server" \
    || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "nvidia-driver-${MIN_DRV}"
  hr
  warn "שודרג ל-דרייבר ${MIN_DRV} — נדרש REBOOT כדי שייטען."
  echo "  הרץ:  ${c_step}sudo reboot${c_off}   ואז התחבר מחדש והרץ שוב את הסקריפט."
  hr
  exit 10
fi
ok "דרייבר NVIDIA $DRV_VER (>=${MIN_DRV}, תואם ל-Ollama)"

# ---------- שלב 3: Ollama (נדרש ל-embeddings) ----------
say "שלב 3/9 — Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "  מתקין Ollama…"
  curl -fsSL https://ollama.com/install.sh | sh
fi
# חשוב: לרענן את השירות כדי שירוץ הבינארי החדש.
sudo systemctl enable ollama >/dev/null 2>&1 || true
sudo systemctl daemon-reload
sudo systemctl restart ollama
say "  ממתין ש-Ollama יענה…"
for _ in $(seq 1 30); do curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || die "Ollama אינו מגיב על 11434"
ok "Ollama פעיל (גרסה $(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1))"

# ---------- שלב 4: קובץ .env + טוקן HF ----------
say "שלב 4/9 — הגדרות (.env)"
cd "$PROJECT_DIR"
[[ -f .env ]] || { cp .env.example .env; ok ".env נוצר מ-.env.example"; }
chmod 600 .env
HF_TOKEN="$(grep -E '^PUAH_HF_TOKEN=' .env | head -1 | cut -d= -f2- || true)"
if [[ -z "$HF_TOKEN" || "$HF_TOKEN" == hf_xxxx* ]]; then
  hr
  die "חסר PUAH_HF_TOKEN ב-.env. ערוך:  nano .env  (טוקן מ-https://huggingface.co/settings/tokens), ואז הרץ שוב."
fi
# עדכון שם מודל ה-embedding בהתאם להגדרות הסקריפט
sed -i "s|^PUAH_EMBED_MODEL=.*|PUAH_EMBED_MODEL=${EMBED_MODEL}|" .env
ok "טוקן HF מוגדר (${#HF_TOKEN} תווים)"

# ---------- שלב 5: תלויות Python (.venv + .venv-pii) ----------
say "שלב 5/9 — תלויות Python (10–15 דק' בפעם הראשונה)"
PII_ARGS=(); [[ "$SETUP_PII" == "1" ]] || PII_ARGS+=(--skip-pii)
python3 rebuild.py --install-only "${PII_ARGS[@]}"
[[ -x "$VENV_PY" ]] || die ".venv לא נוצר כראוי"
ok "תלויות מותקנות ($([[ -d "$PROJECT_DIR/.venv-pii" ]] && echo 'כולל .venv-pii ל-PII' || echo 'ללא PII'))"

# ---------- שלב 6: אימות ש-torch רואה את ה-GPU ----------
say "שלב 6/9 — אימות torch↔GPU"
CUDA_OK="$("$VENV_PY" -c 'import torch;print("yes" if torch.cuda.is_available() else "no")' 2>/dev/null || echo "err")"
if [[ "$CUDA_OK" != "yes" ]]; then
  warn "torch לא רואה את ה-GPU (CUDA=$CUDA_OK) — מתקין מחדש torch עם CUDA ($CUDA_INDEX)…"
  "$VENV_PY" -m pip uninstall -y torch torchaudio
  "$VENV_PY" -m pip install torch torchaudio --index-url "$CUDA_INDEX"
  CUDA_OK="$("$VENV_PY" -c 'import torch;print("yes" if torch.cuda.is_available() else "no")' 2>/dev/null || echo "err")"
fi
[[ "$CUDA_OK" == "yes" ]] || die "torch עדיין על CPU. ודא ש-nvidia-smi עובד (אולי חסר reboot) והרץ שוב."
ok "torch על GPU: $("$VENV_PY" -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0))')"

# ---------- שלב 7: משיכת מודלים ----------
say "שלב 7/9 — משיכת מודל ה-embedding"
for M in "$EMBED_MODEL"; do
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$M"; then
    ok "  $M כבר קיים"
  else
    echo "  מושך $M…"; ollama pull "$M"
  fi
done

# אימות ש-Ollama מריץ את ה-LLM על ה-GPU (ולא נופל ל-CPU בשקט → סיכומים איטיים/timeout)
say "  מאמת ש-Ollama על ה-GPU…"
curl -fsS localhost:11434/api/embed -d "{\"model\":\"$EMBED_MODEL\",\"input\":\"שלום\"}" >/dev/null 2>&1 || true
VRAM_B="$(curl -fsS localhost:11434/api/ps 2>/dev/null | "$VENV_PY" -c 'import sys,json;m=json.load(sys.stdin).get("models",[]);print(m[0]["size_vram"] if m else 0)' 2>/dev/null || echo 0)"
if [[ "${VRAM_B:-0}" -gt 0 ]]; then
  ok "  Ollama על ה-GPU (VRAM: $(( VRAM_B / 1000000000 )).$(( (VRAM_B/100000000) % 10 )) GB)"
else
  warn "  Ollama רץ על CPU! (size_vram=0) — הסיכומים יהיו איטיים/יעשו timeout."
  warn "  בדוק:  sudo journalctl -u ollama | grep -i 'driver too old\\|library=cpu'"
  warn "  לרוב הסיבה: דרייבר NVIDIA < 550. שדרג:  sudo apt-get install -y nvidia-driver-550-server && sudo reboot"
fi

# ---------- שלב 8: אימות טוקן HF + אישור מודלי pyannote ----------
say "שלב 8/9 — בדיקת גישה ל-HuggingFace (זיהוי דוברים)"
"$VENV_PY" - "$HF_TOKEN" <<'PY' || warn "בדיקת ה-HF נכשלה (ראה למעלה)"
import sys, json, urllib.request, urllib.error
tok = sys.argv[1]
def head(url):
    r = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"}, method="HEAD")
    try:
        with urllib.request.urlopen(r, timeout=20) as x: return x.status
    except urllib.error.HTTPError as e: return e.code
    except Exception: return None
# whoami — מזהה טוקן פג-תוקף (401)
w = urllib.request.Request("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {tok}"})
try:
    with urllib.request.urlopen(w, timeout=20) as x:
        who = json.loads(x.read()).get("name")
        print(f"  חשבון HF: {who}")
except urllib.error.HTTPError as e:
    print(f"  [שגיאה] הטוקן לא תקין (HTTP {e.code}) — ככל הנראה פג תוקף. צור טוקן חדש.")
    sys.exit(1)
bad = []
for repo in ("pyannote/speaker-diarization-3.1","pyannote/segmentation-3.0","pyannote/speaker-diarization-community-1"):
    st = head(f"https://huggingface.co/{repo}/resolve/main/config.yaml")
    okk = st in (200, 302)
    print(f"  {'OK ' if okk else 'NO '} [{st}] {repo}")
    if not okk: bad.append(repo)
if bad:
    print("  [פעולה נדרשת] אשר את תנאי השימוש (Agree) בדפים הבאים, מחובר לאותו חשבון:")
    for r in bad: print(f"      https://huggingface.co/{r}")
    sys.exit(2)
print("  ✓ גישה מלאה לכל מודלי pyannote — זיהוי הדוברים יעבוד.")
PY

# ---------- שלב 9: שירותי systemd (עולה אוטומטית, שורד reboot של Spot) ----------
say "שלב 9/9 — הפעלה"
if [[ "$SETUP_SERVICES" == "1" ]]; then
  for svc in worker api; do
    EXEC="$VENV_PY -m app.worker"
    [[ "$svc" == "api" ]] && EXEC="$VENV_PY -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
    sudo tee "/etc/systemd/system/puah-$svc.service" >/dev/null <<EOF
[Unit]
Description=Puah $svc
After=network-online.target ollama.service
Wants=ollama.service
[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUTF8=1
Environment=PYTHONUNBUFFERED=1
Environment=HF_HUB_DISABLE_SYMLINKS_WARNING=1
ExecStart=$EXEC
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
  done
  sudo systemctl daemon-reload
  sudo systemctl enable --now puah-worker puah-api
  sleep 6
  ok "שירותים הופעלו (יעלו אוטומטית גם אחרי reboot של Spot)"
else
  warn "מצב ללא-שירותים — הרץ ידנית:  python3 rebuild.py --no-browser --host 0.0.0.0"
fi

# ---------- בדיקת בריאות סופית ----------
hr
if curl -fsS http://127.0.0.1:8000/api/health 2>/dev/null | grep -q '"status":"ok"'; then
  ok "המערכת פעילה!  $(curl -s http://127.0.0.1:8000/api/health)"
else
  warn "ה-API עדיין לא ענה — בדוק:  journalctl -u puah-api -n 40 --no-pager"
fi
hr
say "צפייה בממשק (אין הרשאת firewall → מנהרת SSH מ-Cloud Shell):"
echo "  ${c_step}gcloud compute ssh <vm> --zone=<zone> -- -L 127.0.0.1:8000:localhost:8000 -N${c_off}"
echo "  ואז ב-Cloud Shell:  Web Preview → Preview on port 8000"
echo
echo "  לוגים:   journalctl -u puah-worker -f     |    journalctl -u puah-api -f"
ok "סיום."
