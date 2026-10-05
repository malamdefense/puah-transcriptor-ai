# מדריך פריסה לשרת ייצור — מערכת התמלול של מכון פוע"ה

מדריך זה מתאר פריסה **On-Prem** של המערכת על שרת ייצור עם GPU. כל העיבוד מקומי; תוכן השיחות
לעולם אינו עוזב את השרת (ראו [README → Air-Gapped](README.md)).

> **המסלול המומלץ הוא פריסה native** (לא Docker) — הוא נבדק מקצה-לקצה וכולל את כל היכולות
> (תמלול, זיהוי דוברים, סיכום, סיכום-ללקוח, משוב, והסרת PII). פריסת Docker מכוסה בסוף כחלופה
> חלקית (ראו המגבלות שם).

---

## 1. ארכיטקטורה בקצרה

המערכת מורכבת משלושה תהליכים מקומיים שחולקים את תיקיית `data/`:

| תהליך | תפקיד | משאב |
|-------|-------|------|
| **API + Web** (`uvicorn app.main:app`) | קבלת העלאות, ממשק, ייצוא, חיפוש | CPU קל |
| **Worker** (`python -m app.worker`) | תמלול → דוברים → סיכום → Word → PII | **GPU** (סדרתי) |
| **Ollama** (`ollama serve`) | מודל השפה לסיכום/משוב | GPU |

בנוסף: שתי סביבות PII מבודדות שה-worker קורא להן כתת-תהליכים, בזו אחר זו:
`.venv-pii` (Python 3.10) ל-HebSafeHarbor (NER עברי — שמות, מקומות, ארגונים, תאריכים עבריים),
ו-`.venv-presidio` (Python 3.10–3.14) ל-[Presidio](https://github.com/data-privacy-stack/presidio) בגרסה
העדכנית (מזהים מבוססי checksum/תבניות — ת"ז, טלפונים ישראליים, אימייל, כרטיסי אשראי, IBAN, חשבון בנק,
מספר רכב, דרכון, תאריכים מספריים). Presidio רץ על הפלט של HebSafeHarbor; אם אחד המנועים המופעלים חסר —
ההסרה נכשלת (fail-closed) ולא נשלח תוכן ל-Gemini. כיבוי: `PUAH_PII_PRESIDIO_ENABLED=false`.

> **worker יחיד בלבד.** ה-GPU הוא משאב סדרתי — אל תריצו כמה workers במקביל (OOM/התנגשות).
> מספר העלאות בו-זמנית מסתדרות בתור; ה-worker מעבד אחת-אחת.

---

## 2. דרישות שרת

**חומרה:**
- **GPU של NVIDIA** עם ≥12GB VRAM (מומלץ 16GB+; מריצים במקביל Whisper, pyannote ומודל 12B).
  לכרטיסי **Blackwell** (RTX PRO / RTX 50xx) נדרש **CUDA 12.8** (חבילות `cu128`).
- 16GB+ RAM, ו-50GB+ דיסק פנוי (מודלים ~15GB + הקלטות + תמלולים).
- דיסק מוצפן מומלץ (התוכן רגיש).

**מערכת הפעלה:** Windows Server 2019+/Windows 11, או Linux (Ubuntu 22.04+). שניהם נתמכים.

**תוכנה (מותקנת מראש):**
- מנהלי התקן NVIDIA עדכניים (`nvidia-smi` עובד).
- **Python 3.13** (לאפליקציה) **וגם Python 3.10** (ל-`.venv-pii` של HebSafeHarbor — spaCy 3.2 לא רץ על 3.11+).
  `.venv-presidio` נבנה אוטומטית מהאינטרפרטר של האפליקציה (כל Python 3.10–3.14) — אין דרישה נוספת.
- **ffmpeg** ב-PATH.
- **Ollama** (https://ollama.com). ב-Linux נדרש גם `zstd` להתקנתו: `sudo apt-get install -y zstd`.

---

## 3. התקנה

```powershell
# 1. העתיקו/שכפלו את הפרויקט לשרת, למשל C:\puah  (או /opt/puah ב-Linux)

# 2. הקמה אוטומטית — יוצר .venv, מתקין torch+CUDA + תלויות, ומקים את .venv-pii ואת .venv-presidio
python rebuild.py            # Windows
# python3 rebuild.py         # Linux
#   ל-GPU שאינו Blackwell:  python rebuild.py --cuda-index https://download.pytorch.org/whl/cu124

# 3. הורדת מודל ה-embedding (פעם אחת; הסיכום מופק ב-Gemini בענן)
ollama pull bge-m3

# 4. עריכת .env — חובה PUAH_HF_TOKEN (ראו סעיף 4)

# 5. אישור תנאי מודלי pyannote ב-HuggingFace (חד-פעמי, עם החשבון של הטוקן):
#    https://huggingface.co/pyannote/speaker-diarization-3.1
#    https://huggingface.co/pyannote/segmentation-3.0
#    https://huggingface.co/pyannote/speaker-diarization-community-1
```

ההרצה הראשונה של עבודה תוריד את מודלי התמלול והזיהוי (~כמה GB, נשמר במטמון). מומלץ להעלות
הקלטת-בדיקה אחת ולוודא מסלול מלא לפני העלייה לאוויר.

---

## 4. הגדרות ייצור (`.env`)

```ini
PUAH_HF_TOKEN=hf_xxxxxxxx          # חובה (זיהוי דוברים)
PUHA_GEMINI_API_KEY=...            # חובה (מנוע הסיכום — Gemini)
PUAH_LLM_MAX_TOKENS=6144           # אורך פלט מרבי
# האזינו רק ל-localhost; חשיפה החוצה דרך reverse proxy בלבד (ראו סעיף 6)
PUAH_CORS_ORIGINS=["https://puah-transcribe.internal"]
PUAH_MAX_UPLOAD_MB=2048
PUAH_DATA_DIR=data                 # או נתיב מוחלט לכונן נתונים מוצפן
```

> הגנו על `.env` — הוא מכיל את טוקן ה-HF. הרשאות קובץ: `chmod 600 .env` (Linux) /
> הסרת ירושה והגבלה למשתמש השירות (Windows).

---

## 5. הרצה כשירות (שיעלה אוטומטית ויתאושש מקריסה)

### Linux — systemd

קבצים מוכנים תחת [`deploy/systemd/`](deploy/systemd/). התקנה:
```bash
sudo cp deploy/systemd/puah-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ollama          # אם לא הותקן כשירות ע"י המתקין
sudo systemctl enable --now puah-worker puah-api
sudo systemctl status puah-worker puah-api  # בדיקה
journalctl -u puah-worker -f                # לוגים
```
(ערכו ב-service את `WorkingDirectory`, `User`, ונתיב `.venv` בהתאם לשרת.)

### Windows — NSSM (מומלץ) או Task Scheduler

עם [NSSM](https://nssm.cc) (Non-Sucking Service Manager):
```powershell
# Worker
nssm install PuahWorker "C:\puah\.venv\Scripts\python.exe" "-m app.worker"
nssm set PuahWorker AppDirectory "C:\puah"
nssm set PuahWorker AppEnvironmentExtra "PYTHONUTF8=1" "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
nssm set PuahWorker Start SERVICE_AUTO_START

# API
nssm install PuahApi "C:\puah\.venv\Scripts\python.exe" "-m uvicorn app.main:app --host 127.0.0.1 --port 8000"
nssm set PuahApi AppDirectory "C:\puah"
nssm set PuahApi Start SERVICE_AUTO_START

nssm start PuahWorker; nssm start PuahApi
```
Ollama ב-Windows מותקן כשירות שעולה אוטומטית עם המערכת. ודאו זאת ב-`services.msc`.

> חלופה ללא NSSM: Task Scheduler → משימה "At startup" שמריצה את `rebuild.py --offline`
> (לאחר התקנה ראשונית), אך NSSM נותן ניהול שירות, הפעלה-מחדש אוטומטית ולוגים.

---

## 6. Reverse Proxy + HTTPS

האזינו ל-`127.0.0.1:8000` בלבד, וחשפו דרך nginx/IIS עם TLS. דוגמת nginx מוכנה ב-
[`deploy/nginx/puah.conf`](deploy/nginx/puah.conf). נקודות קריטיות:
- **`client_max_body_size 2048m;`** — אחרת העלאות גדולות (עד 2GB) ייחסמו.
- **`proxy_read_timeout 3600s;`** — עיבוד ארוך/העלאות גדולות.
- תעודת TLS (פנים-ארגונית מספיקה לרשת פנימית).

---

## 7. אבטחה ⚠️

- **אין כרגע אימות משתמשים מובנה באפליקציה.** חובה אחד מאלה:
  - להגביל לרשת הפנימית/VPN בלבד (firewall — לא לחשוף לאינטרנט), **וגם**
  - להוסיף אימות ב-reverse proxy (Basic Auth / SSO ארגוני / mTLS).
- בִּנדו את ה-API ל-`127.0.0.1` (לא `0.0.0.0`) כשהוא מאחורי proxy.
- הצפנת דיסק לכונן `data/` (תמלולים, הקלטות, DB — תוכן רגיש ביותר).
- הרשאות: `.env` ו-`data/` נגישים רק למשתמש השירות.
- מצב מנותק-רשת בזמן ריצה: `python rebuild.py --offline` (ראו README).

---

## 8. גיבוי ושחזור

מה לגבות: **כל `data/`** — `puah.db` (מטא-דאטה + אינדקס חיפוש), `results/` (תמלולים, סיכומים,
Word, PII), ו-`uploads/` (הקלטות מקור).

```bash
# גיבוי עקבי של ה-DB (גם בזמן ריצה, ללא עצירת השירות):
sqlite3 data/puah.db ".backup '/backup/puah-$(date +%F).db'"
# גיבוי הקבצים:
rsync -a data/results data/uploads /backup/puah-data/
```
מומלץ cron יומי + שמירת הגיבויים מוצפנים ומחוץ לשרת. שחזור: החזרת הקבצים והפעלת השירותים.

---

## 9. תפעול ותחזוקה

- **לוגים:** `journalctl -u puah-worker -f` (Linux) / לוגי NSSM (Windows). שלבי העיבוד נרשמים שם.
- **דיסק:** `uploads/` ו-`results/` והמטמון (`~/.cache/huggingface`, `~/.ollama`) גדלים. נטרו,
  ושקלו מדיניות מחיקה/ארכוב של הקלטות ישנות.
- **Ollama:** ודאו שפועל לפני שמצפים לסיכום (`curl localhost:11434/api/tags`). אם סיכום חוזר ריק — זו הבדיקה הראשונה.
- **עדכון גרסה:**
  ```bash
  # עדכנו את הקוד, ואז:
  python rebuild.py --reinstall   # רק אם requirements השתנו; אחרת רק הפעילו מחדש את השירותים
  sudo systemctl restart puah-worker puah-api
  ```
- **בריאות:** `GET /api/health` (סטטוס + מודלים) · `GET /api/llm/status` (זמינות Ollama).

---

## 10. פריסת Docker (חלופה — חלקית)

קיימים [`Dockerfile`](Dockerfile) ו-[`docker-compose.yml`](docker-compose.yml) להרצה על Linux עם
`nvidia-container-toolkit`. עודכנו ל-cu128 (ניתן לשנות ל-`CUDA_INDEX` אחר ב-build) ולסדר התקנה נכון,
ונוסף שירות **Ollama** ל-compose.

```bash
cp .env.example .env      # ערכו PUAH_HF_TOKEN
docker compose up -d --build
docker compose logs -f ollama-init   # מושך אוטומטית את bge-m3 (פעם אחת)
```

שירות `ollama-init` מושך את המודלים בעלייה הראשונה ומסיים. תיקיית `scripts/` נכללת ב-image,
כך שניתן להריץ גם את הערכת ה-RAG:
```bash
docker compose run --rm worker python scripts/eval_rag.py --only-gt
```

> **מגבלת Docker:** הסרת ה-PII (HebSafeHarbor + Presidio) **אינה** כלולה בתמונה (היא דורשת סביבות
> Python מבודדות). אם נדרשת הסרת PII בייצור — השתמשו בפריסה native, או בנו תמונת-לוואי ייעודית ל-PII.
> תמלול, זיהוי דוברים וסיכום עובדים ב-Docker.

---

## 11. פריסה ב-Google Cloud (GCP) 🌩️

### 11.0 לפני הכול — פרטיות ותאימות

הארכיטקטורה נשארת **self-hosted**: Whisper, pyannote, Ollama, HebSafeHarbor ו-Presidio רצים כולם על ה-VM
שלכם. **שום נתון אינו נשלח ל-API של צד שלישי** (לא Google AI, לא OpenAI). מה שמשתנה: התוכן יושב
פיזית על תשתית Google. לתוכן רפואי/הלכתי רגיש — הקפידו על:

- **אזור `me-west1` (תל אביב)** → data residency בישראל.
- **ללא IP ציבורי** (`--no-address`) + גישה דרך IAP בלבד (מוגדר למטה).
- **CMEK** (מפתחות הצפנה בניהולכם) לדיסקים — אופציונלי אך מומלץ.
- אימות משתמשים דרך **IAP** — פותר את הפער שאין אימות מובנה באפליקציה (סעיף 7).

### 11.1 הכנות

```bash
export PROJECT_ID="puah-prod"          # ← שנו
export REGION="me-west1"               # תל אביב
export ZONE="me-west1-a"
export VM="puah-vm"
export BUCKET="${PROJECT_ID}-puah-app"
export SA_EMAIL="puah-vm@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com iap.googleapis.com \
    secretmanager.googleapis.com storage.googleapis.com
```

> **מכסת GPU:** ודאו שיש לכם מכסת `NVIDIA_L4_GPUS` באזור (Console → IAM → Quotas). אם L4 אינו
> זמין ב-`me-west1`, בדקו זמינות או בחרו אזור חלופי (ושקלו את השלכות ה-data residency).

**חשבון שירות עם הרשאות מינימליות:**
```bash
gcloud iam service-accounts create puah-vm --display-name="Puah VM"
for ROLE in roles/secretmanager.secretAccessor roles/storage.objectViewer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" --role="$ROLE" --condition=None
done
```

### 11.2 רשת: VPC + Cloud NAT + Firewall

```bash
gcloud compute networks create puah-vpc --subnet-mode=custom
gcloud compute networks subnets create puah-subnet \
    --network=puah-vpc --region="$REGION" --range=10.10.0.0/24 \
    --enable-private-ip-google-access          # גישה ל-GCS/Secrets בלי IP ציבורי

# Cloud NAT — יציאה לאינטרנט (הורדת חבילות ומודלים) בלי לחשוף את ה-VM
gcloud compute routers create puah-router --network=puah-vpc --region="$REGION"
gcloud compute routers nats create puah-nat --router=puah-router --region="$REGION" \
    --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
```

**Firewall — רק IAP ורק ה-Load Balancer. שום דבר ציבורי:**
```bash
# SSH דרך IAP בלבד
gcloud compute firewall-rules create puah-allow-iap-ssh --network=puah-vpc \
    --direction=INGRESS --action=ALLOW --rules=tcp:22 \
    --source-ranges=35.235.240.0/20 --target-tags=puah-app

# תעבורת ה-Load Balancer + health checks לפורט 8000
gcloud compute firewall-rules create puah-allow-lb --network=puah-vpc \
    --direction=INGRESS --action=ALLOW --rules=tcp:8000 \
    --source-ranges=130.211.0.0/22,35.191.0.0/16 --target-tags=puah-app
```

### 11.3 טוקן HuggingFace ב-Secret Manager

```bash
printf '%s' 'hf_XXXXXXXXXXXX' | gcloud secrets create puah-hf-token --data-file=-
```

### 11.4 העלאת הקוד ל-GCS

```bash
gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access

# מתיקיית הפרויקט (ללא venvs ו-data):
gcloud storage rsync -r . "gs://${BUCKET}/puah" \
    --exclude='^(\.venv|\.venv-pii|\.venv-presidio|data|\.git|__pycache__)/.*$'
```

### 11.5 יצירת ה-VM עם GPU

ה-startup-script ([`deploy/gcp/startup.sh`](deploy/gcp/startup.sh)) מתקין הכול לבד: דרייברי NVIDIA,
ffmpeg, Ollama, את האפליקציה, את שלושת ה-venv (כולל `.venv-pii` ו-`.venv-presidio`), מושך את המודלים, ומרים את השירותים.

```bash
gcloud compute instances create "$VM" --zone="$ZONE" \
    --machine-type=g2-standard-8 \
    --maintenance-policy=TERMINATE --restart-on-failure \
    --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
    --boot-disk-size=250GB --boot-disk-type=pd-balanced \
    --subnet=puah-subnet --no-address \
    --service-account="$SA_EMAIL" --scopes=cloud-platform \
    --tags=puah-app \
    --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
    --metadata=app-bucket="${BUCKET}",hf-secret=puah-hf-token \
    --metadata-from-file=startup-script=deploy/gcp/startup.sh
```

> **`g2-standard-8` כולל אוטומטית GPU אחד מסוג NVIDIA L4 (24GB VRAM)** — אין צורך ב-`--accelerator`.
> 24GB מספיקים ל-Whisper + pyannote *יחד* (מודל השפה רץ בענן). `--maintenance-policy=TERMINATE` הוא חובה ב-GPU.
>
> **בונוס:** Ubuntu 22.04 מגיע עם **Python 3.10** — בדיוק מה שנדרש ל-`.venv-pii` של HebSafeHarbor.
> בניגוד ל-Windows, **אין צורך להתקין Python שני**. שני ה-venv נבנים מאותו מפרש.

### 11.6 מעקב אחרי ההתקנה

ההרצה הראשונה אורכת 15–25 דקות (דרייברים + torch + מודלים):
```bash
gcloud compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap
sudo tail -f /var/log/puah-startup.log        # או: journalctl -u google-startup-scripts -f
curl -s localhost:8000/api/health              # אמור להחזיר status: ok
systemctl status puah-worker puah-api
```

### 11.7 גישה מהירה למנהלים (מנהרת IAP — בלי Load Balancer)

```bash
gcloud compute start-iap-tunnel "$VM" 8000 --local-host-port=localhost:8000 --zone="$ZONE"
# ואז בדפדפן:  http://localhost:8000
```
מצוין לבדיקות ולשימוש אישי. לצוות של פוע"ה — המשיכו לסעיף הבא.

### 11.8 חשיפה לצוות: HTTPS Load Balancer + IAP

כך אנשי הצוות פשוט נכנסים לכתובת ומקבלים **מסך התחברות של Google** — בלי VPN ובלי IP ציבורי ל-VM.

```bash
# קבוצת מכונות + health check
gcloud compute instance-groups unmanaged create puah-ig --zone="$ZONE"
gcloud compute instance-groups unmanaged add-instances puah-ig --instances="$VM" --zone="$ZONE"
gcloud compute instance-groups unmanaged set-named-ports puah-ig --named-ports=http:8000 --zone="$ZONE"
gcloud compute health-checks create http puah-hc --port=8000 --request-path=/api/health

# backend service — timeout ארוך להעלאות (עד 2GB) ולעיבוד
gcloud compute backend-services create puah-backend --global \
    --protocol=HTTP --port-name=http --health-checks=puah-hc --timeout=3600
gcloud compute backend-services add-backend puah-backend --global \
    --instance-group=puah-ig --instance-group-zone="$ZONE"

# HTTPS front-end (תעודה מנוהלת ע"י Google)
gcloud compute url-maps create puah-map --default-service=puah-backend
gcloud compute ssl-certificates create puah-cert --domains=puah.example.org --global   # ← הדומיין שלכם
gcloud compute target-https-proxies create puah-proxy --url-map=puah-map --ssl-certificates=puah-cert
gcloud compute addresses create puah-ip --global
gcloud compute forwarding-rules create puah-fr --global \
    --address=puah-ip --target-https-proxy=puah-proxy --ports=443

# הפניית הדומיין ל-IP:
gcloud compute addresses describe puah-ip --global --format='value(address)'   # → רשומת A ב-DNS
```

**הפעלת IAP והרשאות (זה מה שמחליף את האימות החסר באפליקציה):**
```bash
# דרוש OAuth consent screen מוגדר בפרויקט (Console → APIs & Services → OAuth consent screen)
gcloud compute backend-services update puah-backend --global --iap=enabled

# הרשאה לכל משתמש/קבוצה שצריכים גישה:
gcloud iap web add-iam-policy-binding \
    --resource-type=backend-services --service=puah-backend \
    --member='group:puah-staff@puah.org.il' --role='roles/iap.httpsResourceAccessor'
```

### 11.9 תפעול, גיבוי ועלויות

```bash
# גיבוי אוטומטי יומי של הדיסק (כולל data/)
gcloud compute resource-policies create snapshot-schedule puah-daily \
    --region="$REGION" --max-retention-days=30 --start-time=23:00 \
    --daily-schedule --storage-location="$REGION"
gcloud compute disks add-resource-policies "$VM" --zone="$ZONE" --resource-policies=puah-daily

# עדכון גרסה: סנכרון הקוד ואז ריענון על ה-VM
gcloud storage rsync -r . "gs://${BUCKET}/puah" --exclude='^(\.venv|\.venv-pii|\.venv-presidio|data|\.git|__pycache__)/.*$'
gcloud compute ssh "$VM" --zone="$ZONE" --tunnel-through-iap --command \
  'sudo gcloud storage rsync -r gs://'"$BUCKET"'/puah /opt/puah && sudo systemctl restart puah-worker puah-api'

# חיסכון: כיבוי מחוץ לשעות העבודה (ה-GPU הוא עיקר העלות)
gcloud compute instances stop  "$VM" --zone="$ZONE"
gcloud compute instances start "$VM" --zone="$ZONE"
```

> **עלות:** `g2-standard-8` (L4) הוא כ-**$0.9–1.1 לשעה** on-demand — כ-**$700/חודש** אם דלוק 24/7.
> כיבוי מחוץ לשעות פעילות (או Committed Use Discount) מוריד זאת משמעותית. הדיסק והתעבורה זניחים ביחס.

---

## 12. צ'קליסט עלייה לאוויר

- [ ] `nvidia-smi` עובד; CUDA תואם ל-GPU (Blackwell → cu128).
- [ ] Python 3.13 + Python 3.10 + ffmpeg + Ollama מותקנים.
- [ ] `python rebuild.py` הסתיים בהצלחה (כולל `.venv-pii` ו-`.venv-presidio`).
- [ ] `PUHA_GEMINI_API_KEY` מוגדר · `ollama pull bge-m3` בוצע.
- [ ] `.env` עם `PUAH_HF_TOKEN`, הרשאות מוגבלות.
- [ ] שלושת תנאי מודלי pyannote אושרו ב-HuggingFace.
- [ ] הקלטת-בדיקה עברה מסלול מלא (תמלול+דוברים+סיכום+משוב+PII).
- [ ] שירותים מוגדרים (systemd/NSSM) ועולים אוטומטית.
- [ ] reverse proxy + TLS + `client_max_body_size 2048m`.
- [ ] אימות/הגבלת רשת מוגדרים (אין אימות מובנה!).
- [ ] גיבוי יומי של `data/` מוגדר ונבדק.

**נוסף, אם פורסים ב-GCP (סעיף 11):**
- [ ] אזור `me-west1` (ת"א) — data residency בישראל.
- [ ] מכסת `NVIDIA_L4_GPUS` אושרה באזור.
- [ ] ה-VM ללא IP ציבורי (`--no-address`) + Cloud NAT ליציאה.
- [ ] Firewall: רק IAP (`35.235.240.0/20`) ו-LB (`130.211.0.0/22`, `35.191.0.0/16`).
- [ ] טוקן HF ב-Secret Manager (לא בקוד/ב-image).
- [ ] **IAP מופעל** על ה-backend service + הרשאות לצוות (זה האימות!).
- [ ] snapshot schedule יומי לדיסק.
- [ ] מדיניות כיבוי מחוץ לשעות פעילות (חיסכון בעלות ה-GPU).
