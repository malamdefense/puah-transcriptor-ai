# פריסת Puah ב-GCP — סקריפטים

שלושה סקריפטים שמקפלים את **כל** הלקחים מפריסה ידנית (הרשאות IAM חסרות, Spot, אין firewall,
טוקן פג-תוקף, reboot לדרייבר, torch על CPU, Ollama ישן).

| סקריפט | רץ ב… | תפקיד |
|--------|-------|-------|
| `01-create-vm.sh` | **Cloud Shell** | יוצר VM עם GPU (Spot, ללא service-account, Ubuntu 22.04, לולאת zones) |
| `02-provision.sh` | **על ה-VM** | מתקין הכול: דרייברים, Ollama, תלויות, מודלים, שירותים. אידמפוטנטי. |
| `03-open-tunnel.sh` | **Cloud Shell** | מנהרת SSH לצפייה בממשק (עוקף את חוסר הרשאת ה-firewall) |

## כלל הזהב: איפה מריצים מה

| ה-prompt מתחיל ב… | זה… | מריצים שם |
|---|---|---|
| `@cloudshell` | Cloud Shell (ארעי, בלי GPU) | `gcloud` בלבד — סקריפט 01, 03 |
| `@puah-vm` | ה-VM האמיתי | הכול השאר — סקריפט 02 |

## זרימה מלאה

```bash
# ── ב-Cloud Shell ──
bash deploy/gcp/01-create-vm.sh                              # יוצר את ה-VM
# העתקת הקוד ל-VM *ללא* .env המקומי (שעלול לכלול PUAH_LLM_PROVIDER=ollama), venvs, data וקבצי פיתוח:
tar --exclude='.env' --exclude='.venv*' --exclude='venv' --exclude='data' \
    --exclude='.git' --exclude='.claude' --exclude='.pytest_cache' --exclude='__pycache__' \
    --exclude='bin' --exclude='include' --exclude='lib' --exclude='lib64' --exclude='media' \
    --exclude='pyvenv.cfg' --exclude='*.pyc' --exclude='*.mp3' --exclude='*.mp4' \
    --exclude='*.wav' --exclude='*.m4a' --exclude='*.docx' --exclude='*.xlsx' --exclude='diagram.png' \
    -C ~/<פרויקט> -czf - . | gcloud compute ssh puah-vm --zone=<ZONE> -- 'mkdir -p ~/puah && tar -xzf - -C ~/puah'
# (חלופה עם gcloud compute scp: הקפד למחוק מיד לאחר ההעתקה את ~/puah/.env כדי שלא ידרוס את ברירת המחדל של GCP)
gcloud compute ssh puah-vm --zone=<ZONE>

# ── על ה-VM ──
cd ~/puah
cp -n .env.example .env
nano .env                                                   # שים PUAH_HF_TOKEN + PUHA_GEMINI_API_KEY
bash deploy/gcp/02-provision.sh
#   אם ביקש reboot:  sudo reboot → התחבר מחדש → הרץ שוב את 02 (ימשיך מהנקודה)

# ── ב-Cloud Shell (חלון נוסף) ──
bash deploy/gcp/03-open-tunnel.sh                            # ואז Web Preview → port 8000
```

## שלושה דברים שהסקריפט **לא** יכול לעשות בשבילך (ידני)

1. **טוקן HuggingFace תקף** ב-`.env` (מ-https://huggingface.co/settings/tokens, סוג Read).
   הסקריפט מזהה טוקן פג-תוקף (401) ומתריע.
2. **אישור תנאי pyannote** — היכנס ל-HF (אותו חשבון) ולחץ Agree ב-3 הדפים:
   `speaker-diarization-3.1` · `segmentation-3.0` · `speaker-diarization-community-1`.
   הסקריפט בודק את הגישה ומדפיס בדיוק אילו דפים חסרים.
3. **מפתח Google Gemini (`PUHA_GEMINI_API_KEY`)** — נדרש להפקת סיכום ללקוח ומשוב (ב-`.env`, במטא-דאטה `gemini-secret` של `startup.sh`, או ממסך ההגדרות `⚙` בממשק). בלעדיו התמלול והסרת ה-PII יעבדו אך הסיכומים ידולגו.

## אם bash מתלונן על `$'\r'` (קבצים מ-Windows)
```bash
sed -i 's/\r$//' deploy/gcp/*.sh
```

## הערות

- **אל תעתיק `.env` מהפיתוח המקומי:** בסביבת הפיתוח המקומית `.env` עשוי להגדיר `PUAH_LLM_PROVIDER=ollama` (`muse-glimmer`), בעוד פריסת GCP מריצה את הסיכום ב-Gemini ומושכת ל-Ollama רק את `bge-m3`. תמיד התחל מ-`.env.example` (או הגדר `PUAH_LLM_PROVIDER=gemini`).
- **זיכרון RAM:** סביבות ה-PII המתמידות (`--serve`) חוסכות 45–180 שניות לשיחה, וה-API משחרר אוטומטית את עותק ה-PII שלו לאחר 3 דקות חוסר פעילות (`PUAH_PII_API_IDLE_TIMEOUT_SEC=180`). לייצור שוטף מומלצת מכונה עם 30–32GB RAM (`MACHINE=g2-standard-8` ל-L4 או `MACHINE=n1-standard-8` ל-T4); 15–16GB (`g2-standard-4` / `n1-standard-4`) מספיק ל-POC.
- **Spot:** המכונה זולה אך גוגל עלולה לכבותה. הדיסק נשמר; `03-open-tunnel.sh` מפעיל אותה שוב לבד.
  השירותים (systemd) עולים אוטומטית לאחר reboot.
- **פרטיות:** ל-data residency בישראל השאר את ה-zones תחת `me-west1` (אין שם L4 — הסקריפט יפול
  ל-T4; הגדר `ACCEL='type=nvidia-tesla-t4,count=1' MACHINE=n1-standard-4` או `n1-standard-8`). us = POC בלבד.
- **אבטחה:** אין אימות מובנה. הגישה דרך מנהרת SSH בלבד (אישית). אל תעלה תוכן רגיש ב-POC.
