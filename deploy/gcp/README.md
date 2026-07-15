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
gcloud compute scp --recurse ~/<פרויקט> puah-vm:~/puah --zone=<ZONE>
gcloud compute ssh puah-vm --zone=<ZONE>

# ── על ה-VM ──
cd ~/puah
nano .env                                                   # שים PUAH_HF_TOKEN תקף
bash deploy/gcp/02-provision.sh
#   אם ביקש reboot:  sudo reboot → התחבר מחדש → הרץ שוב את 02 (ימשיך מהנקודה)

# ── ב-Cloud Shell (חלון נוסף) ──
bash deploy/gcp/03-open-tunnel.sh                            # ואז Web Preview → port 8000
```

## שני דברים שהסקריפט **לא** יכול לעשות בשבילך (ידני)

1. **טוקן HuggingFace תקף** ב-`.env` (מ-https://huggingface.co/settings/tokens, סוג Read).
   הסקריפט מזהה טוקן פג-תוקף (401) ומתריע.
2. **אישור תנאי pyannote** — היכנס ל-HF (אותו חשבון) ולחץ Agree ב-3 הדפים:
   `speaker-diarization-3.1` · `segmentation-3.0` · `speaker-diarization-community-1`.
   הסקריפט בודק את הגישה ומדפיס בדיוק אילו דפים חסרים.

## אם bash מתלונן על `$'\r'` (קבצים מ-Windows)
```bash
sed -i 's/\r$//' deploy/gcp/*.sh
```

## הערות

- **Spot:** המכונה זולה אך גוגל עלולה לכבותה. הדיסק נשמר; `03-open-tunnel.sh` מפעיל אותה שוב לבד.
  השירותים (systemd) עולים אוטומטית לאחר reboot.
- **פרטיות:** ל-data residency בישראל השאר את ה-zones תחת `me-west1` (אין שם L4 — הסקריפט יפול
  ל-T4; הגדר `ACCEL='type=nvidia-tesla-t4,count=1' MACHINE=n1-standard-4`). us = POC בלבד.
- **אבטחה:** אין אימות מובנה. הגישה דרך מנהרת SSH בלבד (אישית). אל תעלה תוכן רגיש ב-POC.
