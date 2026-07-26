#!/usr/bin/env bash
# =============================================================================
#  Puah — מנהרת גישה לממשק  (מריצים *ב-Cloud Shell*)
# -----------------------------------------------------------------------------
#  כשאין הרשאת firewall (compute.firewalls.create) — לא ניתן לפתוח את פורט 8000
#  ולא ניתן להשתמש ב-IAP tunnel (גם הוא דורש firewall). הפתרון: מעבירים את
#  הפורט דרך חיבור ה-SSH הקיים (פורט 22 פתוח כברירת מחדל ב-GCP). אפס הרשאות נוספות.
#
#  שימוש:
#     bash deploy/gcp/03-open-tunnel.sh                 # ברירת מחדל puah-vm
#     VM=my-vm ZONE=us-east1-c bash deploy/gcp/03-open-tunnel.sh
#
#  אחרי שהמנהרה עולה (החלון "נתקע" — זה תקין), ב-Cloud Shell:
#     Web Preview (אייקון בפינה הימנית-עליונה) → Preview on port 8000
# =============================================================================
set -Eeuo pipefail

VM="${VM:-puah-vm}"
ZONE="${ZONE:-}"

# גילוי אוטומטי של ה-zone. אם נכשל (למשל שגיאת Gaia id) — אפשר לתת ZONE=... ידנית.
if [[ -z "$ZONE" ]]; then
  ZONE="$(gcloud compute instances list --filter="name=$VM" --format='value(zone)' 2>/dev/null \
          | grep -vi 'error\|gaia\|boundary' | head -1 || true)"
fi
if [[ -z "$ZONE" ]]; then
  echo "⚠ לא הצלחתי לזהות את ה-zone אוטומטית (כנראה בגלל רעש ה-Gaia id)."
  echo "  הרץ עם zone מפורש, למשל:"
  echo "     ZONE=us-east1-c bash deploy/gcp/03-open-tunnel.sh"
  exit 1
fi

# ודא שהמכונה רצה (Spot עלול להיכבות)
STATUS="$(gcloud compute instances describe "$VM" --zone="$ZONE" --format='value(status)' 2>/dev/null | grep -vi 'error\|gaia\|boundary' || true)"
if [[ -n "$STATUS" && "$STATUS" != "RUNNING" ]]; then
  echo "⚠ המכונה במצב '$STATUS'. מפעיל…"
  gcloud compute instances start "$VM" --zone="$ZONE" || true
  echo "  ממתין ~30 שניות לעלייה…"; sleep 30
fi

echo "» פותח מנהרה: localhost:8000 → $VM:8000  (השאר חלון זה פתוח)"
echo "  ואז ב-Cloud Shell:  Web Preview → Preview on port 8000"
exec gcloud compute ssh "$VM" --zone="$ZONE" -- -L 127.0.0.1:8000:localhost:8000 -N
