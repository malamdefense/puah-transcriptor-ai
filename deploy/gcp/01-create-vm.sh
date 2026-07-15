#!/usr/bin/env bash
# =============================================================================
#  Puah — יצירת VM עם GPU ב-GCP  (מריצים *ב-Cloud Shell*)
# -----------------------------------------------------------------------------
#  מקפל את כל הלקחים מהפריסה הידנית:
#   • --no-service-account  → פותר את חסם ה-SSH (iam.serviceAccountUser)
#   • SPOT + לולאת zones    → משיג מלאי GPU (on-demand נכשל ב-STOCKOUT)
#   • Ubuntu 22.04          → Python 3.10 מובנה ל-.venv-pii (בלי deadsnakes)
#   • termination-action=STOP → אם Spot נכבה, הדיסק נשמר
#
#  שימוש:
#     bash deploy/gcp/01-create-vm.sh
# =============================================================================
set -Eeuo pipefail

# ---------- הגדרות (ערוך לפי הצורך) ----------
VM="${VM:-puah-vm}"
MACHINE="${MACHINE:-g2-standard-4}"          # כולל NVIDIA L4 (24GB) — מספיק לכל הצינור
DISK_GB="${DISK_GB:-250}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
# סדר עדיפויות ל-zones (data residency: me-west1 = תל אביב; us = POC בלבד)
ZONES="${ZONES:-me-west1-b me-west1-c me-west1-a us-east1-c us-east1-b us-central1-a us-central1-b us-west1-b us-west4-a}"
# GPU: ל-me-west1 אין L4 — נשתמש ב-T4 (16GB, מספיק). ל-us: L4 מובנה ב-g2.
# הסקריפט מנסה קודם את המכונה כפי שהיא; אם צריך T4 מפורש, הגדר:  ACCEL="type=nvidia-tesla-t4,count=1"
ACCEL="${ACCEL:-}"

PROJECT="$(gcloud config get-value project 2>/dev/null)"
echo "» פרויקט: $PROJECT   |  מכונה: $MACHINE   |  דיסק: ${DISK_GB}GB"
[[ -n "$PROJECT" ]] || { echo "✗ אין פרויקט פעיל. הרץ: gcloud config set project <ID>"; exit 1; }

# ---------- אם כבר קיימת מכונה בשם הזה ----------
EXISTING="$(gcloud compute instances list --filter="name=$VM" --format='value(zone,status)' 2>/dev/null | head -1 || true)"
if [[ -n "$EXISTING" ]]; then
  echo "⚠ כבר קיימת מכונה '$VM' ($EXISTING)."
  echo "  להתחברות:  gcloud compute ssh $VM --zone=$(echo "$EXISTING" | awk '{print $1}')"
  echo "  למחיקה:    gcloud compute instances delete $VM --zone=$(echo "$EXISTING" | awk '{print $1}') --quiet"
  exit 0
fi

# ---------- לולאה: מנסה zone אחרי zone עד שיש מלאי ----------
for Z in $ZONES; do
  echo "── מנסה ליצור ב-$Z ..."
  ARGS=(
    --zone="$Z" --machine-type="$MACHINE"
    --provisioning-model=SPOT --instance-termination-action=STOP
    --no-service-account --no-scopes
    --maintenance-policy=TERMINATE
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT"
    --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced
    --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring
  )
  [[ -n "$ACCEL" ]] && ARGS+=(--accelerator="$ACCEL")
  if gcloud compute instances create "$VM" "${ARGS[@]}" 2>/tmp/vm_err; then
    echo
    echo "✅✅ נוצר בהצלחה ב-$Z"
    echo "────────────────────────────────────────────────"
    echo "השלבים הבאים (מ-Cloud Shell):"
    echo "  1. העתק את הקוד ל-VM:"
    echo "       gcloud compute scp --recurse ~/<תיקיית-הפרויקט> $VM:~/puah --zone=$Z"
    echo "  2. התחבר:"
    echo "       gcloud compute ssh $VM --zone=$Z"
    echo "  3. על ה-VM — ערוך טוקן HF והרץ את ההתקנה:"
    echo "       cd ~/puah && nano .env    # אם כבר קיים; אחרת ייווצר בהתקנה"
    echo "       bash deploy/gcp/02-provision.sh"
    echo "  4. לצפייה בממשק (מנהרת SSH מ-Cloud Shell):"
    echo "       gcloud compute ssh $VM --zone=$Z -- -L 127.0.0.1:8000:localhost:8000 -N"
    echo "       ואז Web Preview → port 8000"
    echo "────────────────────────────────────────────────"
    exit 0
  fi
  if grep -q "RESOURCE_POOL_EXHAUSTED\|STOCKOUT\|does not have enough resources\|does not exist in zone" /tmp/vm_err; then
    echo "   (אין מלאי/סוג לא זמין ב-$Z — ממשיך)"
  else
    echo "✗ שגיאה שאינה מלאי:"; cat /tmp/vm_err; exit 1
  fi
done

echo "✗ לא נמצא מלאי GPU באף zone שנוסה. אפשרויות: נסה שוב בעוד שעה, הוסף zones ל-ZONES,"
echo "  או שנה MACHINE/ACCEL (למשל T4:  ACCEL='type=nvidia-tesla-t4,count=1' MACHINE=n1-standard-4)."
exit 1
