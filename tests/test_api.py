"""בדיקות שכבת ה-API + DB + ייצוא — ללא GPU/מודלים.

מדמה השלמת משימה (במקום ה-worker) כדי לבדוק את כל מסלול ה-API.
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

# נתוני בדיקה בתיקייה זמנית, וטוקן דמה
_tmp = tempfile.mkdtemp(prefix="puah-test-")
os.environ["PUAH_DATA_DIR"] = _tmp
os.environ["PUAH_HF_TOKEN"] = "hf_dummy"
# מצביע מודל השפה לפורט לא-קיים כדי שהסיכום ייכשל באופן צפוי (503)
os.environ["PUAH_LLM_BASE_URL"] = "http://127.0.0.1:9"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "מכון פוע" in r.text


def test_full_job_lifecycle():
    # רשימה ריקה בהתחלה
    assert client.get("/api/jobs").json() == []

    # העלאת "קובץ" (תוכן דמה — אין עיבוד אמיתי כאן)
    files = {"file": ("ייעוץ.mp4", b"\x00\x01\x02fake-mp4-bytes", "video/mp4")}
    r = client.post("/api/jobs", files=files, data={"num_speakers": "2"})
    assert r.status_code == 201, r.text
    job = r.json()
    job_id = job["id"]
    assert job["status"] == "queued"
    assert job["num_speakers"] == 2

    # סטטוס
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    # אין תוצאה עדיין
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409

    # סוג קובץ לא נתמך
    bad = {"file": ("x.txt", b"hello", "text/plain")}
    assert client.post("/api/jobs", files=bad).status_code == 400

    # --- דימוי השלמת ה-worker ---
    result = {
        "job_id": job_id,
        "language": "he",
        "duration_sec": 12.0,
        "num_speakers": 2,
        "speakers": {"SPEAKER_00": "דובר 1", "SPEAKER_01": "דובר 2"},
        "segments": [
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "שלום, איך אפשר לעזור?", "words": []},
            {"start": 3.2, "end": 7.0, "speaker": "SPEAKER_01", "text": "תודה שקיבלתם אותי.", "words": []},
        ],
        "asr_model": "ivrit-ai/whisper-large-v3-turbo-ct2",
        "diarization_model": "pyannote/speaker-diarization-3.1",
    }
    db.save_result(job_id, result)
    db.index_job_segments(job_id, result)  # כמו שה-worker עושה
    db.update_job(job_id, status="done", progress=100.0, num_speakers=2, duration_sec=12.0)

    # תוצאה
    res = client.get(f"/api/jobs/{job_id}/result")
    assert res.status_code == 200
    assert res.json()["num_speakers"] == 2
    assert len(res.json()["segments"]) == 2

    # ייצוא בכל הפורמטים
    txt = client.get(f"/api/jobs/{job_id}/export?format=txt")
    assert txt.status_code == 200
    assert "דובר 1:" in txt.text
    assert client.get(f"/api/jobs/{job_id}/export?format=srt").status_code == 200
    assert "-->" in client.get(f"/api/jobs/{job_id}/export?format=srt").text
    assert "WEBVTT" in client.get(f"/api/jobs/{job_id}/export?format=vtt").text
    assert client.get(f"/api/jobs/{job_id}/export?format=json").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/export?format=bad").status_code == 400

    # ייצוא Word (נוצר on-the-fly מהתוצאה)
    docx = client.get(f"/api/jobs/{job_id}/export?format=docx")
    assert docx.status_code == 200
    assert docx.content[:2] == b"PK"  # docx = zip
    assert "filename*=UTF-8''" in docx.headers.get("content-disposition", "")

    # גרסת PII לא נוצרה בסימולציה (אין .venv-pii) -> 404
    assert client.get(f"/api/jobs/{job_id}/pii?format=json").status_code == 404

    # --- דשבורד סטטיסטיקות ---
    st = client.get("/api/stats")
    assert st.status_code == 200
    sd = st.json()
    assert sd["total"] >= 1
    assert sd["by_status"].get("done", 0) >= 1
    assert sd["segments"] >= 1
    assert "2" in sd["speaker_distribution"]  # num_speakers=2 בעבודת הבדיקה
    assert isinstance(sd["top_keywords"], list) and isinstance(sd["per_day"], list)
    assert "avg_processing_sec" in sd and "with_pii" in sd

    # --- ייצוא Excel מצרפי (סיכום+משוב) ---
    xlsx = client.get("/api/export/feedback.xlsx")
    assert xlsx.status_code == 200
    assert xlsx.content[:2] == b"PK"  # xlsx = zip

    # --- עריכת פרומפטים ---
    prompts = client.get("/api/prompts").json()["prompts"]
    assert "client_summary_task" in prompts and prompts["summary_system"]["is_custom"] is False
    assert client.put("/api/prompts", json={"prompts": {"summary_system": "בדיקה"}}).status_code == 200
    after = client.get("/api/prompts").json()["prompts"]["summary_system"]
    assert after["value"] == "בדיקה" and after["is_custom"] is True
    client.post("/api/prompts/reset", params={"key": "summary_system"})
    assert client.get("/api/prompts").json()["prompts"]["summary_system"]["is_custom"] is False

    # --- לוגואים ---
    assert client.get("/api/branding").json() == {"right": False, "left": False}
    assert client.get("/api/branding/right").status_code == 404
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    up = client.post("/api/branding/right", files={"file": ("logo.png", png, "image/png")})
    assert up.status_code == 200
    assert client.get("/api/branding").json()["right"] is True
    assert client.get("/api/branding/right").status_code == 200
    assert client.delete("/api/branding/right").status_code == 204
    assert client.get("/api/branding").json()["right"] is False

    # שינוי שם דובר
    r = client.put(f"/api/jobs/{job_id}/speakers", json={"speakers": {"SPEAKER_00": "יועץ"}})
    assert r.status_code == 200
    assert r.json()["speakers"]["SPEAKER_00"] == "יועץ"
    assert "יועץ:" in client.get(f"/api/jobs/{job_id}/export?format=txt").text

    # הגשת מדיה
    assert client.get(f"/api/jobs/{job_id}/media").status_code == 200

    # --- חיפוש גלובלי ---
    sr = client.get("/api/search", params={"q": "לעזור"})
    assert sr.status_code == 200
    body = sr.json()
    assert body["count"] == 1
    assert body["matches"][0]["job_id"] == job_id
    # החיפוש מחזיר את שם הדובר המעודכן ("יועץ")
    assert body["matches"][0]["speaker"] == "יועץ"
    # שאילתה ריקה -> אפס תוצאות
    assert client.get("/api/search", params={"q": ""}).json()["count"] == 0
    # מילה שלא קיימת
    assert client.get("/api/search", params={"q": "מטוס"}).json()["count"] == 0

    # --- סטטוס מודל השפה + סיכום (אמור להיכשל ב-503 כי אין שרת LLM) ---
    assert client.get("/api/llm/status").status_code == 200
    summ = client.post(f"/api/jobs/{job_id}/summarize")
    assert summ.status_code == 503

    # מחיקה
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    # לאחר מחיקה — אין יותר תוצאות חיפוש
    assert client.get("/api/search", params={"q": "לעזור"}).json()["count"] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nכל {len(fns)} בדיקות ה-API עברו.")
