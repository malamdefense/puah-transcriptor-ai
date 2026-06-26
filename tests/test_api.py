"""Tests for the API + DB + export layer — without GPU/models.

Simulates the completion of a job (instead of the worker) to test the entire API path.
"""

import os
import sys
import tempfile
from pathlib import Path

# Test data in a temporary directory, and dummy token
_tmp = tempfile.mkdtemp(prefix="puah-test-")
os.environ["PUAH_DATA_DIR"] = _tmp
os.environ["PUAH_HF_TOKEN"] = "hf_dummy"
# Point language model to a non-existent port to fail the summary as expected (503)
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
    # Empty list at the beginning
    assert client.get("/api/jobs").json() == []

    # Upload "file" (dummy content — no real processing here)
    files = {"file": ("ייעוץ.mp4", b"\x00\x01\x02fake-mp4-bytes", "video/mp4")}
    r = client.post("/api/jobs", files=files, data={"num_speakers": "2"})
    assert r.status_code == 201, r.text
    job = r.json()
    job_id = job["id"]
    assert job["status"] == "queued"
    assert job["num_speakers"] == 2

    # Status
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    # No result yet
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409

    # Unsupported file type
    bad = {"file": ("x.txt", b"hello", "text/plain")}
    assert client.post("/api/jobs", files=bad).status_code == 400

    # --- Simulate worker completion ---
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
    db.index_job_segments(job_id, result)  # Like the worker does
    db.update_job(job_id, status="done", progress=100.0, num_speakers=2, duration_sec=12.0)

    # Result
    res = client.get(f"/api/jobs/{job_id}/result")
    assert res.status_code == 200
    assert res.json()["num_speakers"] == 2
    assert len(res.json()["segments"]) == 2

    # Export in all formats
    txt = client.get(f"/api/jobs/{job_id}/export?format=txt")
    assert txt.status_code == 200
    assert "דובר 1:" in txt.text
    assert client.get(f"/api/jobs/{job_id}/export?format=srt").status_code == 200
    assert "-->" in client.get(f"/api/jobs/{job_id}/export?format=srt").text
    assert "WEBVTT" in client.get(f"/api/jobs/{job_id}/export?format=vtt").text
    assert client.get(f"/api/jobs/{job_id}/export?format=json").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/export?format=bad").status_code == 400

    # Export Word (created on-the-fly from the result)
    docx = client.get(f"/api/jobs/{job_id}/export?format=docx")
    assert docx.status_code == 200
    assert docx.content[:2] == b"PK"  # docx = zip
    assert "filename*=UTF-8''" in docx.headers.get("content-disposition", "")

    # PII version not created in simulation (no .venv-pii) -> 404
    assert client.get(f"/api/jobs/{job_id}/pii?format=json").status_code == 404

    # --- Statistics dashboard ---
    st = client.get("/api/stats")
    assert st.status_code == 200
    sd = st.json()
    assert sd["total"] >= 1
    assert sd["by_status"].get("done", 0) >= 1
    assert sd["segments"] >= 1
    assert "2" in sd["speaker_distribution"]  # num_speakers=2 in the test job
    assert isinstance(sd["top_keywords"], list) and isinstance(sd["per_day"], list)
    assert "avg_processing_sec" in sd and "with_pii" in sd

    # --- Export Excel (summary+feedback) ---
    xlsx = client.get("/api/export/feedback.xlsx")
    assert xlsx.status_code == 200
    assert xlsx.content[:2] == b"PK"  # xlsx = zip

    # Change speaker name
    r = client.put(f"/api/jobs/{job_id}/speakers", json={"speakers": {"SPEAKER_00": "יועץ"}})
    assert r.status_code == 200
    assert r.json()["speakers"]["SPEAKER_00"] == "יועץ"
    assert "יועץ:" in client.get(f"/api/jobs/{job_id}/export?format=txt").text

    # Media download
    assert client.get(f"/api/jobs/{job_id}/media").status_code == 200

    # --- Global search ---
    sr = client.get("/api/search", params={"q": "לעזור"})
    assert sr.status_code == 200
    body = sr.json()
    assert body["count"] == 1
    assert body["matches"][0]["job_id"] == job_id
    # The search returns the updated speaker name ("יועץ")
    assert body["matches"][0]["speaker"] == "יועץ"
    # Empty query -> zero results
    assert client.get("/api/search", params={"q": ""}).json()["count"] == 0
    # Word that doesn't exist
    assert client.get("/api/search", params={"q": "מטוס"}).json()["count"] == 0

    # --- Language model status + summary (should fail with 503 because no LLM server) ---
    assert client.get("/api/llm/status").status_code == 200
    summ = client.post(f"/api/jobs/{job_id}/summarize")
    assert summ.status_code == 503

    # Delete
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    # After deletion — no more search results
    assert client.get("/api/search", params={"q": "לעזור"}).json()["count"] == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nAll {len(fns)} API tests passed.")
