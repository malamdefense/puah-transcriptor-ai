"""Tests for full-text search (FTS5) and summary model processing — without real GPU/LLM."""

import os
import sys
import tempfile
from pathlib import Path

os.environ["PUAH_DATA_DIR"] = tempfile.mkdtemp(prefix="puah-search-")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.config import Settings  # noqa: E402
from app.pipeline import summarize as sm  # noqa: E402

db.init_db()


def _sample_result(job_id):
    return {
        "job_id": job_id,
        "language": "he",
        "duration_sec": 30.0,
        "num_speakers": 2,
        "speakers": {"SPEAKER_00": "יועץ", "SPEAKER_01": "מטופל"},
        "segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "שלום וברוך הבא לפגישת ייעוץ", "words": []},
            {"start": 5.0, "end": 12.0, "speaker": "SPEAKER_01", "text": "תודה רבה, רציתי לשאול על טיפולי פוריות", "words": []},
            {"start": 12.0, "end": 20.0, "speaker": "SPEAKER_00", "text": "בוודאי, נסביר על האפשרויות ההלכתיות", "words": []},
        ],
        "asr_model": "x",
        "diarization_model": "y",
        "summary": None,
    }


# ---------- FTS ----------

def test_search_finds_hebrew_terms():
    db.create_job("a.mp4", None, "he")
    # Random job ID — we'll use the one that was created
    jobs = db.list_jobs()
    job_id = jobs[0]["id"]
    db.index_job_segments(job_id, _sample_result(job_id))

    res = db.search_segments("פוריות")
    assert len(res) == 1
    assert res[0]["job_id"] == job_id
    assert "פוריות" in res[0]["text"]
    assert res[0]["speaker"] == "מטופל"
    assert res[0]["original_name"] == "a.mp4"


def test_search_prefix_and_and():
    job_id = db.list_jobs()[0]["id"]
    # Full word search
    assert len(db.search_segments("ייעוץ")) >= 1
    # Substring within a word with prefix: "הלכ" finds "ההלכתיות" (Hebrew word prefix issue)
    assert len(db.search_segments("הלכ")) == 1
    # Two words = AND: both in the same segment
    assert len(db.search_segments("טיפולי פוריות")) == 1
    # Word that doesn't exist
    assert db.search_segments("מטוס") == []
    # Empty query
    assert db.search_segments("   ") == []


def test_delete_removes_from_index():
    job_id = db.list_jobs()[0]["id"]
    db.delete_job(job_id)
    assert db.search_segments("פוריות") == []


# ---------- Summary response processing ----------

def test_parse_json_with_code_fence():
    raw = '```json\n{"summary": "תקציר", "key_points": ["א", "ב"]}\n```'
    parsed = sm._parse_json(raw)
    assert parsed["summary"] == "תקציר"
    assert parsed["key_points"] == ["א", "ב"]


def test_parse_json_with_surrounding_text():
    raw = 'הנה הסיכום שביקשת:\n{"summary": "ש", "topics": ["נושא"]} בהצלחה!'
    parsed = sm._parse_json(raw)
    assert parsed["topics"] == ["נושא"]


def test_normalize_coerces_types():
    norm = sm._normalize_summary({"summary": "x", "key_points": "נקודה אחת", "keywords": None})
    assert norm["key_points"] == ["נקודה אחת"]  # String -> list
    assert norm["keywords"] == []
    assert norm["action_items"] == []


def test_blank_optional_env_becomes_none():
    # Regression: PUAH_DEFAULT_NUM_SPEAKERS= (empty) in .env doesn't fall through the config
    s = Settings(default_num_speakers="", hf_token="")
    assert s.default_num_speakers is None
    assert s.hf_token is None
    # Real value still converts correctly
    assert Settings(default_num_speakers="3").default_num_speakers == 3


def test_summarize_single_shot(monkeypatch):
    settings = Settings(summary_max_chars=100000)
    calls = []

    def fake_chat(s, messages, temperature=0.2):
        calls.append(messages)
        return '{"summary": "סיכום השיחה", "key_points": ["נקודה"], "action_items": ["לתאם פגישה"], "topics": ["פוריות"], "keywords": ["הלכה"]}'

    monkeypatch.setattr(sm, "_chat", fake_chat)
    out = sm.summarize(settings, _sample_result("j1"))
    assert out["summary"] == "סיכום השיחה"
    assert out["action_items"] == ["לתאם פגישה"]
    assert out["model"] == settings.llm_model
    assert len(calls) == 1  # Single-path


def test_generate_client_summary_and_feedback(monkeypatch):
    settings = Settings()
    captured = {}

    def fake_chat(s, messages, temperature=0.2):
        captured["sys"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return "  סיכום לפונה\nשלום וברכה, ...  "  # With whitespace around for strip test

    monkeypatch.setattr(sm, "_chat", fake_chat)
    r = _sample_result("j3")
    cs = sm.generate_client_summary(settings, r)
    assert cs == "סיכום לפונה\nשלום וברכה, ..."  # Stripped
    assert "מכון פוע" in captured["sys"]  # System of Puah
    assert "סיכום לפונה" in captured["user"]  # From client summary task
    fb = sm.generate_feedback(settings, r)
    assert fb and "תמלול השיחה" in captured["user"]  # Promped into the system


def test_summarize_mapreduce(monkeypatch):
    settings = Settings(summary_max_chars=20)  # Smaller for map-reduce testing
    n = {"calls": 0}

    def fake_chat(s, messages, temperature=0.2):
        n["calls"] += 1
        # Last call (union) returns JSON; previous calls return text
        if "JSON" in messages[-1]["content"]:
            return '{"summary": "מאוחד", "key_points": ["נ"]}'
        return "סיכום חלקי"

    monkeypatch.setattr(sm, "_chat", fake_chat)
    out = sm.summarize(settings, _sample_result("j2"))
    assert out["summary"] == "מאוחד"
    assert n["calls"] >= 2  # At least one mapping + union


if __name__ == "__main__":
    # Direct run without pytest monkeypatch
    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    import inspect

    # Ordering (not alphabetical) — search tests have order dependency: create -> search -> delete
    fns = [v for v in globals().values() if callable(v) and getattr(v, "__name__", "").startswith("test_")]
    fns.sort(key=lambda f: f.__code__.co_firstlineno)
    passed = 0
    for fn in fns:
        if "monkeypatch" in inspect.signature(fn).parameters:
            fn(_MP())
        else:
            fn()
        print(f"  ok: {fn.__name__}")
        passed += 1
    print(f"\nAll {passed} search/summary tests passed.")
