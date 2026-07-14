"""בדיקות לחיפוש מלא (FTS5) ולעיבוד תשובת מודל הסיכום — ללא GPU/LLM אמיתי."""

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
    # מזהה המשימה אקראי — נשתמש בזה שנוצר
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
    # חיפוש מילה מלאה
    assert len(db.search_segments("ייעוץ")) >= 1
    # תת-מחרוזת בתוך מילה עם תחילית: "הלכ" מוצא את "ההלכתיות" (בעיית התחיליות בעברית)
    assert len(db.search_segments("הלכ")) == 1
    # שתי מילים = AND: שתיהן באותו קטע
    assert len(db.search_segments("טיפולי פוריות")) == 1
    # מילה שלא קיימת
    assert db.search_segments("מטוס") == []
    # שאילתה ריקה
    assert db.search_segments("   ") == []


def test_delete_removes_from_index():
    job_id = db.list_jobs()[0]["id"]
    db.delete_job(job_id)
    assert db.search_segments("פוריות") == []


# ---------- עיבוד תשובת הסיכום ----------

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
    assert norm["key_points"] == ["נקודה אחת"]  # מחרוזת -> רשימה
    assert norm["keywords"] == []
    assert norm["action_items"] == []


def test_blank_optional_env_becomes_none():
    # רגרסיה: PUAH_DEFAULT_NUM_SPEAKERS= (ריק) ב-.env לא יפיל את הקונפיג
    s = Settings(default_num_speakers="", hf_token="")
    assert s.default_num_speakers is None
    assert s.hf_token is None
    # ערך אמיתי עדיין מומר כראוי
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
    assert len(calls) == 1  # מסלול חד-פעמי


_LONG_OUT = ("סיכום לפונה\nשלום וברכה,\n" + "בשיחתנו עסקנו בבירור מקיף בנושא. " * 8).strip()


def test_generate_client_summary_and_feedback(monkeypatch):
    settings = Settings()
    captured = {}

    def fake_chat(s, messages, temperature=0.2):
        captured["sys"] = messages[0]["content"]
        captured["user"] = messages[1]["content"]
        return "  " + _LONG_OUT + "  "  # רווחים מסביב לבדיקת strip

    monkeypatch.setattr(sm, "_chat", fake_chat)
    r = _sample_result("j3")
    cs = sm.generate_client_summary(settings, r)
    assert cs == _LONG_OUT  # נחתך (strip)
    assert "מכון פוע" in captured["sys"]  # נעשה שימוש ב-system של פוע"ה
    assert "סיכום לפונה" in captured["user"]  # ושל משימת הסיכום-ללקוח
    fb = sm.generate_feedback(settings, r)
    assert fb and "תמלול השיחה" in captured["user"]  # התמלול הוזרק לפרומפט


def test_retries_on_degenerate_output(monkeypatch):
    """רגרסיה: gemma פולט לעיתים <unused50> — חייבים לזהות ולנסות שוב."""
    calls = {"n": 0}

    def flaky(s, messages, temperature=0.2):
        calls["n"] += 1
        if calls["n"] == 1:
            return "<unused50>" * 30  # הפלט המנוון שראינו בייצור
        return _LONG_OUT

    monkeypatch.setattr(sm, "_chat", flaky)
    out = sm.generate_client_summary(Settings(), _sample_result("j9"))
    assert calls["n"] == 2  # ניסה שוב לאחר הפלט הפגום
    assert "<unused" not in out and out == _LONG_OUT


def test_raises_after_all_attempts_fail(monkeypatch):
    monkeypatch.setattr(sm, "_chat", lambda s, m, temperature=0.2: "<unused50>" * 30)
    try:
        sm.generate_feedback(Settings(), _sample_result("j10"))
        assert False, "expected LLMError"
    except sm.LLMError as exc:
        assert "לא תקין" in str(exc)


def test_looks_valid_rules():
    assert sm._looks_valid("א" * 200) is False  # מעט תווים ייחודיים = מנוון
    assert sm._looks_valid("קצר") is False
    assert sm._looks_valid(_LONG_OUT) is True
    assert sm._looks_valid(_LONG_OUT + "<unused50>") is False


def test_summarize_mapreduce(monkeypatch):
    settings = Settings(summary_max_chars=20)  # קטן בכוונה -> כפיית map-reduce
    n = {"calls": 0}

    def fake_chat(s, messages, temperature=0.2):
        n["calls"] += 1
        # הקריאה האחרונה (האיחוד) מחזירה JSON; קודמות מחזירות טקסט
        if "JSON" in messages[-1]["content"]:
            return '{"summary": "מאוחד", "key_points": ["נ"]}'
        return "סיכום חלקי של קטע זה, עם מספיק תוכן כדי לעבור ולידציה."

    monkeypatch.setattr(sm, "_chat", fake_chat)
    out = sm.summarize(settings, _sample_result("j2"))
    assert out["summary"] == "מאוחד"
    assert n["calls"] >= 2  # לפחות מיפוי אחד + איחוד


def test_prompts_store_override_and_reset():
    from app import prompts_store

    # ברירת מחדל
    assert "סיכום לפונה" in prompts_store.get("client_summary_task")
    assert prompts_store.all_prompts()["client_summary_task"]["is_custom"] is False
    default = prompts_store.all_prompts()["client_summary_task"]["default"]

    # שמירת override
    prompts_store.save({"client_summary_task": "פרומפט מותאם אישית"})
    assert prompts_store.get("client_summary_task") == "פרומפט מותאם אישית"
    assert prompts_store.all_prompts()["client_summary_task"]["is_custom"] is True

    # שמירת ערך זהה לברירת המחדל מסירה את ה-override
    prompts_store.save({"client_summary_task": default})
    assert prompts_store.all_prompts()["client_summary_task"]["is_custom"] is False

    # reset של הכול
    prompts_store.save({"summary_system": "x", "feedback_task": "y"})
    prompts_store.reset()
    assert all(not p["is_custom"] for p in prompts_store.all_prompts().values())


if __name__ == "__main__":
    # הרצה ידנית עם מוק פשוט במקום pytest monkeypatch
    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    import inspect

    # סדר הגדרה (לא אלפביתי) — לבדיקות החיפוש יש תלות סדר: יצירה -> חיפוש -> מחיקה
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
    print(f"\nכל {passed} בדיקות החיפוש/הסיכום עברו.")
