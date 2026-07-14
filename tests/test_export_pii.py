"""בדיקות לייצוא Word ולוגיקת הסרת ה-PII (ה-PII עצמו ממוקיי — ללא HebSafeHarbor)."""

import os
import sys
import tempfile
from pathlib import Path

os.environ["PUAH_DATA_DIR"] = tempfile.mkdtemp(prefix="puah-docx-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.pipeline import pii as pii_mod  # noqa: E402
from app.pipeline.export_docx import build_docx, build_docx_bytes  # noqa: E402


def _result():
    return {
        "job_id": "j1", "language": "he", "duration_sec": 65.0, "num_speakers": 2,
        "speakers": {"SPEAKER_00": "יועץ", "SPEAKER_01": "דוד כהן"},
        "segments": [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "שלום, איך אפשר לעזור?", "words": []},
            {"start": 5.0, "end": 12.0, "speaker": "SPEAKER_01", "text": "שמי דוד כהן ואני גר ברמת גן", "words": []},
        ],
        "asr_model": "x", "diarization_model": "y",
        "summary": {
            "summary": "שיחת ייעוץ עם דוד כהן.",
            "key_points": ["דוד כהן פנה לייעוץ"],
            "action_items": ["ליצור קשר עם דוד כהן"],
            "topics": ["ייעוץ"],
            "keywords": ["פוריות"],
            "model": "gemma4:12b",
        },
    }


# ---------- ייצוא Word ----------

def test_build_docx_creates_file_with_content():
    from docx import Document

    out = Path(os.environ["PUAH_DATA_DIR"]) / "out.docx"
    build_docx(_result(), out, title="שיחת בדיקה.mp4")
    assert out.exists() and out.stat().st_size > 0
    text = "\n".join(p.text for p in Document(str(out)).paragraphs)
    assert "שיחת בדיקה.mp4" in text
    assert "שמי דוד כהן ואני גר ברמת גן" in text  # קטע תמלול
    assert "סיכום אוטומטי" in text                  # מקטע הסיכום
    assert "דוד כהן" in text                          # שם הדובר


def test_build_docx_bytes_is_valid_docx():
    data = build_docx_bytes(_result(), title="t", deidentified=True)
    assert data[:2] == b"PK"  # docx = zip
    assert len(data) > 500


# ---------- מיפוי הסרת PII (ממוקיי) ----------

def test_deidentify_result_maps_all_fields(monkeypatch):
    # מדמה את HebSafeHarbor: מחליף כל טקסט ב-[REDACTED:<len>] כדי לאמת מיפוי 1:1
    def fake_run(settings, texts):
        assert isinstance(texts, list)
        return [f"<<{i}>>" for i, _ in enumerate(texts)]

    monkeypatch.setattr(pii_mod, "_run_pii", fake_run)

    res = _result()
    out = pii_mod.deidentify_result(Settings(), res)

    # מקור לא השתנה (deepcopy)
    assert res["segments"][1]["text"] == "שמי דוד כהן ואני גר ברמת גן"
    # כל הקטעים הוחלפו
    assert out["segments"][0]["text"] == "<<0>>"
    assert out["segments"][1]["text"] == "<<1>>"
    # שדות הסיכום הוחלפו ושמרו על המבנה (אורכי רשימות)
    assert out["summary"]["summary"] == "<<2>>"
    assert out["summary"]["key_points"] == ["<<3>>"]
    assert out["summary"]["action_items"] == ["<<4>>"]
    assert out["summary"]["topics"] == ["<<5>>"]
    assert out["summary"]["keywords"] == ["<<6>>"]
    assert out["pii_deidentified"] is True


def test_deidentify_includes_client_and_feedback(monkeypatch):
    monkeypatch.setattr(pii_mod, "_run_pii", lambda s, texts: [f"<<{i}>>" for i, _ in enumerate(texts)])
    r = _result()
    r["client_summary"] = "שלום דוד כהן, סיכום השיחה."
    r["feedback"] = "ציון 4 מתוך 5. היועץ דוד עשה עבודה טובה."
    out = pii_mod.deidentify_result(Settings(), r)
    # שני השדות החדשים אמורים להיות מנוקים (מוחלפים)
    assert out["client_summary"].startswith("<<")
    assert out["feedback"].startswith("<<")
    # המקור לא השתנה
    assert "דוד כהן" in r["client_summary"]


def test_deidentify_empty_raises(monkeypatch):
    monkeypatch.setattr(pii_mod, "_run_pii", lambda s, t: t)
    empty = {"segments": [], "summary": None}
    try:
        pii_mod.deidentify_result(Settings(), empty)
        assert False, "expected PIIError"
    except pii_mod.PIIError:
        pass


def test_pii_python_path():
    s = Settings()
    p = pii_mod.pii_python(s)
    assert s.pii_venv in str(p)
    assert p.name.startswith("python")


if __name__ == "__main__":
    import inspect

    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    fns = [v for v in globals().values() if callable(v) and getattr(v, "__name__", "").startswith("test_")]
    fns.sort(key=lambda f: f.__code__.co_firstlineno)
    for fn in fns:
        if "monkeypatch" in inspect.signature(fn).parameters:
            fn(_MP())
        else:
            fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nכל {len(fns)} בדיקות Word/PII עברו.")
