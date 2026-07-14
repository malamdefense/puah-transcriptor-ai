"""בדיקות ל-RAG הסמנטי — עם embeddings ממוקים (ללא Ollama/GPU)."""

import os
import sys
import tempfile
from pathlib import Path

os.environ["PUAH_DATA_DIR"] = tempfile.mkdtemp(prefix="puah-rag-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402
from app.config import Settings  # noqa: E402

_TMP = Path(os.environ["PUAH_DATA_DIR"])


def _fake_embed(settings, texts):
    """embedding דטרמיניסטי: כל מילת-מפתח = ציר נפרד."""
    keys = ("פוריות", "הלכה", "אלף", "בית")
    return [[1.0 if k in t else 0.0 for k in keys] for t in texts]


def _make_xlsx() -> Path:
    import openpyxl

    path = _TMP / "gt.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["name of the record", "סיכום ללקוח", "משוב"])
    ws.append(["rec_a", "סיכום בנושא פוריות", "משוב אלף ליועץ"])
    ws.append(["rec_b", "סיכום בנושא הלכה", "משוב בית ליועץ"])
    wb.save(path)
    return path


def test_read_excel_rows():
    rows = rag.read_excel_rows(_make_xlsx())
    assert len(rows) == 2
    assert rows[0]["recording"] == "rec_a"
    assert "פוריות" in rows[0]["summary"] and "אלף" in rows[0]["feedback"]


def test_build_and_search(monkeypatch):
    monkeypatch.setattr(rag, "embed_texts", _fake_embed)
    s = Settings()
    info = rag.build_from_excel(_make_xlsx(), s)
    assert info["rows"] == 2 and info["entries"] == 4  # 2 שורות × (סיכום + משוב)
    assert rag.stats(s) == {
        "summary": {"entries": 2, "recordings": 2},
        "feedback": {"entries": 2, "recordings": 2},
    }

    # שליפה סמנטית — הסיכום הנכון
    top = rag.search("טקסט שנוצר על פוריות", "summary", top_k=2, settings=s)
    assert top[0]["recording"] == "rec_a" and top[0]["score"] > 0.9
    assert rag.search("טקסט שנוצר על הלכה", "summary", top_k=1, settings=s)[0]["recording"] == "rec_b"

    # שליפה סמנטית — המשוב הנכון
    assert rag.search("משהו אלף", "feedback", top_k=1, settings=s)[0]["recording"] == "rec_a"
    assert rag.search("משהו בית", "feedback", top_k=1, settings=s)[0]["recording"] == "rec_b"


def test_kind_isolation(monkeypatch):
    """שאילתת סיכום לא תחזיר רשומות משוב (הפרדה לפי kind)."""
    monkeypatch.setattr(rag, "embed_texts", _fake_embed)
    s = Settings()
    rag.build_from_excel(_make_xlsx(), s)
    res = rag.search("אלף", "summary", top_k=99, settings=s)
    assert len(res) == 2  # רק 2 הסיכומים, לא המשובים
    assert all(r["recording"] in ("rec_a", "rec_b") for r in res)


def test_bad_kind_raises(monkeypatch):
    monkeypatch.setattr(rag, "embed_texts", _fake_embed)
    try:
        rag.search("x", "not_a_kind", settings=Settings())
        assert False, "expected RagError"
    except rag.RagError:
        pass


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
    print(f"\nכל {len(fns)} בדיקות ה-RAG עברו.")
