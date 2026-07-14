#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""רץ הסרת-PII עצמאי — מורץ ע"י הפייתון של .venv-pii (לא של האפליקציה).

מייבא אך ורק את HebSafeHarbor ו-stdlib, כדי שלא ייגע בתלויות האפליקציה
(pydantic v2 וכו'). קורא JSON {"texts": [...]} מ-stdin, מחזיר {"texts": [...]}
מנוקה ל-stdout. טוען את מנוע HebSafeHarbor פעם אחת (הטעינה היקרה).

שימוש (ע"י pii.py כתת-תהליך):
    .venv-pii/python app/pii_runner.py  < input.json  > output.json
"""

import json
import sys

# תאימות Python 3.10+: HebSafeHarbor 0.1.1 משתמש ב-`from collections import Set`,
# כינוי שהוסר ב-Python 3.10. משחזרים את הכינויים מ-collections.abc לפני הייבוא,
# כדי לתמוך ב-Python 3.10 (וגם 3.9/3.8). פתרון רפרודוקטיבי שאינו נוגע בקובצי הספרייה.
import collections
import collections.abc as _abc

for _name in (
    "Set", "Mapping", "Sequence", "Iterable", "Callable", "Hashable",
    "MutableMapping", "MutableSet", "MutableSequence", "MappingView",
):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(_abc, _name))


def main() -> int:
    # קריאת stdin כ-UTF-8 במפורש (לא תלוי בקידוד ברירת המחדל של Windows/locale)
    try:
        raw = sys.stdin.buffer.read().decode("utf-8")
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"invalid input json: {exc}", file=sys.stderr)
        return 2

    texts = data.get("texts", [])
    if not isinstance(texts, list):
        print("'texts' must be a list", file=sys.stderr)
        return 2

    try:
        from hebsafeharbor import HebSafeHarbor
    except Exception as exc:  # noqa: BLE001
        print(f"hebsafeharbor not available: {exc}", file=sys.stderr)
        return 3

    hsh = HebSafeHarbor()  # טעינת המודל (he_ner_news_trf) — יקר, פעם אחת

    # HebSafeHarbor עלול להיכשל על מחרוזת ריקה — נשלח רווח ונחזיר ריק חזרה.
    nonempty_idx = [i for i, t in enumerate(texts) if t and str(t).strip()]
    docs = [{"text": str(texts[i])} for i in nonempty_idx]

    anon = list(texts)  # ברירת מחדל: ערכים ריקים נשארים כפי שהם
    if docs:
        output = hsh(docs)
        for pos, out in zip(nonempty_idx, output):
            try:
                anon[pos] = out.anonymized_text.text
            except Exception:  # noqa: BLE001 — אם משהו לא צפוי, עדיף ריק מאשר טקסט מקורי
                anon[pos] = ""

    # כתיבת stdout כ-UTF-8 במפורש
    sys.stdout.buffer.write(json.dumps({"texts": anon}, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
