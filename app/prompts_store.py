"""מאגר פרומפטים הניתן לעריכה מהממשק.

ברירות המחדל מוגדרות בקוד (app/pipeline/puah_prompts.py). עריכות מהממשק נשמרות
לקובץ data/prompts.json ודורסות את ברירות המחדל בזמן ריצה — בלי לגעת בקוד.
"""

from __future__ import annotations

import json
from typing import Any

from .config import get_settings
from .pipeline import puah_prompts

# Metadata for each editable prompt: key -> (Hebrew title, default value)
# Note: "Summary general — system instructions" (SUMMARY_SYSTEM) is not editable from the interface —
# it is fixed in code (app/pipeline/puah_prompts.py). Only the output instructions (JSON) are exposed for editing.
FIELDS: dict[str, dict[str, str]] = {
    "summary_instructions": {
        # "label": "סיכום כללי — הוראות פלט (JSON)",
        # "default": puah_prompts.SUMMARY_INSTRUCTIONS,
    },
    "puah_system": {
        "label": "עקרונות ליבה — סיכום ללקוח + משוב (הנחיית מערכת)",
        "default": puah_prompts.SYSTEM,
    },
    "client_summary_task": {
        "label": "סיכום ללקוח — הנחיית משימה",
        "default": puah_prompts.CLIENT_SUMMARY_TASK,
    },
    "feedback_task": {
        "label": "משוב ליועץ/רב — הנחיית משימה",
        "default": puah_prompts.FEEDBACK_TASK,
    },
}


def _store_path():
    return get_settings().data_dir / "prompts.json"


def _overrides() -> dict[str, str]:
    path = _store_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in FIELDS and isinstance(v, str)}
    except Exception:
        return {}


def get(key: str) -> str:
    return _overrides().get(key, FIELDS[key]["default"])


def all_prompts() -> dict[str, dict[str, Any]]:
    ov = _overrides()
    return {
        key: {
            "label": meta["label"],
            "value": ov.get(key, meta["default"]),
            "default": meta["default"],
            "is_custom": key in ov,
        }
        for key, meta in FIELDS.items()
    }


def save(updates: dict[str, str]) -> None:
    ov = _overrides()
    for key, val in updates.items():
        if key not in FIELDS:
            continue
        val = str(val)
        if val.strip() == FIELDS[key]["default"].strip():
            ov.pop(key, None)
        else:
            ov[key] = val
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ov, ensure_ascii=False, indent=2), encoding="utf-8")


def reset(key: str | None = None) -> None:
    if key is None:
        _store_path().unlink(missing_ok=True)
        return
    ov = _overrides()
    ov.pop(key, None)
    _store_path().write_text(json.dumps(ov, ensure_ascii=False, indent=2), encoding="utf-8")
