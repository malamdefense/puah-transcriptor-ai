"""Secrets store that can be edited from the interface (overrides .env).

Similar to prompts_store, but for API keys: the value saved here (data/secrets.json) *overrides*
the value in .env at runtime. Intended for the case where the system manager wants to enter/replace a key
from the interface without touching the server's configuration files.

Security: the value is saved in plain text in a local file (like .env itself), but *never* returned
in full through the API — only in a masked form (see masked()).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Editable keys from the interface: name in the store -> corresponding field in Settings
FIELDS: dict[str, str] = {
    "gemini_api_key": "gemini_api_key",
}


def _store_path(data_dir: Path) -> Path:
    return Path(data_dir) / "secrets.json"


def _load(data_dir: Path) -> dict[str, str]:
    path = _store_path(data_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {k: v for k, v in data.items() if k in FIELDS and isinstance(v, str) and v.strip()}


def apply_overrides(settings: Any, data_dir: Path) -> None:
    """Apply the saved overrides to the settings object (instead of the value from .env).

    data_dir is passed explicitly to avoid circular dependency with config.get_settings().
    """
    for key, field in FIELDS.items():
        value = _load(data_dir).get(key)
        if value:
            setattr(settings, field, value)


def masked(value: str | None) -> str | None:
    """Safe representation for display: only the beginning and end of the key."""
    if not value:
        return None
    v = value.strip()
    if len(v) <= 12:
        return v[:2] + "…" + v[-2:]
    return v[:6] + "…" + v[-4:]


def status(settings: Any) -> dict[str, Any]:
    """Status of the keys for the interface — without revealing the full value."""
    from .config import Settings  # ייבוא עצל: קריאת ערכי ה-.env "הנקיים"

    env_only = Settings()
    overrides = _load(settings.data_dir)
    out: dict[str, Any] = {}
    for key, field in FIELDS.items():
        override = overrides.get(key)
        env_value = getattr(env_only, field, None)
        effective = override or env_value
        out[key] = {
            "is_custom": bool(override),         
            "env_present": bool(env_value),      
            "configured": bool(effective),        
            "source": "ui" if override else ("env" if env_value else "none"),
            "masked": masked(effective),
        }
    return out


def save(settings: Any, key: str, value: str) -> None:
    """Save a key entered from the interface and apply it immediately to the active settings."""
    if key not in FIELDS:
        raise KeyError(key)
    value = (value or "").strip()
    if not value:
        raise ValueError("Empty value")
    data_dir = Path(settings.data_dir)
    data = _load(data_dir)
    data[key] = value
    path = _store_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    setattr(settings, FIELDS[key], value)


def clear(settings: Any, key: str) -> None:
    if key not in FIELDS:
        raise KeyError(key)
    from .config import Settings

    data_dir = Path(settings.data_dir)
    data = _load(data_dir)
    data.pop(key, None)
    path = _store_path(data_dir)
    if data:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)
    setattr(settings, FIELDS[key], getattr(Settings(), FIELDS[key], None))
