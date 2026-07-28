"""Export the transcript to formats: text, SRT, WebVTT, JSON."""

from __future__ import annotations

import json
from typing import Any


def _fmt_ts(seconds: float, sep: str = ",") -> str:
    """HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _fmt_clock(seconds: float) -> str:
    """[MM:SS] for display in text."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _speaker_name(result: dict[str, Any], spk_id: str) -> str:
    return result.get("speakers", {}).get(spk_id, spk_id)


def to_txt(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for seg in result["segments"]:
        name = _speaker_name(result, seg["speaker"])
        lines.append(f"[{_fmt_clock(seg['start'])}] {name}: {seg['text']}")
    return "\n".join(lines) + "\n"


def to_srt(result: dict[str, Any]) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(result["segments"], start=1):
        name = _speaker_name(result, seg["speaker"])
        blocks.append(
            f"{i}\n"
            f"{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n"
            f"{name}: {seg['text']}\n"
        )
    return "\n".join(blocks)


def to_vtt(result: dict[str, Any]) -> str:
    out = ["WEBVTT", ""]
    for seg in result["segments"]:
        name = _speaker_name(result, seg["speaker"])
        out.append(f"{_fmt_ts(seg['start'], '.')} --> {_fmt_ts(seg['end'], '.')}")
        out.append(f"<v {name}>{seg['text']}")
        out.append("")
    return "\n".join(out)


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


EXPORTERS = {
    "txt": ("text/plain; charset=utf-8", to_txt),
    "srt": ("text/plain; charset=utf-8", to_srt),
    "vtt": ("text/vtt; charset=utf-8", to_vtt),
    "json": ("application/json; charset=utf-8", to_json),
}
