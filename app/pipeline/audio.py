"""חילוץ והמרת אודיו מ-mp4 באמצעות ffmpeg.

גם Whisper וגם pyannote עובדים הכי טוב על WAV מונו ב-16kHz, אז אנחנו ממירים פעם
אחת ומשתמשים באותו קובץ בשני השלבים.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegError(
            "ffmpeg לא נמצא ב-PATH. התקן ffmpeg (לדוגמה: winget install Gyan.FFmpeg "
            "או choco install ffmpeg) והפעל מחדש."
        )
    return path


def _ffprobe_bin() -> str | None:
    return shutil.which("ffprobe")


def extract_audio(src: Path, dst: Path) -> None:
    """ממיר קובץ וידאו/אודיו ל-WAV מונו 16kHz (PCM 16-bit)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i", str(src),
        "-vn",            # ללא וידאו
        "-ac", "1",       # מונו
        "-ar", "16000",   # 16kHz
        "-acodec", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(
            f"חילוץ האודיו נכשל (קוד {proc.returncode}).\n"
            f"{proc.stderr[-1500:]}"
        )
    if not dst.exists() or dst.stat().st_size == 0:
        raise FFmpegError("חילוץ האודיו הסתיים אך לא נוצר קובץ אודיו תקין.")


def get_duration(path: Path) -> float:
    """משך הקובץ בשניות (0.0 אם לא ניתן לקבוע)."""
    ffprobe = _ffprobe_bin()
    if not ffprobe:
        return 0.0
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0
