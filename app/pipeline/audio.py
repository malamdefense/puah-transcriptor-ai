"""**Audio extraction and conversion from mp4 using ffmpeg.**

Both Whisper and pyannote perform best on mono WAV at 16kHz, so we convert once and reuse the same file for both stages.

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
            "ffmpeg not found in PATH. Install ffmpeg (e.g. winget install Gyan.FFmpeg "
            "or choco install ffmpeg) and run again."
        )
    return path


def _ffprobe_bin() -> str | None:
    return shutil.which("ffprobe")


def extract_audio(src: Path, dst: Path) -> None:
    """Convert video/audio file to mono 16kHz WAV (PCM 16-bit)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i", str(src),
        "-vn",            # no video
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz
        "-acodec", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"Audio extraction failed (code {proc.returncode}).\n"
            f"{proc.stderr[-1500:]}"
        )
    if not dst.exists() or dst.stat().st_size == 0:
        raise FFmpegError("Audio extraction completed but no valid audio file was created.")


def get_duration(path: Path) -> float:
    """File duration in seconds (0.0 if not determinable)."""
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0
