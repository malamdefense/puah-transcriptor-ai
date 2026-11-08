"""Recording-quality check, and gated fixes applied only to the copy of the audio the ASR hears.

Measured on 29 real calls and on controlled degradations of them (see the 'Recording-quality
check' block in config.py for the thresholds and what each one rests on). Two entry points:

prepare_asr_input(wav, diar, settings, work_dir) -> AsrInput
    After extract_audio and diarization, before ASR. One Silero pass over the original WAV gives
    the call metrics (speech level, SNR, noise floor, clipping, speech spectral flatness,
    bandwidth, mains hum, VAD kept share at thresholds 0.5 and 0.2) and, with the diarization
    turns, each speaker's acoustic metrics. Then the fixes, in the measured order
    hum notch -> low level -> per-speaker gain, each only when its own condition holds on the
    signal as it stands after the previous step. The processed copy {work_dir}/{stem}_asr.wav is
    written only when a fix applied (complete or not at all); the original is never modified and
    no fix moves the time axis, so diarization, merge and timestamps keep using the original.
    Last, the VAD sensitivity is decided on the audio the ASR will hear, and the VAD chunks the
    ASR will use are known, so each speaker's share of untranscribed speech is computed here.

finalize_report(report, asr_segments, diar, settings) -> dict
    After ASR: each speaker's word confidence (words assigned to speakers by overlap with the
    turns), then every flag with its fixed Hebrew message. Returns result['audio_quality'].

The report holds numbers, codes and fixed Hebrew messages only, never transcript text. Messages
name a speaker as the literal placeholder {speaker}; the UI substitutes the current display name.

numpy is required. soundfile is used to read when it loads (the stdlib wave module otherwise),
ffmpeg runs only when the hum or the low-level gate fires, and Silero (faster_whisper.vad,
onnxruntime) is imported lazily, so this module imports cheaply.
"""

from __future__ import annotations

import bisect
import logging
import math
import os
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("puah.audio_quality")

VERSION = 1
SR = 16000
SILERO_WIN = 512  # Silero VAD hop / window (samples)
BLOCK = 80  # 5 ms energy blocks; every frame below is a whole number of them
FRAME_WIN, FRAME_HOP = 400, 160  # 25 ms frames every 10 ms (levels, masks)
FRAME_CENTER = FRAME_WIN // 2
ACTIVE_FRAME = 320  # 20 ms frames of the active level (louder half of the frames)
NEAR_FS = 31130  # |int16| at or above 0.95 of full scale
LOWP = 0.5  # a word below this probability counts as low-confidence
SPEECH_PROB, NONSPEECH_PROB = 0.5, 0.2  # Silero probability: speech frame / non-speech frame
MIN_ACTIVE_FRAMES = 100  # a speaker needs 1 s of Silero speech frames to be judged
FFT_N = 512
FLAT_BAND_HZ = (100.0, 3800.0)
FLAT_STRIDE = 3  # flatness on every 3rd Silero speech window (measured cost/precision trade-off)
HUM_BAND_HZ = (45.0, 55.0)
VOICE_BAND_HZ = (300.0, 3400.0)
DETECT_THRESHOLDS = (0.5, 0.2)  # VAD kept share is compared at these two thresholds
NOTCH_FILTER = ",".join(f"bandreject=f={f}:width_type=h:w=8" for f in (50, 100, 150, 200, 250))
LOUDNORM_FILTER = "loudnorm=I=-20:TP=-2"
_LENGTH_TOLERANCE = SR // 20  # an ffmpeg step may end a few samples off; more is an error
_FFMPEG_TIMEOUT_S = 900

TITLE_HE = "בדיקת איכות הקלטה"
NO_ISSUES_HE = "לא נמצאו בעיות באיכות ההקלטה."
FOOTNOTE_HE = "הבדיקה אוטומטית; ההקלטה המקורית לא שונתה."

# Fixed wording. {speaker} stays a placeholder for the UI; {p} (percent) and {d} (dB) are filled
# with whole numbers when the message is built.
MESSAGES_HE: dict[tuple[str, str], str] = {
    ("noisy_speaker", "warn"): "יש רעש רקע בקטעים של {speaker} — ייתכנו שגיאות בתמלול דבריו.",
    ("noisy_speaker", "severe"): (
        "יש רעש רקע חזק בקטעים של {speaker} — התמלול של דבריו עלול להיות חלקי ולא מדויק. "
        "מומלץ לבדוק מול ההקלטה."
    ),
    ("quiet_speaker", "warn"): "{speaker} נשמע חלש בהרבה מהצד השני (בכ-{d} dB) — ייתכן שחלק מדבריו לא תומלל.",
    ("quiet_speaker", "severe"): (
        "{speaker} כמעט לא נשמע בהקלטה (חלש בכ-{d} dB מהצד השני) — חלקים מדבריו כנראה חסרים בתמלול."
    ),
    ("clipping", "speaker"): (
        "בקטעים של {speaker} ההקלטה מעוותת (עוצמה גבוהה מדי). "
        "עיוות כזה בדרך כלל אינו פוגע משמעותית בתמלול."
    ),
    ("clipping", "call"): (
        "ההקלטה מעוותת בחלקים (עוצמה גבוהה מדי). עיוות כזה בדרך כלל אינו פוגע משמעותית בתמלול."
    ),
    ("narrowband", "info"): (
        "איכות השמע בהקלטה נמוכה מהרגיל לשיחת טלפון (רוחב פס צר) — ייתכנו שגיאות בתמלול."
    ),
    ("speech_not_transcribed", "warn"): (
        "כ-{p}% מזמן הדיבור של {speaker} לא זוהה כדיבור ולכן לא תומלל. "
        "מומלץ להאזין להקלטה בקטעים שבהם חסר תמלול."
    ),
    ("low_confidence_speaker", "warn"): (
        "ביטחון התמלול נמוך בדברי {speaker} ({p}% מהמילים בביטחון נמוך) — מומלץ לבדוק מול ההקלטה. "
        "סיבות אפשריות: איכות שמע, דיבור חופף או מילים בשפה אחרת."
    ),
    # the mean-probability rule alone (few words below 0.5, but the average is low)
    ("low_confidence_speaker", "mean"): (
        "ביטחון התמלול נמוך בדברי {speaker} — מומלץ לבדוק מול ההקלטה. "
        "סיבות אפשריות: איכות שמע, דיבור חופף או מילים בשפה אחרת."
    ),
    # Applied fixes: the 'fix' wording goes in report['fixes'], the 'flag' wording in report['flags'].
    ("speaker_gain", "fix"): "דובר אחד הוקלט חלש במיוחד — עוצמתו הוגברה אוטומטית לצורך התמלול.",
    ("speaker_gain", "flag"): "{speaker} הוקלט חלש במיוחד — עוצמתו הוגברה אוטומטית לצורך התמלול.",
    ("hum_notch", "fix"): "זוהה זמזום חשמלי בהקלטה — סונן אוטומטית לצורך התמלול.",
    ("low_level", "fix"): "עוצמת ההקלטה נמוכה מאוד — הוגברה אוטומטית לצורך התמלול.",
    ("vad_sensitivity", "fix"): (
        "בהקלטה זו חלק מהדיבור לא זוהה בבירור, ולכן התמלול הורץ ברגישות גבוהה יותר לזיהוי דיבור."
    ),
}


class AudioQualityError(RuntimeError):
    """The check could not run on this audio; the caller transcribes the original."""


@dataclass
class AsrInput:
    """What the ASR should hear.

    path — the original WAV, or the processed copy {work_dir}/{stem}_asr.wav when a fix applied.
    vad_threshold — per-job VAD threshold for Transcriber.transcribe; None = the setting.
    report — the audio_quality dict so far (version, timing_s, call, speakers, fixes; no flags).
    processed — True when path is the processed copy (the caller deletes it after ASR).
    """

    path: Path
    vad_threshold: Optional[float]
    report: dict
    processed: bool = False


# ----------------------------------------------------------------------------- small helpers
def _np():
    import numpy as np  # noqa: PLC0415

    return np


def _num(v: Any, nd: int) -> Optional[float]:
    """A JSON-safe rounded float (None for missing, NaN or infinite values)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _db10(p: float) -> float:
    return 10.0 * math.log10(p + 1e-12)


def _fill(template: str, **numbers: int) -> str:
    """Fill {p}/{d} with whole numbers and leave {speaker} for the UI (no str.format: a
    placeholder the template does not use must not raise)."""
    out = template
    for key, val in numbers.items():
        out = out.replace("{" + key + "}", str(int(val)))
    return out


# ----------------------------------------------------------------------------- WAV I/O
def _read_wav_int16(path: Path):
    """Mono 16 kHz PCM16 samples as int16 (soundfile when it loads, the wave module otherwise)."""
    np = _np()
    data = sr = channels = None
    try:
        import soundfile as sf  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — absent, or its native library refused to load
        sf = None
    if sf is not None:
        arr, sr = sf.read(str(path), dtype="int16", always_2d=True)
        channels = arr.shape[1]
        data = arr[:, 0] if channels else arr.reshape(-1)
    else:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise AudioQualityError("expected 16-bit PCM audio")
            sr, channels = w.getframerate(), w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            data = raw.reshape(-1, channels)[:, 0] if channels > 1 else raw
    if sr != SR:
        raise AudioQualityError(f"expected {SR} Hz audio, got {sr}")
    if channels != 1:
        raise AudioQualityError("expected mono audio")
    return np.ascontiguousarray(data, dtype=np.int16)


def _write_wav_int16(path: Path, x16) -> None:
    np = _np()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.ascontiguousarray(x16, dtype="<i2").tobytes())


def _ffmpeg(src: Path, dst: Path, af: str) -> None:
    from .audio import FFmpegError, _ffmpeg_bin  # noqa: PLC0415

    cmd = [_ffmpeg_bin(), "-y", "-v", "error", "-nostdin", "-i", str(src), "-af", af,
           "-ac", "1", "-ar", str(SR), "-acodec", "pcm_s16le", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, timeout=_FFMPEG_TIMEOUT_S)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg filter failed (code {proc.returncode}): {proc.stderr[-400:]}")


def _read_same_length(path: Path, n: int):
    """An ffmpeg output, held to the original's length (the time axis must not move)."""
    np = _np()
    y = _read_wav_int16(path)
    if abs(len(y) - n) > _LENGTH_TOLERANCE:
        raise AudioQualityError(f"filter changed the length by {len(y) - n} samples")
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y), dtype=np.int16)])
    return y[:n]


# ----------------------------------------------------------------------------- Silero + VAD
def _silero_probs(x):
    """Silero speech probability per 512-sample window, computed exactly as faster-whisper's VAD
    does it (same model, same padding), so the chunks below are the ones the ASR will use."""
    np = _np()
    from faster_whisper.vad import get_vad_model  # noqa: PLC0415

    padded = np.pad(x, (0, SILERO_WIN - len(x) % SILERO_WIN))
    return np.asarray(get_vad_model()(padded), dtype=np.float32).reshape(-1)


def speech_chunks(probs, n_samples: int, threshold: float, *, min_speech_ms: int = 0,
                  min_silence_ms: int = 2000, pad_ms: int = 400,
                  max_speech_s: float = float("inf"), neg_threshold: Optional[float] = None,
                  ) -> list[tuple[int, int]]:
    """faster_whisper.vad.get_speech_timestamps (1.2.1) on precomputed probabilities.

    A line-by-line port, so one Silero pass serves every threshold and nothing global is patched.
    Comparisons are made in float32, as numpy does it for the library's float32 probabilities.
    Returns (start, end) sample pairs.
    """
    np = _np()
    window = SILERO_WIN
    min_speech_samples = SR * min_speech_ms / 1000
    speech_pad_samples = SR * pad_ms / 1000
    max_speech_samples = SR * max_speech_s - window - 2 * speech_pad_samples
    min_silence_samples = SR * min_silence_ms / 1000
    min_silence_samples_at_max_speech = SR * 98 / 1000
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    above = (p >= np.float32(threshold)).tolist()
    below = (p < np.float32(neg_threshold)).tolist()

    triggered = False
    speeches: list[dict] = []
    current: dict = {}
    temp_end = 0
    prev_end = next_start = 0
    for i in range(len(above)):
        pos = window * i
        if above[i] and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = pos
        if above[i] and not triggered:
            triggered = True
            current["start"] = pos
            continue
        if triggered and pos - current["start"] > max_speech_samples:
            if prev_end:
                current["end"] = prev_end
                speeches.append(current)
                current = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current["end"] = pos
                speeches.append(current)
                current = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue
        if below[i] and triggered:
            if not temp_end:
                temp_end = pos
            if pos - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if pos - temp_end < min_silence_samples:
                continue
            current["end"] = temp_end
            if current["end"] - current["start"] > min_speech_samples:
                speeches.append(current)
            current = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            continue
    if current and (n_samples - current["start"]) > min_speech_samples:
        current["end"] = n_samples
        speeches.append(current)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - silence_duration // 2))
            else:
                speech["end"] = int(min(n_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
        else:
            speech["end"] = int(min(n_samples, speech["end"] + speech_pad_samples))
    return [(int(s["start"]), int(s["end"])) for s in speeches]


def _vad_options(settings) -> dict:
    """The production VadOptions (Transcriber.transcribe), apart from the threshold."""
    return {"min_speech_ms": int(settings.asr_vad_min_speech_duration_ms),
            "min_silence_ms": int(settings.asr_vad_min_silence_duration_ms),
            "pad_ms": int(settings.asr_vad_speech_pad_ms)}


def _kept_share(chunks: list[tuple[int, int]], n: int) -> float:
    return sum(e - s for s, e in chunks) / n if n else 0.0


# ----------------------------------------------------------------------------- frame arrays
def _block_energy(x):
    """Sum of squares of each 5 ms block (float64), in slices to keep memory flat."""
    np = _np()
    nb = len(x) // BLOCK
    out = np.empty(nb, dtype=np.float64)
    step = BLOCK * 65536
    for a in range(0, nb * BLOCK, step):
        seg = x[a:min(a + step, nb * BLOCK)].astype(np.float64)
        out[a // BLOCK:a // BLOCK + len(seg) // BLOCK] = np.square(seg).reshape(-1, BLOCK).sum(1)
    return out


def _frame_power(b):
    """Mean power of 25 ms frames every 10 ms (frame i = samples [160 i, 160 i + 400))."""
    np = _np()
    if len(b) < 5:
        return np.zeros(0)
    n = (len(b) - 5) // 2 + 1
    c = np.concatenate([[0.0], np.cumsum(b)])
    idx = 2 * np.arange(n)
    return (c[idx + 5] - c[idx]) / FRAME_WIN


def _energy20(b):
    """Mean power of consecutive 20 ms frames."""
    n = len(b) // 4
    return b[:n * 4].reshape(n, 4).sum(1) / ACTIVE_FRAME


def _active_rms(e20, frame_sel=None, top: float = 0.5) -> float:
    """RMS of the louder `top` share of 20 ms frames (only the selected frames when given)."""
    np = _np()
    e = e20 if frame_sel is None else e20[frame_sel]
    if len(e) == 0:
        return float("nan")
    k = max(1, int(len(e) * top))
    return float(math.sqrt(float(np.mean(np.sort(e)[::-1][:k]))))


def _rms_db(r: float) -> float:
    return 20.0 * math.log10(r + 1e-9) if math.isfinite(r) else float("nan")


def _frame_mask(intervals_s, n: int):
    """10 ms frames whose centre lies inside any (start, end) interval in seconds."""
    np = _np()
    m = np.zeros(n, dtype=bool)
    off = FRAME_CENTER / SR
    for s, e in intervals_s:
        a = int(math.ceil((s - off) / 0.01))
        b = int(math.floor((e - off) / 0.01)) + 1
        if b > a:
            m[max(0, a):min(n, b)] = True
    return m


def _abs_counts(x16):
    """Histogram of |sample| over int16 values 0..32768."""
    np = _np()
    counts = np.zeros(65536, dtype=np.int64)
    step = 1 << 22
    u = x16.view(np.uint16)
    for a in range(0, len(u), step):
        counts += np.bincount(u[a:a + step], minlength=65536)
    out = np.zeros(32769, dtype=np.int64)
    out[0] = counts[0]
    out[1:32768] = counts[1:32768] + counts[65535:32768:-1]  # +v and -v
    out[32768] = counts[32768]  # -32768
    return out


def _clip_metrics(counts) -> tuple[Optional[float], Optional[float], int]:
    """(share of samples >= 0.95 FS, histogram pile-up ratio at the peak, peak)."""
    np = _np()
    total = int(counts.sum())
    if total == 0:
        return None, None, 0
    nz = np.flatnonzero(counts)
    peak = int(nz[-1])
    near = float(counts[NEAR_FS:].sum() / total)
    if peak < 1:
        return near, 0.0, peak
    # Same binning as np.histogram(|x|[|x| > 0], bins=100, range=(0, peak + 1)).
    h, _ = np.histogram(np.arange(1, peak + 1), bins=100, range=(0, peak + 1),
                        weights=counts[1:peak + 1].astype(np.float64))
    ratio = float(h[-1] / max(1.0, float(np.median(h[-11:-1]))))
    return near, ratio, peak


@dataclass
class _Hops:
    """Per-10 ms-hop clipping aggregates; hop i = samples [160 i + 120, 160 i + 280), the
    stretch of samples that 10 ms frame i stands for."""

    seg: Any  # int16 samples, shape (n_frames, 160)
    near: Any  # samples at >= 0.95 FS in each hop
    absmax: Any  # largest |sample| in each hop


def _hops(x16, n_frames: int) -> _Hops:
    np = _np()
    off = FRAME_CENTER - FRAME_HOP // 2
    n = max(0, min(n_frames, (len(x16) - off) // FRAME_HOP))
    seg = x16[off:off + n * FRAME_HOP].reshape(n, FRAME_HOP)
    absmax = np.maximum(seg.max(1).astype(np.int32), -seg.min(1).astype(np.int32))
    near = ((seg >= NEAR_FS) | (seg <= -NEAR_FS)).sum(1)
    return _Hops(seg=seg, near=near, absmax=absmax)


def _speaker_clip_metrics(hops: _Hops, m) -> tuple[Optional[float], Optional[float]]:
    """_clip_metrics over the samples of the selected frames, from the hop aggregates.

    The pile-up ratio reads only the top 11 of the 100 bins (>= 89% of the speaker's peak), so
    only the hops that reach that high are histogrammed; the result is identical.
    """
    np = _np()
    m = m[:len(hops.absmax)]
    total = int(m.sum()) * FRAME_HOP
    if total == 0:
        return None, None
    near = float(hops.near[m].sum() / total)
    peak = int(hops.absmax[m].max())
    if peak < 1:
        return near, 0.0
    lo = (89 * (peak + 1)) // 100 - 1
    vals = np.abs(hops.seg[np.flatnonzero(m & (hops.absmax >= lo))].astype(np.int32)).ravel()
    h, _ = np.histogram(vals[vals > 0], bins=100, range=(0, peak + 1))
    return near, float(h[-1] / max(1.0, float(np.median(h[-11:-1]))))


def _hum_db(x) -> float:
    """10 log10(power 45-55 Hz / power 300-3400 Hz), Welch-averaged over consecutive 1 s Hann
    blocks (1 Hz bins; the full-length FFT of the study took seconds on a long call)."""
    np = _np()
    L = SR
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    w = np.hanning(L).astype(np.float32)
    f = np.fft.rfftfreq(L, 1 / SR)
    hum_bins = (f > HUM_BAND_HZ[0]) & (f < HUM_BAND_HZ[1])
    voice_bins = (f > VOICE_BAND_HZ[0]) & (f < VOICE_BAND_HZ[1])
    nb = len(x) // L
    h = s = 0.0
    for a in range(0, nb, 64):
        X = np.fft.rfft(x[a * L:min(nb, a + 64) * L].reshape(-1, L) * w, axis=1)
        Xh, Xv = X[:, hum_bins], X[:, voice_bins]
        h += float((Xh.real ** 2 + Xh.imag ** 2).sum())
        s += float((Xv.real ** 2 + Xv.imag ** 2).sum())
    return float(10 * np.log10((h / s if s > 0 else 0.0) + 1e-12))


def _spectral(x, probs):
    """Speech spectral flatness (100-3800 Hz) per analysed window and the -40 dB bandwidth edge.

    Windows: every 3rd 512-sample Silero window with probability >= 0.5, Hann, 512-point FFT.
    Returns (window starts, flatness per window, edge Hz or None).
    """
    np = _np()
    w = np.hanning(FFT_N).astype(np.float32)
    f = np.fft.rfftfreq(FFT_N, 1 / SR)
    band = (f >= FLAT_BAND_HZ[0]) & (f <= FLAT_BAND_HZ[1])
    st = np.flatnonzero(probs >= SPEECH_PROB)[::FLAT_STRIDE] * SILERO_WIN
    st = st[st + FFT_N <= len(x)]
    flat = []
    L = np.zeros(len(f))
    ar = np.arange(FFT_N)
    for a in range(0, len(st), 4096):
        P = np.abs(np.fft.rfft(x[st[a:a + 4096, None] + ar[None, :]] * w, axis=1)) ** 2 + 1e-12
        Pb = P[:, band]
        flat.append(np.exp(np.mean(np.log(Pb), 1)) / np.mean(Pb, 1))
        L += P.sum(0)
    flat_arr = np.concatenate(flat) if flat else np.zeros(0)
    edge = None
    if len(st):
        Ls = 10 * np.log10(np.convolve(L, np.ones(3) / 3, mode="same")[1:] + 1e-20)
        edge = float(f[1:][np.flatnonzero(Ls >= Ls.max() - 40)[-1]])
    return st, flat_arr, edge


# ----------------------------------------------------------------------------- analysis
@dataclass
class _Frames:
    """Per-10 ms-frame arrays of one signal."""

    pw: Any  # mean power of each 25 ms frame
    fdb: Any  # its dB
    nz: Any  # not digital silence
    fprob: Any  # Silero probability at the frame centre
    e20: Any  # 20 ms frame powers (active level)


def _frames(x, probs) -> _Frames:
    np = _np()
    b = _block_energy(x)
    pw = _frame_power(b)
    n = len(pw)
    centers = np.arange(n) * FRAME_HOP + FRAME_CENTER
    fprob = probs[np.minimum(centers // SILERO_WIN, len(probs) - 1)] if len(probs) else np.zeros(n)
    return _Frames(pw=pw, fdb=10 * np.log10(np.maximum(pw, 1e-12)), nz=pw > 1e-10, fprob=fprob,
                   e20=_energy20(b))


def _call_metrics(x16, x, probs, fr: _Frames) -> tuple[dict, Any, Any]:
    """Call-level acoustic metrics of the original. Returns (metrics, flat window starts, flatness)."""
    np = _np()
    near, ratio, peak = _clip_metrics(_abs_counts(x16))
    sp = (fr.fprob >= SPEECH_PROB) & fr.nz
    ns = (fr.fprob < NONSPEECH_PROB) & fr.nz
    speech = _db10(float(fr.pw[sp].mean())) if sp.any() else None
    floor = float(np.median(fr.fdb[ns])) if ns.any() else None
    st, flat, edge = _spectral(x, probs)
    m = {
        "duration_s": len(x16) / SR,
        "speech_level_db": speech,
        "snr_db": (speech - floor) if (speech is not None and floor is not None) else None,
        "noise_floor_dbfs": floor,
        "active_level_dbfs": _rms_db(_active_rms(fr.e20)),
        "peak_dbfs": 20 * math.log10(max(peak, 1) / 32768),
        "near_fs_frac": near,
        "clip_hist_ratio": ratio,
        "flat_speech_median": float(np.median(flat)) if len(flat) else None,
        "bw_edge40_hz": edge,
        "hum_db": _hum_db(x),
    }
    return m, st, flat


def _speaker_labels(diar) -> list[str]:
    labels = {t["speaker"] for t in (getattr(diar, "turns", None) or [])}
    labels |= {t["speaker"] for t in (getattr(diar, "exclusive_turns", None) or [])}
    return sorted(labels)


def _speaker_frame_masks(diar, n: int) -> tuple[dict, Any]:
    """Each speaker's non-overlapping diarized frames (regular turns minus overlap)."""
    np = _np()
    turns = getattr(diar, "turns", None) or []
    masks = {}
    cnt = np.zeros(n, dtype=np.int16)
    for spk in _speaker_labels(diar):
        m = _frame_mask([(t["start"], t["end"]) for t in turns if t["speaker"] == spk], n)
        masks[spk] = m
        cnt += m
    overlap = cnt > 1
    return {spk: m & ~overlap for spk, m in masks.items()}, overlap


def _speaker_acoustics(x16, fr: _Frames, flat_st, flat, masks: dict, settings) -> dict:
    """Per-speaker metrics of the original audio (no ASR needed)."""
    np = _np()
    n = len(fr.pw)
    flat_frame = np.clip(np.rint((flat_st + SILERO_WIN // 2 - FRAME_CENTER) / FRAME_HOP).astype(np.int64),
                         0, max(0, n - 1)) if len(flat_st) else np.zeros(0, dtype=np.int64)
    hops = _hops(x16, n) if masks else None
    out: dict[str, dict] = {}
    for spk, m in masks.items():
        speech_s = float(m.sum()) * FRAME_HOP / SR
        act = m & (fr.fprob >= SPEECH_PROB) & fr.nz
        r: dict[str, Any] = {"speech_s": speech_s,
                             "active_speech_s": float(act.sum()) * FRAME_HOP / SR,
                             "level_db": None, "gap_to_loudest_db": None,
                             "snr_in_turns_db": None, "flat_speech_median": None,
                             "near_fs_frac": None, "clip_hist_ratio": None,
                             "silero_mean_prob": float(fr.fprob[m].mean()) if m.any() else None,
                             "vad_drop_share": None, "n_words": 0, "mean_wprob": None,
                             "lowp_share": None, "gain_db": 0.0}
        if act.any():
            r["level_db"] = _db10(float(fr.pw[act].mean()))
        inside = m & fr.nz
        if inside.any() and r["level_db"] is not None:
            r["snr_in_turns_db"] = r["level_db"] - float(np.percentile(fr.fdb[inside], 10))
        if len(flat):
            sel = act[flat_frame]
            if sel.any():
                r["flat_speech_median"] = float(np.median(flat[sel]))
        if m.any():
            r["near_fs_frac"], r["clip_hist_ratio"] = _speaker_clip_metrics(hops, m)
        out[spk] = r
    majors = [s for s, r in out.items() if _acoustically_judged(r, settings)]
    if majors:
        loudest = max(out[s]["level_db"] for s in majors)
        for s in majors:
            out[s]["gap_to_loudest_db"] = loudest - out[s]["level_db"]
    return out


# ----------------------------------------------------------------------------- per-speaker gain
def _hard_masks(turns, n: int, labels) -> dict:
    """0/1 sample masks of each speaker's turns (int8)."""
    np = _np()
    out = {}
    for spk in labels:
        m = np.zeros(n, dtype=np.int8)
        for t in turns:
            if t["speaker"] != spk:
                continue
            a, b = int(t["start"] * SR), min(n, int(t["end"] * SR))
            if b > a:
                m[max(0, a):b] = 1
        out[spk] = m
    return out


def _mask_frames20(hard):
    """20 ms frames more than half covered by the mask."""
    np = _np()
    nf = len(hard) // ACTIVE_FRAME
    return hard[:nf * ACTIVE_FRAME].reshape(nf, ACTIVE_FRAME).sum(1, dtype=np.int32) > ACTIVE_FRAME // 2


def soft_mask(hard, ramp_samples: int):
    """The hard 0/1 mask convolved with a normalised Hann kernel of 2 r + 1 taps ('same' mode),
    clipped to 0..1: raised-cosine ramps of about r samples at each turn edge.

    Equal to np.convolve(hard, k, mode='same') but built from the mask's edges, so it costs
    O(edges * r + n) instead of O(n * r). Computed in place in float32 (about 9 bytes per sample
    at peak; a float64 build took about 29, i.e. 0.8 GB on a 30-minute call). Away from the
    ramps the value is exactly 0 or 1, so the gain there is exactly 1 or g.
    """
    np = _np()
    n = len(hard)
    r = int(ramp_samples)
    if r <= 1:
        return hard.astype(np.float32)
    k = np.hanning(2 * r + 1).astype(np.float64)
    k /= k.sum()
    tail = np.cumsum(k[::-1])[::-1]  # tail[a] = sum(k[a:])
    ramp = tail[1:][::-1].astype(np.float32)  # value at n = p - r .. p + r - 1 for an edge at p
    d = np.diff(hard.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
    edges = np.flatnonzero(d)
    signs = d[edges].astype(np.int32)
    del d
    step = np.zeros(n, dtype=np.int32)
    out = np.zeros(n, dtype=np.float32)
    for p, s in zip(edges.tolist(), signs.tolist()):
        q = p + r
        if q < n:
            step[q] += s
        lo, hi = p - r, p + r
        a, b = max(lo, 0), min(hi, n)
        if b > a:
            out[a:b] += np.float32(s) * ramp[a - lo:b - lo]
    np.cumsum(step, out=step)  # 0/1 (or a small count) once every ramp has passed
    np.add(out, step, out=out)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _solo_masks(diar, n: int) -> dict:
    """Each speaker's exclusive turns minus the regular turns' overlap: where only they speak.

    The other party's overlapping speech and back-channels sit at full level inside a quiet
    party's exclusive turns; measured there, a party 16-20 dB quieter read only ~9 dB quieter on
    real calls (and the other voice set the peak cap), so the gate and the lift use solo speech.
    """
    np = _np()
    exclusive = getattr(diar, "exclusive_turns", None) or getattr(diar, "turns", None) or []
    regular = getattr(diar, "turns", None) or exclusive
    labels = sorted({t["speaker"] for t in exclusive})
    reg = _hard_masks(regular, n, sorted({t["speaker"] for t in regular}))
    count = np.zeros(n, dtype=np.int16)
    for m in reg.values():
        count += m
    del reg
    solo = (count <= 1).view(np.int8)
    del count
    hard = _hard_masks(exclusive, n, labels)
    for s in labels:
        hard[s] &= solo
    return hard


def _gain_plan(x, e20, diar, settings) -> dict:
    """Decide the per-speaker gain (perspk_gain of the fixes study) on the current signal.

    Gate: among speakers with >= audio_fix_speaker_gate_min_speech_s of solo speech (exclusive
    turns minus overlap), the active level of the loudest minus the quietest >=
    audio_fix_speaker_gain_min_gap_db. Speakers with >= audio_fix_speaker_gain_min_speech_s may
    then be lifted to the loudest of the gate speakers (a short loud third voice does not raise
    the target); never attenuated; capped at max_db and so that the speaker's 99.9th-percentile
    peak stays at audio_fix_speaker_gain_peak. The lift covers solo speech only, so the other
    party's overlapping voice is never raised.
    """
    np = _np()
    n = len(x)
    hard = _solo_masks(diar, n)
    labels = sorted(hard)
    dur = {s: float(hard[s].sum()) / SR for s in labels}
    sel20 = {s: _mask_frames20(hard[s]) for s in labels}
    lv = {s: _active_rms(e20, sel20[s]) for s in labels if dur[s] >= settings.audio_fix_speaker_gain_min_speech_s}
    gate = {s: _rms_db(lv[s]) for s in lv if dur[s] >= settings.audio_fix_speaker_gate_min_speech_s
            and math.isfinite(lv[s])}
    plan = {"apply": False, "gap_db": None, "gains": {}, "hard": hard, "sel20": sel20, "gate": list(gate)}
    if len(gate) < 2:
        return plan
    plan["gap_db"] = max(gate.values()) - min(gate.values())
    if plan["gap_db"] < settings.audio_fix_speaker_gain_min_gap_db:
        return plan
    lv = {s: r for s, r in lv.items() if math.isfinite(r)}
    if len(lv) < 2:
        return plan
    target = max(lv[s] for s in gate if s in lv)
    gmax = 10 ** (settings.audio_fix_speaker_gain_max_db / 20)
    peak = float(settings.audio_fix_speaker_gain_peak)
    for s, r in lv.items():
        p999 = float(np.percentile(np.abs(x[hard[s].view(bool)]), 99.9))
        g = min(target / max(r, 1e-6), gmax, max(1.0, peak / max(p999, 1e-6)))
        plan["gains"][s] = max(g, 1.0)
    plan["apply"] = any(g > 1.0 for g in plan["gains"].values())
    return plan


def apply_speaker_gain(x16, plan: dict, ramp_ms: float):
    """int16 output of y = clip(x * G, -1, 1), G(t) = 1 + sum_s (g_s - 1) * soft_s(t).

    Where no speaker is boosted G is exactly 1, so those samples come out bit-identical.
    Computed in place in int16 units (x * G with x in int16 units is exactly 32768 times the
    product in float units, so this equals clipping x * G / 32768 to -1..1).
    """
    np = _np()
    G = np.ones(len(x16), dtype=np.float32)
    r = int(round(ramp_ms / 1000 * SR))
    for s, g in plan["gains"].items():
        if g > 1.0:
            m = soft_mask(plan["hard"][s], r)
            m *= np.float32(g - 1.0)
            G += m
            del m
    np.multiply(G, x16, out=G)
    np.clip(G, -32768.0, 32767.0, out=G)
    np.rint(G, out=G)
    return G.astype(np.int16)


# ----------------------------------------------------------------------------- entry point 1
def prepare_asr_input(wav_path: Path, diar, settings, work_dir: Path) -> AsrInput:
    """Pre-ASR check, gated fixes on the ASR copy, and the job's VAD threshold.

    Raises on any internal failure (the caller then transcribes the original); never leaves a
    partial file behind: the copy is written under a temporary name and renamed last.
    """
    np = _np()
    t0 = time.perf_counter()
    wav_path, work_dir = Path(wav_path), Path(work_dir)
    out_path = work_dir / f"{wav_path.stem}_asr.wav"
    # a copy or temp file left by an interrupted earlier run of this job
    for stale in [out_path, *work_dir.glob(f"{wav_path.stem}_asr.*.tmp.wav")]:
        if stale.exists():
            stale.unlink()
    x16 = _read_wav_int16(wav_path)
    if len(x16) < SR:
        raise AudioQualityError("audio shorter than 1 s")
    x = x16.astype(np.float32) / np.float32(32768)
    probs = _silero_probs(x)
    fr = _frames(x, probs)
    call, flat_st, flat = _call_metrics(x16, x, probs, fr)
    masks, _overlap = _speaker_frame_masks(diar, len(fr.pw))
    speakers = _speaker_acoustics(x16, fr, flat_st, flat, masks, settings)
    t_pre = time.perf_counter() - t0

    t1 = time.perf_counter()
    fixes: list[dict] = []
    temps: list[Path] = []
    cur16, cur_path, changed = x16, wav_path, False
    try:
        # 1) Mains hum: notch chain (hurts clean audio, hence the gate).
        if settings.audio_fix_hum_enabled and call["hum_db"] > settings.audio_fix_hum_threshold_db:
            tmp = work_dir / f"{wav_path.stem}_asr.hum.tmp.wav"
            temps.append(tmp)
            _ffmpeg(cur_path, tmp, NOTCH_FILTER)
            cur16, cur_path, changed = _read_same_length(tmp, len(x16)), tmp, True
            after = _hum_db(cur16.astype(np.float32) / np.float32(32768))
            fixes.append({"code": "hum_notch",
                          "detail": {"hum_db": _num(call["hum_db"], 2), "hum_db_after": _num(after, 2),
                                     "threshold_db": float(settings.audio_fix_hum_threshold_db)},
                          "message_he": MESSAGES_HE[("hum_notch", "fix")]})
        # 2) Whole call very quiet: loudnorm, re-measured on the signal as it now stands.
        cur = cur16.astype(np.float32) / np.float32(32768) if changed else x
        e20 = _energy20(_block_energy(cur)) if changed else fr.e20
        level = _rms_db(_active_rms(e20))
        # (only when there is speech at all: silence or bare noise must not be lifted 40 dB)
        if settings.audio_fix_low_level_enabled and math.isfinite(level) \
                and level < settings.audio_fix_low_level_dbfs and call.get("speech_level_db") is not None:
            tmp = work_dir / f"{wav_path.stem}_asr.level.tmp.wav"
            temps.append(tmp)
            _ffmpeg(cur_path, tmp, LOUDNORM_FILTER)
            cur16, cur_path, changed = _read_same_length(tmp, len(x16)), tmp, True
            cur = cur16.astype(np.float32) / np.float32(32768)
            e20 = _energy20(_block_energy(cur))
            fixes.append({"code": "low_level",
                          "detail": {"active_level_dbfs": _num(level, 2),
                                     "active_level_dbfs_after": _num(_rms_db(_active_rms(e20)), 2),
                                     "threshold_dbfs": float(settings.audio_fix_low_level_dbfs)},
                          "message_he": MESSAGES_HE[("low_level", "fix")]})
        # 3) One party much quieter: per-speaker gain on the original audio's turns.
        gain_applied = False
        if settings.audio_fix_speaker_gain_enabled:
            gain_turns = getattr(diar, "exclusive_turns", None) or getattr(diar, "turns", None) or []
            plan = _gain_plan(cur, e20, diar, settings) if gain_turns else None
            if plan and plan["apply"]:
                cur16 = apply_speaker_gain(cur16, plan, settings.audio_fix_speaker_gain_ramp_ms)
                changed = gain_applied = True
                cur = cur16.astype(np.float32) / np.float32(32768)
                e20 = _energy20(_block_energy(cur))
                after = [_rms_db(_active_rms(e20, plan["sel20"][s])) for s in plan["gate"]]
                gains_db = {s: 20 * math.log10(g) for s, g in plan["gains"].items()}
                for s, gdb in gains_db.items():
                    if s in speakers:
                        speakers[s]["gain_db"] = gdb
                fixes.append({"code": "speaker_gain",
                              "detail": {"gap_db": _num(plan["gap_db"], 2),
                                         "gap_db_after": _num(max(after) - min(after), 2),
                                         "threshold_db": float(settings.audio_fix_speaker_gain_min_gap_db),
                                         "max_gain_db": _num(max(gains_db.values()), 2),
                                         "speakers_lifted": sum(1 for g in gains_db.values() if g > 0.0)},
                              "message_he": MESSAGES_HE[("speaker_gain", "fix")]})
        # 4) VAD sensitivity, decided on the audio the ASR will hear.
        fprobs = _silero_probs(cur) if changed else probs
        opts = _vad_options(settings)
        n = len(x16)
        kept50 = _kept_share(speech_chunks(fprobs, n, DETECT_THRESHOLDS[0], **opts), n)
        kept20 = _kept_share(speech_chunks(fprobs, n, DETECT_THRESHOLDS[1], **opts), n)
        gain20 = kept20 - kept50
        vad_threshold: Optional[float] = None
        used: Optional[float] = float(settings.asr_vad_threshold) if settings.vad_filter else None
        # never together with the speaker gain: the fixes study found VAD 0.35 cancelled its benefit
        if (settings.vad_filter and settings.audio_fix_vad_enabled and not gain_applied
                and gain20 >= settings.audio_fix_vad_gain_trigger
                and settings.audio_fix_vad_threshold < settings.asr_vad_threshold):
            vad_threshold = used = float(settings.audio_fix_vad_threshold)
            fixes.append({"code": "vad_sensitivity",
                          "detail": {"vad_gain_th20": _num(gain20, 4),
                                     "trigger": float(settings.audio_fix_vad_gain_trigger),
                                     "threshold": used,
                                     "default_threshold": float(settings.asr_vad_threshold)},
                          "message_he": MESSAGES_HE[("vad_sensitivity", "fix")]})
        chunks = speech_chunks(fprobs, n, used, **opts) if used is not None else [(0, n)]
        vadm = _frame_mask([(s / SR, e / SR) for s, e in chunks], len(fr.pw))
        for spk, m in masks.items():
            if m.any():
                speakers[spk]["vad_drop_share"] = float(np.mean(~vadm[m]))
        call.update({"vad_kept_th50": kept50, "vad_kept_th20": kept20, "vad_gain_th20": gain20,
                     "asr_vad_threshold_used": used})
        if changed:
            part = work_dir / f"{wav_path.stem}_asr.part.tmp.wav"
            temps.append(part)
            _write_wav_int16(part, cur16)
            os.replace(part, out_path)
    finally:
        for p in temps:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    t_fix = time.perf_counter() - t1

    report = {
        "version": VERSION,
        "timing_s": {"precheck": _num(t_pre, 3), "fixes": _num(t_fix, 3), "post": None},
        "call": _round_call(call),
        "speakers": {s: _round_speaker(r) for s, r in speakers.items()},
        "fixes": fixes,
        "flags": [],
    }
    log.info("Audio check %.1f s + %.1f s: fixes %s, VAD threshold %s", t_pre, t_fix,
             ",".join(f["code"] for f in fixes) or "none",
             "setting" if vad_threshold is None else vad_threshold)
    return AsrInput(path=out_path if changed else wav_path, vad_threshold=vad_threshold,
                    report=report, processed=changed)


def _round_call(c: dict) -> dict:
    nd = {"duration_s": 2, "speech_level_db": 2, "snr_db": 2, "noise_floor_dbfs": 2,
          "active_level_dbfs": 2, "peak_dbfs": 2, "near_fs_frac": 6, "clip_hist_ratio": 3,
          "flat_speech_median": 5, "bw_edge40_hz": 1, "hum_db": 2, "vad_kept_th50": 4,
          "vad_kept_th20": 4, "vad_gain_th20": 4, "asr_vad_threshold_used": 3}
    return {k: _num(c.get(k), d) for k, d in nd.items()}


def _judged(r: dict, settings) -> bool:
    """A speaker with enough non-overlapping diarized speech to be judged at all."""
    return (r.get("speech_s") or 0.0) >= float(settings.audio_check_min_speaker_s)


def _acoustically_judged(r: dict, settings) -> bool:
    """... and with enough Silero speech frames for level, noise and clipping metrics."""
    return (_judged(r, settings) and r.get("level_db") is not None
            and (r.get("active_speech_s") or 0.0) >= MIN_ACTIVE_FRAMES * FRAME_HOP / SR)


_SPEAKER_ND = {"speech_s": 2, "active_speech_s": 2, "level_db": 2, "gap_to_loudest_db": 2, "snr_in_turns_db": 2,
               "flat_speech_median": 5, "near_fs_frac": 6, "clip_hist_ratio": 3,
               "silero_mean_prob": 4, "vad_drop_share": 4, "mean_wprob": 4, "lowp_share": 4,
               "gain_db": 2}


def _round_speaker(r: dict) -> dict:
    out: dict[str, Any] = {k: _num(r.get(k), d) for k, d in _SPEAKER_ND.items()}
    out["n_words"] = int(r.get("n_words") or 0)
    if out["gain_db"] is None:
        out["gain_db"] = 0.0
    return out


# ----------------------------------------------------------------------------- entry point 2
def _interval_index(turns):
    """Sorted turn arrays for fast overlap queries."""
    ts = sorted(turns, key=lambda t: t["start"])
    starts = [float(t["start"]) for t in ts]
    run_max, cur = [], -math.inf
    for t in ts:
        cur = max(cur, float(t["end"]))
        run_max.append(cur)
    return ts, starts, run_max


def assign_words(words, exclusive_turns, turns) -> list[Optional[str]]:
    """Speaker of each (start, end) word: largest overlap with the exclusive turns, else with the
    regular turns, else the nearest regular turn edge within 1 s (None otherwise)."""
    views = [_interval_index(v) for v in (exclusive_turns or [], turns or []) if v]
    edges = sorted([(float(t["start"]), t["speaker"]) for t in (turns or [])]
                   + [(float(t["end"]), t["speaker"]) for t in (turns or [])])
    edge_t = [e[0] for e in edges]
    out: list[Optional[str]] = []
    for ws, we in words:
        if we <= ws:
            we = ws + 0.02
        best = None
        for ts, starts, run_max in views:
            hi = bisect.bisect_left(starts, we)
            lo = bisect.bisect_right(run_max, ws)
            ov: dict[str, float] = {}
            for t in ts[lo:hi]:
                if t["end"] <= ws or t["start"] >= we:
                    continue
                ov[t["speaker"]] = ov.get(t["speaker"], 0.0) + min(we, t["end"]) - max(ws, t["start"])
            if ov:
                best = max(ov, key=ov.get)
                break
        if best is None and edges:
            # the nearest edge on each side (with every edge at that same time); ties by label
            mid = (ws + we) / 2
            i = bisect.bisect_left(edge_t, mid)
            cands = []
            for j0, step in ((i - 1, -1), (i, 1)):
                j = j0
                while 0 <= j < len(edges) and edge_t[j] == edge_t[j0]:
                    cands.append((abs(mid - edge_t[j]), edges[j][1]))
                    j += step
            dd, sp = min(cands)
            best = sp if dd <= 1.0 else None
        out.append(best)
    return out


def _word_confidence(asr_segments, diar) -> dict[str, list[float]]:
    words, probs = [], []
    for seg in asr_segments or []:
        for w in seg.get("words") or []:
            p = w.get("probability")
            if p is None:
                continue
            try:
                ws, we, pv = float(w["start"]), float(w["end"]), float(p)
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(pv):
                continue
            words.append((ws, we))
            probs.append(pv)
    spk = assign_words(words, getattr(diar, "exclusive_turns", None), getattr(diar, "turns", None))
    out: dict[str, list[float]] = {}
    for s, p in zip(spk, probs):
        if s is not None:
            out.setdefault(s, []).append(p)
    return out


def _flag(code, severity, speaker, value, threshold, message) -> dict:
    return {"code": code, "severity": severity, "speaker": speaker,
            "value": _num(value, 4), "threshold": _num(threshold, 4), "message_he": message}


def build_flags(report: dict, settings) -> list[dict]:
    """Every flag for a report whose per-speaker word confidence is filled in. Pure function."""
    call = report.get("call") or {}
    speakers = report.get("speakers") or {}
    flags: list[dict] = []

    # applied fixes first (info), in the order they were applied
    for fx in report.get("fixes") or []:
        code, det = fx.get("code"), fx.get("detail") or {}
        if code == "speaker_gain":
            lifted = sorted(((r.get("gain_db") or 0.0, s) for s, r in speakers.items()), reverse=True)
            spk = lifted[0][1] if lifted and lifted[0][0] > 0 else None
            msg = MESSAGES_HE[("speaker_gain", "flag")] if spk else MESSAGES_HE[("speaker_gain", "fix")]
            flags.append(_flag(code, "info", spk, det.get("gap_db"), det.get("threshold_db"), msg))
        elif code == "hum_notch":
            flags.append(_flag(code, "info", None, det.get("hum_db"), det.get("threshold_db"), fx["message_he"]))
        elif code == "low_level":
            flags.append(_flag(code, "info", None, det.get("active_level_dbfs"), det.get("threshold_dbfs"),
                               fx["message_he"]))
        elif code == "vad_sensitivity":
            flags.append(_flag(code, "info", None, det.get("vad_gain_th20"), det.get("trigger"), fx["message_he"]))

    for spk, r in sorted(speakers.items()):
        if not _judged(r, settings):
            continue
        drop = r.get("vad_drop_share")
        if drop is not None and drop >= settings.audio_check_vad_drop_warn:
            flags.append(_flag("speech_not_transcribed", "warn", spk, drop, settings.audio_check_vad_drop_warn,
                               _fill(MESSAGES_HE[("speech_not_transcribed", "warn")], p=round(100 * drop))))
        if _acoustically_judged(r, settings):
            gap = r.get("gap_to_loudest_db")
            if gap is not None and gap >= settings.audio_check_quiet_gap_warn_db:
                severe = gap >= settings.audio_check_quiet_gap_severe_db
                sev = "severe" if severe else "warn"
                thr = settings.audio_check_quiet_gap_severe_db if severe else settings.audio_check_quiet_gap_warn_db
                flags.append(_flag("quiet_speaker", sev, spk, gap, thr,
                                   _fill(MESSAGES_HE[("quiet_speaker", sev)], d=round(gap))))
            flat = r.get("flat_speech_median")
            if flat is not None and flat >= settings.audio_check_noise_flat_warn:
                severe = flat >= settings.audio_check_noise_flat_severe
                sev = "severe" if severe else "warn"
                thr = settings.audio_check_noise_flat_severe if severe else settings.audio_check_noise_flat_warn
                flags.append(_flag("noisy_speaker", sev, spk, flat, thr, MESSAGES_HE[("noisy_speaker", sev)]))
        lowp, wprob = r.get("lowp_share"), r.get("mean_wprob")
        if lowp is not None and wprob is not None and (
                lowp >= settings.audio_check_lowconf_lowp or wprob < settings.audio_check_lowconf_wprob):
            by_share = lowp >= settings.audio_check_lowconf_lowp
            msg = (_fill(MESSAGES_HE[("low_confidence_speaker", "warn")], p=round(100 * lowp)) if by_share
                   else MESSAGES_HE[("low_confidence_speaker", "mean")])
            flags.append(_flag("low_confidence_speaker", "warn", spk, lowp if by_share else wprob,
                               settings.audio_check_lowconf_lowp if by_share else settings.audio_check_lowconf_wprob,
                               msg))

    clipped = [(s, r) for s, r in sorted(speakers.items())
               if _acoustically_judged(r, settings) and _clipped(r, settings)]
    for spk, r in clipped:
        flags.append(_flag("clipping", "info", spk, r.get("near_fs_frac"), settings.audio_check_clip_frac,
                           MESSAGES_HE[("clipping", "speaker")]))
    if not clipped and _clipped(call, settings):
        flags.append(_flag("clipping", "info", None, call.get("near_fs_frac"), settings.audio_check_clip_frac,
                           MESSAGES_HE[("clipping", "call")]))
    edge = call.get("bw_edge40_hz")
    if edge is not None and edge < settings.audio_check_narrowband_hz:
        flags.append(_flag("narrowband", "info", None, edge, settings.audio_check_narrowband_hz,
                           MESSAGES_HE[("narrowband", "info")]))
    return flags


def _clipped(r: dict, settings) -> bool:
    near, ratio = r.get("near_fs_frac"), r.get("clip_hist_ratio")
    return (near is not None and ratio is not None and near >= settings.audio_check_clip_frac
            and ratio >= settings.audio_check_clip_hist)


def finalize_report(report: dict, asr_segments: list[dict], diar, settings) -> dict:
    """The complete audio_quality dict: per-speaker word confidence and every flag.

    asr_segments: the transcriber's segments (words carry 'probability'; words without it are
    left out). The input report is not modified.
    """
    t0 = time.perf_counter()
    out = {k: v for k, v in report.items()}
    out["timing_s"] = dict(report.get("timing_s") or {})
    out["speakers"] = {s: dict(r) for s, r in (report.get("speakers") or {}).items()}
    out["fixes"] = [dict(f) for f in report.get("fixes") or []]
    conf = _word_confidence(asr_segments, diar)
    for spk, probs in conf.items():
        r = out["speakers"].setdefault(spk, _round_speaker({}))
        r["n_words"] = len(probs)
        r["mean_wprob"] = _num(sum(probs) / len(probs), 4)
        r["lowp_share"] = _num(sum(1 for p in probs if p < LOWP) / len(probs), 4)
    out["flags"] = build_flags(out, settings)
    out["timing_s"]["post"] = _num(time.perf_counter() - t0, 3)
    return out
