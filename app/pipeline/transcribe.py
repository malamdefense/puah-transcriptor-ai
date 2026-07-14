"""תמלול עברית באמצעות faster-whisper עם מודל ivrit.ai.

המודל נטען פעם אחת ונשאר תושב בזיכרון ה-GPU כל עוד ה-worker חי.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from ..config import Settings


def resolve_device_and_compute(settings: Settings) -> tuple[str, str]:
    """בוחר device ו-compute_type מתאימים (מכבד הגדרות מפורשות)."""
    device = settings.device
    compute = settings.compute_type

    if device == "auto":
        try:
            import torch  # noqa: PLC0415

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    return device, compute


class Transcriber:
    """עוטף את faster-whisper. נטען עצלן בקריאה הראשונה ל-load()."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self.device, self.compute_type = resolve_device_and_compute(settings)

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self._model = WhisperModel(
            self.settings.asr_model,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(
        self,
        audio_path: Path,
        duration_sec: float,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> list[dict]:
        """מחזיר רשימת קטעים: {start, end, text, words:[{start,end,word}]}.

        progress_cb מקבל אחוז (0-100) לפי התקדמות לאורך הזמן.
        """
        self.load()
        assert self._model is not None

        segments_iter, _info = self._model.transcribe(
            str(audio_path),
            language=self.settings.language,
            beam_size=self.settings.beam_size,
            vad_filter=self.settings.vad_filter,
            word_timestamps=True,
        )

        results: list[dict] = []
        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    text = (w.word or "").strip()
                    if not text:
                        continue
                    words.append({"start": float(w.start), "end": float(w.end), "word": text})
            results.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": (seg.text or "").strip(),
                    "words": words,
                }
            )
            if progress_cb and duration_sec > 0:
                pct = min(100.0, (seg.end / duration_sec) * 100.0)
                progress_cb(pct)

        if progress_cb:
            progress_cb(100.0)
        return results
