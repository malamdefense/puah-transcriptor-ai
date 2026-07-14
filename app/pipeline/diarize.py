"""זיהוי דוברים (speaker diarization) באמצעות pyannote.audio.

המודל gated ב-HuggingFace — נדרש לאשר את תנאי השימוש פעם אחת וליצור טוקן.
ראו הוראות ב-README.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from ..config import Settings

# torchcodec נכשל בטעינה ב-Windows (חסרים DLLs של FFmpeg בגרסת shared), ואנחנו עוקפים
# אותו לגמרי — טוענים את ה-WAV לזיכרון עם soundfile. משתיקים את האזהרה הרועשת (לא כשל).
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")


class DiarizationError(RuntimeError):
    pass


class Diarizer:
    """עוטף את pipeline הזיהוי של pyannote. נטען עצלן ונשאר תושב בזיכרון."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise DiarizationError(
                "pyannote.audio אינו מותקן. הריצו: pip install pyannote.audio"
            ) from exc

        if not self.settings.hf_token:
            raise DiarizationError(
                "חסר טוקן HuggingFace (PUAH_HF_TOKEN). יש לאשר את תנאי השימוש של "
                f"'{self.settings.diarization_model}' ושל 'pyannote/segmentation-3.0' "
                "ב-HuggingFace וליצור טוקן קריאה. ראו README."
            )

        # pyannote.audio החדש משתמש ב-token=; ישן יותר ב-use_auth_token=. תומכים בשניהם.
        try:
            pipeline = Pipeline.from_pretrained(
                self.settings.diarization_model, token=self.settings.hf_token
            )
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                self.settings.diarization_model, use_auth_token=self.settings.hf_token
            )
        if pipeline is None:
            raise DiarizationError(
                "טעינת מודל הזיהוי נכשלה — ודאו שאישרתם את תנאי השימוש וש-PUAH_HF_TOKEN תקין."
            )

        # העברה ל-GPU אם זמין
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available() and self.settings.device != "cpu":
                pipeline.to(torch.device("cuda"))
        except Exception:
            pass

        self._pipeline = pipeline

    def diarize(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None,
    ) -> list[dict]:
        """מחזיר רשימת תורות דיבור: {start, end, speaker}.

        num_speakers — רמז מספר הדוברים (None = זיהוי אוטומטי).
        """
        self.load()
        assert self._pipeline is not None

        kwargs: dict = {}
        if num_speakers and num_speakers > 0:
            kwargs["num_speakers"] = int(num_speakers)

        annotation = self._pipeline(self._load_audio(audio_path), **kwargs)
        return self._collect(annotation)

    @staticmethod
    def _load_audio(audio_path: Path):
        """טוען את ה-WAV לזיכרון ומחזיר dict ל-pyannote, לעקיפת torchcodec.

        torchcodec (תלות של pyannote) נכשל בטעינה ב-Windows כשאין DLLs של FFmpeg
        בגרסת shared. אנחנו כבר מחלצים WAV עם ffmpeg, ו-soundfile קורא אותו ישירות
        (libsndfile, ללא FFmpeg). אם soundfile לא זמין — נופלים חזרה לנתיב הקובץ.
        """
        try:
            import soundfile as sf  # noqa: PLC0415
            import torch  # noqa: PLC0415

            data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T).contiguous()  # (channel, time)
            return {"waveform": waveform, "sample_rate": int(sample_rate)}
        except Exception:
            return str(audio_path)

    @staticmethod
    def _collect(output) -> list[dict]:
        # pyannote ישן מחזיר Annotation ישירות; חדש מחזיר DiarizeOutput עם .speaker_diarization
        annotation = output
        if not hasattr(annotation, "itertracks"):
            for attr in ("speaker_diarization", "diarization", "annotation"):
                cand = getattr(output, attr, None)
                if cand is not None and hasattr(cand, "itertracks"):
                    annotation = cand
                    break
        if not hasattr(annotation, "itertracks"):
            raise DiarizationError(
                f"פלט זיהוי דוברים בלתי מוכר: {type(output).__name__} (אין itertracks)"
            )

        turns: list[dict] = []
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)}
            )
        turns.sort(key=lambda t: t["start"])
        return turns
