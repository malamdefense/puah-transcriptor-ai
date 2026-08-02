"""Speaker diarization using pyannote.audio.

The gated model in HuggingFace — you need to accept the terms of use once and create a token.
See instructions in README.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from ..config import Settings

# torchcodec fails to load on Windows (missing FFmpeg shared DLLs); we bypass it
# entirely by loading the WAV into memory with soundfile. The noisy warning is
# silenced (it is not an actual failure).
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")


class DiarizationError(RuntimeError):
    pass


class Diarizer:
    """Wrap the diarization pipeline of pyannote. Load lazily and keep in memory."""

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
                "pyannote.audio is not installed. Run: pip install pyannote.audio"
            ) from exc

        if not self.settings.hf_token:
            raise DiarizationError(
                "Missing HuggingFace token (PUAH_HF_TOKEN). You need to accept the terms of use of "
                f"'{self.settings.diarization_model}' ושל 'pyannote/segmentation-3.0' "
                "in HuggingFace and create a token. See README."
            )

        # pyannote.audio new uses token=; old uses use_auth_token=. Both are supported.
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
                "Model loading failed — make sure you accepted the terms of use and PUAH_HF_TOKEN is valid."
            )

        # Transfer to GPU if available
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
        """Return a list of turns: {start, end, speaker}.

        num_speakers — hint for the number of speakers (None = automatic detection).
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
        """Load the WAV into memory and return a dict for pyannote, bypassing torchcodec.

        torchcodec (pyannote dependency) fails to load on Windows when there are no DLLs of FFmpeg
        in the shared version. We already extract WAV with ffmpeg, and soundfile reads it directly
        (libsndfile, without FFmpeg). If soundfile is not available — we fall back to the file path.
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
        # pyannote old returns Annotation directly; new returns DiarizeOutput with .speaker_diarization
        annotation = output
        if not hasattr(annotation, "itertracks"):
            for attr in ("speaker_diarization", "diarization", "annotation"):
                cand = getattr(output, attr, None)
                if cand is not None and hasattr(cand, "itertracks"):
                    annotation = cand
                    break
        if not hasattr(annotation, "itertracks"):
            raise DiarizationError(
                f"Unknown diarization output: {type(output).__name__} (no itertracks)"
            )

        turns: list[dict] = []
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)}
            )
        turns.sort(key=lambda t: t["start"])
        return turns
