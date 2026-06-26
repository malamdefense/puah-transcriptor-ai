"""Speaker diarization using pyannote.audio.

The gated model in HuggingFace — requires approval of the usage terms once and create a token.
See instructions in README.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from ..config import Settings

# torchcodec failed to load on Windows (missing DLLs of FFmpeg in shared version), and we bypass it entirely — loading the WAV into memory with soundfile. We suppress the annoying warning (not a failure).
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")


class DiarizationError(RuntimeError):
    pass


class Diarizer:
    """Wraps the diarization pipeline of pyannote. Loads lazily and stays in memory."""

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
                "Missing HuggingFace token (PUAH_HF_TOKEN). Please approve the usage terms of "
                f"'{self.settings.diarization_model}' and 'pyannote/segmentation-3.0' "
                "in HuggingFace and create a token. See README."
            )

        # pyannote.audio newer uses token=; older uses use_auth_token=. Both are supported.
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
                "Model loading failed — verify that you approved the usage terms and that PUAH_HF_TOKEN is valid."
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
        """Returns a list of speech turns: {start, end, speaker}.

        num_speakers — hint for number of speakers (None = automatic diarization).
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
        """Loads the WAV into memory and returns a dict for pyannote, bypassing torchcodec.

        torchcodec (dependency of pyannote) failed to load on Windows when there are no DLLs of FFmpeg in shared version.
        We already extract WAV with ffmpeg, and soundfile reads it directly (libsndfile, without FFmpeg).
        If soundfile is not available — fallback to the file path.
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
        # pyannote older returns Annotation directly; newer returns DiarizeOutput with .speaker_diarization
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
