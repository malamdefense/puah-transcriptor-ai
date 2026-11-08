"""Speaker diarization using pyannote.audio.

The gated model in HuggingFace — you need to accept the terms of use once and create a token.
See instructions in README.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import Settings
from . import import_failure_reason

# pyannote.audio >= 4 sends OTLP telemetry (pipeline name, file duration, requested speaker
# counts) to otel.pyannote.ai unless this is "false". The system is on-prem: nothing may leave
# the machine. It must be set before pyannote is first imported (pyannote fills the variable from
# its own config only when it is unset; pyannote is imported lazily in Diarizer.load, never above).
# An explicit value in the environment still wins.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

# torchcodec fails to load on Windows (missing FFmpeg shared DLLs); we bypass it
# entirely by loading the WAV into memory with soundfile. The noisy warning is
# silenced (it is not an actual failure).
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")
# A speaker active for only a frame or two inside a 10 s window (a breath, a click, a "כן" over
# the other party) leaves the embedding's statistics pooling one frame to take a std over. torch
# warns, the embedding comes out NaN, and pyannote drops NaN embeddings before clustering
# (pipelines/clustering.py filter_embeddings). Nothing is lost: the warning is noise in the log.
warnings.filterwarnings("ignore", message=r"(?s)std\(\): degrees of freedom is <= 0", category=UserWarning)


class DiarizationError(RuntimeError):
    pass


@dataclass
class DiarizationResult:
    """Both views of one diarization run.

    turns — speaker_diarization: {start, end, speaker}; turns of different speakers may overlap.
    exclusive_turns — exclusive_speaker_diarization: one speaker at a time, made by pyannote for
    aligning with a transcript. Same SPEAKER_xx labels. None when pyannote returns a bare
    Annotation (pyannote.audio < 4).
    """

    turns: list[dict]
    exclusive_turns: Optional[list[dict]] = None


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
                import_failure_reason(exc, "pyannote.audio", "pip install pyannote.audio")
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

        self._apply_min_cluster_size(pipeline)
        self._pipeline = pipeline

    def _apply_min_cluster_size(self, pipeline) -> None:
        """Fold "fragment" clusters into the real speakers.

        pyannote reassigns any cluster with fewer training embeddings than min_cluster_size to the
        nearest large one. With 3.1's 12, a two-party call whose details were spelled out in short
        replies came out as 4 speakers (each party plus a cluster of its own 0.5-1 s answers); 30
        gave the right 2 and left the other measured calls' turns byte-identical. pyannote itself
        caps the value at 10% of a call's embeddings, so short calls are not over-merged. Clustering
        without this hyper-parameter (VBx in newer pipelines) is left alone.
        """
        mcs = self.settings.diarization_min_cluster_size
        clustering = getattr(pipeline, "clustering", None)
        if mcs and clustering is not None and hasattr(clustering, "min_cluster_size"):
            clustering.min_cluster_size = int(mcs)

    def diarize(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None,
    ) -> list[dict]:
        """Return a list of turns: {start, end, speaker}.

        num_speakers — hint for the number of speakers (None = automatic detection, within
        diarization_min_speakers / diarization_max_speakers when those are set).
        """
        return self.diarize_full(audio_path, num_speakers).turns

    def diarize_full(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """Run the pipeline once and return both the regular and the exclusive turns.

        num_speakers — hint for the number of speakers (None = automatic detection, bounded by
        the diarization_min_speakers / diarization_max_speakers settings when they are set).
        """
        self.load()
        assert self._pipeline is not None

        output = self._pipeline(self._load_audio(audio_path), **self._speaker_kwargs(num_speakers))
        return self._collect_full(output)

    def _speaker_kwargs(self, num_speakers: Optional[int]) -> dict:
        """Speaker-count arguments for the pipeline call.

        A positive hint is passed alone (pyannote lets num_speakers override min/max anyway).
        Otherwise each configured bound is passed; an unset bound is left out, so pyannote
        keeps its own choice.
        """
        if num_speakers and num_speakers > 0:
            return {"num_speakers": int(num_speakers)}
        kwargs: dict = {}
        if self.settings.diarization_min_speakers is not None:
            kwargs["min_speakers"] = int(self.settings.diarization_min_speakers)
        if self.settings.diarization_max_speakers is not None:
            kwargs["max_speakers"] = int(self.settings.diarization_max_speakers)
        return kwargs

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
        """The regular (possibly overlapping) turns of a pipeline output."""
        return Diarizer._collect_full(output).turns

    @staticmethod
    def _collect_full(output) -> DiarizationResult:
        # pyannote old returns Annotation directly; new returns DiarizeOutput with
        # .speaker_diarization and .exclusive_speaker_diarization (same speaker labels)
        if hasattr(output, "itertracks"):
            return DiarizationResult(turns=Diarizer._annotation_to_turns(output), exclusive_turns=None)

        annotation = None
        for attr in ("speaker_diarization", "diarization", "annotation"):
            cand = getattr(output, attr, None)
            if cand is not None and hasattr(cand, "itertracks"):
                annotation = cand
                break
        if annotation is None:
            raise DiarizationError(
                f"Unknown diarization output: {type(output).__name__} (no itertracks)"
            )

        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        return DiarizationResult(
            turns=Diarizer._annotation_to_turns(annotation),
            exclusive_turns=(
                Diarizer._annotation_to_turns(exclusive) if hasattr(exclusive, "itertracks") else None
            ),
        )

    @staticmethod
    def _annotation_to_turns(annotation) -> list[dict]:
        turns: list[dict] = []
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)}
            )
        turns.sort(key=lambda t: t["start"])
        return turns
