"""Transcription of Hebrew speech using faster-whisper with the ivrit.ai model.

The model is loaded once and remains resident in GPU memory as long as the worker is alive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ..config import Settings
from . import asr_terms

log = logging.getLogger("puah.transcribe")


def resolve_device_and_compute(settings: Settings) -> tuple[str, str]:
    """Select the appropriate device and compute_type (respecting explicit settings)."""
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
    """Wraps faster-whisper. Lazily loaded by the first call to load()."""

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

    def count_tokens(self, text: str) -> int:
        """Tokens `text` costs as hotwords — encoded exactly as faster-whisper's get_prompt does
        (" " + text.strip(), no special tokens), so the hotword cap is exact."""
        self.load()
        assert self._model is not None
        return len(self._model.hf_tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids)

    def transcribe(
        self,
        audio_path: Path,
        duration_sec: float,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> list[dict]:
        """Return a list of segments: {start, end, text, words:[{start,end,word}]}.

        progress_cb receives a percentage (0-100) based on progress along the time axis.
        """
        self.load()
        assert self._model is not None
        s = self.settings

        hotwords, dropped = asr_terms.plan_hotwords(s, s.language, self.count_tokens)
        if hotwords is not None:
            log.info(
                "ASR hotwords: %d terms, %d tokens (cap %d), dropped to fit: %s",
                len(hotwords.split(asr_terms.SEPARATOR)), self.count_tokens(hotwords),
                s.asr_hotwords_max_tokens, ", ".join(dropped) or "none",
            )
        elif s.asr_hotwords_enabled:
            log.info("ASR hotwords: none passed (language %s, %d terms over the cap)",
                     s.language, len(dropped))

        cond = s.asr_condition_on_previous_text
        if cond is None:
            # Prompt and output share Whisper's 448-token window, and carried-over text already
            # fills it on dense speech: with hotwords, carry-over would cut the transcript.
            cond = hotwords is None
        elif cond and hotwords is not None:
            log.warning("ASR hotwords with condition_on_previous_text=True: dense windows hit the "
                        "448-token limit and lose speech")
        if s.asr_no_repeat_ngram_size > 0:
            log.warning("asr_no_repeat_ngram_size=%d blocks repeated Hebrew sub-word tokens and "
                        "corrupts words; keep it 0", s.asr_no_repeat_ngram_size)

        vad_parameters = None
        if s.vad_filter:
            # Keys must be faster_whisper.vad.VadOptions field names (it is built as VadOptions(**d)).
            vad_parameters = {
                "threshold": s.asr_vad_threshold,
                "min_speech_duration_ms": s.asr_vad_min_speech_duration_ms,
                "min_silence_duration_ms": s.asr_vad_min_silence_duration_ms,
                "speech_pad_ms": s.asr_vad_speech_pad_ms,
            }

        segments_iter, _info = self._model.transcribe(
            str(audio_path),
            language=s.language,
            beam_size=s.beam_size,
            vad_filter=s.vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=True,
            temperature=s.asr_temperature_list,
            compression_ratio_threshold=s.asr_compression_ratio_threshold,
            log_prob_threshold=s.asr_log_prob_threshold,
            no_speech_threshold=s.asr_no_speech_threshold,
            condition_on_previous_text=cond,
            prompt_reset_on_temperature=s.asr_prompt_reset_on_temperature,
            repetition_penalty=s.asr_repetition_penalty,
            no_repeat_ngram_size=s.asr_no_repeat_ngram_size,
            hallucination_silence_threshold=s.asr_hallucination_silence_threshold,
            hotwords=hotwords,
        )

        results: list[dict] = []
        fallback_segments = 0
        for seg in segments_iter:
            # A window re-decoded by temperature sampling (the compression/logprob checks failed)
            if (getattr(seg, "temperature", None) or 0.0) > 0.0:
                fallback_segments += 1
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
        log.info("ASR: %d segments, %d decoded with temperature fallback",
                 len(results), fallback_segments)
        return results
