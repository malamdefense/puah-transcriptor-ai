"""Transcription of Hebrew speech using faster-whisper with the ivrit.ai model.

The model is loaded once and remains resident in GPU memory as long as the worker is alive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from ..config import Settings
from . import asr_config, asr_terms, asr_words

log = logging.getLogger("puah.transcribe")


def _opt_float(value) -> Optional[float]:
    return None if value is None else float(value)


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
        # Facts about the last transcribe() call (no text): see transcribe()
        self.last_decode: dict = {}
        # {"repo_id", "commit"} of the weights load() loaded, read once right after loading
        self._identity: Optional[dict] = None

    def loaded_identity(self) -> dict:
        """Device and compute type the loaded model actually uses (the requested ones before a
        load), plus the model's repo id and snapshot commit. Never a filesystem path.

        The commit is the one the cache resolved when the weights were loaded, not re-read per job:
        if another process later moves the cache's refs/main, jobs keep reporting the loaded one."""
        ct2 = getattr(self._model, "model", None) if self._model is not None else None
        device = compute = None
        if ct2 is not None:
            try:
                device = getattr(ct2, "device", None)
                compute = getattr(ct2, "compute_type", None)
            except Exception:  # noqa: BLE001 - provenance only, never fails a job
                device = compute = None
        ident = self._identity if self._identity is not None else asr_config.model_identity(self.settings.asr_model)
        return {
            "device": str(device) if device else self.device,
            "compute_type": str(compute) if compute else self.compute_type,
            **ident,
        }

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self._model = WhisperModel(
            self.settings.asr_model,
            device=self.device,
            compute_type=self.compute_type,
        )
        # The snapshot faster-whisper just resolved the repo id to (refs/main, read now). Provenance
        # only: a failure here never fails the load.
        try:
            self._identity = asr_config.model_identity(self.settings.asr_model)
        except Exception:  # noqa: BLE001
            self._identity = {"repo_id": None, "commit": None}

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
        *,
        vad_threshold: Optional[float] = None,
    ) -> list[dict]:
        """Return the decoder segments, one dict per faster-whisper segment:

        {start, end, text (the decoder text, stripped), words, seek, temperature, avg_logprob,
         compression_ratio, no_speech_prob, window_flag}
        words: [{start, end, word (stripped), probability (when faster-whisper gives one), lead,
                 window_flag (only when not None)}] — the decoder's word PIECES: nothing is glued
        here (the worker's post-processing glues pieces with lead=False, asr_words), so the raw
        sidecar keeps them. `seek` is the decode window's start frame in the VAD-collapsed
        timeline (shared by every segment of one window); `temperature` is the LAST temperature
        tried (faster-whisper reports 1.0 for a window where every temperature failed, whatever
        result it kept); `window_flag` is asr_words.window_flag of the kept result under the
        configured thresholds ("gate_failed" / "sampled_hi" / None), None when asr_window_flags
        is off. Values a segment does not carry are None.

        progress_cb receives a percentage (0-100) based on progress along the time axis.
        vad_threshold overrides settings.asr_vad_threshold for this call only (the recording-quality
        check lowers it for a job whose speech the VAD would cut); None keeps the setting.
        After the call, self.last_decode = {condition_on_previous_text (effective), hotwords (bool),
        vad_threshold (effective, None without VAD), windows, fallback_segments}.
        """
        self.load()
        assert self._model is not None
        s = self.settings
        self.last_decode = {}

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
                "threshold": s.asr_vad_threshold if vad_threshold is None else float(vad_threshold),
                "min_speech_duration_ms": s.asr_vad_min_speech_duration_ms,
                "min_silence_duration_ms": s.asr_vad_min_silence_duration_ms,
                "speech_pad_ms": s.asr_vad_speech_pad_ms,
            }
            if vad_threshold is not None:
                log.info("ASR VAD threshold %.2f for this job (setting %.2f)",
                         vad_parameters["threshold"], s.asr_vad_threshold)

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

        flags_on = getattr(s, "asr_window_flags", True)
        results: list[dict] = []
        fallback_segments = 0
        seeks: set[int] = set()
        for seg in segments_iter:
            temperature = _opt_float(getattr(seg, "temperature", None))
            avg_logprob = _opt_float(getattr(seg, "avg_logprob", None))
            compression_ratio = _opt_float(getattr(seg, "compression_ratio", None))
            seek = getattr(seg, "seek", None)
            seek = None if seek is None else int(seek)
            if seek is not None:
                seeks.add(seek)
            # A window re-decoded by temperature sampling (the compression/logprob checks failed)
            if (temperature or 0.0) > 0.0:
                fallback_segments += 1
            flag = None
            if flags_on:
                flag = asr_words.window_flag(
                    avg_logprob, compression_ratio, temperature,
                    compression_ratio_threshold=s.asr_compression_ratio_threshold,
                    log_prob_threshold=s.asr_log_prob_threshold,
                )
            words = []
            if seg.words:
                raw = list(seg.words)
                leads = asr_words.lead_flags([w.word or "" for w in raw])
                for w, lead in zip(raw, leads):
                    if lead is None:  # whitespace-only: its space belongs to the next piece
                        continue
                    word = {"start": float(w.start), "end": float(w.end), "word": (w.word or "").strip()}
                    prob = getattr(w, "probability", None)
                    if prob is not None:
                        word["probability"] = float(prob)
                    word["lead"] = lead
                    if flag is not None:
                        word["window_flag"] = flag
                    words.append(word)
            results.append(
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": (seg.text or "").strip(),
                    "words": words,
                    "seek": seek,
                    "temperature": temperature,  # the last temperature tried
                    "avg_logprob": avg_logprob,
                    "compression_ratio": compression_ratio,
                    "no_speech_prob": _opt_float(getattr(seg, "no_speech_prob", None)),
                    "window_flag": flag,
                }
            )
            if progress_cb and duration_sec > 0:
                pct = min(100.0, (seg.end / duration_sec) * 100.0)
                progress_cb(pct)

        if progress_cb:
            progress_cb(100.0)
        self.last_decode = {
            "condition_on_previous_text": bool(cond),
            "hotwords": hotwords is not None,
            "vad_threshold": vad_parameters["threshold"] if vad_parameters else None,
            "windows": len(seeks),
            "fallback_segments": fallback_segments,
        }
        log.info("ASR: %d segments, %d decoded with temperature fallback (%d windows)",
                 len(results), fallback_segments, len(seeks))
        return results
