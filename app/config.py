"""Application settings — loaded from environment variables / the .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PUAH_",
        extra="ignore",
        # Required so that fields with an explicit validation_alias (gemini_api_key
        # and friends — whose .env names fall outside the PUAH_ prefix) can also be
        # constructed by their Python field name (Settings(gemini_api_key=...)),
        # the same way every other field already behaves via the env_prefix mechanism.
        populate_by_name=True,
    )

    # --- Storage paths (On-Prem, everything local) ---
    data_dir: Path = Field(default=Path("data"))

    # --- Transcription model (ivrit.ai, CTranslate2 format for faster-whisper) ---
    # turbo-ct2 = fast and accurate for Hebrew. For maximum accuracy: ivrit-ai/whisper-large-v3-ct2
    asr_model: str = Field(default="ivrit-ai/whisper-large-v3-turbo-ct2")
    language: str = Field(default="he")
    # cuda / cpu / auto
    device: str = Field(default="auto")
    # float16 for GPU, int8 for CPU. auto picks automatically.
    compute_type: str = Field(default="auto")
    beam_size: int = Field(default=5)
    # Filter out silence before transcribing — improves accuracy and shortens runtime
    vad_filter: bool = Field(default=True)

    # --- ASR decoding (faster-whisper options; measured on the institute's calls) ---
    # faster-whisper's 2.4 assumes 1-byte text: Hebrew is 2 bytes per letter in UTF-8, so normal
    # 30 s Hebrew windows score 1.9-2.55 and a correct window was needlessly re-decoded by random
    # sampling (up to temperature 1.0). 2.8 removed every false fallback and still catches loops
    # that fill most of a window. None disables the check.
    asr_compression_ratio_threshold: float | None = Field(default=2.8)
    asr_log_prob_threshold: float | None = Field(default=-1.0)
    asr_no_speech_threshold: float | None = Field(default=0.6)
    # Temperature fallback, comma-separated ("0.0,0.2,0.4"); see the asr_temperature_list property
    asr_temperatures: str = Field(default="0.0,0.2,0.4,0.6,0.8,1.0")
    # None / "auto" = automatic: False when hotwords are passed (prompt + output share Whisper's
    # 448-token window, and carried-over text already fills it), True otherwise (the library default).
    asr_condition_on_previous_text: bool | None = Field(default=None)
    asr_prompt_reset_on_temperature: float = Field(default=0.5)
    # Keep 1.0 / 0: on Hebrew sub-word tokens a penalty removed genuine "לא, לא, לא" and commas,
    # and no_repeat_ngram_size corrupted words (even emitting invalid UTF-8).
    asr_repetition_penalty: float = Field(default=1.0, ge=1.0)
    asr_no_repeat_ngram_size: int = Field(default=0, ge=0)
    # Barely fires with VAD on (silences are already shortened) and created new loops — keep None.
    asr_hallucination_silence_threshold: float | None = Field(default=None)
    # Silero VAD options (faster-whisper VadOptions; defaults = the library's)
    asr_vad_threshold: float = Field(default=0.5)
    asr_vad_min_speech_duration_ms: int = Field(default=0, ge=0)
    asr_vad_min_silence_duration_ms: int = Field(default=2000, ge=0)
    asr_vad_speech_pad_ms: int = Field(default=400, ge=0)

    # --- Domain terms (drug/device names) ---
    # faster-whisper `hotwords`: a short term list prepended to every 30 s window. It fixes brand
    # names but costs speech: on 2 calls 2.6-4% of words were lost, and it pulled ordinary words
    # toward the list. Off by default; when on, condition_on_previous_text turns off (see above).
    asr_hotwords_enabled: bool = Field(default=False)
    asr_hotwords_extra: str = Field(default="")  # comma-separated, appended after the curated list
    asr_hotwords_max_tokens: int = Field(default=64, ge=0)
    # Closed table of known ASR garbles -> canonical spelling ("נורד ג 'יסטון" -> "נורג'סטון").
    # Exact match only: never fuzzy, never a word that is also a first name or an ordinary word.
    asr_term_normalization_enabled: bool = Field(default=True)

    # --- Repetition clean-up (between transcription and merge) ---
    # Removes repeated copies the word alignment found no audio for, and caps runaway loops.
    asr_cleanup_enabled: bool = Field(default=True)
    asr_cleanup_max_ngram: int = Field(default=12, ge=1)
    asr_cleanup_collapsed_word_sec: float = Field(default=0.03, ge=0.0)
    asr_cleanup_min_collapsed_unigram_copies: int = Field(default=3, ge=1)
    asr_cleanup_max_unigram_copies: int = Field(default=10, ge=0)  # 0 = no cap
    asr_cleanup_max_phrase_copies: int = Field(default=5, ge=0)  # 0 = no cap

    # --- Language detection (after merge; decoding stays `language`) ---
    # The ivrit-ai model's language-ID head always answers "he", even on English speech, so the
    # language is read from the transcript's script instead. Annotates the result only.
    language_detection: bool = Field(default=True)
    asr_languages: str = Field(default="he,en")  # the languages that may be reported; first = default

    # --- Speaker diarization (pyannote) ---
    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1")
    # Requires a HuggingFace token after accepting the model's terms of use (one-time).
    hf_token: str | None = Field(default=None)
    # Hint for the number of speakers. None = auto-detect. Can also be overridden per job.
    default_num_speakers: int | None = Field(default=None)
    # Bounds for the automatic count when there is no hint (None = unbounded, pyannote's own choice)
    diarization_min_speakers: int | None = Field(default=None, ge=1)
    diarization_max_speakers: int | None = Field(default=None, ge=1)
    # pyannote clustering: smaller clusters are folded into the nearest speaker (3.1's own value: 12,
    # which let short spelled-out replies form phantom speakers). None keeps the model's value.
    diarization_min_cluster_size: int | None = Field(default=30, ge=1)
    # Assign words on pyannote's exclusive (one speaker at a time) diarization, made for aligning
    # with a transcript; short back-channel replies are kept from the regular (overlapping) one.
    merge_use_exclusive: bool = Field(default=True)
    # Conservative, punctuation-aware relabelling of 1-2 word islands and of sentence tails.
    # Turning both merge switches off falls back to overlap assignment on the regular turns; it is
    # close to, but not byte-identical with, the merge before October 2026 (summed per-speaker
    # overlap, tie-break and padding of zero-length words always apply).
    merge_smoothing: bool = Field(default=True)

    # --- Summaries (Google Gemini — see the "LLM provider" section below) ---
    # 2048 was too small: long summaries/feedback were cut off mid-text. Raised to 6144.
    llm_max_tokens: int = Field(default=6144)
    # The two deliverables, each with its own switch. (There is no master switch: the
    # former summary_enabled gated an AI-generated structured summary that no longer
    # exists — topics/keywords are now derived locally in pipeline/keywords.py.)
    client_summary_enabled: bool = Field(default=True)
    feedback_enabled: bool = Field(default=True)
    # Redaction placeholders (<שם_>, <ארגון_>...) must never reach the caller: a tagged answer is
    # sent back for repair, and as a last resort the tags are removed with Hebrew-aware rules.
    # False = observe only: no repair turn and no removal at generation (the check is still recorded
    # and logged). The fixed "never copy a placeholder" rule in the prompt and the clean-up of the
    # Word/Excel exports stay on either way.
    llm_tag_guard_enabled: bool = Field(default=True)

    # --- Local Ollama server — currently used only for embeddings (RAG/evaluation) ---
    # Ollama is still required for the embedding model (bge-m3).
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_timeout_sec: float = Field(default=600.0)

    # --- LLM provider: Google Gemini (cloud) ---
    # The local provider was removed, so producing summaries requires a valid API key
    # and internet connectivity, and the call content (after mandatory local PII
    # redaction) leaves to Google's servers. A key can also be entered from the
    # settings screen (⚙), which overrides the .env value at runtime.
    gemini_api_key: str | None = Field(default=None, validation_alias="PUHA_GEMINI_API_KEY")
    # gemini-1.5-flash (originally requested) was fully retired by Google (404), and
    # gemini-2.5-flash returns "no longer available to new users" — both verified
    # directly against the API (list_models + a real generateContent call).
    # gemini-flash-latest is a Google-maintained alias that always points at the
    # current flash model, so it survives similar deprecations. Pin a concrete
    # version via PUHA_GEMINI_MODEL when reproducibility matters.
    gemini_model: str = Field(default="gemini-flash-latest", validation_alias="PUHA_GEMINI_MODEL")
    # Gemini "thinking budget" (thinking_config.thinking_budget). 0 = internal
    # thinking off — needed to avoid the truncated/meta-text answers observed when
    # thinking was left uncontrolled. Use -1 for dynamic, or an explicit token count.
    gemini_thinking_budget: int = Field(default=0, validation_alias="PUHA_GEMINI_THINKING_BUDGET")

    # --- Semantic RAG (local embeddings via Ollama + ChromaDB vector storage) ---
    # bge-m3: multilingual, good Hebrew coverage. Pull with: ollama pull bge-m3
    embed_model: str = Field(default="bge-m3")
    # Chroma: None = embedded (data/chroma/, On-Prem). Set host for a separate Chroma server.
    chroma_host: str | None = Field(default=None)
    chroma_port: int = Field(default=8000)

    # --- Word export + PII removal (after saving the transcript) ---
    # Create a Word document for each transcript ({job_id}.docx)
    word_export_enabled: bool = Field(default=True)
    # Remove PII (HebSafeHarbor) and save a clean version ({job_id}.pii.json/.docx)
    pii_enabled: bool = Field(default=True)
    # The isolated venv where HebSafeHarbor is installed (relative to the repo root)
    pii_venv: str = Field(default=".venv-pii")
    pii_timeout_sec: float = Field(default=900.0)
    # Second PII engine: Presidio (https://github.com/data-privacy-stack/presidio, current release).
    # Runs *after* HebSafeHarbor on its already-redacted output and adds checksum/pattern
    # recognizers (Israeli ID, Israeli phones, IBAN, credit cards, emails, URLs, IPs, crypto
    # wallets, dates, bank accounts, license plates). Lives in its own isolated venv because the
    # current Presidio needs numpy<2.5 / spaCy>=3.4 — incompatible with both .venv (numpy 2.5,
    # CUDA torch) and .venv-pii (HebSafeHarbor pins Presidio 2.2.31).
    pii_presidio_enabled: bool = Field(default=True)
    pii_presidio_venv: str = Field(default=".venv-presidio")
    # Minimum confidence (0–1) for a Presidio detection to be redacted
    pii_presidio_score_threshold: float = Field(default=0.4)

    # --- worker ---
    # How often the worker checks for a new task in the queue
    poll_interval_sec: float = Field(default=2.0)
    # Load the models into memory as soon as the worker starts (otherwise lazily in the first task)
    preload_models: bool = Field(default=True)

    # --- API ---
    max_upload_mb: int = Field(default=2048)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # An empty environment variable (e.g. PUAH_DEFAULT_NUM_SPEAKERS= in .env) means "unset": the
    # field's own default — None for most, but e.g. 2.8 for the compression threshold and 30 for the
    # cluster size, so a blank line can never silently switch a check off. Otherwise pydantic would
    # fail converting "" to a number. "none"/"off" switch an optional check off explicitly, and
    # "auto" means the automatic carry-over rule.
    @field_validator(
        "default_num_speakers", "hf_token", "gemini_api_key",
        "asr_compression_ratio_threshold", "asr_log_prob_threshold", "asr_no_speech_threshold",
        "asr_condition_on_previous_text", "asr_hallucination_silence_threshold",
        "diarization_min_speakers", "diarization_max_speakers", "diarization_min_cluster_size",
        mode="before",
    )
    @classmethod
    def _blank_to_default(cls, v: object, info: ValidationInfo) -> object:
        if isinstance(v, str):
            word = v.strip().lower()
            if word == "":
                return cls.model_fields[info.field_name].get_default(call_default_factory=True)
            if word in ("none", "off") or (word == "auto" and info.field_name == "asr_condition_on_previous_text"):
                return None
        return v

    @field_validator("asr_temperatures", mode="before")
    @classmethod
    def _temperatures(cls, v: object) -> object:
        """Accept "0.0,0.2", "[0, 0.2]" or a list; store the canonical comma-separated form."""
        if isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)
        if isinstance(v, str):
            body = v.strip().strip("[]")
            if not body.strip():
                return cls.model_fields["asr_temperatures"].get_default()
            return ",".join(str(float(x)) for x in body.split(",") if x.strip())
        return v

    @model_validator(mode="after")
    def _speaker_bounds(self) -> "Settings":
        lo, hi = self.diarization_min_speakers, self.diarization_max_speakers
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("diarization_min_speakers must not exceed diarization_max_speakers")
        return self

    @property
    def asr_temperature_list(self) -> list[float]:
        return [float(x) for x in self.asr_temperatures.split(",") if x.strip()] or [0.0]

    @property
    def asr_language_list(self) -> list[str]:
        """The languages language detection may report; the first one is the default."""
        langs = [x.strip().lower() for x in (self.asr_languages or "").split(",") if x.strip()]
        return langs or [(self.language or "he").lower()]

    # --- Derived properties ---
    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "puah.db"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.audio_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """The active settings of the application (singleton).

    Secrets entered from the interface (data/secrets.json) override the .env values here.
    The initialization is done only through this entry point — Settings() "clean" continues to reflect the .env only,
    so the repository can compare it against and restore it on preference deletion.
    """
    from . import secrets_store
    settings = Settings()
    settings.ensure_dirs()
    secrets_store.apply_overrides(settings, settings.data_dir)
    return settings
