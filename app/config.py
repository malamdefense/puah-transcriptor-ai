"""Application settings — loaded from environment variables / the .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
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

    # --- Speaker diarization (pyannote) ---
    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1")
    # Requires a HuggingFace token after accepting the model's terms of use (one-time).
    hf_token: str | None = Field(default=None)
    # Hint for the number of speakers. None = auto-detect. Can also be overridden per job.
    default_num_speakers: int | None = Field(default=None)

    # --- Automatic summary (Google Gemini — see the "LLM provider" section below) ---
    summary_enabled: bool = Field(default=True)
    # 2048 was too small: long summaries/feedback were cut off mid-text. Raised to 6144.
    llm_max_tokens: int = Field(default=6144)
    # Above this length (characters) the summary is produced via map-reduce
    # (split into chunks, then merge).
    summary_max_chars: int = Field(default=12000)
    # The dedicated PUAH products (per the prompt): "client summary" and "consultant feedback"
    client_summary_enabled: bool = Field(default=True)
    feedback_enabled: bool = Field(default=True)

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

    # --- worker ---
    # How often the worker checks for a new task in the queue
    poll_interval_sec: float = Field(default=2.0)
    # Load the models into memory as soon as the worker starts (otherwise lazily in the first task)
    preload_models: bool = Field(default=True)

    # --- API ---
    max_upload_mb: int = Field(default=2048)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Empty environment variable (e.g. PUAH_DEFAULT_NUM_SPEAKERS= in .env) will be considered unset (None),
    # otherwise pydantic will fail trying to convert "" to a number/optional value.
    @field_validator("default_num_speakers", "hf_token", "gemini_api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

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
