"""**System settings loaded from environment variables / .env file.**"""

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
    )

    # --- Storage paths (On-Prem, everything local) ---
    data_dir: Path = Field(default=Path("data"))

    # --- Transcription model (ivrit.ai, CTranslate2 format for faster-whisper) ---
    # turbo-ct2 = fast and accurate in Hebrew. For maximum accuracy: ivrit-ai/whisper-large-v3-ct2
    asr_model: str = Field(default="ivrit-ai/whisper-large-v3-turbo-ct2")
    language: str = Field(default="he")
    # cuda / cpu / auto
    device: str = Field(default="auto")
    # float16 for GPU, int8 for CPU. auto chooses automatically.
    compute_type: str = Field(default="auto")
    beam_size: int = Field(default=5)
    # VAD filtering before transcription — improves accuracy and shortens time
    vad_filter: bool = Field(default=True)

    # --- Speaker diarization (pyannote) ---
    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1")
    # HuggingFace token required after approving usage terms for the model (one-time).
    hf_token: str | None = Field(default=None)
    # Hint for the number of speakers. None = automatic diarization. Can also be overridden per task.
    default_num_speakers: int | None = Field(default=None)

    # --- Automatic summary (local language model, OpenAI-compatible interface: Ollama / vLLM / LM Studio) ---
    summary_enabled: bool = Field(default=True)
    # Default: Ollama local. Replace with vLLM/LM Studio as needed.
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    # Default: Gemma 4.0 12B it — pulled from Ollama registry (ollama pull gemma4:12b),
    # multilingual with good Hebrew and 256K context window. Reliable and works out of the box.
    # Hebrew-specific alternative: DictaLM 3.0. Pulling via hf.co fails currently (Ollama bug with HuggingFace),
    # so we import it manually (see README) and set the name here, e.g.: dictalm3
    llm_model: str = Field(default="gemma4:12b")
    llm_api_key: str = Field(default="ollama")  # Ollama ignores; vLLM/other may require it
    llm_timeout_sec: float = Field(default=600.0)
    # Provider: "ollama" (native /api/chat — controls num_ctx and disables reasoning) or "openai"
    # (matches /v1/chat/completions — vLLM/LM Studio). The /v1 of Ollama ignores num_ctx!
    llm_provider: str = Field(default="ollama")
    # Context window. The default of Ollama (4096) is too small for transcription — must increase.
    llm_num_ctx: int = Field(default=16384)
    llm_max_tokens: int = Field(default=2048)
    # Disabling "thinking" (reasoning) of models like gemma4 — direct JSON output, faster.
    llm_think: bool = Field(default=False)
    # Above this length (characters) the summary is performed in map-reduce (splitting into segments and merging).
    summary_max_chars: int = Field(default=12000)
    # PUAH-specific outputs (based on the prompts): "client summary" and "advisor/rab feedback"
    client_summary_enabled: bool = Field(default=True)
    feedback_enabled: bool = Field(default=True)

    # --- Export Word + PII removal (after transcription is saved) ---
    # Create Word document for each transcription ({job_id}.docx)
    word_export_enabled: bool = Field(default=True)
    # De-identifying PII (HebSafeHarbor) and saving a clean version ({job_id}.pii.json/.docx)
    pii_enabled: bool = Field(default=True)
    # Isolated venv where HebSafeHarbor is installed (relative to the root)
    pii_venv: str = Field(default=".venv-pii")
    pii_timeout_sec: float = Field(default=900.0)

    # --- worker ---
    # How often the worker checks for a new task in the queue
    poll_interval_sec: float = Field(default=2.0)
    # Preloading models to memory on startup (otherwise lazy on first task)
    preload_models: bool = Field(default=True)

    # --- API ---
    max_upload_mb: int = Field(default=2048)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Empty environment variable (e.g. PUAH_DEFAULT_NUM_SPEAKERS= in .env) is considered unset (None),
    # otherwise pydantic fails trying to convert "" to a number/optional value.
    @field_validator("default_num_speakers", "hf_token", mode="before")
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
    settings = Settings()
    settings.ensure_dirs()
    return settings
