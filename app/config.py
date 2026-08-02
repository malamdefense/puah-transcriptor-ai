"""הגדרות מערכת — נטענות ממשתני סביבה / קובץ .env."""

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
        populate_by_name=True,
    )

    data_dir: Path = Field(default=Path("data"))

    asr_model: str = Field(default="ivrit-ai/whisper-large-v3-turbo-ct2")
    language: str = Field(default="he")
    device: str = Field(default="auto")
    compute_type: str = Field(default="auto")
    beam_size: int = Field(default=5)
    vad_filter: bool = Field(default=True)

    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1")
    hf_token: str | None = Field(default=None)
    default_num_speakers: int | None = Field(default=None)

    summary_enabled: bool = Field(default=True)
    llm_max_tokens: int = Field(default=6144)
    summary_max_chars: int = Field(default=12000)
    client_summary_enabled: bool = Field(default=True)
    feedback_enabled: bool = Field(default=True)

    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_timeout_sec: float = Field(default=600.0)

    gemini_api_key: str | None = Field(default=None, validation_alias="PUHA_GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-flash-latest", validation_alias="PUHA_GEMINI_MODEL")
    gemini_thinking_budget: int = Field(default=0, validation_alias="PUHA_GEMINI_THINKING_BUDGET")

    embed_model: str = Field(default="bge-m3")
    chroma_host: str | None = Field(default=None)
    chroma_port: int = Field(default=8000)

    word_export_enabled: bool = Field(default=True)
    pii_enabled: bool = Field(default=True)
    pii_venv: str = Field(default=".venv-pii")
    pii_timeout_sec: float = Field(default=900.0)

    poll_interval_sec: float = Field(default=2.0)
    preload_models: bool = Field(default=True)

    max_upload_mb: int = Field(default=2048)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("default_num_speakers", "hf_token", "gemini_api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

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
