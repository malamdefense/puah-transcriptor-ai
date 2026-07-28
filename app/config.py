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
        # נדרש כדי ששדות עם validation_alias מפורש (gemini_api_key וכד' — ששמם
        # ב-.env יוצא מכלל קידומת PUAH_) יהיו ניתנים לבנייה גם לפי שם השדה
        # בפייתון (Settings(gemini_api_key=...)), כפי שכל שאר השדות כבר
        # מתנהגים דרך מנגנון ה-env_prefix האוטומטי.
        populate_by_name=True,
    )

    # --- נתיבי אחסון (On-Prem, הכל מקומי) ---
    data_dir: Path = Field(default=Path("data"))

    # --- מודל תמלול (ivrit.ai, פורמט CTranslate2 ל-faster-whisper) ---
    # turbo-ct2 = מהיר ומדויק בעברית. לדיוק מקסימלי: ivrit-ai/whisper-large-v3-ct2
    asr_model: str = Field(default="ivrit-ai/whisper-large-v3-turbo-ct2")
    language: str = Field(default="he")
    # cuda / cpu / auto
    device: str = Field(default="auto")
    # float16 ל-GPU, int8 ל-CPU. auto בוחר אוטומטית.
    compute_type: str = Field(default="auto")
    beam_size: int = Field(default=5)
    # סינון שתיקות לפני התמלול — משפר דיוק ומקצר זמן
    vad_filter: bool = Field(default=True)

    # --- זיהוי דוברים (pyannote) ---
    diarization_model: str = Field(default="pyannote/speaker-diarization-3.1")
    # נדרש טוקן HuggingFace לאחר אישור תנאי השימוש במודל (חד-פעמי).
    hf_token: str | None = Field(default=None)
    # רמז למספר דוברים. None = זיהוי אוטומטי. ניתן גם לדרוס פר-משימה.
    default_num_speakers: int | None = Field(default=None)

    # --- סיכום אוטומטי (Google Gemini — ראו סעיף "ספק LLM" בהמשך) ---
    summary_enabled: bool = Field(default=True)
    # 2048 היה קטן מדי: סיכומים/משובים ארוכים נקטעו באמצע. שולש ל-6144.
    llm_max_tokens: int = Field(default=6144)
    # מעל אורך זה (תווים) הסיכום מתבצע במפה-רדוקציה (חלוקה לקטעים ואיחוד).
    summary_max_chars: int = Field(default=12000)
    # תוצרי פוע"ה הייעודיים (לפי הפרומפט): "סיכום ללקוח" ו"משוב ליועץ/רב"
    client_summary_enabled: bool = Field(default=True)
    feedback_enabled: bool = Field(default=True)

    # --- Local Ollama server — currently used only for embeddings (RAG/evaluation) ---
    # Ollama is still required for the embedding model (bge-m3).
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_timeout_sec: float = Field(default=600.0)

    # --- LLM provider: Google Gemini (cloud) ---
    # Note: the local provider was removed at the request of the client, so generating summaries requires a valid API key and internet connectivity, and the call content (after local PII enforcement is required) goes to Google servers. You can enter a key from the settings (⚙) and it overrides the .env.
    gemini_api_key: str | None = Field(default=None, validation_alias="PUHA_GEMINI_API_KEY")
    # gemini-1.5-flash (originally requested) was completely removed by Google (404), and gemini-2.5-flash "no longer available to new users" — directly verified against the API (genai.list_models() + actual generateContent call). gemini-flash-latest is an alias maintained by Google that always points to the currently available flash model — stable against similar future deprecations. For permanent versions, you can set a specific version via PUHA_GEMINI_MODEL.
    gemini_model: str = Field(default="gemini-flash-latest", validation_alias="PUHA_GEMINI_MODEL")
    # "Thinking budget" for Gemini (thinking_config.thinking_budget). 0 = internal thinking off — required to prevent truncation/cutting of output (truncated/internal metadata responses when thinking was not controlled). Can be 1 (dynamic) or a number of tokens.
    gemini_thinking_budget: int = Field(default=0, validation_alias="PUHA_GEMINI_THINKING_BUDGET")

    # --- Semantic RAG (local embeddings via Ollama + ChromaDB vector storage) ---
    # bge-m3: multi
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
