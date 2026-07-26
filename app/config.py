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

    # --- סיכום אוטומטי (מודל שפה מקומי, ממשק תואם OpenAI: Ollama / vLLM / LM Studio) ---
    summary_enabled: bool = Field(default=True)
    # ברירת מחדל: Ollama מקומי. החליפו ל-vLLM/LM Studio לפי הצורך.
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    # ברירת מחדל: Gemma 4.0 12B it — נמשך מ-registry של Ollama (ollama pull gemma4:12b),
    # רב-לשוני עם עברית טובה וחלון הקשר 256K. אמין ועובד מהקופסה.
    # חלופה מותאמת-עברית: DictaLM 3.0. משיכה דרך hf.co נכשלת כיום (באג Ollama מול HuggingFace),
    # לכן מייבאים אותה ידנית (ראו README) ומגדירים כאן את השם, למשל: dictalm3
    llm_model: str = Field(default="gemma4:12b")
    llm_api_key: str = Field(default="ollama")  # Ollama מתעלם; vLLM/אחר עשוי לדרוש
    llm_timeout_sec: float = Field(default=600.0)
    # ספק: "ollama" (native /api/chat — שולט ב-num_ctx ומכבה reasoning) או "openai"
    # (תואם /v1/chat/completions — vLLM/LM Studio). ה-/v1 של Ollama מתעלם מ-num_ctx!
    llm_provider: str = Field(default="ollama")
    # חלון הקשר. ה-default של Ollama (4096) קטן מדי לתמלול — חייבים להגדיל.
    llm_num_ctx: int = Field(default=16384)
    llm_max_tokens: int = Field(default=2048)
    # כיבוי "חשיבה" (reasoning) של מודלים כמו gemma4 — פלט JSON ישיר, מהיר יותר.
    llm_think: bool = Field(default=False)
    # מעל אורך זה (תווים) הסיכום מתבצע במפה-רדוקציה (חלוקה לקטעים ואיחוד).
    summary_max_chars: int = Field(default=12000)
    # תוצרי פוע"ה הייעודיים (לפי הפרומפט): "סיכום ללקוח" ו"משוב ליועץ/רב"
    client_summary_enabled: bool = Field(default=True)
    feedback_enabled: bool = Field(default=True)

    # --- RAG סמנטי (embeddings מקומיים דרך Ollama + אחסון וקטורי ChromaDB) ---
    # bge-m3: רב-לשוני עם עברית טובה. משיכה:  ollama pull bge-m3
    embed_model: str = Field(default="bge-m3")
    # Chroma: ריק = embedded (data/chroma/, On-Prem). הגדירו host לשרת Chroma נפרד.
    chroma_host: str | None = Field(default=None)
    chroma_port: int = Field(default=8000)

    # --- ייצוא Word + הסרת PII (אחרי שמירת התמלול) ---
    # יצירת מסמך Word לכל תמלול ({job_id}.docx)
    word_export_enabled: bool = Field(default=True)
    # הסרת פרטים מזהים (HebSafeHarbor) ושמירת גרסה מנוקה ({job_id}.pii.json/.docx)
    pii_enabled: bool = Field(default=True)
    # סביבת ה-venv המבודדת שבה מותקן HebSafeHarbor (יחסית לשורש הריפו)
    pii_venv: str = Field(default=".venv-pii")
    pii_timeout_sec: float = Field(default=900.0)

    # --- worker ---
    # כל כמה שניות ה-worker בודק אם יש משימה חדשה בתור
    poll_interval_sec: float = Field(default=2.0)
    # טעינת המודלים לזיכרון כבר בעלייה (אחרת בעצלתיים במשימה הראשונה)
    preload_models: bool = Field(default=True)

    # --- API ---
    max_upload_mb: int = Field(default=2048)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # משתנה סביבה ריק (למשל PUAH_DEFAULT_NUM_SPEAKERS= ב-.env) ייחשב כלא-מוגדר (None),
    # אחרת pydantic ייכשל בניסיון להמיר "" למספר/לערך אופציונלי.
    @field_validator("default_num_speakers", "hf_token", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # --- נגזרות ---
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
