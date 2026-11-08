"""Pydantic schemas for API responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class JobSummary(BaseModel):
    id: str
    original_name: str
    status: str
    stage: Optional[str] = None
    progress: float = 0.0
    num_speakers: Optional[int] = None
    language: str = "he"
    duration_sec: Optional[float] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @classmethod
    def from_row(cls, row: dict) -> "JobSummary":
        return cls(**{k: row.get(k) for k in cls.model_fields})


class Word(BaseModel):
    start: float
    end: float
    word: str
    speaker: Optional[str] = None
    # Whisper's word probability (3 decimals); None for results from before it was kept
    probability: Optional[float] = None
    # "gate_failed" / "sampled_hi": the word's decode window should be checked against the audio
    window_flag: Optional[str] = None
    # Per-word language, when a language-aware stage set one
    lang: Optional[str] = None


class Segment(BaseModel):
    """Continuous speech segment of a single speaker."""

    start: float
    end: float
    speaker: str
    text: str
    words: list[Word] = []
    language: Optional[str] = None  # detected from the segment's script ("he"/"en"); None for old results


class Summary(BaseModel):
    summary: str = ""
    key_points: list[str] = []
    action_items: list[str] = []
    topics: list[str] = []
    keywords: list[str] = []
    model: Optional[str] = None


class TranscriptResult(BaseModel):
    job_id: str
    language: str
    duration_sec: float
    num_speakers: int
    speakers: dict[str, str]  # Internal identifier (SPEAKER_00) -> display name
    segments: list[Segment]
    asr_model: str
    diarization_model: str
    summary: Optional[Summary] = None
    client_summary: Optional[str] = None
    feedback: Optional[str] = None
    # Language detection (after merge): every language with a real share, and the shares themselves
    languages: list[str] = []
    language_shares: dict[str, float] = {}
    # Placeholder-tag check of the generated texts (labels and counts only, never text)
    client_summary_check: Optional[dict] = None
    feedback_check: Optional[dict] = None
    # Re-identification record of the generated texts (labels, counts and character spans only)
    client_summary_reid: Optional[dict] = None
    feedback_reid: Optional[dict] = None
    # Recording-quality check: numbers, flag codes and fixed Hebrew messages only (never transcript text)
    audio_quality: Optional[dict] = None
    # Provenance and post-processing audit of the transcription (ids, settings, counts; never a
    # path or text) and how word confidence may be displayed. None for older results.
    asr_config: Optional[dict] = None
    confidence_display: Optional[dict] = None
    asr_postprocess: Optional[dict] = None


class SpeakerRename(BaseModel):
    speakers: dict[str, str]  # Internal identifier -> new name


class SearchMatch(BaseModel):
    job_id: str
    original_name: str
    seg_index: int
    start: float
    speaker: str
    text: str


class SearchResponse(BaseModel):
    query: str
    count: int
    matches: list[SearchMatch]
