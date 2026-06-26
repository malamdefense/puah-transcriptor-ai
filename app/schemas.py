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


class Segment(BaseModel):
    """Continuous speech segment of a single speaker."""

    start: float
    end: float
    speaker: str
    text: str
    words: list[Word] = []


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
    speakers: dict[str, str]  # Internal ID (SPEAKER_00) -> display name
    segments: list[Segment]
    asr_model: str
    diarization_model: str
    summary: Optional[Summary] = None
    client_summary: Optional[str] = None
    feedback: Optional[str] = None


class SpeakerRename(BaseModel):
    speakers: dict[str, str]  # Internal ID -> new name


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
