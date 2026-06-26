"""Tests for the core merge logic — mapping words to speakers. Runs without GPU/models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.merge import merge, _overlap, _speaker_for_interval  # noqa: E402


def _w(start, end, word):
    return {"start": start, "end": end, "word": word}


def test_overlap_basic():
    assert _overlap(0, 10, 5, 15) == 5
    assert _overlap(0, 5, 10, 15) == 0
    assert _overlap(0, 10, 2, 8) == 6


def test_speaker_picks_max_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    assert _speaker_for_interval(0.5, 1.5, turns) == "SPEAKER_00"
    assert _speaker_for_interval(6.0, 7.0, turns) == "SPEAKER_01"
    # Word on the boundary with more overlap for 01
    assert _speaker_for_interval(4.6, 6.0, turns) == "SPEAKER_01"


def test_speaker_no_overlap_falls_back_to_nearest():
    turns = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    assert _speaker_for_interval(20.0, 21.0, turns) == "SPEAKER_00"


def test_merge_groups_by_speaker_and_gap():
    asr = [
        {
            "start": 0.0,
            "end": 9.0,
            "text": "שלום מה שלומך טוב מאוד",
            "words": [
                _w(0.0, 1.0, "שלום"),
                _w(1.0, 2.0, "מה"),
                _w(2.0, 3.0, "שלומך"),
                _w(6.0, 7.0, "טוב"),   # Other speaker
                _w(7.0, 8.0, "מאוד"),
            ],
        }
    ]
    turns = [
        {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"},
        {"start": 5.5, "end": 9.0, "speaker": "SPEAKER_01"},
    ]
    segments, speakers = merge(asr, turns)

    assert speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert len(segments) == 2
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[0]["text"] == "שלום מה שלומך"
    assert segments[1]["speaker"] == "SPEAKER_01"
    assert segments[1]["text"] == "טוב מאוד"


def test_merge_without_word_timestamps():
    """Nefila kinnanit le-r'et keshif word timestamps."""
    asr = [
        {"start": 0.0, "end": 4.0, "text": "קטע ראשון", "words": []},
        {"start": 4.0, "end": 8.0, "text": "קטע שני", "words": []},
    ]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01"},
    ]
    segments, speakers = merge(asr, turns)
    assert len(segments) == 2
    assert segments[0]["text"] == "קטע ראשון"
    assert segments[1]["speaker"] == "SPEAKER_01"


def test_merge_empty():
    assert merge([], []) == ([], [])


if __name__ == "__main__":
    # Direct run without pytest
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
