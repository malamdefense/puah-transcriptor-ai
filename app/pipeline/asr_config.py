"""Provenance of a job's transcription: which model, which settings (I2; recorded per job).

`asr_config` lets a result say which model weights decoded it and with which ASR settings, so
jobs scored by different models are never pooled and an experiment snapshot can be matched to a
production job. It holds ids, settings, counts and hashes only: never a filesystem path (a local
model directory is reported by its Hugging Face repo id and snapshot commit, or not at all) and
never transcript text. `confidence_display` says how the displayed word confidence was scored and
that the threshold is not validated yet.

Standard library only (importable by the test interpreter without the ML stack). The study runner
keeps identical copies of ASR_SETTINGS_FIELDS, canonical_json, settings_subset and settings_hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

# The settings that decide what the ASR stage and its post-processing produce. data_dir, asr_model,
# device and compute_type are deliberately absent (paths and machine facts; the model is recorded
# by repo id and commit instead).
ASR_SETTINGS_FIELDS: tuple[str, ...] = (
    "language", "beam_size", "vad_filter", "asr_compression_ratio_threshold", "asr_log_prob_threshold",
    "asr_no_speech_threshold", "asr_temperatures", "asr_condition_on_previous_text", "asr_prompt_reset_on_temperature",
    "asr_repetition_penalty", "asr_no_repeat_ngram_size", "asr_hallucination_silence_threshold", "asr_vad_threshold",
    "asr_vad_min_speech_duration_ms", "asr_vad_min_silence_duration_ms", "asr_vad_speech_pad_ms",
    "asr_hotwords_enabled", "asr_hotwords_extra", "asr_hotwords_max_tokens", "asr_term_normalization_enabled",
    "asr_cleanup_enabled", "asr_cleanup_max_ngram", "asr_cleanup_collapsed_word_sec",
    "asr_cleanup_min_collapsed_unigram_copies", "asr_cleanup_max_unigram_copies", "asr_cleanup_max_phrase_copies",
    "merge_use_exclusive", "merge_smoothing", "language_detection", "asr_languages",
    "asr_join_split_words", "asr_window_flags", "asr_confidence_mode", "confidence_display_threshold",
)

# A Hugging Face repo id ("org/name") or a bare model name; anything else (a drive letter, a
# backslash, an absolute path) is treated as a local path and never reported.
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_native(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, str, float)):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_native(x) for x in v]
    return str(v)


def settings_subset(settings: Any) -> dict:
    """The ASR_SETTINGS_FIELDS the object has, as JSON-native values. A non-empty hotword extra
    list is replaced by its SHA-256 (operator terms may be names)."""
    out: dict = {}
    for f in ASR_SETTINGS_FIELDS:
        if not hasattr(settings, f):
            continue
        v = getattr(settings, f)
        if f == "asr_hotwords_extra":
            v = sha256_hex(v) if v else None
        out[f] = _json_native(v)
    return out


def settings_hash(settings: Any) -> str:
    return sha256_hex(canonical_json(settings_subset(settings)))


def _default_hf_cache() -> Path:
    """The hub cache huggingface_hub uses (its constants.py): HF_HUB_CACHE, else the legacy
    HUGGINGFACE_HUB_CACHE, else <HF_HOME>/hub with HF_HOME defaulting to
    <XDG_CACHE_HOME or ~/.cache>/huggingface."""
    def expand(value: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(value)))

    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            return expand(os.environ[name])
    if os.environ.get("HF_HOME"):
        return expand(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return expand(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_identity(asr_model: str, *, hf_cache: Optional[Path] = None) -> dict:
    """{"repo_id", "commit"} of the configured model, without network access or a path.

    A snapshot directory (<cache>/models--<org>--<name>/snapshots/<commit>) gives its repo id and
    commit; any other directory, or a string that is not a repo id, gives None/None. A repo id
    gets the commit its local cache resolves `main` to (refs/main), or None.
    """
    p = Path(str(asr_model))
    try:
        is_dir = p.is_dir()
    except (OSError, ValueError):
        is_dir = False
    if is_dir:
        parent, grand = p.parent, p.parent.parent
        if parent.name == "snapshots" and grand.name.startswith("models--"):
            parts = grand.name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                return {"repo_id": f"{parts[0]}/{parts[1]}", "commit": p.name}
        return {"repo_id": None, "commit": None}
    repo = str(asr_model or "")
    if not _REPO_ID.match(repo):
        return {"repo_id": None, "commit": None}
    if "/" not in repo:
        return {"repo_id": repo, "commit": None}
    org, name = repo.split("/", 1)
    cache = Path(hf_cache) if hf_cache is not None else _default_hf_cache()
    try:
        commit = (cache / f"models--{org}--{name}" / "refs" / "main").read_text(encoding="utf-8").strip() or None
    except (OSError, ValueError):
        commit = None
    return {"repo_id": repo, "commit": commit}


def _versions() -> dict:
    from importlib import metadata  # noqa: PLC0415

    out: dict = {}
    for key, dist in (("faster_whisper", "faster-whisper"), ("ctranslate2", "ctranslate2")):
        try:
            out[key] = metadata.version(dist)
        except Exception:  # noqa: BLE001 - not installed (e.g. the test interpreter)
            out[key] = None
    return out


def build_asr_config(
    settings: Any,
    *,
    model_repo_id: Optional[str],
    model_commit: Optional[str],
    device: Optional[str],
    compute_type: Optional[str],
    condition_on_previous_text: Optional[bool],
    english_stage: bool = False,
    versions: Optional[dict] = None,
) -> dict:
    """The job's `asr_config`: ids, settings and hashes only (no path, no text)."""
    v = _versions() if versions is None else versions
    cfg = {
        "v": 1,
        "model_repo_id": model_repo_id,
        "model_commit": model_commit,
        "device": None if device is None else str(device),
        "compute_type": None if compute_type is None else str(compute_type),
        "condition_on_previous_text": None if condition_on_previous_text is None else bool(condition_on_previous_text),
        "confidence_mode": getattr(settings, "asr_confidence_mode", "mean"),
        "english_stage": bool(english_stage),
        "join_split_words": bool(getattr(settings, "asr_join_split_words", False)),
        "window_flags": bool(getattr(settings, "asr_window_flags", False)),
        "faster_whisper": v.get("faster_whisper"),
        "ctranslate2": v.get("ctranslate2"),
        "settings_hash": settings_hash(settings),
    }
    cfg["config_hash"] = sha256_hex(canonical_json(cfg))
    return cfg


def build_confidence_display(settings: Any, *, model_commit: Optional[str]) -> dict:
    """How the UI may show word confidence: the score, the provisional threshold, the weights that
    scored it, and validated=False until a reference set validates the marks."""
    return {
        "score": getattr(settings, "asr_confidence_mode", "mean"),
        "threshold": getattr(settings, "confidence_display_threshold", 0.5),
        "scored_by_model_commit": model_commit,
        "validated": False,
    }
