# -*- coding: utf-8 -*-
"""Does a whole-call low level (or hum) change the diarization, and does diarizing the call-wide
fixed copy (loudnorm / notch) restore it? Numbers only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

E2E = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\User\projects\puah")
WORK = E2E / "diar_work"
WORK.mkdir(exist_ok=True)
os.environ.update({"PUAH_DATA_DIR": str(E2E / "data"), "HF_HUB_OFFLINE": "1", "PYANNOTE_METRICS_ENABLED": "false"})
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402,F401
from scipy.optimize import linear_sum_assignment  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.pipeline import audio_quality  # noqa: E402
from app.pipeline.diarize import Diarizer  # noqa: E402

SR = 16000
settings = get_settings()
diarizer = Diarizer(settings)
INP = E2E / "inputs"


def frames(turns, n):
    labs = sorted({t["speaker"] for t in turns})
    out = np.full(n, -1, dtype=np.int32)
    for t in turns:
        out[int(t["start"] * 100):int(t["end"] * 100)] = labs.index(t["speaker"])
    return out, labs


def agreement(ref_turns, hyp_turns, dur):
    n = int(dur * 100) + 1
    r, rl = frames(ref_turns, n)
    h, hl = frames(hyp_turns, n)
    m = np.zeros((len(rl), len(hl)))
    for i in range(len(rl)):
        for j in range(len(hl)):
            m[i, j] = np.sum((r == i) & (h == j))
    ri, ci = linear_sum_assignment(-m)
    ref_speech = np.sum(r >= 0)
    return round(float(m[ri, ci].sum() / ref_speech), 3)


def diar(path: Path):
    d = diarizer.diarize_full(path, None)
    ex = d.exclusive_turns
    talk = {}
    for t in ex:
        talk[t["speaker"]] = talk.get(t["speaker"], 0.0) + t["end"] - t["start"]
    return d, {s: round(v, 1) for s, v in sorted(talk.items())}


def fixed_copy(path: Path, d) -> Path | None:
    """The ASR copy the production check would write (call-wide fixes + gain)."""
    res = audio_quality.prepare_asr_input(path, d, settings, WORK)
    return res.path if res.processed else None


out = {}
for call in ("call16", "call09"):
    src = INP / f"{call}.wav"
    x, _ = sf.read(src, dtype="float32")
    dur = len(x) / SR
    variants = {"orig": src}
    for db in (-12, -18, -24):
        p = WORK / f"{call}_low{-db}.wav"
        sf.write(p, x * 10 ** (db / 20), SR, subtype="PCM_16")
        variants[f"low{-db}"] = p
    if call == "call16":
        variants["hum"] = INP / "call16_hum.wav"
    ref, ref_talk = diar(src)
    out[call] = {"orig": {"talk_s": ref_talk}}
    for name, p in variants.items():
        if name == "orig":
            continue
        d, talk = diar(p)
        row = {"talk_s": talk, "agree": agreement(ref.exclusive_turns, d.exclusive_turns, dur)}
        fc = fixed_copy(p, d)
        if fc is not None:
            d2, talk2 = diar(fc)
            row["fixed_talk_s"] = talk2
            row["fixed_agree"] = agreement(ref.exclusive_turns, d2.exclusive_turns, dur)
            fc.unlink()
        out[call][name] = row
        print(call, name, json.dumps(row))
(E2E / "diar_level.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("done")
