#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run PII removal independently — run by the Python in .venv-pii (not the application's).

Import only HebSafeHarbor and stdlib, to avoid interfering with the application's dependencies
(pydantic v2 and others).

Modes:
  * Single-shot (default): reads JSON {"texts": [...]} from stdin, writes {"texts": [...]} to stdout.
  * Persistent daemon (--serve): loads HebSafeHarbor once, writes {"ready": true} line to stdout,
    then processes JSON-line requests from stdin and writes one JSON-line response per request.
  * Probe (--probe): imports HebSafeHarbor without loading the transformer model and exits 0 or 3.
"""

import json
import sys

# Python 3.10+ compatibility: HebSafeHarbor 0.1.1 uses `from collections import Set`,
# a name that was removed in Python 3.10. Restore the names from collections.abc before importing,
# to support Python 3.10 (and 3.9/3.8). A non-invasive solution that doesn't touch the library files.
import collections
import collections.abc as _abc

for _name in (
    "Set", "Mapping", "Sequence", "Iterable", "Callable", "Hashable",
    "MutableMapping", "MutableSet", "MutableSequence", "MappingView",
):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(_abc, _name))


def keep_countries() -> None:
    """Leave country names as they are.

    HebSafeHarbor's default COUNTRY operator does not tag a country — it *rewrites* it as its
    region ("אוסטרליה" -> "אוסטרליה ואוקיאניה"), inserting words the speaker never said into the
    de-identified transcript and into the text sent to the summariser. A country is not an
    identifier (HIPAA Safe Harbor keeps it), and where a caller lives abroad is useful context.
    Patched here, in our runner, rather than in the installed library.
    """
    from hebsafeharbor.anonymizer.custom_operators.country_anonymizer_operator import (
        CountryAnonymizerOperator,
    )

    CountryAnonymizerOperator.operate = lambda self, text, params=None: text


def redact_batch(hsh, texts: list) -> list:
    """Redact all non-empty strings in `texts` using an initialized HebSafeHarbor instance."""
    nonempty_idx = [i for i, t in enumerate(texts) if t and str(t).strip()]
    docs = [{"text": str(texts[i])} for i in nonempty_idx]

    anon = list(texts)
    if docs:
        output = hsh(docs)
        for pos, out in zip(nonempty_idx, output):
            try:
                anon[pos] = out.anonymized_text.text
            except Exception:
                anon[pos] = ""
    return anon


def probe() -> int:
    """--probe: can HebSafeHarbor be imported here at all?

    Imports only — constructing HebSafeHarbor() loads a transformer and costs tens of seconds.
    app/pipeline/pii.py uses this to tell "the files are on disk" apart from "the engine runs".
    """
    try:
        from hebsafeharbor import HebSafeHarbor  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"hebsafeharbor not available: {exc}", file=sys.stderr)
        return 3
    return 0


def serve() -> int:
    """--serve: load HebSafeHarbor once and answer JSON-line requests over stdin/stdout."""
    proto_out = sys.__stdout__.buffer
    sys.stdout = sys.stderr

    try:
        from hebsafeharbor import HebSafeHarbor
        keep_countries()
        hsh = HebSafeHarbor()
    except Exception as exc:  # noqa: BLE001
        print(f"hebsafeharbor not available: {exc}", file=sys.stderr, flush=True)
        return 3

    proto_out.write((json.dumps({"ready": True, "engine": "hebsafeharbor"}) + "\n").encode("utf-8"))
    proto_out.flush()

    for raw_line in sys.stdin.buffer:
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            resp = {"error": f"invalid input json: {exc}"}
            proto_out.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            proto_out.flush()
            continue

        cmd = data.get("cmd") if isinstance(data, dict) else None
        if cmd == "ping":
            proto_out.write(b'{"ok": true}\n')
            proto_out.flush()
            continue
        if cmd == "shutdown":
            proto_out.write(b'{"ok": true}\n')
            proto_out.flush()
            return 0

        texts = data.get("texts", []) if isinstance(data, dict) else None
        if not isinstance(texts, list):
            resp = {"error": "'texts' must be a list"}
            proto_out.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            proto_out.flush()
            continue

        try:
            anon = redact_batch(hsh, texts)
            resp = {"texts": anon}
        except Exception as exc:  # noqa: BLE001
            resp = {"error": f"hebsafeharbor execution failed: {exc}"}
        proto_out.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        proto_out.flush()
    return 0


def main() -> int:
    # Answered before stdin is touched — a probe sends no payload.
    if "--probe" in sys.argv[1:]:
        return probe()
    if "--serve" in sys.argv[1:]:
        return serve()

    # Read stdin as UTF-8 explicitly (not dependent on the default encoding of Windows/locale)
    try:
        raw = sys.stdin.buffer.read().decode("utf-8")
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"invalid input json: {exc}", file=sys.stderr)
        return 2

    texts = data.get("texts", [])
    if not isinstance(texts, list):
        print("'texts' must be a list", file=sys.stderr)
        return 2

    try:
        from hebsafeharbor import HebSafeHarbor
    except Exception as exc:  # noqa: BLE001
        print(f"hebsafeharbor not available: {exc}", file=sys.stderr)
        return 3

    keep_countries()
    hsh = HebSafeHarbor()
    anon = redact_batch(hsh, texts)

    sys.stdout.buffer.write(json.dumps({"texts": anon}, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
