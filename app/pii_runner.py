#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent PII removal runner — run by .venv-pii/python (not the application's).

Imports only HebSafeHarbor and stdlib, to avoid conflicts with application dependencies
(pydantic v2 and others). Reads JSON {"texts": [...]} from stdin, returns {"texts": [...]} to stdout.
Loads the HebSafeHarbor engine once (the download is expensive).

Usage (by pii.py as a subprocess):
    .venv-pii/python app/pii_runner.py  < input.json  > output.json
"""

import json
import sys

# Compatibility with Python 3.10+: HebSafeHarbor 0.1.1 uses `from collections import Set`,
# alias removed in Python 3.10. Restore the aliases from collections.abc before import,
# to support Python 3.10 (and 3.9/3.8). A more maintainable solution that doesn't touch library files.
import collections
import collections.abc as _abc

for _name in (
    "Set", "Mapping", "Sequence", "Iterable", "Callable", "Hashable",
    "MutableMapping", "MutableSet", "MutableSequence", "MappingView",
):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(_abc, _name))


def main() -> int:
    # Read stdin as UTF-8 explicitly (not dependent on Windows/locale default encoding)
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

    hsh = HebSafeHarbor()  # Load the model (he_ner_news_trf) — expensive, once

    # HebSafeHarbor may fail on empty string — send a space and return empty.
    nonempty_idx = [i for i, t in enumerate(texts) if t and str(t).strip()]
    docs = [{"text": str(texts[i])} for i in nonempty_idx]

    anon = list(texts)  # Default: empty values remain as is
    if docs:
        output = hsh(docs)
        for pos, out in zip(nonempty_idx, output):
            try:
                anon[pos] = out.anonymized_text.text
            except Exception:  # noqa: BLE001 — if something unexpected, prefer empty over original text
                anon[pos] = ""

    # Write stdout as UTF-8 explicitly
    sys.stdout.buffer.write(json.dumps({"texts": anon}, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
