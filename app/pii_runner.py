#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run PII removal independently — run by the Python in .venv-pii (not the application's).

Import only HebSafeHarbor and stdlib, to avoid interfering with the application's dependencies
(pydantic v2 and others). Reads JSON {"texts": [...]} from stdin, returns {"texts": [...]} to stdout.
Loads HebSafeHarbor once (the expensive load).

Usage (by pii.py as a subprocess):
    .venv-pii/python app/pii_runner.py  < input.json  > output.json
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


def main() -> int:
    # Answered before stdin is touched — a probe sends no payload.
    if "--probe" in sys.argv[1:]:
        return probe()

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

    sys.stdout.buffer.write(json.dumps({"texts": anon}, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
