#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Second PII pass — Presidio (https://github.com/data-privacy-stack/presidio), current release.

Run by the Python in .venv-presidio (not the application's, not HebSafeHarbor's). Reads JSON
{"texts": [...], "score_threshold": 0.4} from stdin and writes
{"texts": [...], "entities": {"PHONE_NUMBER": 3, ...}, "engine": "presidio 2.2.x"} to stdout.

The worker feeds it the output of HebSafeHarbor (already redacted — names, places, organisations,
Hebrew dates …), so this pass only adds what a checksum/pattern engine is good at:

  * Presidio built-ins — EMAIL_ADDRESS, PHONE_NUMBER (python-phonenumbers, IL first), URL,
    IP_ADDRESS, CREDIT_CARD (Luhn), IBAN_CODE (mod-97), CRYPTO (wallets), DATE_TIME (numeric dates).
  * Israeli recognizers added here — ISRAELI_ID (9 digits + national checksum), Israeli phone
    formats as a fallback, ISRAELI_BANK_ACCOUNT, ISRAELI_LICENSE_PLATE, ISRAELI_PASSPORT
    (context-only).

Every recognizer carries Hebrew (and English) context words, so "טלפון 052-1234567" scores higher
than a bare number. Replacement tags mirror HebSafeHarbor's vocabulary (<קשר_>, <מזהה_>,
<תאריך_>) so the de-identified document reads uniformly.

Only Presidio + stdlib are imported here (lazily, inside build_engines) so that the pure helpers
(israeli_id_valid, tag_for_entity, tokenize) can be imported and unit-tested from the application's
venv. Tokenization is pure Python (no spaCy model, no compiled extension) — see import_presidio for
why that matters on machines with Application Control policies.

Usage (by app/pipeline/pii.py as a subprocess):
    .venv-presidio/python app/presidio_runner.py  < input.json  > output.json
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

# ---------- Pure helpers (no Presidio import) ----------

DEFAULT_SCORE_THRESHOLD = 0.4

# Replacement tag per entity — same vocabulary as HebSafeHarbor's ReplaceInHebrew operator.
_TAG_CONTACT = "<קשר_>"
_TAG_ID = "<מזהה_>"
_TAG_DATE = "<תאריך_>"

TAG_FOR_ENTITY: dict[str, str] = {
    "EMAIL_ADDRESS": _TAG_CONTACT,
    "PHONE_NUMBER": _TAG_CONTACT,
    "URL": _TAG_CONTACT,
    "IP_ADDRESS": _TAG_CONTACT,
    "DATE_TIME": _TAG_DATE,
    "CREDIT_CARD": _TAG_ID,
    "IBAN_CODE": _TAG_ID,
    "CRYPTO": _TAG_ID,
    "ISRAELI_ID": _TAG_ID,
    "ISRAELI_BANK_ACCOUNT": _TAG_ID,
    "ISRAELI_LICENSE_PLATE": _TAG_ID,
    "ISRAELI_PASSPORT": _TAG_ID,
}


def tag_for_entity(entity_type: str) -> str:
    return TAG_FOR_ENTITY.get(entity_type, _TAG_ID)


def israeli_id_valid(digits: str) -> bool:
    """Israeli ID (teudat zehut) checksum: weights 1,2,1,2… over 9 digits (left-padded with
    zeros), each product reduced to a single digit by subtracting 9 when >9; the sum must be
    divisible by 10."""
    digits = "".join(ch for ch in digits if ch.isdigit())
    if not 5 <= len(digits) <= 9:
        return False
    digits = digits.zfill(9)
    total = 0
    for i, ch in enumerate(digits):
        n = int(ch) * (1 if i % 2 == 0 else 2)
        if n > 9:
            n -= 9
        total += n
    return total % 10 == 0


# Words/numbers (hyphens, dots, @, + kept inside a token so phone numbers and e-mails stay whole)
# and single punctuation marks.
_TOKEN_RE = re.compile(r"""[^\s,;:!?()\[\]{}"'«»]+|[,;:!?()\[\]{}"'«»]""")


def tokenize(text: str) -> list[tuple[str, int]]:
    """Pure-Python tokenizer. Returns (token, offset-in-text) pairs."""
    return [(m.group(0), m.start()) for m in _TOKEN_RE.finditer(text)]


# ---------- Hebrew / English context words ----------

CTX_PHONE = ["טלפון", "טל", "נייד", "סלולרי", "פלאפון", "פקס", "להתקשר", "תתקשר", "תתקשרי",
             "חייג", "מספר", "וואטסאפ", "whatsapp", "phone", "mobile", "cell", "call"]
CTX_EMAIL = ["אימייל", "מייל", "דואל", "דואר אלקטרוני", "כתובת", "email", "mail"]
CTX_URL = ["אתר", "קישור", "לינק", "כתובת", "website", "link", "url"]
CTX_IP = ["ip", "כתובת ip", "איי פי", "שרת", "כתובת"]
CTX_CARD = ["כרטיס", "אשראי", "ויזה", "מאסטרקארד", "ישראכרט", "תשלום", "card", "credit", "visa"]
CTX_IBAN = ["iban", "חשבון", "בנק", "העברה", "העברה בנקאית", "swift", "bank"]
CTX_CRYPTO = ["ביטקוין", "ארנק", "קריפטו", "bitcoin", "wallet", "crypto", "btc", "eth"]
CTX_DATE = ["תאריך", "לידה", "הולדת", "נולד", "נולדה", "נולדתי", "ביום", "בתאריך", "date", "born"]
CTX_ID = ["תעודת זהות", "ת.ז", "ת\"ז", "תז", "זהות", "מספר זהות", "תעודה", "id", "identity"]
CTX_BANK = ["חשבון", "חשבון בנק", "בנק", "סניף", "מספר חשבון", "account", "bank", "branch"]
CTX_PLATE = ["רכב", "מספר רכב", "לוחית", "לוחית רישוי", "אוטו", "מכונית", "רישוי", "plate", "car"]
CTX_PASSPORT = ["דרכון", "מספר דרכון", "passport"]


# ---------- Importing Presidio (tolerating blocked spaCy extensions) ----------

def _probe_blocked_spacy_extensions() -> list[str]:
    """Return dotted names of spaCy compiled extensions that Windows refuses to load.

    On enterprise Windows machines with Application Control (WDAC) policies, individual .pyd
    files may be rejected ("An Application Control policy has blocked this file", WinError 4551)
    — observed for spacy.matcher.dependencymatcher and spacy.kb.kb_in_memory. Loading is probed
    with LoadLibrary *before* spaCy is imported: a failed import leaves half-initialised Cython
    modules behind that cannot be initialised a second time, so retry-after-failure is not an option.
    """
    import os

    if os.name != "nt":
        return []
    import ctypes
    import importlib.util
    import pathlib

    spec = importlib.util.find_spec("spacy")
    if not spec or not spec.origin:
        return []
    base = pathlib.Path(spec.origin).parent
    blocked: list[str] = []
    for pyd in sorted(base.rglob("*.pyd")):
        try:
            ctypes.WinDLL(str(pyd))
        except OSError as exc:
            if getattr(exc, "winerror", None) == 4551 or "Application Control" in str(exc):
                rel = pyd.relative_to(base.parent).with_suffix("")
                blocked.append(".".join(rel.parts).split(".cp3")[0])
    return blocked


def _stub_module(full: str) -> None:
    """Install a stub for `full` whose attributes are dummy classes (constructing one raises)."""
    import types

    stub = types.ModuleType(full)

    def _getattr(name: str, _full: str = full):
        if name.startswith("__"):
            raise AttributeError(name)

        def _init(self, *a, **k):
            raise RuntimeError(f"{_full}.{name} is unavailable (blocked extension stub)")

        return type(name, (), {"__init__": _init})

    stub.__getattr__ = _getattr  # type: ignore[attr-defined]
    sys.modules[full] = stub


_PRESIDIO: tuple | None = None


def import_presidio():
    """Import presidio_analyzer / presidio_anonymizer, working around blocked spaCy extensions.

    presidio_analyzer imports spaCy at package import time. None of the extensions that
    Application Control policies were seen to block is needed here (tokenization is pure Python,
    no NER), so they are stubbed up-front and the import runs once (cached for later calls).
    """
    global _PRESIDIO
    if _PRESIDIO is None:
        for full in _probe_blocked_spacy_extensions():
            _stub_module(full)
            print(f"presidio_runner: stubbed blocked spaCy extension {full}", file=sys.stderr)
        import presidio_analyzer
        import presidio_anonymizer
        _PRESIDIO = (presidio_analyzer, presidio_anonymizer)
    return _PRESIDIO


def presidio_version() -> str:
    try:
        from importlib.metadata import version
        return version("presidio-analyzer")
    except Exception:  # noqa: BLE001
        return "?"


# ---------- Engine construction (Presidio imported here only) ----------

def build_engines(score_threshold: float = DEFAULT_SCORE_THRESHOLD):
    """Create (AnalyzerEngine, AnonymizerEngine, operators) configured for Hebrew."""
    import_presidio()
    from presidio_analyzer import (
        AnalyzerEngine,
        Pattern,
        PatternRecognizer,
        RecognizerRegistry,
    )
    from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        CryptoRecognizer,
        DateRecognizer,
        EmailRecognizer,
        IbanRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        UrlRecognizer,
    )
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    class HebrewTokenizerNlpEngine(NlpEngine):
        """Minimal NLP engine: tokens + lower-cased "lemmas" for Presidio's context enhancer.

        NER itself is HebSafeHarbor's job (first pass). Presidio only needs tokens here so that
        Hebrew context words ("טלפון", "תעודת זהות", "דרכון" …) around a match raise its score.
        Pure Python — no spaCy model, no compiled extension, works air-gapped.
        """

        engine_name = "hebrew_tokenizer"
        is_available = True

        def __init__(self):
            self._loaded = False

        def load(self) -> None:
            self._loaded = True

        def is_loaded(self) -> bool:
            return self._loaded

        def process_text(self, text: str, language: str) -> NlpArtifacts:
            toks = tokenize(text)
            return NlpArtifacts(
                entities=[],
                tokens=[t for t, _ in toks],
                tokens_indices=[i for _, i in toks],
                lemmas=[t.lower() for t, _ in toks],
                nlp_engine=self,
                language=language,
                scores=[],
            )

        def process_batch(self, texts, language: str, **kwargs):
            for t in texts:
                yield t, self.process_text(str(t), language)

        def is_stopword(self, word: str, language: str) -> bool:
            return False

        def is_punct(self, word: str, language: str) -> bool:
            return not any(ch.isalnum() for ch in word)

        def get_supported_entities(self) -> list[str]:
            return []

        def get_supported_languages(self) -> list[str]:
            return ["he"]

    class IsraeliIdRecognizer(PatternRecognizer):
        """9-digit (or 8-digit, leading zero dropped) number that passes the national checksum."""

        def __init__(self):
            super().__init__(
                supported_entity="ISRAELI_ID",
                name="IsraeliIdRecognizer",
                supported_language="he",
                patterns=[Pattern("israeli_id", r"(?<!\d)\d{8,9}(?!\d)", 0.3)],
                context=CTX_ID,
            )

        def validate_result(self, pattern_text: str):
            return True if israeli_id_valid(pattern_text) else False

    lang = "he"
    recognizers = [
        EmailRecognizer(supported_language=lang, context=CTX_EMAIL),
        UrlRecognizer(supported_language=lang, context=CTX_URL),
        IpRecognizer(supported_language=lang, context=CTX_IP),
        CreditCardRecognizer(supported_language=lang, context=CTX_CARD),
        IbanRecognizer(supported_language=lang, context=CTX_IBAN),
        CryptoRecognizer(supported_language=lang, context=CTX_CRYPTO),
        DateRecognizer(supported_language=lang, context=CTX_DATE),
        PhoneRecognizer(
            supported_language=lang,
            context=CTX_PHONE,
            supported_regions=("IL", "US", "GB", "FR", "DE"),
            leniency=1,
        ),
        IsraeliIdRecognizer(),
        # Israeli phone formats as a fallback to python-phonenumbers (which may reject numbers
        # ASR wrote with unusual spacing). Mobile 05X / 07X, landline 02/03/04/08/09, +972.
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="IsraeliPhoneRecognizer",
            supported_language=lang,
            patterns=[
                Pattern("il_mobile", r"(?<!\d)0[57]\d[- .]?\d{3}[- .]?\d{4}(?!\d)", 0.5),
                Pattern("il_landline", r"(?<!\d)0[2-4,8-9][- .]?\d{3}[- .]?\d{4}(?!\d)", 0.4),
                Pattern("il_intl", r"\+972[- .]?\d{1,2}[- .]?\d{3}[- .]?\d{4}(?!\d)", 0.6),
            ],
            context=CTX_PHONE,
        ),
        PatternRecognizer(
            supported_entity="ISRAELI_BANK_ACCOUNT",
            name="IsraeliBankAccountRecognizer",
            supported_language=lang,
            # bank(2) - branch(3) - account(6–9), as printed on cheques/statements
            patterns=[Pattern("il_bank_account", r"(?<!\d)\d{2}[- ]\d{3}[- ]\d{6,9}(?!\d)", 0.4)],
            context=CTX_BANK,
        ),
        PatternRecognizer(
            supported_entity="ISRAELI_LICENSE_PLATE",
            name="IsraeliLicensePlateRecognizer",
            supported_language=lang,
            # 12-345-67 (7 digits) or 123-45-678 (8 digits)
            patterns=[Pattern("il_plate", r"(?<!\d)(?:\d{2}-\d{3}-\d{2}|\d{3}-\d{2}-\d{3})(?!\d)", 0.4)],
            context=CTX_PLATE,
        ),
        PatternRecognizer(
            supported_entity="ISRAELI_PASSPORT",
            name="IsraeliPassportRecognizer",
            supported_language=lang,
            # A bare 8-digit number is far too generic — only the "דרכון" context lifts it above
            # the threshold (Presidio's context enhancer adds +0.35). The base score must leave
            # real margin: at 0.05 the boosted score landed on exactly 0.4, so raising
            # pii_presidio_score_threshold to 0.41 silently stopped redacting passports. At 0.15
            # the boosted score is 0.50, while an unsupported 8-digit number stays at 0.15 and is
            # still ignored.
            patterns=[Pattern("il_passport", r"(?<!\d)\d{8}(?!\d)", 0.15)],
            context=CTX_PASSPORT,
        ),
    ]

    registry = RecognizerRegistry(supported_languages=[lang])
    for rec in recognizers:
        registry.add_recognizer(rec)

    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=HebrewTokenizerNlpEngine(),
        supported_languages=[lang],
        default_score_threshold=score_threshold,
    )
    anonymizer = AnonymizerEngine()
    operators = {"DEFAULT": OperatorConfig("replace", {"new_value": _TAG_ID})}
    for entity, tag in TAG_FOR_ENTITY.items():
        operators[entity] = OperatorConfig("replace", {"new_value": tag})
    return analyzer, anonymizer, operators


def redact(texts: list[Any], score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> tuple[list, dict[str, int]]:
    """Redact every non-empty text; returns (texts, per-entity counts)."""
    analyzer, anonymizer, operators = build_engines(score_threshold)
    out = list(texts)
    counts: dict[str, int] = {}
    for i, t in enumerate(texts):
        if not t or not str(t).strip():
            continue
        text = str(t)
        results = analyzer.analyze(text=text, language="he", score_threshold=score_threshold)
        if not results:
            continue
        for r in results:
            counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
        out[i] = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
    return out, counts


def probe() -> int:
    """--probe: can Presidio be imported here at all? No engine is built, no text is analysed.

    app/pipeline/pii.py uses this to tell "the files are on disk" apart from "the engine runs".
    """
    try:
        import_presidio()
    except Exception as exc:  # noqa: BLE001
        print(f"presidio not available: {exc}", file=sys.stderr)
        return 3
    return 0


def main() -> int:
    # Answered before stdin is touched — a probe sends no payload.
    if "--probe" in sys.argv[1:]:
        return probe()

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
        threshold = float(data.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_SCORE_THRESHOLD

    try:
        import_presidio()
    except Exception as exc:  # noqa: BLE001
        print(f"presidio not available: {exc}", file=sys.stderr)
        return 3

    out, counts = redact(texts, threshold)
    payload = {"texts": out, "entities": counts, "engine": f"presidio {presidio_version()}"}
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
