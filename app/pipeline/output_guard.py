"""Output guard: redaction placeholders must never reach the caller.

Every message sent to Gemini goes through the PII engines first (llm_providers), so the model
sees "<שם_>", "<ארגון_>" … where identifying details were — and can copy one into its answer.
Above all when the placeholder sits in the instruction text itself: the engines once tagged the
institute's name inside the client-summary template, and the model reproduced the tagged closing
line ("ניתן לפנות שוב <ארגון_> ונשמח לסייע") in every summary. This module guards the output:

  * PLACEHOLDER_RULE, REMINDER, repair_message — fixed prompt text, appended in code by
    summarize._messages (never user-editable);
  * find_tags / has_tags — every surface form the engines or the model produce: angle, square,
    curly and typographic brackets, nesting ("<<ארגון_>_>"), glued prefix letters ("ב<מיקום_>"),
    HTML/markdown escaping, bidi marks, English labels ("<PERSON_1>"), bare remnants ("שם_");
  * sanitize_caller_text — deterministic, Hebrew-aware rewrite without placeholders. Never raises;
  * check_record / legacy_check — the record stored next to the text.

Pure stdlib: no settings, no I/O. Only canonical labels (CANONICAL_LABELS) are ever returned,
recorded or logged — never the matched surface, which in a malformed bracket could hold anything.
"""

from __future__ import annotations

import logging
import re
import time

log = logging.getLogger(__name__)

# Every label find_tags can return. "<אחר_>" stands for a bracket of unknown kind.
CANONICAL_LABELS: tuple[str, ...] = (
    "<שם_>", "<מיקום_>", "<ארגון_>", "<קשר_>", "<מזהה_>", "<תאריך_>", "<מדינה_>", "<MISC__AFF>", "<אחר_>",
)

_H = "א-ת"  # Hebrew letters
_BIDI = "‎‏‪-‮⁦-⁩؜"
_B = "[" + _BIDI + "]*"  # optional bidi control characters around/inside a tag
# Glued prefix letters (ו/ה/ב/כ/ל/מ/ש, up to 3) + an optional hyphen/maqaf: "ב<מיקום_>", "ב-<מיקום_>"
_GLUE = rf"(?<![{_H}A-Za-z0-9])[והבכלמש]{{1,3}}[-־]?"

# ---------- Detection ----------
# The three redaction layers emit <שם_> <מיקום_> <ארגון_> <קשר_> <מזהה_> <תאריך_> <יום_> <חודש_>
# <שנה_> <מדינה_> (HebSafeHarbor, Presidio runner, pii_context) plus <TIME> and <MISC__AFF>
# (Presidio's default replace for entity types HebSafeHarbor leaves unmapped). On top of those,
# the descriptive / translated placeholders a model plausibly invents:
_VOCAB_HE = (
    r"שם(?:[ _]+(?:ה?פרטי|ה?משפחה|ה?מלא|הפונה|הפונים|היועץ|היועצת|הרב|הרבנית|הרופא|הרופאה"
    r"|האישה|האשה|הבעל|בן[ _]הזוג|בת[ _]הזוג|המכון|הארגון|המוסד|המקום|העיר|הילד|הילדה))?"
    r"|מיקום|מקום(?:[ _]+מגורים)?|עיר|יישוב|ישוב|כתובת|ארגון|מוסד|בית[ _]חולים|קופת[ _]חולים"
    r"|קשר|פרטי[ _]קשר|טלפון|מספר[ _]טלפון|טלפון[ _]נייד|נייד|מייל|אימייל|דוא\"?ל|דואל"
    r"|מזהה|מספר[ _]מזהה|תעודת[ _]זהות|ת\"?ז|מספר[ _]זהות|דרכון"
    r"|תאריך|יום|חודש|שנה|מדינה|אדם|פרט[ _]מזהה|פרטים[ _]מזהים|מידע[ _]מוסתר|הוסר|הושמט|חסוי|מוסתר"
)
_VOCAB_EN = (
    r"PERSON|PERS|PER|NAME|FIRST_NAME|LAST_NAME|FULL_NAME|LOCATION|LOC|GPE|CITY|ADDRESS|COUNTRY"
    r"|ORG|ORGANI[SZ]ATION|FAC|PHONE|PHONE_NUMBER|EMAIL|EMAIL_ADDRESS|URL|IP_ADDRESS|CONTACT"
    r"|ID|ID_NUMBER|ISRAELI_ID|ISRAELI_ID_NUMBER|CREDIT_CARD|IBAN_CODE|PASSPORT|ISRAELI_PASSPORT"
    r"|DATE|DATE_TIME|TIME|BIRTH_DATE|MEDICAL_DATE|MISC__AFF|MISC_AFF|REDACTED|PII|NRP"
)
# The English labels that hold "_" — the only upper-case words recognised without brackets
# ("PHONE_NUMBER"); "COVID_19" or "IVF_" in prose are left alone.
_VOCAB_EN_US = "|".join(sorted((w for w in _VOCAB_EN.split("|") if "_" in w), key=len, reverse=True))

_OPEN_ANGLE = r"(?:\\?<|&lt;|＜)"
_CLOSE_ANGLE = r"(?:\\?>|&gt;|＞)"
_SUFFIX = r"(?:_\d*)?"  # numbered placeholders: <שם_1>, <PERSON_2>

# 1) Angle brackets: any short Hebrew body (no digits) or an UPPER_CASE label. Nesting
#    ("<<ארגון_>_>"), spaces and bidi marks inside the brackets are tolerated. The blanks before
#    the closing bracket are bounded: the body may end in blanks too, and two unbounded "\s*"
#    side by side made a lone "<שם" before a long blank run quadratic (0.3 s per 1000 blanks).
_ANGLE = (
    rf"{_OPEN_ANGLE}{{1,3}}{_B}\s*"
    rf"(?P<a>[{_H}][{_H}\"'״׳ _\-]{{0,30}}?|[A-Z][A-Z_]{{1,30}}?){_SUFFIX}"
    rf"\s{{0,4}}{_B}(?:_?\s{{0,4}}{_CLOSE_ANGLE}){{1,3}}"
)
# 2) Typographic angle quotes: only a known label — « » are also Hebrew quotation marks.
_FANCY = rf"[‹«〈⟨《]{_B}\s*(?P<f>{_VOCAB_HE}|{_VOCAB_EN})_?{_SUFFIX}\s*{_B}[›»〉⟩》]"
# 3) Square / curly brackets: a known label ("[שם]", "[שם_הפונה]", "[NAME]", "{{name}}") —
#    "[נושא הפנייה בקצרה]" is template text, "[1]" a reference.
_SQUARE = (
    rf"(?:\\?\[|\{{{{1,2}})\s*(?P<s>{_VOCAB_HE}|{_VOCAB_EN}|name|person)_?{_SUFFIX}\s*(?:\\?\]|\}}{{1,2}})"
)
# 4) Round brackets: only with the underscore that marks a tag ("(שם_)", "(PERSON_1)") —
#    "(IVF)" and "(שם נבדקה)" stay prose.
_ROUND = rf"\(\s*(?P<r>(?:{_VOCAB_HE})_\d*|(?:{_VOCAB_EN})_\d*|(?:{_VOCAB_EN_US})_?\d*)\s*\)"
# 5) Bare remnants: a Hebrew word glued to "_" never occurs in prose ("שם_", "מספר_טלפון"), nor
#    does an upper-case label holding "_" ("PHONE_NUMBER", "MISC__AFF"). Every repeated chunk ends
#    with "_", so a long run of letters cannot be split in exponentially many ways.
_BARE = (
    rf"(?<![{_H}A-Za-z0-9_])(?P<b>[{_H}]{{2,12}}_(?:[{_H}]{{2,12}}_)*(?:[{_H}]{{2,12}})?\d*"
    rf"|(?:{_VOCAB_EN_US})\d*)(?![{_H}A-Za-z0-9])"
)
# 6) The engines' own tag shape (pii_context._TAG_BODY), with no length limit — last, so the
#    forms above (nesting, a trailing "_>") win wherever they match.
_ENGINE = rf"<(?P<e>[{_H}]+_|[A-Z][A-Z_]*)>"

_ANY = re.compile(rf"(?P<g>{_GLUE})?(?:{_ANGLE}|{_FANCY}|{_SQUARE}|{_ROUND}|{_BARE}|{_ENGINE})")

# Canonical label per body, first rule that matches.
_KIND_RULES = [
    ("<קשר_>", re.compile(r"קשר|טלפון|נייד|מייל|דוא|דואל|PHONE|EMAIL|URL|IP_|CONTACT", re.I)),
    ("<מזהה_>", re.compile(r"מזהה|זהות|ת\"?ז|דרכון|\bID\b|_ID|ID_|CREDIT|IBAN|PASSPORT", re.I)),
    ("<תאריך_>", re.compile(r"תאריך|^יום|^חודש|^שנה|DATE|TIME", re.I)),
    ("<ארגון_>", re.compile(r"ארגון|מוסד|חולים|המכון|ORG|FAC", re.I)),
    ("<מיקום_>", re.compile(r"מיקום|מקום|עיר|ישוב|יישוב|כתובת|LOC|GPE|CITY|ADDRESS", re.I)),
    ("<מדינה_>", re.compile(r"מדינה|COUNTRY", re.I)),
    ("<שם_>", re.compile(r"שם|NAME|PER|אדם", re.I)),
    ("<MISC__AFF>", re.compile(r"MISC", re.I)),
]
_CODE = {"<שם_>": "PER", "<מיקום_>": "LOC", "<ארגון_>": "ORG", "<קשר_>": "CON", "<מזהה_>": "IDN",
         "<תאריך_>": "DAT", "<מדינה_>": "LOC", "<MISC__AFF>": "OTH", "<אחר_>": "OTH"}


def _body(m: re.Match) -> str:
    body = next((g for g in (m.group(k) for k in ("a", "f", "s", "r", "b", "e")) if g), "")
    return body.strip(" _")


def _kind(m: re.Match) -> str:
    body = _body(m)
    for label, rx in _KIND_RULES:
        if rx.search(body):
            return label
    return "<אחר_>"


def find_tags(text: object) -> list[str]:
    """Canonical label of every placeholder in `text`, in order (duplicates kept).

    [] for a non-string or empty input. Never raises.
    """
    if not isinstance(text, str) or not text:
        return []
    return [_kind(m) for m in _ANY.finditer(text)]


def has_tags(text: object) -> bool:
    return isinstance(text, str) and bool(text) and _ANY.search(text) is not None


# ---------- Sanitisation ----------
# Every placeholder is first rewritten to a sentinel (private-use characters) that keeps its
# glued prefix letters in place: "ב<מיקום_>" -> "ב" + S0 + "LOC" + S1. The grammar rules below
# are written against the sentinels, so they hold for every surface variant at once.
S0, S1 = "", ""


def _S(*codes: str) -> str:
    return S0 + "(?:" + "|".join(codes) + ")" + S1


_PER, _LOC, _ORG, _CON, _IDN, _DAT = (_S(c) for c in ("PER", "LOC", "ORG", "CON", "IDN", "DAT"))
_ALLS = _S("PER", "LOC", "ORG", "CON", "IDN", "DAT", "OTH")
_NB = rf"(?<![{_H}])"
_NA = rf"(?![{_H}])"
_P = "[והבכלמש]{0,3}"
_PERS = rf"(?:{_P}){_PER}(?:\s*(?:,|ו)\s*(?:{_P}){_PER})*"  # one or more names: "<שם_> ו<שם_>"

# A placeholder that names a role ("[שם הפונה]", "<שם_הרופא>") is replaced by that role.
_ROLES = {"הפונה": "פונה", "הפונים": "פונים", "היועץ": "יועץ", "היועצת": "יועצת", "הרב": "רב",
          "הרבנית": "רבנית", "הרופא": "רופא/ה", "הרופאה": "רופאה", "האישה": "אישה", "האשה": "אישה",
          "הבעל": "בעל", "הילד": "ילד", "הילדה": "ילדה"}
_ROLE_RX = re.compile(r"^שם[ _]+(?:(?P<r>" + "|".join(_ROLES) + r")|(?P<z>ב[ןת])[ _]הזוג)$")


def _def(prefix: str, noun: str) -> str:
    """Definite noun after prefix letters: ב/ל/כ absorb the article ("לרופא/ה"), ו/מ/ש keep it."""
    if prefix and prefix[-1] in "בלכ":
        return prefix + noun
    return prefix + "ה" + noun


def _to_sentinels(text: str) -> tuple[str, list[str]]:
    labels: list[str] = []

    def sub(m: re.Match) -> str:
        label = _kind(m)
        labels.append(label)
        pre = re.sub("[-־]", "", m.group("g") or "")
        role = _ROLE_RX.match(_body(m))
        if role:
            if role.group("z"):  # "בן הזוג": the article sits on the second word
                return pre + role.group("z") + " הזוג"
            return _def(pre, _ROLES[role.group("r")])
        return pre + S0 + _CODE[label] + S1

    return _ANY.sub(sub, text), labels


_TITLES = {"הרב": "רב", "רב": "רב", "הרבנית": "רבנית", "רבנית": "רבנית",
           'ד"ר': "רופא/ה", "ד״ר": "רופא/ה", "דר'": "רופא/ה", "דוקטור": "רופא/ה",
           "פרופ'": "רופא/ה", "פרופ׳": "רופא/ה", "פרופסור": "רופא/ה",
           "גב'": "", "גב׳": "", "גברת": "", "מר": ""}
_TITLE_RX = "|".join(re.escape(t) for t in sorted(_TITLES, key=len, reverse=True))
_CONSTRUCT = {"מרפאת": "מרפאה", "מכון": "מכון", "מרכז": "מרכז", "יחידת": "יחידה", "מחלקת": "מחלקה",
              "עמותת": "עמותה", "סניף": "סניף", "ארגון": "ארגון", "מוסד": "מוסד"}
_CONSTRUCT_RX = "|".join(_CONSTRUCT)
_GREG = "ינואר|פברואר|מרץ|מרס|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר"
_HEBM = "תשרי|חשוון|חשון|מרחשוון|כסלו|טבת|שבט|אדר|ניסן|אייר|סיון|סיוון|תמוז|אב|אלול"

_RULES: list[tuple[re.Pattern, object]] = []


def _rule(pattern: str, repl) -> None:
    _RULES.append((re.compile(pattern), repl))


def _title(m: re.Match) -> str:
    role = _TITLES[m.group("t")]
    return _def(m.group("p"), role) if role else ""


# R1  The template's closing line — "ניתן לפנות שוב <ארגון_> ונשמח לסייע". The speaker ("נשמח") is
#     the institute itself, so "אלינו" is right whatever organisation the tag stood for, and it
#     never re-inserts a name. Only inside a sentence that goes on to "נשמח"; elsewhere an
#     organisation tag may be a hospital and must not become "us".
_rule(rf"(?P<v>לפנות|לחזור|לשוב\s+ולפנות)(?P<s>\s+שוב)?\s+(?:אל\s+|עם\s+)?(?:{_P}){_ORG}(?=[^.\n]{{0,80}}נשמח)",
      lambda m: m.group("v") + " אלינו" + (m.group("s") or ""))
_rule(rf"ליצור\s+קשר(?P<s>\s+שוב)?\s+(?:עם\s+)?(?:{_P}){_ORG}(?=[^.\n]{{0,80}}נשמח)",
      lambda m: "ליצור איתנו קשר" + (m.group("s") or ""))
_rule(rf"{_NB}(?P<p>[ובלמש]?)צוות\s+(?:{_P}){_ORG}", lambda m: _def(m.group("p"), "צוות"))
# R2  "בבית החולים <ארגון_>" -> "בבית החולים"; "במרפאת <ארגון_>" -> "במרפאה"
_rule(rf"{_NB}(?P<p>[ובלמש]{{0,2}})(?P<n>בית|קופת)\s+ה?חולים\s+(?:{_P})(?:{_ORG}|{_LOC}|{_PER})(?:\s+(?:{_P}){_LOC})?",
      lambda m: m.group("p") + m.group("n") + " החולים")
_rule(rf"{_NB}(?P<p>[ובלמש]{{0,2}})(?P<n>{_CONSTRUCT_RX})\s+(?:{_P})(?:{_ORG}|{_LOC}|{_PER})(?:\s+(?:{_P}){_LOC})?",
      lambda m: _def(m.group("p"), _CONSTRUCT[m.group("n")]))
# R3  Greetings and vocatives: "שלום <שם_>," -> "שלום,"   "<שם_> שלום," -> "שלום,"   "לכבוד <שם_>" -> line dropped
#     "שלום רב" / "בכבוד רב" are greetings, not the title "רב" (R4 would make them "שלום הרב")
_rule(rf"{_NB}(?P<g>שלום(?:\s+(?:וברכה|רב))?|ברכה|היי|בוקר\s+טוב|ערב\s+טוב|בכבוד\s+רב|לכבוד)\s+{_PERS}"
      rf"(?:\s+(?:היקר|היקרה|היקרים|היקרות|ה?נכבד|ה?נכבדה){_NA})?",
      lambda m: "" if m.group("g") == "לכבוד" else m.group("g"))
_rule(rf"^(?P<i>\s*){_PERS}\s*(?:היקר|היקרה|היקרים|היקרות)?\s*,?\s*(?=שלום)", r"\g<i>")
_rule(rf"(?P<b>^\s*|[.!?]\s+){_PERS}\s*,\s*", r"\g<b>")
# R4  Titles keep their role: "הרב <שם_>" -> "הרב", "לד\"ר <שם_>" -> "לרופא/ה", "גב' <שם_>" -> dropped
_rule(rf"{_NB}(?P<p>[ובלמש]{{0,2}})(?P<t>{_TITLE_RX})\s+{_PERS}", _title)
# R5  A person as a complement goes with its preposition: "של <שם_>", "עם <שם_>", "את <שם_>"
_rule(rf"\s*{_NB}(?:של|עם|את|אצל|מול|בשם|הנקרא|הנקראת|ע\"י|על\s+ידי)\s+{_PERS}", "")
_rule(rf"\s*{_NB}(?:דרך|באמצעות|מטעם|מול|אצל|עם|של)\s+(?:{_P})(?:{_ORG}|{_LOC})(?:\s+(?:{_P})(?:{_ORG}|{_LOC}))*", "")
# R6  Places: "באזור <מיקום_>" -> "באזור מגוריכם"; "בעיר <מיקום_>" goes with its noun
_rule(rf"{_NB}(?P<p>[בלמ]?)(?:אזור|איזור)\s+(?:{_P}){_LOC}", lambda m: m.group("p") + "אזור מגוריכם")
_rule(rf"\s*{_NB}[ובלמ]{{0,2}}(?:העיר|עיר|היישוב|יישוב|הישוב|ישוב|שכונת|רחוב|המושב|הקיבוץ)\s+(?:{_P}){_LOC}", "")
# R7  Dates: "בתאריך <תאריך_>", "<יום_>/03/2020", "<יום_> במרץ 2020" — the whole expression goes.
#     Only "/" takes blanks before a number: "<תאריך_>. 3 מחזורים" ends a sentence and
#     "<תאריך_> - 3 מחזורים" is a clause, neither is the rest of the date.
_rule(rf"\s*{_NB}(?:{_P}(?:תאריך|יום|מועד)\s+)?(?:(?:עד|מאז|לקראת|סביב|בסביבות|בשעה|לפני|אחרי)\s+|החל\s+מ)?(?:{_P}){_DAT}"
      rf"(?:\s*/\s*(?:{_DAT}|\d{{1,4}})|\s*-\s*{_DAT}|[.\-](?:{_DAT}|\d{{1,4}})){{0,2}}"
      rf"(?:\s+[בל]?(?:{_GREG}|{_HEBM}){_NA})?"
      rf"(?:\s*,?\s*(?:{_DAT}|(?:19|20)\d\d(?!\d)))?", "")
# R8  Contact / ID: "במספר <קשר_>" goes whole; "בטלפון <קשר_>" keeps "בטלפון"
_rule(rf"\s*{_NB}{_P}(?:מספר|מס'|כתובת)(?:\s+ה?(?:טלפון|נייד|מייל|אימייל|דוא\"ל|תעודת\s+(?:ה)?זהות|זהות|ת\"ז|דרכון))?\s*:?\s*(?:{_CON}|{_IDN})", "")
_rule(rf"\s*{_NB}{_P}(?:תעודת\s+זהות|ת\"ז|ת\.ז\.?|דרכון)\s*:?\s*{_IDN}", "")
_rule(rf"(?P<n>{_NB}{_P}(?:טלפון|נייד|מייל|אימייל|דוא\"ל|וואטסאפ|ווטסאפ))\s*:?\s*{_CON}", r"\g<n>")
# R9a A first conjunct with the same preposition as the second: "עסקנו ב<X> ובשאלת" -> "עסקנו בשאלת"
_rule(rf"[ \t]*{_NB}(?P<p>[בלמ]){_ALLS}\s+ו(?=(?P=p))", " ")
# R9  Anything else: the placeholder together with its glued prefix letters
_rule(rf"[ \t]*{_NB}{_P}{_ALLS}", "")

_CLEANUP = [
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"[ \t]+([,.;:!?])"), r"\1"),
    (re.compile(r"\(\s*\)|\[\s*\]|\"\s*\"|״\s*״|\*\*\s*\*\*|``"), ""),
    # a preposition or "ו" left hanging before punctuation: "דיברנו עם." -> "דיברנו."
    (re.compile(rf"(?<![{_H}])(?:של|עם|את|אל|על|מול|אצל|בשם|ע\"י|ו)[ \t]*(?=[,.;:!?]|$)"), ""),
    (re.compile(r"([,;:])[ \t]*([,.;:!?])"), r"\2"),
    (re.compile(r"[ \t]+[—–-][ \t]*(?=[.,;]|$)"), ""),
    (re.compile(r"^[ \t]*[,;:.]+[ \t]*"), ""),
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"[ \t]+([,.;:!?])"), r"\1"),
]
_HAS_CONTENT = re.compile(rf"[{_H}A-Za-z0-9]")
_SENTINEL = re.compile(_ALLS)
# Blank runs in a tagged line are collapsed before the rules (the cleanup collapses them anyway):
# several rules put "\s*" next to "\s*", which over a long run of blanks — degenerate model
# output — backtracks polynomially (a 2000-blank run took about 20 s).
_BLANK_RUN = re.compile(r"\s{2,}")
# Template restore (R0): a line split into at most this many literal parts — each gap is a lazy
# ".{1,40}?", so more gaps would make a failing match expensive.
_TEMPLATE_MAX_PARTS = 4
_TEMPLATE_MIN_LITERAL = 25


def _cleanup(line: str) -> str:
    for rx, repl in _CLEANUP:
        line = rx.sub(repl, line)
    return line.strip(" \t")


def _strip_tags(text: str) -> str:
    """Plain removal, repeated until nothing matches (removing one tag can complete another,
    "<ש<שם_>ם_>"). Terminates: every pass that changes the text makes it shorter."""
    prev = None
    while prev != text:
        prev, text = text, _ANY.sub("", text)
    return text


def _last_resort(text: str) -> str:
    out = []
    for line in text.split("\n"):
        stripped = _strip_tags(line)
        if stripped == line:
            out.append(line)
            continue
        stripped = _cleanup(stripped)
        if _HAS_CONTENT.search(stripped):
            out.append(stripped)
    # The cleanup can complete a tag ("<()שם_>"), and a tag can span a line break.
    return _strip_tags("\n".join(out))


def _template_line(line: str, template: str | None) -> str | None:
    """R0: a tagged line that is a verbatim copy of a fixed line of the (raw, unredacted) prompt
    template — the tag being that template's own words, redacted on the way to the model — gets
    the template line back. Template lines with fill-in brackets ("[נושא הפנייה]") never qualify."""
    if not template:
        return None
    parts = re.split(rf"{_NB}{_P}{_ALLS}", line.strip())
    if not 2 <= len(parts) <= _TEMPLATE_MAX_PARTS or sum(len(p) for p in parts) < _TEMPLATE_MIN_LITERAL:
        return None
    rx = re.compile(r"\s*" + r".{1,40}?".join(map(re.escape, parts)) + r"\s*")
    for cand in template.split("\n"):
        cand = cand.strip()
        if cand and "[" not in cand and rx.fullmatch(cand) and not has_tags(cand):
            return cand
    return None


def sanitize_caller_text(text, template: str | None = None) -> tuple[str, list[str]]:
    """Rewrite `text` without any placeholder. Returns (clean_text, canonical labels removed).

    template: the raw prompt the text was generated from; a line copied from it is restored
    verbatim (R0) before the grammar rules run. Only lines that held a placeholder are touched;
    a line left with no letter or digit is dropped. Idempotent. Never raises: on an internal error
    it falls back to plain removal, so the job still yields a text.
    """
    if not isinstance(text, str) or not text:
        return text, []
    try:
        work, labels = _to_sentinels(text)
        if not labels:
            return text, []
        out_lines = []
        for line in work.split("\n"):
            if S0 not in line:
                out_lines.append(line)
                continue
            restored = _template_line(line, template)
            if restored is not None:
                out_lines.append(restored)
                continue
            line = _BLANK_RUN.sub(" ", line)
            for rx, repl in _RULES:
                line = rx.sub(repl, line)
            # A sentinel no rule took (glued to a word: "מספרים<שם_>") goes with its code
            line = _cleanup(_SENTINEL.sub("", line).replace(S0, "").replace(S1, ""))
            if not _HAS_CONTENT.search(line):
                continue  # the line held nothing but the placeholder (a signature, "לכבוד <שם_>")
            out_lines.append(line)
        clean = "\n".join(out_lines)
        if has_tags(clean):  # defence in depth — should never trigger
            clean = _last_resort(clean)
        return clean, labels
    except Exception as exc:  # noqa: BLE001 — a summary must still be produced
        # The exception type only: a message could quote the text.
        log.warning("Placeholder sanitisation failed (%s); falling back to plain removal", type(exc).__name__)
        try:
            return _last_resort(text), find_tags(text)
        except Exception:  # noqa: BLE001
            return text.translate(_NO_BRACKETS), ["<אחר_>"]


_NO_BRACKETS = str.maketrans("", "", "<>＜＞[]{}‹›«»〈〉⟨⟩《》_")


# ---------- Fixed prompt text (in code, never user-editable) ----------
# It passes through the PII engines like every other message, so it is worded to survive them:
# no proper noun, none of the words the engines are known to over-tag ("שלום", "בלי", "כאילו",
# first-name-like words, the institute's name; the raw engines also tag "מוסד"). Checked through
# the real HebSafeHarbor + Presidio chain, without the restore step: every word comes back
# untouched. The example tags in it are harmless: a tag the engines tag again is collapsed back
# by pii_context.unnest_tags.
PLACEHOLDER_RULE = (
    "כלל פרטיות קבוע: לפני שהתמלול הגיע אליך הוסרו ממנו פרטים מזהים, וכל פרט כזה הוחלף בתגית הסתרה, "
    "למשל <שם_>, <מיקום_>, <ארגון_>, <קשר_>, <מזהה_>, <תאריך_>, <יום_>, <חודש_>, <שנה_>, <TIME>. "
    "אסור להעתיק תגית הסתרה לטקסט שאתה כותב, בשום צורה: לא בסוגריים משולשים, לא בסוגריים מרובעים, "
    "לא ללא סוגריים, ולא צמודה לאות שימוש (כמו ב<מיקום_> או ל<ארגון_>). אל תנחש מה הוסתר. "
    "במקום תגית כתוב ניסוח כללי או השמט את הפרט: אדם יתואר לפי תפקידו (הפונה, בן הזוג, הרופא המטפל) "
    "והפנייה אל הפונה תהיה ללא שם; מקומות, מספרי טלפון, כתובות מייל, מספרי זהות ותאריכים יושמטו; "
    "ארגון, מרפאה או בית חולים יתוארו כ\"הגורם המטפל\" או יושמטו. "
    "אם תגית מופיעה בתוך נוסח קבוע שאתה מתבקש להחזיר, השמט אותה ונסח את המשפט מחדש באופן טבעי."
)
REMINDER = "תזכורת: אסור שתגית הסתרה כמו <שם_> או <ארגון_> תופיע בתשובה."


def repair_message(labels: list[str]) -> dict[str, str]:
    """The user turn that asks the model to rewrite its previous (tagged) answer."""
    shown = ", ".join(dict.fromkeys(labels)) or "<שם_>"
    return {"role": "user", "content": (
        "בתשובה הקודמת הופיעו תגיות הסתרה (" + shown + "). כתוב את אותה תשובה מחדש, באותו מבנה "
        "ובאותו תוכן, ללא אף תגית הסתרה: החלף כל תגית בניסוח כללי או השמט אותה, כך שהמשפטים "
        "יישארו תקינים וטבעיים. החזר רק את הטקסט המתוקן."
    )}


# ---------- The check record (stored next to the text: labels only, never text) ----------

def check_record(
    *,
    tags_found: list[str],
    sanitized: bool,
    attempts: int | None,
    residual: list[str],
    legacy: bool = False,
) -> dict:
    """tags_found: every label seen in any model answer; residual: labels still in the final
    text (should always be empty); attempts: model calls made (None for a legacy check)."""
    return {
        "version": 1,
        "tags_found": sorted(set(tags_found)),
        "sanitized": bool(sanitized),
        "attempts": attempts,
        "residual": sorted(set(residual)),
        "legacy": bool(legacy),
        "checked_at": time.time(),
    }


def legacy_check(text: str | None) -> dict | None:
    """Check of a text produced before the guard existed (computed on the fly, not persisted)."""
    if not isinstance(text, str) or not text.strip():
        return None
    tags = find_tags(text)
    return check_record(tags_found=tags, sanitized=False, attempts=None, residual=tags, legacy=True)
