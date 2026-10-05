"""Conversation-level PII pass — what per-segment NER structurally cannot see.

HebSafeHarbor and Presidio are handed one transcript segment at a time. In a counselling call
that loses exactly the details that matter most:

  * answers are one-word turns. A first name said alone on its line gives NER nothing to go
    on — and ASR may spell it with the wrong letter (ס for ש), so no name list contains it;
  * numbers are dictated in pieces across turns: "050." / "414 -6488." never form a phone
    pattern inside any single segment;
  * e-mail addresses are spoken: spelled letters, digits and "שטרודל" spread over several
    turns, never a single x@y.com string;
  * a name the engines do catch in one place is left untouched where it recurs.

This pass runs over the whole ordered conversation — after the engines for the de-identified
copy (pii.deidentify_result) and in front of them for the Gemini prompt
(summarize.build_transcript_text). It is deterministic, stdlib-only and runs in the
application process: no model, nothing for Windows Application Control to block.

contextual_redact only ever removes text. The one thing that puts text back is
restore_safe_terms, which undoes the engines' over-redaction on a closed list — drug and device
names, function words, the health funds, countries, the institute's own name — and never a word
that can be a first name. It runs right after the engines (pii._run_pii), before this pass learns
names from their output.

It is deliberately narrow about *where* it acts: inside a request for personal details
("מה השם הפרטי של אשתך?", "המייל שלה?") it removes the answer and every number; outside one it
removes only what has the unmistakable shape of a phone/ID number, a spelled-out address or a
self-introduction. Ordinary counselling content — a test's or syndrome's name, ages, weeks,
risk ratios, gene names — must reach the summariser intact.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Sequence

TAG_NAME = "<שם_>"
TAG_CONTACT = "<קשר_>"
TAG_ID = "<מזהה_>"

_H = "א-ת"  # Hebrew letters; ASR emits no niqqud
_NLB = rf"(?<![{_H}A-Za-z])"
_NLA = rf"(?![{_H}A-Za-z])"
_GERESH = "'׳"
_PREFIXES = "והבכלמש"
_P = f"[{_PREFIXES}]{{0,2}}"  # prefix letters glued to a Hebrew word

# Only the engines' own tag vocabulary counts as an existing redaction: a Hebrew word plus "_"
# (<שם_>, <קשר_>, <מיקום_> …) or an upper-case label (<MISC__AFF>). A bare "<" … ">" in a
# summary ("פחות מ < 5% … >") must not shield what lies between.
_TAG_BODY = rf"<(?:[{_H}]+_|[A-Z][A-Z_]*)>"
_SPLIT_TAGS = re.compile(rf"({_TAG_BODY})")
_IS_TAG = re.compile(_TAG_BODY)
_TOKEN = re.compile(rf"{_TAG_BODY}|[{_H}A-Za-z0-9]+(?:[{_GERESH}\"״][{_H}A-Za-z]+)*")
_WORD = re.compile(rf"[{_H}A-Za-z]+(?:[{_GERESH}\"״][{_H}A-Za-z]+)*")


def _w(alternatives: str) -> str:
    """A whole word, optionally carrying glued prefix letters (ו/ה/ב/ל/מ/ש/כ)."""
    return rf"{_NLB}{_P}(?:{alternatives}){_NLA}"


# ---------- requests for a personal detail ----------
# A request needs a person or a possessive: "המייל שלה", "השם של אשתך". The bare word is not
# enough — "המיילדת", "אשלח במייל", "מחשב נייד", "מה השם של הבדיקה?" are not requests.

_POSS = r"(?:שלך|שלכם|שלכן|שלה|שלו|שלי|שלנו)"
_PERSON = (
    r"(?:אשתך|בעלך|אשתו|בעלה|אשתי|בעלי|הבעל|האישה|האשה|הבן|הבת|הילד|הילדה|האמא|האבא|אמא|אבא"
    r"|הרופא|הרופאה|הרב|הרבנית|היועץ|היועצת|המטפל|המטפלת|האחות|הבחור|הבחורה|החתן|הכלה|הפונה)"
)

# Built outside the f-strings below: Python 3.10 allows no quote/backslash inside {…}.
_DOAL = _w('דוא"?ל')

_CUES: dict[str, re.Pattern] = {
    "id": re.compile(
        rf"תעודת\s+זהות|{_NLB}ו?ת\s*[.\"״{_GERESH}]?\s*ז(?:\s*\.)?{_NLA}"
        rf"|{_NLB}מס[{_GERESH}.]?\s*(?:ה)?זהות{_NLA}|מספר\s+(?:ה)?זהות|{_w('זהות|דרכון')}"
    ),
    "email": re.compile(
        rf"{_w('(?:אי)?מייל|email|e-mail')}\s+{_POSS}"
        rf"|מה\s+(?:ה)?(?:אי)?מייל|כתובת\s+(?:ה)?(?:אי)?מייל|{_DOAL}"
        rf"|(?:יש|אין)\s+(?:ל[{_H}]+\s+)?(?:גם\s+)?(?:אי)?מייל{_NLA}"
        rf"|{_w('(?:אי)?מייל')}\s*\?"
    ),
    "phone": re.compile(
        rf"{_w('טלפון|נייד|פלאפון|סלולרי|סלולארי')}\s+{_POSS}"
        rf"|מספר\s+(?:ה)?(?:טלפון|נייד|פלאפון)|{_w('המספר')}\s+(?:ה?טלפון\s+)?{_POSS}"
        rf"|טלפון\s+נייד|לחזור\s+(?:אלי|אליי|אליך|אלייך|אליכם|אלייכם)|ליצור\s+קשר"
        rf"|{_w('וואטסאפ|ואטסאפ|ווטסאפ|whatsapp')}"
    ),
    "name": re.compile(
        rf"{_NLB}{_P}שם\s+(?:ה?פרטי|ה?משפחה|המלא|מלא){_NLA}"
        rf"|{_NLB}[והב]?(?:ה)?שם\s+{_POSS}"
        rf"|{_NLB}[ו]?(?:ה)?שם\s+(?:\S+\s+)?של\s+{_PERSON}"
        rf"|איך\s+קוראים|עם\s+מי\s+(?:אני\s+)?(?:מדבר|מדברת)|{_NLB}מי\s+(?:מדבר|מדברת){_NLA}"
        rf"|{_w('שמך|שמכם|שמכן')}|{_NLB}(?:ה)?שם\s+(?:מלא\s+)?לברכה"
        rf"|{_NLB}(?:ה)?שם\s+ו(?:ה)?(?:טלפון|מספר|מייל|תעודת)"
    ),
    "address": re.compile(
        rf"{_w('כתובת')}\s+(?:ה?מגורים\s+)?{_POSS}|כתובת\s+(?:ה)?מגורים|מקום\s+(?:ה)?מגורים"
        rf"|איפה\s+(?:אתם\s+|את\s+|אתה\s+)?(?:גר|גרה|גרים|גרות){_NLA}|באיזו?\s+(?:עיר|יישוב|ישוב)"
    ),
}
# The same requests in English — calls from immigrants switch languages mid-sentence.
_CUES_EN: dict[str, re.Pattern] = {
    "email": re.compile(r"\b(?:your|my|his|her)\s+e-?\s?mail\b|\be-?\s?mail\s+address\b", re.I),
    "phone": re.compile(r"\b(?:your|my|his|her)\s+(?:phone|cell|mobile)(?:\s+number)?\b|\bphone\s+number\b"
                        r"|\bcall\s+you\s+back\b", re.I),
    "name": re.compile(r"\bwhat(?:'s|\s+is)\s+(?:your|his|her)\s+(?:first\s+|last\s+|full\s+)?name\b"
                       r"|\b(?:your|his|her)\s+(?:first\s+|last\s+|full\s+)?name\b|\bwho\s+am\s+i\s+speaking\b", re.I),
    "address": re.compile(r"\b(?:your|my)\s+(?:home\s+)?address\b|\bwhere\s+do\s+you\s+live\b", re.I),
}
_WORD_ANSWER_KINDS = frozenset({"name", "email", "address"})

_WINDOW = 8          # turns after a request that may hold its answer
_LONG_TURN = 20      # a turn this long means the conversation has moved on
_ANSWER_SCAN = 3     # an answer comes within this many turns of the request
_MAX_ANSWER_WORDS = 6
_PHONE_RUN = 7       # digits dictated across adjacent turns that make a phone/ID number

# Words that are never the personal detail itself.
# ("ברוך" is deliberately absent: it is a first name. "ברוך השם" is kept by _NO_PROPAGATE and
# by the request cues needing a person, not by pretending the word can never be a name.)
_STOP = frozenset(
    """
    כן לא אוקיי אוקי אוק בסדר גמור תודה רבה אה אהה אההה אמ אממ הממ נו יופי טוב נכון בדיוק רגע
    שנייה שניה מה מי איך למה מתי איפה אז גם זה זאת זו זהו שלה שלו שלי שלנו שלכם שלך שלהם יש אין
    לה לו לי לך לנו לכם את אתה אתם אתן אני הוא היא אנחנו הם הן עם על אל של עוד כבר רק מאוד וואו
    ממש סבבה בבקשה סליחה שלום היי הלו ביי להתראות כאילו פשוט אולי בטח ברור אמן השם מייל
    אימייל טלפון נייד מספר שם פרטי משפחה כתובת דוקטור דר ד"ר פרופסור הרב רב רבנית גברת מר אדון
    רופא רופאה אחות אשתי בעלי אשתך בעלך אישה איש בעל אשה אבא אמא ילד ילדה דוט נקודה קום שטרודל
    בן בת עכשיו מחר היום אתמול פה כאן כמובן מעולה הבנתי באמת עדיין שוב
    בכל לכל מכל שכל וכל הכל הכול בלי שמה שמו
    """.split()
)
# A short reply holding one of these is a sentence, not a dictated detail ("אין לי מייל").
_SENTENCE_MARKERS = frozenset("יש אין מאוד גם רק כבר היה היתה הייתה היו".split())
# Leading pronouns that may introduce a dictated name ("אני שלומית", "היא יהודית") …
_LEAD_PRONOUNS = frozenset("אני אתה את הוא היא אנחנו אתם הם זה זאת זו".split())
# … unless what follows is a verb about the call itself ("אני מדבר", "היא שומעת").
_CALL_VERBS = frozenset("מדבר מדברת שומע שומעת מתקשר מתקשרת מחכה פה כאן".split())
# Counselling vocabulary a short reply may well consist of, and that is never a name.
_COMMON = frozenset(
    """
    שבוע שבועות בדיקה בדיקות גנטי גנטית גנטיות סקר מורחב צריך לבדוק הריון נשא נשאית תקין תקינה
    תוצאה תוצאות טיפול טיפולים מרפאה בית חולים זרע ביוץ מחזור הפריה תרופה תרופות כדורים
    """.split()
)
# First names that are also everyday Hebrew words: removed where they are given as an answer and
# within that exchange, but not hunted across the whole conversation ("ברוך השם", "חיים",
# "שני בני הזוג" would otherwise vanish).
_NO_PROPAGATE = frozenset(
    """
    ברוך חיים אור שמחה יפה דבורה צפורה ציפורה אריה זאב דב צבי יונה רון נחמה מלכה שלום אמת תמר
    עדן שקד הדר אביב מתנה נס טובה זהבה פנינה ישראל אמונה שירה רינה גילה אורה ברכה שושנה ורד טל
    גל חן שחר רז שני אלה גיל עמית הדס מעיין רעות אושר עדי מור ים שיר נוי חנה יעל
    """.split()
)

# ---------- self-introductions ----------
# Anchored phrases that name a person outright, with or without a question being asked.
_NAME_WORD = rf"[{_H}A-Za-z]{{2,}}(?:[{_GERESH}][{_H}]+)?"
_TITLE = r"(?:ה?רב|ה?רבנית|ד\"?ר|דוקטור|פרופסור)\s+"
_SELF_INTRO = [
    re.compile(rf"{_NLB}(?:קוראים|קורים)\s+(?:לי|לו|לה|לנו|לאשתי|לבעלי|לאשתו|לבעלה|לבן\s+שלי|לבת\s+שלי)"
               rf"\s+(?P<n>{_NAME_WORD}(?:\s+{_NAME_WORD})?)"),
    re.compile(rf"{_NLB}(?:לי|לו|לה|לנו|לאשתי|לבעלי|לאשתו|לבעלה)\s+(?:קוראים|קורים)"
               rf"\s+(?P<n>{_NAME_WORD}(?:\s+{_NAME_WORD})?)"),
    # Only "שמי": in speech "שמה"/"שמו" are as often "there" ("שנמצאת שמה בכל התהליך") as "her/his
    # name", and the third person is covered by "קוראים לה X" / "השם שלה X".
    re.compile(rf"{_NLB}שמי\s+(?:הוא\s+|היא\s+|זה\s+)?(?P<n>{_NAME_WORD}(?:\s+{_NAME_WORD})?)"),
    re.compile(rf"{_NLB}(?:ה)?שם\s+(?:שלי|שלו|שלה)\s+(?:הוא\s+|היא\s+|זה\s+)?"
               rf"(?P<n>{_NAME_WORD}(?:\s+{_NAME_WORD})?)"),
    re.compile(rf"{_NLB}(?:זה|זאת|זו|כאן|פה)\s+(?:{_TITLE})?(?P<n>{_NAME_WORD})\s+(?:מדבר|מדברת){_NLA}"),
    re.compile(rf"(?:^|[.,!?]\s*)(?:שלום\s*,?\s*)?(?:מדבר|מדברת)\s+(?:{_TITLE})?(?P<n>{_NAME_WORD}){_NLA}"),
    # English, as immigrants say it
    re.compile(r"\bmy\s+(?:first\s+|last\s+|full\s+)?name\s+is\s+(?P<n>[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})?)", re.I),
    re.compile(r"\bthis\s+is\s+(?P<n>[A-Z][a-z]+)\s+(?:speaking|calling|here)\b"),
    re.compile(r"\bcall\s+me\s+(?P<n>[A-Za-z]{2,})\b", re.I),
]
# English words that end a captured name ("my name is Sarah and …").
_EN_STOP = frozenset(
    "is the a an and not sure here just so from calling speaking fine good okay ok yes no sorry glad "
    "happy very really but or with about i i'm im you back later tomorrow when if again".split()
)

# ---------- numbers ----------
# Decimals stay one value; a sentence-ending period is not part of a number ("052.").
_NUM = re.compile(r"(?<!\d)(?<!\d\.)\d+(?:\.\d+)?(?!\d)(?!\.\d)")
# Groups separated only by spaces, hyphens or commas read as one dictated number.
_NUM_SEQ = re.compile(r"(?<!\d)(?<!\d\.)\d+(?:[ \t\-,]+\d+)+(?!\d)(?!\.\d)")
_DIGIT_WORDS = "אפס|אחת|אחד|שתיים|שתים|שניים|שלוש|שלש|ארבע|חמש|שש|שבע|שמונה|תשע"
_DW_RUN = re.compile(rf"{_NLB}ו?(?:{_DIGIT_WORDS}){_NLA}(?:[\s,\-]+ו?(?:{_DIGIT_WORDS}){_NLA})*")
_DW_ONE = re.compile(rf"(?:{_DIGIT_WORDS})")
_UNIT_BEFORE = re.compile(
    rf"(?:{_NLB}{_P}(?:גיל|בת|בן|שבוע|שבועות|שנה|שנים|שנת|חודש|חודשים|יום|ימים|פי|סבב|מחזור|ניסיון"
    rf"|רמה|רמת|ערך|תור|חדר|קומה|מקום){_NLA}|[A-Za-z]{{2,}})\s*-?\s*$"
)
_UNIT_AFTER = re.compile(
    rf"\s*-?\s*(?:%|{_NLB}{_P}(?:אחוז|אחוזים|שקל|שקלים|ש\"ח|מ\"ג|מיליגרם|גרם|ק\"ג|ס\"מ|יחידות|שבוע"
    rf"|שבועות|שנה|שנים|ימים|יום|חודש|חודשים|פעמים|ביציות|עוברים|זרעונים|פגישות|ילדים){_NLA})"
)
_PERCENT = re.compile(r"\s*(?:%|אחוז)")
_RATIO = re.compile(r"(?<!\d)(?<!\d\.)\d+\s+ל\s*-?\s*\d+(?!\d)(?!\.\d)")   # "1 ל -350"
_YEAR = re.compile(r"(?:19|20)\d\d")

# ---------- dates ----------
# A specific date in a caller's life (a birth, a consultation, a procedure) identifies; a year on
# its own does not. Hebrew-calendar dates ("ט' תמוז תשפ״ד", "בכ"ב בטבת") and spelled Gregorian ones
# ("14 במרץ 2024") are what the engines miss in speech. Day + month of a fixed holiday or fast is
# not a personal date when no year is given ("ט' באב", "ט"ו בשבט") — halachic talk is full of them.
TAG_DATE = "<תאריך_>"
_HEB_MONTHS = r"תשרי|מרחשון|מרחשוון|חשון|חשוון|כסלו|כסליו|טבת|שבט|אדר(?:\s+[אב][׳'])?|ניסן|אייר|סיון|סיוון|תמוז|אב|אלול"
_HEB_DAY = r"(?:[יכ][\"״][א-ט]|ט[\"״][וז]|[א-טיכל])[׳'\"״]?"
_HEB_YEAR = r"(?:ה[׳'])?תש[א-ת]?[\"״]?[א-ת]"
_HEB_DATE = re.compile(
    rf"{_NLB}[ובלמה]?(?P<day>{_HEB_DAY})\s*(?:ב|ל)?\s*(?P<month>{_HEB_MONTHS}){_NLA}"
    rf"(?P<year>\s*,?\s*{_HEB_YEAR}{_NLA})?"
)
_HOLIDAYS = frozenset({
    ("א", "תשרי"), ("ב", "תשרי"), ("ג", "תשרי"), ("י", "תשרי"), ("טו", "תשרי"), ("כב", "תשרי"),
    ("כה", "כסלו"), ("י", "טבת"), ("טו", "שבט"), ("יג", "אדר"), ("יד", "אדר"), ("טו", "אדר"),
    ("ז", "אדר"), ("טו", "ניסן"), ("כא", "ניסן"), ("כז", "ניסן"), ("ה", "אייר"), ("יד", "אייר"),
    ("יח", "אייר"), ("כח", "אייר"), ("ו", "סיון"), ("ו", "סיוון"), ("יז", "תמוז"), ("ט", "אב"),
    ("טו", "אב"), ("כט", "אלול"),
})
_GREG_MONTHS = r"ינואר|פברואר|מרץ|מרס|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר"
_GREG_DATE = re.compile(
    rf"(?<!\d)\d{{1,2}}\s*(?:ב|ל|ה)?\s*-?\s*(?:{_GREG_MONTHS}){_NLA}(?:\s*,?\s*(?:19|20)\d\d)?(?!\d)"
)
_NUMERIC_DATE = re.compile(r"(?<!\d)\d{1,2}[/.]\d{1,2}[/.](?:19|20)?\d\d(?!\d)")
# Month + year without a day ("תמוז תשפ״ד") is still more specific than a year.
_HEB_MONTH_YEAR = re.compile(
    rf"{_NLB}[ובלמ]?(?:חודש\s+)?(?:{_HEB_MONTHS}){_NLA}\s*,?\s*{_HEB_YEAR}{_NLA}"
)
# English months with a day or a year ("July 14", "14th of July", "July 2023", "July, last year").
# A month on its own ("In January") is too vague to identify and is kept.
_EN_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|march|apr(?:il)?|may|june?|july?|aug(?:ust)?"
              r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_EN_DATE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_EN_MONTHS}\b(?:,?\s+(?:19|20)\d\d)?"
    rf"|{_EN_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+(?:19|20)\d\d)?"
    rf"|{_EN_MONTHS},?\s+(?:(?:19|20)\d\d|(?:last|this|next)\s+year)\b"
    rf"|(?:last|this|next)\s+year,?\s+(?:in\s+)?{_EN_MONTHS}\b)",
    re.I,
)


def _redact_dates(text: str, counts: dict[str, int]) -> str:
    def fn(chunk: str) -> str:
        def heb(m: re.Match) -> str:
            day = re.sub(r"[׳'\"״]", "", m.group("day"))
            month = m.group("month").split()[0]
            if not m.group("year") and (day, month) in _HOLIDAYS:
                return m.group()
            counts["dates"] += 1
            lead = m.group()[: m.start("day") - m.start()]  # keep a glued prefix letter
            return lead + TAG_DATE

        chunk = _HEB_DATE.sub(heb, chunk)

        def plain(m: re.Match) -> str:
            counts["dates"] += 1
            return TAG_DATE

        chunk = _HEB_MONTH_YEAR.sub(plain, chunk)
        chunk = _GREG_DATE.sub(plain, chunk)
        chunk = _EN_DATE.sub(plain, chunk)
        return _NUMERIC_DATE.sub(plain, chunk)
    return _outside_tags(text, fn)


# ---------- spelled letters, handles and spoken addresses ----------
_LETTER_NAMES = (
    "דאבליו|אייץ'|אייץ|ג'יי|גיי|ג'י|גי|איי|בי|סי|די|אי|אף|קיי|אל|אם|אן|או|פי|קיו|אר|אס|טי|יו|וי"
    "|אקס|וואי|זד|זי"
)
_SPELL_UNIT = rf"(?:{_LETTER_NAMES}|[{_H}A-Za-z](?:\s?[{_GERESH}]\s?[{_H}]?)?)"
# Outside a request: three or more units joined by hyphens ("ס -א -ג 'י").
_SPELLED_HYPHEN = re.compile(rf"{_NLB}{_SPELL_UNIT}(?:\s*-\s*{_SPELL_UNIT}){{2,}}{_NLA}")
# Inside an e-mail/name request: three or more units, any separator ("S, H, L", "ש ל ו מ"). Three,
# not two: in English speech "a" and "I" are words, and two of them may well stand side by side.
_SPELLED_LOOSE = re.compile(rf"{_NLB}{_SPELL_UNIT}(?:[\s,.\-]+{_SPELL_UNIT}){{2,}}{_NLA}")
# Spoken medical acronyms that look exactly like spelled letters (DNA, IVF, PGD, PGT, CVS, IUI, hCG …).
_MEDICAL_SPELLED = frozenset({
    "דיאןאיי", "דיאנאיי", "איויאף", "פיג'ידי", "פיגידי", "פיג'יטי", "פיגיטי", "סיויאס", "אייויאף",
    "איייואיי", "אייואיי", "אייץ'סיג'י", "אייצסיגי", "אמאראיי", "סיטיג'י",
})
# A login-like token: three or more Latin letters and two or more digits ("noa77", "sagit880").
# Starts and ends on a letter or digit, so a sentence-ending period stays outside ("47XXY.").
_HANDLE = re.compile(
    r"(?<![A-Za-z0-9._])(?=(?:[A-Za-z0-9._]*[A-Za-z]){3})(?=(?:[A-Za-z0-9._]*\d){2})"
    r"[A-Za-z0-9](?:[A-Za-z0-9._]*[A-Za-z0-9])?(?![A-Za-z0-9])"
)
_MEDICAL_TOKEN = re.compile(
    r"(?:covid-?19|sars-?cov-?2|brca\d|g6pd|fmr1|hba1c|cftr|mthfr|smn\d|\d{2}[XY]+)", re.I
)
# Inside an e-mail request: a Latin token joined by "." or "_" reads as an address part
# ("shlomit.levinson"). Plain English words are not — in a call held in English they are the content.
_ADDRESS_TOKEN = re.compile(r"(?<![A-Za-z0-9._])[A-Za-z][A-Za-z0-9]*(?:[._][A-Za-z0-9]+)+(?![A-Za-z0-9])")
# A spoken e-mail: "@"/"שטרודל"/"at" followed by a domain word.
_AT = re.compile(rf"@|{_NLB}[שס][טת]רודל{_NLA}|(?<![A-Za-z])at(?![A-Za-z])|{_NLB}(?:אט|עט){_NLA}")
_DOMAIN = re.compile(
    rf"(?:gmail|walla|hotmail|yahoo|outlook|icloud|\bcom\b|\bnet\b|\borg\b|\bdot\b"
    rf"|{_NLB}(?:ג[{_GERESH}]?י?מייל|ג[{_GERESH}]?ימל|וואלה|הוטמייל|יאהו|נקודה|דוט|קום|נט){_NLA})",
    re.I,
)

# ASR confuses letters that sound alike; a name heard as "סגית" in one turn may be "שגית" in another.
_SOUNDALIKE = {"ס": "[סש]", "ש": "[שס]", "ת": "[תט]", "ט": "[טת]", "כ": "[כחק]", "ח": "[חכ]",
               "ק": "[קכ]", "א": "[אע]", "ע": "[עא]", "ב": "[בו]"}
_PROPAGATION_PREFIXES = "ובכלמש"  # never "ה": a first name does not take the definite article


def _new_counts() -> dict[str, int]:
    return {"answers": 0, "propagated": 0, "numbers": 0, "spelled": 0, "alnum": 0, "intro": 0,
            "dates": 0}


# ---------- helpers ----------

def _str(text) -> str:
    return text if isinstance(text, str) else ("" if text is None else str(text))


def _outside_tags(text: str, fn) -> str:
    """Apply fn to every stretch of text that is not an existing redaction tag."""
    return "".join(p if _IS_TAG.fullmatch(p) else fn(p) for p in _SPLIT_TAGS.split(text))


def _plain(text: str) -> str:
    return _IS_TAG.sub(" ", text)


def _words(text: str) -> list[str]:
    return _WORD.findall(_plain(text))


def _bare(word: str) -> str:
    return word.replace("'", "").replace("׳", "").replace('"', "").replace("״", "")


def _in(word: str, vocab: frozenset) -> bool:
    w = _bare(word)
    if w in vocab or word in vocab:
        return True
    # Strip a glued prefix only when a real word of 3+ letters remains: "כהן" is not כ + "הן",
    # "לאה" is not ל + "אה" — read that way, the most common surname would never be redacted.
    return len(w) > 3 and w[0] in _PREFIXES and w[1:] in vocab


def _letters(word: str) -> int:
    return sum(1 for ch in word if ch.isalpha())


def _is_digit_word(word: str) -> bool:
    return bool(re.fullmatch(rf"ו?(?:{_DIGIT_WORDS})", _bare(word)))


def _propagates(term: str) -> bool:
    return (
        _letters(term) >= 3
        and not _in(term, _STOP)
        and term.lower() not in _EN_STOP
        and not _in(term, _COMMON)
        and _bare(term) not in _NO_PROPAGATE
        and not any(ch.isdigit() for ch in term)
    )


def _term_alt(term: str) -> str:
    """Regex for one name, tolerant of ASR spelling: soundalike letters, and for longer names
    an optional or extra mater lectionis (י/ו/ה) — "יהודית"/"יודית", "רבקה"/"ריבקה"."""
    t = _bare(term)
    if len(t) < 4:
        return "".join(_SOUNDALIKE.get(ch, re.escape(ch)) for ch in t)
    parts = []
    for k, ch in enumerate(t):
        piece = _SOUNDALIKE.get(ch, re.escape(ch))
        if 0 < k < len(t) - 1 and ch in "יוה":
            piece = f"(?:{piece})?"
        parts.append(piece)
    return "[יו]?".join(parts)


# Spelling-tolerant matching is two steps: the regex finds candidates, then a candidate must be
# within one edit of the name once soundalike letters are folded together. The regex alone was too
# loose — with optional matres lectionis "אליהו" also matched "אילו", and "כאילו" became "כ<שם_>".
_CANON = str.maketrans({"ש": "ס", "ט": "ת", "ח": "כ", "ק": "כ", "ע": "א", "ב": "ו",
                        "ם": "מ", "ן": "נ", "ץ": "צ", "ף": "פ", "ך": "כ"})


def _canon(word: str) -> str:
    return _bare(word).lower().translate(_CANON)


def _within_one_edit(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) > len(b):
            i += 1
        elif len(b) > len(a):
            j += 1
        else:
            i += 1
            j += 1
    return edits + (len(a) - i) + (len(b) - j) <= 1


class _TermMatcher:
    def __init__(self, rx: re.Pattern, canon: set[str]):
        self.rx = rx
        self.canon = canon

    def accepts(self, surface: str) -> bool:
        c = _canon(surface)
        return any(_within_one_edit(c, t) for t in self.canon)


def _term_regex(terms: Iterable[str]) -> _TermMatcher | None:
    terms = [t for t in terms if t and _letters(t) >= 2]
    alts = {_term_alt(t) for t in terms}
    if not alts:
        return None
    body = "|".join(sorted(alts, key=len, reverse=True))
    # "ברוך השם", "חסדי השם": a name-word followed by God's name is a blessing, not a person.
    rx = re.compile(
        rf"{_NLB}(?P<pre>[{_PROPAGATION_PREFIXES}]{{0,2}})(?P<term>{body}){_NLA}(?!\s+השם{_NLA})"
    )
    return _TermMatcher(rx, {_canon(t) for t in terms})


def _cue_kinds(text: str) -> set[str]:
    plain = _plain(text)
    kinds = {kind for kind, rx in _CUES.items() if rx.search(plain)}
    return kinds | {kind for kind, rx in _CUES_EN.items() if rx.search(plain)}


def _answer_words(text: str, *, after_question: bool = False) -> list[str] | None:
    """The words of a turn that read as a dictated detail.

    None = not an answer. [] = an answer the engines already redacted (nothing left to take).
    Only the first sentence that carries content is judged; a sentence ending in "?" is a
    question or an echo, not an answer.
    """
    plain = _str(text)
    if after_question:
        idx = plain.rfind("?")
        plain = plain[idx + 1:] if idx >= 0 else ""
    for m in re.finditer(r"[^.!?]+[.!?]?", plain):
        sentence = m.group()
        words = _words(sentence)
        has_tag = bool(_IS_TAG.search(sentence))
        content = [w for w in words if not _in(w, _STOP) and w.lower() not in _EN_STOP]
        if not content and not has_tag:
            continue
        if sentence.rstrip().endswith("?"):
            return None
        if re.search(r"\d", _plain(sentence)) or len(words) > _MAX_ANSWER_WORDS:
            return None
        if any(_bare(w) in _SENTENCE_MARKERS for w in words):
            return None
        if words and _bare(words[0]) in _LEAD_PRONOUNS and any(_bare(w) in _CALL_VERBS for w in words):
            return None
        content = [w for w in content if not _in(w, _CALL_VERBS) and not _in(w, _COMMON)
                   and not _in(w, _FUNCTION_WORDS)
                   and not _is_digit_word(w) and _letters(w) >= 2]
        distinct = list(dict.fromkeys(content))
        if not distinct:
            return [] if has_tag else None
        return distinct if len(distinct) <= 3 else None
    return None


def _self_intro_names(text: str) -> list[str]:
    plain = _plain(_str(text))
    found: list[str] = []
    for rx in _SELF_INTRO:
        for m in rx.finditer(plain):
            for w in m.group("n").split():
                if (_in(w, _STOP) or _in(w, _CALL_VERBS) or _in(w, _COMMON) or w.lower() in _EN_STOP
                        or _in(w, _FUNCTION_WORDS)
                        or _bare(w) in _LEAD_PRONOUNS):
                    break
                found.append(w)
    return list(dict.fromkeys(found))


def _engine_name_terms(original: str, redacted: str, known_words: set[str]) -> set[str]:
    """Words the engines replaced one-for-one with a name tag.

    Only 1:1 replacements are trusted: a block that maps three words onto two tags, or that also
    holds an organisation/location tag, cannot say which word was the name — and learning the
    wrong one ("פוריות" from "ממכון פוריות") would erase it from the whole conversation.
    """
    a = _TOKEN.findall(original)
    b = _TOKEN.findall(redacted)
    if len(a) + len(b) > 800:  # keep the alignment cheap on pathological segments
        return set()
    terms: set[str] = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "replace" or (i2 - i1) != (j2 - j1):
            continue
        block = b[j1:j2]
        if any(_IS_TAG.fullmatch(t) and t != TAG_NAME for t in block):
            continue
        for k, tag in enumerate(block):
            tok = a[i1 + k]
            if tag != TAG_NAME or not _WORD.fullmatch(tok):
                continue
            terms.add(tok)
            # The engines replace whole tokens, prefix included ("לנורית"); learn the bare name
            # too when it also occurs on its own.
            for cut in (1, 2):
                rest = tok[cut:]
                if len(rest) >= 3 and all(c in _PREFIXES for c in tok[:cut]) and rest in known_words:
                    terms.add(rest)
    return terms


# ---------- rewriting ----------

def _redact_spoken_email(text: str, counts: dict[str, int]) -> tuple[str, list[str]]:
    """A turn that says an address out loud ("X שטרודל ג'ימייל נקודה קום"): every word that is
    not a function word goes, and the local part's words are returned as names to propagate."""
    plain = _plain(text)
    at = _AT.search(plain)
    if not at or not _DOMAIN.search(plain[at.end():at.end() + 40]):
        return text, []
    local = [w for w in _words(plain[:at.start()])[-3:] if not _in(w, _STOP) and _WORD.fullmatch(w)]

    def fn(chunk: str) -> str:
        def word(m: re.Match) -> str:
            if _in(m.group(), _STOP):
                return m.group()
            counts["spelled"] += 1
            return TAG_CONTACT
        chunk = chunk.replace("@", f" {TAG_CONTACT} ")
        return _WORD.sub(word, chunk)
    return _outside_tags(text, fn), local


def _redact_shapes(
    text: str, counts: dict[str, int], *, email_window: bool, spell_window: bool | None = None
) -> str:
    """Spelled-out letters and login-like handles.

    spell_window: a name or e-mail is being asked for, so letters spelled one by one go.
    email_window: an e-mail is being asked for, so dotted address parts go too.
    """
    if spell_window is None:
        spell_window = email_window
    def fn(chunk: str) -> str:
        def spelled(m: re.Match) -> str:
            if re.sub(r"[\s,.\-]", "", m.group()) in _MEDICAL_SPELLED:
                return m.group()
            counts["spelled"] += 1
            return TAG_CONTACT

        chunk = _SPELLED_HYPHEN.sub(spelled, chunk)
        if spell_window:
            chunk = _SPELLED_LOOSE.sub(spelled, chunk)

        def handle(m: re.Match) -> str:
            if _MEDICAL_TOKEN.fullmatch(m.group()):
                return m.group()
            counts["alnum"] += 1
            return TAG_CONTACT

        chunk = _HANDLE.sub(handle, chunk)
        if email_window:
            def address(m: re.Match) -> str:
                counts["alnum"] += 1
                return TAG_CONTACT
            chunk = _ADDRESS_TOKEN.sub(address, chunk)
        return chunk
    return _outside_tags(text, fn)


def _protected_spans(plain: str, *, years: bool = True) -> list[tuple[int, int]]:
    """Numbers that are clinical facts, not identifiers: ratios, decimals, percentages, ages,
    weeks, prices, lab values — and, outside a request, years."""
    spans = [m.span() for m in _RATIO.finditer(plain)]
    for m in _NUM.finditer(plain):
        g = m.group()
        if "." in g or (years and _YEAR.fullmatch(g)):
            spans.append(m.span())
        elif _UNIT_AFTER.match(plain, m.end()) or _UNIT_BEFORE.search(plain[: m.start()]):
            spans.append(m.span())
    return spans


def _covered(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def _redact_numbers(text: str, counts: dict[str, int], *, everything: bool, tag: str) -> str:
    """everything = inside a personal-details request, or part of a number dictated across turns.

    Otherwise only phone/ID shapes go: groups that read as one number of 7+ digits ("052-731-2290"),
    a lone 7+ digit group, or a 3+ digit group starting with 0 ("050"). Clinical values stay, and
    a percentage or a decimal is never part of a phone or ID number.
    """
    def fn(chunk: str) -> str:
        # "בת 34", "שבוע 12", "1 ל -350" stay even inside a request; a bare number there goes.
        spans = _protected_spans(chunk, years=not everything)
        doomed: list[tuple[int, int]] = []
        for m in _NUM_SEQ.finditer(chunk):
            if sum(len(g) for g in re.findall(r"\d+", m.group())) >= _PHONE_RUN:
                doomed += [g.span() for g in _NUM.finditer(chunk, m.start(), m.end())
                           if "." not in g.group()]
        for m in _NUM.finditer(chunk):
            g = m.group()
            if "." in g or _PERCENT.match(chunk, m.end()) or _covered(m.start(), spans):
                continue
            if everything or len(g) >= 7 or (g[0] == "0" and len(g) >= 3):
                doomed.append(m.span())
        # Digits dictated as words: any pair inside a request, a full number's worth outside one.
        need = 2 if everything else _PHONE_RUN
        for m in _DW_RUN.finditer(chunk):
            if len(_DW_ONE.findall(m.group())) >= need:
                doomed.append(m.span())
        if not doomed:
            return chunk
        out, pos = [], 0
        for s, e in sorted(set(doomed)):
            if s < pos:
                continue
            out.append(chunk[pos:s])
            out.append(tag)
            counts["numbers"] += 1
            pos = e
        out.append(chunk[pos:])
        return "".join(out)
    return _outside_tags(text, fn)


def _propagate(text: str, matcher: _TermMatcher | None, counts: dict[str, int], key: str) -> str:
    if matcher is None:
        return text

    def fn(chunk: str) -> str:
        def sub(m: re.Match) -> str:
            if not matcher.accepts(m.group("term")):
                return m.group()
            counts[key] += 1
            return m.group("pre") + TAG_NAME
        return matcher.rx.sub(sub, chunk)
    return _outside_tags(text, fn)


# ---------- public API ----------

def contextual_redact(
    texts: Sequence[str],
    speakers: Sequence[str] | None = None,
    originals: Sequence[str] | None = None,
) -> tuple[list[str], set[str], dict[str, int]]:
    """Remove what only the conversation's structure reveals.

    texts      the segment texts, in order — raw, or already redacted by the engines.
    speakers   speaker id per segment. Used to prefer the other party's reply; an answer that
               diarization gave the asker's label is still taken when it directly follows.
    originals  the pre-engine texts, when `texts` are engine output; names the engines found
               are then removed wherever else they occur.

    Returns (new_texts, terms, counts). `terms` are the identifying words found — pass them to
    redact_freeform() for summaries and keywords. They are PII: never log them. `counts` is
    safe to log.
    """
    texts = [_str(t) for t in texts]
    n = len(texts)
    if speakers is None:
        speakers = [f"#{i}" for i in range(n)]  # unknown: every turn its own speaker
    speakers = [_str(s) for s in speakers]
    if len(speakers) != n or (originals is not None and len(originals) != n):
        raise ValueError("texts, speakers and originals must be the same length")

    counts = _new_counts()
    word_counts = [len(_words(t)) for t in texts]
    kinds = [_cue_kinds(t) for t in texts]

    # 1. Personal-details requests: which turns they cover, and the answer they got.
    window: list[set[str]] = [set() for _ in range(n)]
    window_terms: list[set[str]] = [set() for _ in range(n)]
    terms: set[str] = set()
    for i in range(n):
        if not kinds[i]:
            continue
        span = [i]
        for j in range(i + 1, min(n, i + 1 + _WINDOW)):
            if word_counts[j] > _LONG_TURN:
                break
            span.append(j)
        for j in span:
            window[j] |= kinds[i]
        if not kinds[i] & _WORD_ANSWER_KINDS:
            continue

        found: list[str] = []
        same_turn = _answer_words(texts[i], after_question=True)   # "מה השם שלך? שלומית."
        answered = same_turn is not None
        answer_speaker = speakers[i] if same_turn else None
        found += same_turn or []
        examined = 0
        for j in span[1:]:
            if answered:
                # at most one continuation by the same speaker: "שלומית." / "לוינסון."
                if answer_speaker is not None and speakers[j] == answer_speaker and not kinds[j]:
                    more = _answer_words(texts[j])
                    if more:
                        found += more
                break
            if kinds[j]:
                continue  # a new request opens its own window
            examined += 1
            if examined > _ANSWER_SCAN:
                break
            # The other party's reply; the asker's own label only right after the question,
            # which is where a mislabelled short answer actually sits.
            if speakers[j] == speakers[i] and j != i + 1:
                continue
            got = _answer_words(texts[j])
            if got is None:
                continue
            found += got
            answered = True
            answer_speaker = speakers[j]
        if found:
            for j in span:
                window_terms[j] |= set(found)
            terms |= {w for w in found if _propagates(w)}

    # 2. Self-introductions anywhere: "קוראים לי X", "שמי X", "זאת X מדברת".
    intro_terms: list[set[str]] = [set() for _ in range(n)]
    for i, t in enumerate(texts):
        names = _self_intro_names(t)
        if names:
            intro_terms[i] |= set(names)
            terms |= {w for w in names if _propagates(w)}

    # 3. Names the engines caught somewhere, so they also go where they were missed.
    if originals is not None:
        originals = [_str(t) for t in originals]
        known = {w for t in originals for w in _WORD.findall(t)}
        for orig, red in zip(originals, texts):
            if orig != red:
                terms |= {t for t in _engine_name_terms(orig, red, known) if _propagates(t)}

    # 4. Numbers dictated across adjacent short turns ("050." / "כן" / "414 -6488.") are one number.
    def dictated(i: int) -> int:
        if word_counts[i] > _LONG_TURN:
            return 0
        plain = _plain(texts[i])
        spans = _protected_spans(plain)
        digits = sum(len(m.group()) for m in _NUM.finditer(plain)
                     if "." not in m.group() and not _PERCENT.match(plain, m.end())
                     and not _covered(m.start(), spans))
        digits += sum(len(_DW_ONE.findall(m.group())) for m in _DW_RUN.finditer(plain))
        if not digits:
            return 0
        others = [w for w in _words(plain) if not _in(w, _STOP) and not _is_digit_word(w)]
        return digits if len(others) <= 3 else 0

    def bridge(i: int) -> bool:  # "כן" / "אוקיי" between two dictated pieces
        words = _words(texts[i])
        return 0 < len(words) <= 3 and all(_in(w, _STOP) for w in words) and not re.search(r"\d", texts[i])

    digit_count = [dictated(i) for i in range(n)]
    in_run = [False] * n
    i = 0
    while i < n:
        if not digit_count[i]:
            i += 1
            continue
        members = [i]
        j = i + 1
        while j < n:
            if digit_count[j]:
                members.append(j)
                j += 1
            elif bridge(j) and j + 1 < n and digit_count[j + 1]:
                j += 1
            else:
                break
        if sum(digit_count[k] for k in members) >= _PHONE_RUN:
            for k in members:
                in_run[k] = True
        i = j

    # 5. Rewrite.
    out: list[str] = []
    for i, text in enumerate(texts):
        t, local_part = _redact_spoken_email(text, counts)
        terms |= {w for w in local_part if _propagates(w)}
        t = _redact_shapes(t, counts, email_window="email" in window[i],
                           spell_window=bool(window[i] & {"email", "name"}))
        t = _redact_dates(t, counts)
        tag = TAG_ID if "id" in window[i] else TAG_CONTACT
        t = _redact_numbers(t, counts, everything=bool(window[i]) or in_run[i], tag=tag)
        if intro_terms[i]:
            t = _propagate(t, _term_regex(intro_terms[i]), counts, "intro")
        if window_terms[i]:
            t = _propagate(t, _term_regex(window_terms[i]), counts, "answers")
        out.append(t)
    global_rx = _term_regex(terms)  # after step 5: a spoken e-mail may have taught a name
    out = [_propagate(t, global_rx, counts, "propagated") for t in out]
    return out, terms, counts


def redact_freeform(text, terms: Iterable[str] = ()):
    """Same rules for text that is not a conversation turn: summaries, keywords, topics.

    No request/answer structure exists here, so numbers are judged by shape alone, and the
    identifying words found in the conversation are removed wherever they recur. Non-text
    values are returned unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    counts = _new_counts()
    t, local = _redact_spoken_email(text, counts)
    t = _redact_shapes(t, counts, email_window=False)
    t = _redact_dates(t, counts)
    t = _redact_numbers(t, counts, everything=False, tag=TAG_CONTACT)
    intro = _self_intro_names(t)
    return _propagate(t, _term_regex(set(terms) | set(intro) | set(local)), counts, "propagated")


# ---------- undoing the engines' over-redaction, on a closed list ----------
# HebSafeHarbor's NER tags ordinary words in conversational Hebrew — "שלום", "בלי", "כאילו",
# "התהליך" — and, worse for this domain, the names of the very drugs and devices a counselling
# call is about (מירנה, קיילינה, נורי-דיי, נורג'סטון). Those reach Gemini as <שם_>/<מיקום_> and the
# summary loses its subject. What is restored is a closed list of words that are never an
# identifier; everything else the engines removed stays removed. The context pass runs after
# this, so a word restored here that is in fact being dictated as a name ("קוראים לי שלום") is
# still caught.

_FUNCTION_WORDS = frozenset(
    """
    בלי בכל כל כמו כך ככה אחרי לפני בגלל בשביל לגבי כלומר דרך תהליך גבול גבולות מצב בעיה עניין
    מקרה שאלה תשובה כלל בכלל אסור מותר הלכה הלכתי הלכתית רבנים פוסק פסק בעזרת סיבה תקופה
    גריס כתם כתמים וסת וסתות הפסק טהרה נידה מקווה טבילה נקיים הרחקות עונה עונות ליבון חציצה
    פרישה בדיקה בדיקות עד עדים
    """.split()
)
# The institute itself is not an identifier. Its name sits in our own prompt template ("ניתן לפנות
# שוב למכון פוע"ה"); redacted there, Gemini copied "<ארגון_>" into the summary sent to the caller.
_INSTITUTION = frozenset('מכון פוע"ה פועה puah machon'.split())
# Small settlements whose name is an everyday word. HebSafeHarbor's gazetteer tags them wherever
# they occur ("שמרת על גבולות" -> "<מיקום_> על <מיקום_>"). Restored only when tagged as a place —
# settlement names that are also first names (נורית, אביעזר, מירב, הראל …) are deliberately absent.
_SETTLEMENT_WORDS = frozenset(
    """
    גבולות שמרת אחווה איחוד אמונים דורות מוסד מורשת מזור מסד מסילות משמרות משמרת מתת נופים
    ניצנים עלומים קדמה רווחה שובה שורשים שלווה שלוחות שרשרת תושיה תרום רעים מגדלים כרמים
    מגשימים חרוצים העוגן הבונים מנוף מעש יבול עוצם כליל
    """.split()
)
# Blessings: a lone "השם" is restored by the stop list; the phrase as a whole comes back here.
_BLESSING = re.compile(r'^(?:ברוך|בעזרת|בע"ה|אם ירצה|ירצה|חסדי|ישתבח)\s+השם$')
# Never restored, whatever list they appear on: words that can be a first name. Redacting the
# greeting "שלום" by mistake costs nothing; restoring a person called Shalom breaks the rule.
_NEVER_RESTORE = _NO_PROPAGATE | frozenset("שלום שלי בן".split())
# Drug, device and test names of fertility/contraception counselling, tolerant of ASR spelling
# (spaces, geresh, soundalike letters). Names that are also first names (יסמין, דיאנה) are left out
# on purpose: restoring one of those could put a person's name back.
_MEDICAL_NAMES = re.compile(
    r"^[והבלמשכ]{0,3}(?:"
    r"מירנה|מרינה|נירנה|קיילינה|קיילנה|קילינה|ג'?יידס|גיידס|נובה ?רינג|נקספלנון|אימפלנון"
    r"|סרזט|צרזט|סרזאט|סארג ?'?ט|סר ?רזט|סרזית|שרזית|הסרזית|מסרזית|סרזת|סרזס|שרדת|סדר זאת"
    r"|הרמונט|המונט|הרמונה|נורי ?דיי|נורידיי|נורג ?'?סטון|נורגסטון|נורד(?: ?ג)? ?'?יסטון"
    r"|נורדיסטון|נורד|יסטון|מיקרולוט|מיקרולט|מרבלון|מינולט|פמודן|לואט"
    r"|קלומיד|קלומיפן|לטרוזול|פמרה|גונל(?: ?אף)?|מנופור|פרגונל|אוביטרל|פרגניל|צטרוטייד"
    r"|אורגלוטרן|דקפפטיל|פרוגסטרון|פוגסטרום|אוטרוגסטן|קרינון|אנדומטרין|דופסטון|פרוביירה"
    r"|פרוגינובה|אסטרופם|אסטרדיול|אסטרוגן|קלקסן|אספירין|מטפורמין|גלוקופאז'?|אלטרוקסין"
    r"|פרולקטין|חומצה ?פולית"
    r")$"
)
# "מרינה" is the ASR spelling of Mirena and also a first name (Marina). With the definite article
# ("המרינה", "שהמרינה") it can only be the device; bare, it comes back only from a non-person tag.
_DRUG_OR_NAME = re.compile(r"^[ובלמשכ]{0,3}מרינה$")
_HEALTH_FUNDS = frozenset("מכבי כללית מאוחדת לאומית קופה קופת חולים קופ\"ח".split())
# Countries and the country as a whole — restored only when the engine called them a place or an
# organisation, never a person: "ישראל" is also a first name.
_COUNTRIES = frozenset(
    """
    ישראל ישראלי ישראלית ארץ הארץ חו"ל חול ארה"ב אמריקה צרפת אנגליה בריטניה אוסטרליה קנדה רוסיה
    אוקראינה ארגנטינה ברזיל גרמניה בלגיה שוויץ אירופה מקסיקו דרום אפריקה הולנד איטליה ספרד
    israel israeli america usa us uk england britain france australia canada russia ukraine
    argentina brazil germany belgium switzerland europe mexico
    """.split()
)
_NON_PERSON_TAGS = re.compile(r"<(?:מיקום_|ארגון_|MISC[A-Z_]*)>")
_MIXABLE = _HEALTH_FUNDS | _INSTITUTION | _COUNTRIES | _STOP | _FUNCTION_WORDS


def _bare_lower(word: str) -> str:
    return _bare(word).lower()


def _token_is(word: str, vocab: frozenset) -> bool:
    w = _bare_lower(word)
    if w in vocab:
        return True
    for cut in (1, 2):
        if len(w) - cut >= 3 and all(c in _PREFIXES for c in w[:cut]) and w[cut:] in vocab:
            return True
    return False


def _safe_span(span_text: str, tokens: list[str], tags: list[str]) -> bool:
    words = [t for t in tokens if _WORD.fullmatch(t)]
    if not words or len(words) != len(tokens):
        return False  # digits or anything else: never restore
    norm = re.sub(r"\s+", " ", span_text.replace("׳", "'").replace("״", '"')).strip()
    person_tagged = not all(_NON_PERSON_TAGS.fullmatch(t) for t in tags)
    # A whole-span match on the drug list wins over the never-restore check, whose prefix stripping
    # misreads "המרינה" as ה+מ+"רינה". Only a drug name that is also a first name, bare and tagged
    # as a person, stays redacted.
    if _MEDICAL_NAMES.match(norm) and not (person_tagged and _DRUG_OR_NAME.match(norm)):
        return True
    if any(_token_is(w, _NEVER_RESTORE) for w in words):
        return False
    if _BLESSING.match(norm):
        return True
    if all(_token_is(w, _STOP | _FUNCTION_WORDS) for w in words):
        return True
    if all(_token_is(w, _HEALTH_FUNDS) for w in words) or all(_token_is(w, _INSTITUTION) for w in words):
        return True
    if person_tagged:
        return False
    if all(_token_is(w, _COUNTRIES) for w in words) or all(_token_is(w, _SETTLEMENT_WORDS) for w in words):
        return True
    # a stretch mixing those lists ("קופת החולים בארץ"); settlement words never mix in
    return all(_token_is(w, _MIXABLE) for w in words)


def unnest_tags(text: str) -> str:
    """Repair a tag the engines tagged again: run over text that already holds "<ארגון_>", the NER
    reads the word inside it and produces "<<ארגון_>_>". Collapse to the inner tag."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(rf"<+({_TAG_BODY})_?>+", r"\1", text)
    return text


def restore_safe_terms(originals: Sequence[str], redacted: Sequence[str]) -> tuple[list[str], int]:
    """Put back the words on the closed lists above that the engines replaced with a tag.

    Works on whole replaced stretches: a stretch comes back only if every word in it is on a
    list (a drug name may span several tokens — "נורד ג יסטון"). Anything else, and anything
    holding a digit, stays redacted. Returns (texts, number of stretches restored).
    """
    out: list[str] = []
    restored = 0
    for orig, red in zip(originals, redacted):
        orig, red = _str(orig), _str(red)
        if orig == red or not _IS_TAG.search(red):
            out.append(red)
            continue
        a = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(orig)]
        b = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(red)]
        if len(a) + len(b) > 40000:
            out.append(red)
            continue
        sm = difflib.SequenceMatcher(a=[t for t, _, _ in a], b=[t for t, _, _ in b], autojunk=False)
        pieces: list[tuple[int, int, str]] = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "replace":
                continue
            btoks = [t for t, _, _ in b[j1:j2]]
            tags = [t for t in btoks if _IS_TAG.fullmatch(t)]
            # the engine may leave a glued prefix letter beside its tag ("בלי" -> "ב <מיקום_>")
            if not tags or any(not _IS_TAG.fullmatch(t) and not (len(t) <= 2 and all(c in _PREFIXES for c in t))
                               for t in btoks):
                continue
            span = orig[a[i1][1]:a[i2 - 1][2]]
            if _safe_span(span, [t for t, _, _ in a[i1:i2]], tags):
                pieces.append((b[j1][1], b[j2 - 1][2], span))
        if pieces:
            parts, pos = [], 0
            for start, end, text in pieces:
                parts.append(red[pos:start])
                parts.append(text)
                pos = end
            parts.append(red[pos:])
            red = "".join(parts)
            restored += len(pieces)
        out.append(red)
    return out, restored
