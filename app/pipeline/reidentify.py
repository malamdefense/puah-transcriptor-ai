"""Controlled re-identification of the Gemini deliverables (client summary, consultant feedback).

The model only ever sees a de-identified transcript. Before this module every identifying detail
became the same generic tag ("<שם_>"), so nothing could say *which* name a tag stood for, and the
output guard could only delete tags from the answer. Here, the tags of the categories the policy
allows to come back (names and organisations by default) are numbered per entity — "<שם_1>",
"<ארגון_2>" — and the numbered tag -> original value mapping is kept in a Vault, in memory, for
the duration of one deliverable. The model may refer to a numbered tag; after the answer comes
back the tags are replaced locally by the original values.

Where the numbers come from — two layers, both by aligning a text before and after redaction:
  * summarize.build_transcript_text: the conversation-level pass (pii_context.contextual_redact),
    per segment;
  * llm_providers.BaseLLMProvider._redact: the PII engines (HebSafeHarbor + Presidio), per message.
    The engines leave numbered tags untouched (verified on the real chain), so a name caught by the
    first layer keeps its number through the second one.
A replaced stretch whose value cannot be read off unambiguously keeps its generic tag: it can never
be restored, and the output guard removes it from the answer exactly as before. Fail-safe.

Invariants:
  * Gemini never receives a value: numbering happens on text that is already redacted, and the
    repair turn always sends the model's own (tagged) answer, never the restored one.
  * Only (category, number) pairs that exist in the vault *and* whose category the policy allows
    are restored; an invented number, a wrong category or an unnumbered tag stays a defect.
  * Values never reach a log, an exception message or a stored record: the Vault's repr and
    every record hold labels and counts only.

Pure stdlib, no settings, no I/O.
"""

from __future__ import annotations

import contextvars
import difflib
import re
from typing import Iterable

_H = "א-ת"
_PREFIX_LETTERS = "והבכלמש"
_BIDI = "\u200e\u200f\u202a-\u202e\u2066-\u2069\u061c"

# Categories the vault may number, keyed by the engines' Hebrew label.
NAME = "שם"
ORG = "ארגון"
DEFAULT_CATEGORIES: tuple[str, ...] = (NAME, ORG)
_CATEGORY_LABELS = (NAME, ORG, "מיקום", "קשר", "מזהה", "תאריך")

# A generic (unnumbered) engine tag, as a whole token body: "<שם_>".
_GENERIC = re.compile(rf"<(?P<cat>[{_H}]+)_>")
# Any engine-shaped tag, numbered or not: "<שם_>", "<שם_3>", "<PERSON_1>".
_ANY_TAG = re.compile(rf"<(?:[{_H}]+_\d*|[A-Z][A-Z_]*\d*)>")
# One whitespace token split into glued prefix letters + a tag + trailing punctuation:
# "ל<שם_>,"  "ו-<ארגון_>"  "(<שם_>)"
_TAG_TOKEN = re.compile(
    rf"^(?P<lead>[\"'״׳(\[]*)(?P<pre>[{_PREFIX_LETTERS}]{{0,3}}[-־]?)"
    rf"<(?P<cat>[{_H}]+)_(?P<num>\d*)>(?P<post>[^\w<>]*)$"
)
_NUMBERED_ANY = re.compile(rf"<(?P<cat>[{_H}]+)[_\- ]\s*(?P<num>\d+)>")

# Words that open an organisation's name: a tag pair "<שם_> <ארגון_>" over "דוד כהן מבית החולים X"
# is split where the first of these starts.
_ORG_CUES = frozenset(
    """בית ביה"ח בי"ח בית"ח מרפאה מרפאת מכון מרכז קופת קופה עמותת עמותה אוניברסיטת אוניברסיטה
    מחלקת מחלקה יחידת יחידה סניף ארגון מוסד מעבדת מעבדה חברת חברה משרד מדרשת ישיבת ישיבה
    קליניקה קליניקת בית-חולים""".split()
)

_EDGE_PUNCT = " \t\n\"'״׳.,;:!?()[]{}«»‹›-–—־/\\"
_MAX_VALUE_WORDS = 8
_MAX_VALUE_CHARS = 80
_MAX_ALIGN_TOKENS = 60000  # difflib guard, as in pii_context.restore_safe_terms

_current: contextvars.ContextVar["Vault | None"] = contextvars.ContextVar("puah_reid_vault", default=None)


def current_vault() -> "Vault | None":
    """The vault of the deliverable being generated in this thread/context, if any."""
    return _current.get()


class use_vault:
    """Context manager that makes `vault` the current one (llm_providers numbers into it)."""

    def __init__(self, vault: "Vault | None"):
        self._vault = vault
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "Vault | None":
        self._token = _current.set(self._vault)
        return self._vault

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _current.reset(self._token)


def _canon(value: str) -> str:
    v = value.replace("׳", "'").replace("״", '"')
    v = re.sub(r"[\"'\-־]", "", v)
    return re.sub(r"\s+", " ", v).strip().lower()


def _clean_value(value: str) -> str | None:
    v = re.sub(rf"[{_BIDI}]", "", value)
    v = re.sub(r"\s+", " ", v).strip(_EDGE_PUNCT)
    if not v or len(v) > _MAX_VALUE_CHARS or len(v.split()) > _MAX_VALUE_WORDS:
        return None
    if re.search(r"\d", v) or "<" in v or ">" in v:
        return None
    if not re.search(rf"[{_H}A-Za-z]", v):
        return None
    return v


class Vault:
    """Numbered tag -> original value, per category. In memory only; never logged."""

    def __init__(self, categories: Iterable[str] = DEFAULT_CATEGORIES):
        self.categories: frozenset[str] = frozenset(categories)
        self._values: dict[tuple[str, int], str] = {}
        self._by_key: dict[tuple[str, str], int] = {}
        self._next: dict[str, int] = {}
        self._honorifics: dict[tuple[str, int], str] = {}
        self._speakers: dict[int, tuple[str, int]] = {}

    # Never print the values: a repr can end up in a log or a traceback.
    def __repr__(self) -> str:
        counts = {c: self._next.get(c, 0) for c in sorted(self.categories)}
        return f"Vault({counts})"

    __str__ = __repr__

    def __len__(self) -> int:
        return len(self._values)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for cat, _ in self._values:
            out[cat] = out.get(cat, 0) + 1
        return out

    def assign(self, category: str, value: str, *, honorific: str = "") -> int | None:
        """The number of `value` in `category` (a new one if unseen); None if not storable."""
        if category not in self.categories:
            return None
        clean = _clean_value(value)
        if clean is None:
            return None
        key = (category, _canon(clean))
        n = self._by_key.get(key)
        if n is None:
            n = self._next.get(category, 0) + 1
            self._next[category] = n
            self._by_key[key] = n
            self._values[(category, n)] = clean
        if honorific and honorific.strip() and (category, n) not in self._honorifics:
            self._honorifics[(category, n)] = honorific.strip()
        return n

    def assign_speaker(self, speaker_index: int, name: str) -> str:
        """Register a speaker's display name in the in-memory vault and return a safe tagged label.

        Safe role/default labels ("דובר 1", "יועצת", "פונה") pass unchanged. A speaker label holding
        a personal name ("רחל כהן", "הרב משה לוי", "יועצת - שרה כהן") stores the personal name in
        the vault under `<שם_N>` and returns `"<שם_N>"` or `"הרב <שם_N>"` / `"יועצת <שם_N>"`.
        """
        if speaker_label_is_safe(name):
            return name
        parsed = _split_speaker_label(name)
        if parsed is None:
            return f"דובר {speaker_index}"
        role_prefix, person_name, honorific = parsed
        n = self.assign(NAME, person_name, honorific=honorific)
        if n is None:
            if role_prefix and speaker_label_is_safe(role_prefix):
                return role_prefix
            return f"דובר {speaker_index}"
        self._speakers[speaker_index] = (NAME, n)
        return f"{role_prefix} <{NAME}_{n}>" if role_prefix else f"<{NAME}_{n}>"

    def value(self, category: str, n: int) -> str | None:
        return self._values.get((category, n))

    # ---------- numbering (alignment of a text before / after redaction) ----------

    def number(self, original: str, redacted: str) -> str:
        """`redacted` with every generic tag whose value is unambiguous replaced by a numbered one.

        Never raises; on any internal problem the redacted text is returned unchanged (its tags
        then simply cannot be restored)."""
        try:
            return self._number(original, redacted)
        except Exception:  # noqa: BLE001 — fail-safe: unnumbered tags are only unrestorable
            return redacted

    def _number(self, original: str, redacted: str) -> str:
        if not isinstance(original, str) or not isinstance(redacted, str):
            return redacted
        if original == redacted or not _GENERIC.search(redacted):
            return redacted
        a = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", original)]
        b = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", redacted)]
        if len(a) + len(b) > _MAX_ALIGN_TOKENS:
            return redacted
        known = {t.strip(_EDGE_PUNCT) for t, _, _ in a}
        sm = difflib.SequenceMatcher(a=[t for t, _, _ in a], b=[t for t, _, _ in b], autojunk=False)
        edits: dict[int, str] = {}  # b token index -> replacement token ("" = collapsed continuation)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "replace":
                continue
            self._number_block(a[i1:i2], b[j1:j2], j1, known, edits)
        if not edits:
            return redacted
        parts, pos = [], 0
        for j, (tok, start, end) in enumerate(b):
            if j not in edits:
                continue
            if edits[j] == "":
                # Collapsed continuation token of a multi-word span: drop inter-token whitespace too
                pos = end
                continue
            parts.append(redacted[pos:start])
            parts.append(edits[j])
            pos = end
        parts.append(redacted[pos:])
        return "".join(parts)

    def _number_block(self, atoks, btoks, j0: int, known: set[str], edits: dict[int, str]) -> None:
        parsed = [_TAG_TOKEN.match(t) for t, _, _ in btoks]
        tag_idx = [k for k, m in enumerate(parsed) if m and not m.group("num")]
        if not tag_idx:
            return
        # 1:1 — every redacted token stands for one original token
        if len(atoks) == len(btoks):
            # When contextual_redact replaces a multi-word name ("רחל כהן") word-by-word with
            # "<שם_> <שם_>" (same category, no internal punctuation or conjunction prefix), collapse
            # into a single numbered tag "<שם_N>" for the full name so it matches the speaker vault tag.
            if (
                len(tag_idx) == len(btoks) > 1
                and len({parsed[k].group("cat") for k in tag_idx}) == 1
                and all(not parsed[k].group("post") for k in tag_idx[:-1])
                and all(not parsed[k].group("lead") and not parsed[k].group("pre") for k in tag_idx[1:])
                and not any(_is_org_cue(atoks[k][0]) for k in range(1, len(atoks)))
            ):
                self._apply_span(
                    parsed[0], parsed[-1], [t for t, _, _ in atoks], j0, len(btoks), edits, known
                )
                return
            for k in tag_idx:
                m = parsed[k]
                self._apply(m, [atoks[k][0]], j0 + k, edits, known)
            return
        if len(tag_idx) != len(btoks):
            return  # tags mixed with other words, counts differ: ambiguous
        cats = [parsed[k].group("cat") for k in tag_idx]
        runs: list[tuple[str, list[int]]] = []
        for k, c in zip(tag_idx, cats):
            if runs and runs[-1][0] == c:
                runs[-1][1].append(k)
            else:
                runs.append((c, [k]))
        if len(runs) == 1:
            if len(tag_idx) == 1:
                self._apply(parsed[0], [t for t, _, _ in atoks], j0, edits, known)
            # several tags of one kind over a different number of words: which word is whose? ambiguous
            return
        if len(runs) == 2 and runs[1][0] == ORG and len(runs[0][1]) == 1 and len(runs[1][1]) == 1:
            # "<שם_> <ארגון_>" over "דוד כהן מבית החולים שערי צדק": split where the organisation starts
            split = next((i for i in range(1, len(atoks)) if _is_org_cue(atoks[i][0])), None)
            if split is None:
                return
            k1, k2 = runs[0][1][0], runs[1][1][0]
            self._apply(parsed[k1], [t for t, _, _ in atoks[:split]], j0 + k1, edits, known)
            self._apply(parsed[k2], [t for t, _, _ in atoks[split:]], j0 + k2, edits, known, org_cue=True)

    def _apply_span(self, m_first: re.Match, m_last: re.Match, words: list[str], j0: int, count: int,
                    edits: dict[int, str], known: set[str]) -> None:
        cat = m_first.group("cat")
        if cat not in self.categories or not words:
            return
        words = list(words)
        lead, pre, post = m_first.group("lead"), m_first.group("pre"), m_last.group("post")
        first = words[0].lstrip("\"'״׳([")
        bare_pre = re.sub("[-־]", "", pre)
        if bare_pre and first.startswith(bare_pre) and len(first) > len(bare_pre):
            first = first[len(bare_pre):].lstrip("-־")
        elif not bare_pre:
            first = _strip_eaten_prefix(first, known, org_cue=cat == ORG)
        words[0] = first
        if post:
            words[-1] = words[-1].rstrip(_EDGE_PUNCT) or words[-1]
        n = self.assign(cat, " ".join(words))
        if n is None:
            return
        edits[j0] = f"{lead}{pre}<{cat}_{n}>{post}"
        for offset in range(1, count):
            edits[j0 + offset] = ""

    def _apply(self, m: re.Match, words: list[str], j: int, edits: dict[int, str], known: set[str],
               *, org_cue: bool = False) -> None:
        cat = m.group("cat")
        if cat not in self.categories or not words:
            return
        words = list(words)
        lead, pre, post = m.group("lead"), m.group("pre"), m.group("post")
        first = words[0].lstrip("\"'״׳([")
        bare_pre = re.sub("[-־]", "", pre)
        if bare_pre and first.startswith(bare_pre) and len(first) > len(bare_pre):
            first = first[len(bare_pre):].lstrip("-־")
        elif not bare_pre:
            first = _strip_eaten_prefix(first, known, org_cue=org_cue or cat == ORG)
        words[0] = first
        if post:
            words[-1] = words[-1].rstrip(_EDGE_PUNCT) or words[-1]
        n = self.assign(cat, " ".join(words))
        if n is None:
            return
        edits[j] = f"{lead}{pre}<{cat}_{n}>{post}"

    # ---------- restoring ----------

    def _format_restored_value(self, cat: str, num: int, value: str, preceding_text: str) -> str:
        hon = self._honorifics.get((cat, num))
        if not hon:
            return value
        prev = preceding_text.rstrip()
        hon_bare = hon[1:] if len(hon) > 2 and hon.startswith("ה") else hon
        if prev.endswith(hon) or prev.endswith(hon_bare):
            return value
        return f"{hon} {value}"

    def restore(self, text: str, allowed: Iterable[str] | None = None) -> tuple[str, dict]:
        """Replace numbered tags (and speaker fallback labels) by their values from the vault.

        Returns (text, info) — info holds labels and counts only:
          {"restored": {label: count}, "unknown": [labels], "policy_skipped": [labels]}
        """
        info = {"restored": {}, "unknown": [], "policy_skipped": []}
        if not isinstance(text, str) or not text:
            return text, info
        allow = self.categories if allowed is None else (frozenset(allowed) & self.categories)

        def sub(m: re.Match) -> str:
            cat = _label_category(m.group("label") or m.group("label2") or "")
            num = m.group("n") or m.group("n2")
            if cat is None or not num:
                return m.group(0)
            label = f"<{cat}_>"
            if cat not in allow:
                info["policy_skipped"].append(label)
                return m.group(0)
            n_int = int(num)
            value = self._values.get((cat, n_int))
            if value is None:
                info["unknown"].append(label)
                return m.group(0)
            info["restored"][label] = info["restored"].get(label, 0) + 1
            return self._format_restored_value(cat, n_int, value, text[: m.start()])

        out = _RESTORE.sub(sub, text)

        # Also restore fallback "דובר N" / "דוברת N" if the model wrote that for a named speaker
        if NAME in allow and self._speakers:
            def sub_spk(m: re.Match) -> str:
                idx = int(m.group("idx"))
                entry = self._speakers.get(idx)
                if entry is None:
                    return m.group(0)
                cat, n_int = entry
                value = self._values.get((cat, n_int))
                if value is None:
                    return m.group(0)
                label = f"<{cat}_>"
                info["restored"][label] = info["restored"].get(label, 0) + 1
                pre = m.group("pre") or ""
                restored_val = self._format_restored_value(cat, n_int, value, out[: m.start()] + pre)
                return f"{pre}{restored_val}"

            out = _SPEAKER_FALLBACK_REF.sub(sub_spk, out)

        info["unknown"] = sorted(set(info["unknown"]))
        info["policy_skipped"] = sorted(set(info["policy_skipped"]))
        return out, info

    def pseudonymize_speakers(self, text: str) -> str:
        """Replace 'דובר N' references to named speakers with `<שם_N>` for the pseudo copy."""
        if not isinstance(text, str) or not text or not self._speakers:
            return text

        def sub_spk(m: re.Match) -> str:
            idx = int(m.group("idx"))
            entry = self._speakers.get(idx)
            if entry is None:
                return m.group(0)
            cat, n_int = entry
            pre = m.group("pre") or ""
            return f"{pre}<{cat}_{n_int}>"

        return _SPEAKER_FALLBACK_REF.sub(sub_spk, text)

    def spans(self, text: str, info: dict | None = None) -> list[list[int]]:
        """[start, end] of every restored value found in `text` (for highlighting in the UI).
        Positions only — the values themselves are not part of the result."""
        if not isinstance(text, str) or not text or not self._values:
            return []
        candidates = set(self._values.values())
        for k, hon in self._honorifics.items():
            val = self._values.get(k)
            if val:
                candidates.add(f"{hon} {val}")
        values = sorted(candidates, key=len, reverse=True)
        rx = re.compile("|".join(re.escape(v) for v in values))
        return [[m.start(), m.end()] for m in rx.finditer(text)]


def _is_org_cue(token: str) -> bool:
    t = token.strip(_EDGE_PUNCT).replace("״", '"')
    if t in _ORG_CUES:
        return True
    for cut in (1, 2):
        if len(t) - cut >= 3 and all(c in _PREFIX_LETTERS for c in t[:cut]) and t[cut:] in _ORG_CUES:
            return True
    return False


def _strip_eaten_prefix(word: str, known: set[str], *, org_cue: bool) -> str:
    """The engine replaced "לדוד" whole with "<שם_>": the value is "דוד" when the rest also occurs
    on its own in the text (or, for an organisation, when the rest opens an organisation's name:
    "מבית" -> "בית"). Never "ה" for names — a first name does not take the definite article."""
    letters = _PREFIX_LETTERS if org_cue else "ובכלמש"
    for cut in (1, 2):
        if len(word) - cut < 2 or not all(c in letters for c in word[:cut]):
            continue
        rest = word[cut:]
        if rest in known or (org_cue and _is_org_cue(rest)):
            return rest
    return word


# Restorable surface forms: an opening bracket of any kind, a label (Hebrew or English), an
# optional separator, a number, a closing bracket. HTML escaping and bidi marks tolerated. Plus the
# bare "שם_1" a model may write without brackets.
_OPEN = r"(?:<|&lt;|＜|\[|\{\{?|«|‹|⟨|\()"
_CLOSE = r"(?:>|&gt;|＞|\]|\}\}?|»|›|⟩|\))"
_B = f"[{_BIDI}]*"
_LABELS = r"שם|ארגון|מיקום|קשר|מזהה|תאריך|PERSON|PERS|PER|NAME|ORGANI[SZ]ATION|ORG"
_RESTORE = re.compile(
    rf"{_OPEN}{_B}\s?(?P<label>{_LABELS})\s?[_\- ]?\s?(?P<n>\d{{1,3}})\s?_?{_B}{_CLOSE}"
    rf"|(?<![{_H}A-Za-z0-9_<＜;])(?P<label2>שם|ארגון)_(?P<n2>\d{{1,3}})(?![0-9])",
    re.IGNORECASE,
)


def _label_category(label: str) -> str | None:
    lab = label.strip().upper()
    if label.strip() in _CATEGORY_LABELS:
        return label.strip()
    if lab in ("PERSON", "PERS", "PER", "NAME"):
        return NAME
    if lab in ("ORG", "ORGANIZATION", "ORGANISATION"):
        return ORG
    return None


def to_generic(text: str) -> str:
    """"<שם_3>" -> "<שם_>": for copies that must not let a number be matched across texts."""
    if not isinstance(text, str):
        return text
    return _NUMBERED_ANY.sub(lambda m: f"<{m.group('cat')}_>", text)


def reid_record(info: dict, spans: list[list[int]] | None = None, *, enabled: bool = True) -> dict:
    """The record stored next to a re-identified text — labels, counts and positions only."""
    return {
        "version": 1,
        "enabled": bool(enabled),
        "restored": dict(sorted((info or {}).get("restored", {}).items())),
        "unknown": list((info or {}).get("unknown", [])),
        "policy_skipped": list((info or {}).get("policy_skipped", [])),
        "spans": spans or [],
    }


# ---------- speaker display names ----------
# Display names reach the model as the label of every line. A user may rename "דובר 1" to a real
# name; the label used to go to Gemini unfiltered. When `vault` is None (for .pii.json and Excel),
# only default labels and role words pass ("דובר N" otherwise). When `vault` is given (for LLM
# summarization/feedback), personal names in speaker labels are stored in the in-memory Vault and
# replaced by numbered tags ("<שם_1>", "הרב <שם_1>", "יועצת <שם_1>") so they are restored locally.

_DEFAULT_SPEAKER = re.compile(r"^\s*דובר(?:ת)?\s*(?P<idx>\d{1,3})\s*$")
_SPEAKER_FALLBACK_REF = re.compile(
    rf"(?<![{_H}A-Za-z0-9_])(?P<pre>[{_PREFIX_LETTERS}]{{0,2}}[-־]?)דובר(?:ת)?\s+(?P<idx>\d{{1,3}})(?![0-9])"
)
_ROLE_WORDS = frozenset(
    """יועץ יועצת היועץ היועצת פונה הפונה פונים הפונים פונות רב הרב רבנית הרבנית בן בת הזוג בעל הבעל
    אישה אשה האישה האשה מטופל מטופלת המטופל המטופלת רופא רופאה הרופא הרופאה אחות האחות מתקשר
    מתקשרת המתקשר המתקשרת לקוח לקוחה הלקוח הלקוחה מנחה המנחה אב אם אבא אמא האב האם ילד ילדה
    דובר דוברת הדובר הדוברת נציג נציגה הנציג הנציגה מזכירה המזכירה צוות מתאם מתאמת כלה חתן הכלה
    החתן ראשון שני שלישי ראשונה שנייה שניה שלישית נוסף נוספת אחר אחרת גבר הגבר מומחה מומחית
    משיב המשיב בלנית הבלנית יועץ/ת יועצ/ת פונה/ה בן/בת""".split()
)
_HONORIFIC_WORDS = frozenset(
    """רב הרב רבנית הרבנית ד״ר ד"ר דר דר' דוקטור פרופ פרופ' פרופסור""".split()
)


def _is_role_or_honorific_word(word: str) -> bool:
    w = word.strip("\"'״׳.")
    if not w:
        return False
    if w in _ROLE_WORDS or w in _HONORIFIC_WORDS or word.strip(".") in _HONORIFIC_WORDS:
        return True
    if len(w) > 2 and w[0] in "וה" and (w[1:] in _ROLE_WORDS or w[1:] in _HONORIFIC_WORDS):
        return True
    return False


def _is_honorific_word(word: str) -> bool:
    w = word.strip(".")
    w_bare = w.strip("\"'״׳.")
    if w in _HONORIFIC_WORDS or w_bare in _HONORIFIC_WORDS:
        return True
    if len(w_bare) > 2 and w_bare[0] in "וה" and w_bare[1:] in _HONORIFIC_WORDS:
        return True
    return False


def _split_speaker_label(name: str) -> tuple[str, str, str] | None:
    """Split an unsafe speaker label into `(role_prefix, person_name, honorific)` or None if not a name."""
    if not isinstance(name, str) or not name.strip():
        return None
    if re.search(r"\d", name) or "<" in name or ">" in name:
        return None
    tokens = [w.strip(".,;:!?()[]{}") for w in re.split(r"[\s,\-־/():]+", name.strip()) if w.strip(".,;:!?()[]{}")]
    if not tokens:
        return None
    prefix_words: list[str] = []
    idx = 0
    while idx < len(tokens) and _is_role_or_honorific_word(tokens[idx]):
        prefix_words.append(tokens[idx])
        idx += 1
    suffix_words: list[str] = []
    end = len(tokens)
    while end > idx and _is_role_or_honorific_word(tokens[end - 1]):
        suffix_words.insert(0, tokens[end - 1])
        end -= 1
    person_words = tokens[idx:end]
    if not person_words:
        return None
    person_name = _clean_value(" ".join(person_words))
    if not person_name:
        return None
    role_prefix = " ".join(prefix_words) or " ".join(suffix_words)
    hon_words = [w for w in prefix_words if _is_honorific_word(w)]
    honorific = " ".join(hon_words)
    return role_prefix, person_name, honorific


def speaker_label_is_safe(name: str) -> bool:
    """True for the default labels ("דובר 2") and for labels made only of role words."""
    if not isinstance(name, str) or not name.strip():
        return False
    if _DEFAULT_SPEAKER.match(name):
        return True
    if re.search(r"\d", name):
        return False
    words = [w for w in re.split(r"[\s,\-־/()]+", name.strip()) if w]
    if not words:
        return False
    for w in words:
        w = w.strip("\"'״׳.")
        if w in _ROLE_WORDS:
            continue
        if len(w) > 2 and w[0] in "וה" and w[1:] in _ROLE_WORDS:
            continue
        return False
    return True


def safe_speaker_labels(speakers: dict, vault: Vault | None = None) -> dict:
    """speaker id -> a label safe to show the model or a de-identified copy.

    When `vault` is None, any label that is not a default or a role word becomes `"דובר N"`.
    When `vault` is given, personal speaker names are stored in the in-memory `Vault` and replaced
    by numbered tags (`"<שם_1>"`, `"הרב <שם_1>"`, `"יועצת <שם_1>"`) so they are restored locally.
    """
    out: dict = {}
    for i, (sid, name) in enumerate((speakers or {}).items(), start=1):
        if vault is not None:
            out[sid] = vault.assign_speaker(i, name)
        else:
            out[sid] = name if speaker_label_is_safe(name) else f"דובר {i}"
    return out


def redact_speaker_names_in_texts(texts: list[str], speakers: dict) -> list[str]:
    """Replace literal occurrences of speaker personal names in `texts` with `<שם_>` so `Vault.number`
    aligns and numbers them with the same `<שם_N>` tag as the speaker's line label."""
    if not texts or not speakers:
        return list(texts)
    names: list[str] = []
    for raw_name in speakers.values():
        if speaker_label_is_safe(raw_name):
            continue
        parsed = _split_speaker_label(raw_name)
        if parsed is None:
            continue
        _, person_name, _ = parsed
        if len(person_name) >= 2:
            names.append(person_name)
    if not names:
        return list(texts)
    # Sort longest first so "רחל כהן" matches as a whole before any shorter name
    names = sorted(set(names), key=len, reverse=True)
    body = "|".join(re.escape(n) for n in names)
    rx = re.compile(
        rf"(?<![{_H}A-Za-z0-9_])(?P<pre>[{_PREFIX_LETTERS}]{{0,2}}[-־]?)(?P<val>{body})(?![{_H}A-Za-z0-9_])"
    )
    return [rx.sub(lambda m: f"{m.group('pre')}<{NAME}_>", t) if t else t for t in texts]


def reidentify_deliverables_for_speakers(
    result: dict, old_speakers: dict | None = None, *, categories: Iterable[str] = DEFAULT_CATEGORIES
) -> None:
    """When speakers are renamed on an already-summarized job, update `client_summary` and `feedback`
    via an in-memory `Vault` so speaker names are restored without touching `.pii.json` or `*_pseudo`."""
    cats = frozenset(categories)
    if NAME not in cats:
        return
    new_speakers = result.get("speakers") or {}
    if not new_speakers:
        return
    vault = Vault(cats)
    safe_speaker_labels(new_speakers, vault=vault)
    if not vault._speakers:
        return

    # Map old speaker personal names (if a speaker was renamed from one personal name to another)
    old_name_to_idx: list[tuple[str, int]] = []
    if old_speakers:
        for i, (sid, old_label) in enumerate(new_speakers.items(), start=1):
            prev = (old_speakers or {}).get(sid, "")
            if prev and prev != old_label and not speaker_label_is_safe(prev):
                parsed_prev = _split_speaker_label(prev)
                if parsed_prev is not None and i in vault._speakers:
                    _, prev_person, prev_hon = parsed_prev
                    if prev_hon:
                        old_name_to_idx.append((f"{prev_hon} {prev_person}", i))
                    old_name_to_idx.append((prev_person, i))

    for key in ("client_summary", "feedback"):
        text = result.get(key)
        if not isinstance(text, str) or not text.strip():
            continue
        pseudo_base = result.get(f"{key}_pseudo") or text
        # If an old speaker personal name was in `text`, turn it into "דובר {i}" before vault.restore
        working = pseudo_base
        for prev_str, spk_idx in sorted(old_name_to_idx, key=lambda x: len(x[0]), reverse=True):
            working = working.replace(prev_str, f"דובר {spk_idx}")
        restored, info = vault.restore(working)
        if restored != text:
            result[key] = restored
            if f"{key}_pseudo" not in result:
                result[f"{key}_pseudo"] = vault.pseudonymize_speakers(working)
            prev_reid = result.get(f"{key}_reid") if isinstance(result.get(f"{key}_reid"), dict) else {}
            merged_restored = dict(prev_reid.get("restored") or {})
            for k_lab, cnt in info.get("restored", {}).items():
                merged_restored[k_lab] = max(merged_restored.get(k_lab, 0), cnt)
            merged_info = {
                "restored": merged_restored,
                "unknown": prev_reid.get("unknown") or info.get("unknown") or [],
                "policy_skipped": prev_reid.get("policy_skipped") or info.get("policy_skipped") or [],
            }
            result[f"{key}_reid"] = reid_record(merged_info, vault.spans(restored), enabled=True)
