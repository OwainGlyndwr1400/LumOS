"""OCR-garble detection for the knowledge lane.

Some ingested corpus is unrecoverable OCR/encoding noise — letter-digit OCR
salad ("Y31dd0u W 00'os"), MIME/base64 blobs ("=?us-ascii?Q?Zjcw...?="). These
match retrieval on topic but the text is meaningless, so pruning them strictly
improves retrieval.

The operator's corpus is extremely diverse, and several *legitimate* categories
resemble garble. Each was surfaced by dry-run scans and is handled explicitly
(never counts as garble):

  • numbers, ORDINALS ("21st"), SCIENTIFIC NOTATION ("4.999E-1")   -> neutral
  • NON-LATIN scripts (Greek, Arabic, Hebrew, Cyrillic, CJK…)      -> word
  • single letters / initials / short abbreviations ("fr.", "N.J.") -> neutral
  • real Latin words                                               -> word

BROKEN-MATH escape codes ("/H11005", "/H11002" — a maths textbook whose symbols
failed to decode) are a JUDGMENT CALL: the surrounding prose is real, only the
symbols are corrupt. Default = KEEP (treated neutral, Option A: never delete
salvageable prose). The pruner's --drop-broken-math flag flips them to garble
for operators who consider corrupted-formula chunks worse than nothing.

score = mangled / (words + mangled). Pure noise -> ~1.0; genuine content (any
language, number/formula/citation-dense) -> ~0.0. Pure + tested against REAL
false-positive chunks pulled from the live scans.
"""

from __future__ import annotations

import re

_VOWELS = frozenset("aeiouyAEIOUY")
_TOKEN_RE = re.compile(r"\S+")
_STRIP = ".,;:!?()[]{}'\"<>/\\|=+*~`^_-–—…"
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)
_SCI_NOTATION = re.compile(r"^[+-]?\d[\d.,]*[eE][+-]?\d+$")
# Broken font/encoding escapes, e.g. "/H11005", "/H20885" (decoded math symbols).
_HEX_ESCAPE = re.compile(r"/H[0-9A-Fa-f]{4,5}")


def _has_non_latin_letter(core: str) -> bool:
    """A letter from a non-Latin script (Greek, Cyrillic, Hebrew, Arabic, CJK…).
    Latin + Latin-1/Extended accents stay under U+0370, so foreign words read as
    words, not noise."""
    return any(c.isalpha() and ord(c) >= 0x0370 for c in core)


def _classify(tok: str) -> str:
    """word | mangled | skip (neutral)."""
    core = tok.strip(_STRIP)
    if len(core) < 2:
        return "skip"                          # single char / initial / stray punct
    if _ORDINAL.match(core) or _SCI_NOTATION.match(core):
        return "skip"                          # "21st", "4.999E-1"
    if _has_non_latin_letter(core):
        return "word"
    letters = sum(c.isalpha() for c in core)
    if letters == 0:
        return "skip"                          # pure number / symbol run
    if any(c.isdigit() for c in core):
        return "mangled"                       # letter+digit OCR salad (Y31dd0u)
    if any(c in _VOWELS for c in core):
        return "word"
    return "skip" if len(core) <= 3 else "mangled"  # short vowelless = abbreviation


def is_broken_math(text: str) -> bool:
    """A chunk carrying /H##### font-decode escapes is a corrupted-maths chunk
    (real prose + escape codes + digit-suffixed variables like C1/x0/y2). It's
    handled as ONE unit — scoring its individual tokens wrongly flags the maths
    variables as OCR salad."""
    return bool(_HEX_ESCAPE.search(text))


def garble_score(text: str, drop_broken_math: bool = False) -> float:
    """0.0 (clean/any-language/formula-dense) -> 1.0 (pure OCR/encoding noise).

    A chunk with /H##### escapes is treated wholesale: KEPT (0.0) by default —
    the prose is real, only symbols are corrupt — or dropped (1.0) when
    drop_broken_math is set."""
    text = (text or "").strip()
    if not text:
        return 1.0
    if is_broken_math(text):
        return 1.0 if drop_broken_math else 0.0
    words = mangled = 0
    for tok in _TOKEN_RE.findall(text):
        cls = _classify(tok)
        if cls == "word":
            words += 1
        elif cls == "mangled":
            mangled += 1
    denom = words + mangled
    if denom == 0:
        return 0.0                             # only numbers/punct — data, not garble
    return round(mangled / denom, 3)


def is_garbled(text: str, threshold: float = 0.55, drop_broken_math: bool = False) -> bool:
    """True if `text` is OCR/encoding noise at/above `threshold`. Genuine content
    (any language, number/formula-dense) sits near 0; pure noise near 1."""
    return garble_score(text, drop_broken_math) >= threshold
