"""OCR-garble detector — must flag unrecoverable noise while sparing the
operator's genuinely-diverse content. Every _MUST_KEEP string is a REAL chunk
from the live scans that an earlier detector version wrongly flagged; they lock
in the categories (numbers, ordinals, scientific notation, non-Latin scripts,
scholarly codes/abbreviations) that must never be treated as garble."""

from lumos_node.knowledge.garble import garble_score, is_garbled

_MUST_KEEP = [
    # number-dense but genuine
    "Sidereal time is calculated using the formula: (0.46061837 + 360.98564736629 * (jd - 245",
    "Dendera is referenced extensively on pages i. 93, 97, 421, 426, 429, 446, 464, 484 and ii",
    "Consciousness, Reality, and Recursive Harmonics - Ibid. Consciousness is the substrate",
    # ordinals (7th / 21st …)
    "For Aries, proper speculation hours after Sunrise are: Sunday (7th, 14th, 21st), Monday (2nd)",
    # scholarly codes + repeated abbreviations
    "Dead Sea Scroll 4Q381 includes fragments: fr. 1, fr. 15, fr. 17, fr. 24, fr. 31, fr. 33",
    "The Index of Qumran Texts includes entries for 4Q texts from 4Q156 to 4Q186",
    "Scholarly references for 4Q180 include J. M. Allegro and A. A. Anderson, DJD, V, 77-9",
    # scientific notation
    "The asteroid non-gravitational force model parameters are: AMRAT=0., A1=4.999999873689E-1",
    # non-Latin scripts (Greek lexicon, Arabic verse)
    "ἀναγχάζειν ANATKAZE 71,14; 92,5.11; 114, 930249, 5 etc. ἀνάγχη ἈΝΆΓΚΗ 83, 9; 88, 35",
    "The user provided the Arabic verse: وَقَٰتِلُوهُمۡ حَتَّىٰ لَا تَكُونَ فِتۡنَةٞ وَيَكُون",
    # normal citation/address wrongly flagged before (N.J., zip, abbreviations)
    "Electronics Engineers, Inc. (IEEE), 445 Hoes Lane, Piscataway, N. J. 08854, website at",
    "Asteroid non-gravitational force model parameters: AMRAT=0. m^2/kg, A1=4.999999873689E-1",
]

# Broken-math escape-code chunks — KEPT by default (Option A: real prose, only
# the symbols are corrupt), but flagged when --drop-broken-math is passed. The
# second is the REAL chunk that flagged despite Option A (its maths variables
# C1/x0/y2 scored as OCR salad) — the whole-chunk rule must now keep it.
_BROKEN_MATH = [
    "/H20885eaxsinbx dx /H11005 (asinbx/H11002bcosbx) /H11001c /H20885eaxcosbx dx",
    "r(x0)/H11005Yr(x0). y*(x 0)/H11005Y(x0),C1, C2y*(x) /H11005C1y1(x)/H11001C2 y2(x).c1/H11005",
]

_MUST_DROP = [
    "CG 0 09'0 Oh'O 02 0 00'0 02'0- 0r'O- 09'p- OH' ZHO Y31dd0u W 00'os",
    "J wQ UQN N ^= 00 00 r-1 ZT N O 1^ U-1 O00 '-i 00 00 LD O O O LD '--1",
    "=?us-ascii?Q?ZjcwzLJjLjy3VVl+eVgbOLLF3cFnD14Rt4XsfvnuEmwPWeERZWaFMmXtvLFB?=",
]


def test_diverse_real_content_is_kept():
    for c in _MUST_KEEP:
        assert not is_garbled(c), (c, garble_score(c))
        assert garble_score(c) < 0.45, (c, garble_score(c))


def test_ocr_and_encoding_noise_is_flagged():
    # Note the deliberate precision-over-recall trade: short vowelless clusters
    # (ZHO/CG) are neutral so real abbreviations (fr./DJD) survive, so this
    # noise type leans on its letter-digit manglings and lands ~0.57 — flagged
    # at the 0.55 default, but with less margin than a MIME blob (1.0). We'd
    # rather miss a little noise than ever delete a genuine chunk.
    for g in _MUST_DROP:
        assert is_garbled(g), (g, garble_score(g))
        assert garble_score(g) >= 0.55, (g, garble_score(g))


def test_populations_separate_with_margin():
    worst_kept = max(garble_score(c) for c in _MUST_KEEP)
    best_dropped = min(garble_score(g) for g in _MUST_DROP)
    assert worst_kept + 0.15 < best_dropped, (worst_kept, best_dropped)


def test_numbers_ordinals_scinotation_never_garble():
    assert garble_score("16, 20, 22, 26, 28, 32, 38, 40, 44") == 0.0
    assert garble_score("1st 2nd 3rd 7th 14th 21st 22nd") == 0.0
    assert garble_score("4.999999873689E-1 360.98564736629 2.32E-18") == 0.0


def test_non_latin_word_is_not_garble():
    assert not is_garbled("ἀνάγχη ἈΝΆΓΚΗ")          # Greek
    assert not is_garbled("وَيَكُونَ فِتۡنَةٞ")            # Arabic


def test_broken_math_kept_by_default_dropped_on_flag():
    for bm in _BROKEN_MATH:
        # Default = keep the whole chunk (real prose + corrupt symbols/vars).
        assert not is_garbled(bm), (bm, garble_score(bm))
        assert garble_score(bm) == 0.0
        # With the flag, the whole /H##### chunk flags for pruning.
        assert is_garbled(bm, drop_broken_math=True)
        assert garble_score(bm, drop_broken_math=True) == 1.0


def test_bracket_text_does_not_break_scoring():
    # Chunk text with [..] (which crashed the CLI's Rich markup) must score fine.
    assert garble_score("solutions [/H110024, /H1100219, 13] and more") >= 0.0


def test_empty_is_garbled():
    assert garble_score("") == 1.0 and is_garbled("")
