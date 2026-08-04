"""Structural Remembrance — the RHC 3-4-5 lossless geometric codec.

Three pieces: two faithful to the corpus spec, plus an HONEST measurement.

1. encode_rhc / decode_rhc — the corpus's literal per-value codec (Future Math
   "Structural Remembrance"): store the residue from the nearest Pythagorean
   square baseline {9, 16, 25}, with the spec's 6-bit base codes. Lossless BY
   CONSTRUCTION: decode_rhc(encode_rhc(x)) == x. As specified this is a *residue
   transform*, not compression — for x far from {9,16,25} the deviation is as
   large as the input.

2. pmg_lift / pmg_unlift — the Product-Mean-Gap decorrelation (corpus identity
   ab = ((a+b)/2)² − ((a−b)/2)²), implemented as the reversible integer Haar
   S-transform. THIS is what decorrelates correlated data so residuals entropy-
   code small. Byte-exact reversible.

3. measure_ratio — round-trips real data and reports the MEASURED compression
   (RHC lift + zlib) vs raw zlib. Honest: on generic data the ratio is modest,
   NOT the corpus's claimed 1147:1. That headline depends on the UBBM
   "GCD-coupling garbage discard" step, which the corpus describes qualitatively
   ("95.99% informatics garbage") but does NOT give as a concrete REVERSIBLE
   algorithm (Patent AU 2024903694). Flagged, not faked.
"""

from __future__ import annotations

import zlib
from array import array

# ── 1. The corpus literal codec ──────────────────────────────────────────────

BASELINE_SQUARES: tuple[int, int, int] = (9, 16, 25)  # 3², 4², 5²
# 6-bit base codes, verbatim from the spec (a²=9→001001, b²=16→010000, c²=25→011001)
BIT_CODES: dict[int, int] = {9: 0b001001, 16: 0b010000, 25: 0b011001}
_INV_CODES: dict[int, int] = {code: base for base, code in BIT_CODES.items()}


def match_nearest_pythagorean(x: int) -> int:
    """Nearest of the 3-4-5 square baseline {9, 16, 25} to x."""
    return min(BASELINE_SQUARES, key=lambda b: abs(x - b))


def encode_rhc(x: int) -> tuple[int, int]:
    """Encode a value as (6-bit base code, deviation from the geometric ideal)."""
    base = match_nearest_pythagorean(x)
    return BIT_CODES[base], x - base


def decode_rhc(code: int, deviation: int) -> int:
    """Lossless reconstruction: base + deviation == original."""
    return _INV_CODES[code] + deviation


# ── 2. PMG / integer-Haar reversible decorrelation lift ──────────────────────

def pmg_lift(a: int, b: int) -> tuple[int, int]:
    """Reversible integer Haar/PMG lift of a pair → (smooth, detail).

    d = a − b (detail); s = b + (d >> 1) (smooth/approx-mean). The >> is floor
    arithmetic shift — exact and consistent in both directions.
    """
    d = a - b
    s = b + (d >> 1)
    return s, d


def pmg_unlift(s: int, d: int) -> tuple[int, int]:
    """Exact inverse of pmg_lift → (a, b)."""
    b = s - (d >> 1)
    a = b + d
    return a, b


def lift_stream(data: bytes) -> tuple[list[int], list[int], int | None]:
    """Lift a byte stream into (smooths, details, carry). Odd tail byte carried."""
    smooths: list[int] = []
    details: list[int] = []
    n = len(data) - (len(data) % 2)
    for i in range(0, n, 2):
        s, d = pmg_lift(data[i], data[i + 1])
        smooths.append(s)
        details.append(d)
    carry = data[n] if len(data) % 2 else None
    return smooths, details, carry


def unlift_stream(smooths: list[int], details: list[int], carry: int | None) -> bytes:
    """Exact inverse of lift_stream → original bytes."""
    out = bytearray()
    for s, d in zip(smooths, details, strict=True):
        a, b = pmg_unlift(s, d)
        out.append(a)
        out.append(b)
    if carry is not None:
        out.append(carry)
    return bytes(out)


# ── 3. Honest measurement ────────────────────────────────────────────────────

def measure_ratio(data: bytes) -> dict[str, object]:
    """Round-trip `data` through the RHC lift and report MEASURED compression.

    Returns lossless flag + sizes. `rhc_vs_zlib` > 1.0 means the RHC decorrelation
    lift beats plain zlib; `ratio_vs_raw` is the overall compression ratio. These
    are the REAL numbers — no 1147:1 unless the (unspecified) GCD-discard lands.
    """
    smooths, details, carry = lift_stream(data)
    restored = unlift_stream(smooths, details, carry)
    lossless = restored == data

    tail = bytes([carry]) if carry is not None else b""
    payload = array("h", smooths).tobytes() + array("h", details).tobytes() + tail
    rhc_zlib = len(zlib.compress(payload, 9))
    raw_zlib = len(zlib.compress(data, 9))
    return {
        "lossless": lossless,
        "raw_bytes": len(data),
        "raw_zlib": raw_zlib,
        "rhc_lift_zlib": rhc_zlib,
        "ratio_vs_raw": round(len(data) / rhc_zlib, 3) if rhc_zlib else None,
        "rhc_vs_zlib": round(raw_zlib / rhc_zlib, 3) if rhc_zlib else None,
    }
