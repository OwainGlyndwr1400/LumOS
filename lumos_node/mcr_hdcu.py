"""MCR-HDCU — Modular Complex Residue / Hyperdimensional Computing Unit.

The Decimal Phase Fold F₁₀ (Future Math `04_operator_frequency_map` row 10;
`01_theorem_index` row 88 "Residue Hyperdimensional Computing"):

    ρ(d)          = exp(i·2π·d / b)            # one symbol → a unit phasor ("spoke")
    F_b(d_n … d_0) = ∏ ρ(d_i)                  # bind the spokes by complex product
                  = exp(i·2π·(Σ d_i mod b) / b)

The fold collapses a symbol sequence onto a SINGLE unit phasor on the b-spoke
wheel — a *residue* signature (Σ mod b). It is lossy BY DESIGN (a Fold, like the
F = i/2 operator): symbol order and individual values vanish, only the modular
residue survives. This is the "Residue" in Residue-HDC, and the "single angle"
the corpus row describes ("encodes multi-digit decimal sequences into single
angles", radial/spoke-wheel signature).

Binding is the complex product (the ∏ in the corpus formula). Bundling
(superposition) and the nearest-spoke readout are the standard FHRR conventions,
supplied where the corpus is silent — flagged inline so they are not mistaken for
corpus-specified behaviour.

NOT YET BUILT (next increment): the reversible high-dimensional phasor-HDC
z(x) = exp(j·φ·x) over a random basis. This module ships the corpus-explicit
scalar residue fold first; the high-D codec is a follow-on.
"""

from __future__ import annotations

import cmath
import math

DEFAULT_BASE = 10
TAU = 2.0 * math.pi


def phasor(symbol: int, base: int = DEFAULT_BASE) -> complex:
    """ρ(d) = exp(i·2π·d/base) — map one symbol to a unit phasor (a spoke)."""
    if base <= 0:
        raise ValueError("base must be positive")
    return cmath.exp(1j * TAU * (symbol % base) / base)


def digits_of(n: int, base: int = DEFAULT_BASE) -> list[int]:
    """Digits of |n| in the given base, most-significant first. 0 → [0]."""
    n = abs(int(n))
    if n == 0:
        return [0]
    out: list[int] = []
    while n:
        out.append(n % base)
        n //= base
    return list(reversed(out))


def bind(*phasors: complex) -> complex:
    """Bind phasors by complex product — the ∏ of the corpus F-fold.

    Bind is modular on the spoke wheel: phasor(a)·phasor(b) == phasor((a+b) mod b).
    """
    out = 1 + 0j
    for z in phasors:
        out *= z
    return out


def bundle(phasors: list[complex]) -> complex:
    """Superpose (bundle) phasors: the normalized vector sum. STANDARD FHRR
    convention — the corpus specifies only the bind/product. Returns the unit
    resultant, or 0j if the sum cancels."""
    s = sum(phasors, 0j)
    m = abs(s)
    return s / m if m > 1e-12 else 0j


def phase_fold(symbols: list[int], base: int = DEFAULT_BASE) -> complex:
    """F_b = ∏ ρ(symbol) — fold a symbol sequence into one unit phasor."""
    return bind(*(phasor(s, base) for s in symbols)) if symbols else 1 + 0j


def fold_residue(symbols: list[int], base: int = DEFAULT_BASE) -> int:
    """The modular residue the fold encodes: (Σ symbols) mod base."""
    return sum(s % base for s in symbols) % base


def fold_angle(symbols: list[int], base: int = DEFAULT_BASE) -> float:
    """The single fold angle in [0, 2π) — the spoke-wheel position."""
    return cmath.phase(phase_fold(symbols, base)) % TAU


def fold_spoke(symbols: list[int], base: int = DEFAULT_BASE) -> int:
    """Nearest spoke index of the fold (readout). STANDARD nearest-codeword
    decode; exact here because the fold lands on a spoke, so it equals
    fold_residue()."""
    return round(fold_angle(symbols, base) / (TAU / base)) % base


def encode_number(n: int, base: int = DEFAULT_BASE) -> complex:
    """Fold the digits of an integer into its single-phasor signature."""
    return phase_fold(digits_of(n, base), base)


def similarity(z1: complex, z2: complex) -> float:
    """Cosine similarity of two phasors = cos(Δphase) ∈ [−1, 1]."""
    m = abs(z1) * abs(z2)
    return (z1 * z2.conjugate()).real / m if m > 1e-12 else 0.0


# ── High-D phasor HDC: z(x) = exp(j·φ·x), φ = golden angle (increment 3) ──────
# Corpus (Future Math §5): z(x) = exp(j·φ·x), φ = the Golden-Ratio angle (137.5°),
# binding = ELEMENTWISE complex multiplication, on the S³/torus manifold via
# norm-preserving isometries → lossless & REVERSIBLE. The golden angle is the
# "most irrational" rotation, so encoded phasors spread maximally (the sunflower-
# floret packing) → minimal collision. This lifts the scalar F₁₀ residue fold to a
# high-dimensional, reversible associative store:
#   encode_hd  — real vector → unit-phasor vector (each component on S¹)
#   bind_hd    — elementwise complex product (phases add); pair a key with a value
#   unbind_hd  — EXACT inverse (× conj key); the reversibility guarantee
#   bundle_hd  — superpose many vectors into one (the memory trace)
#   similarity_hd — mean cos(Δphase), the readout
# Store k key→value pairs as bundle_hd([bind_hd(key_i, val_i) ...]); recall a value
# via unbind_hd(memory, key_i) then nearest-codebook by similarity_hd.
#
# DESIGN NOTES (corpus-silent choices, flagged): the corpus fixes φ=golden and
# bind=elementwise-multiply. Bundling (normalized superposition), the conj-unbind,
# and the cos readout are standard FHRR conventions supplied here.

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 2.39996 rad = 137.50776° (golden angle)


def encode_hd(x: list[float], phi: float = GOLDEN_ANGLE) -> list[complex]:
    """z(x) = exp(j·φ·x) elementwise — lift a real vector onto unit phasors."""
    return [cmath.exp(1j * phi * xk) for xk in x]


def bind_hd(a: list[complex], b: list[complex]) -> list[complex]:
    """Bind = elementwise complex product (phases add). Reversible via unbind_hd."""
    return [ak * bk for ak, bk in zip(a, b, strict=True)]


def unbind_hd(bound: list[complex], key: list[complex]) -> list[complex]:
    """Exact inverse of bind_hd: × the conjugate key. Unit phasors → conj == key⁻¹,
    so unbind_hd(bind_hd(v, key), key) == v (lossless, norm-preserving)."""
    return [zk * kk.conjugate() for zk, kk in zip(bound, key, strict=True)]


def bundle_hd(vectors: list[list[complex]]) -> list[complex]:
    """Superpose vectors: elementwise sum, each component renormalized to S¹. The
    bundle stays similar to every member — the associative-memory trace."""
    if not vectors:
        return []
    dim = len(vectors[0])
    out: list[complex] = []
    for k in range(dim):
        s = sum(v[k] for v in vectors)
        mag = abs(s)
        out.append(s / mag if mag > 1e-12 else 0j)
    return out


def similarity_hd(a: list[complex], b: list[complex]) -> float:
    """Mean cos(Δphase) over components ∈ [−1, 1] — the phasor-vector readout."""
    if not a:
        return 0.0
    total = 0.0
    for ak, bk in zip(a, b, strict=True):
        mag = abs(ak) * abs(bk)
        total += (ak * bk.conjugate()).real / mag if mag > 1e-12 else 0.0
    return total / len(a)
