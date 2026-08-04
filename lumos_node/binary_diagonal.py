"""Binary Diagonal — θ = arctan(ones/zeros), Stern-Brocot, and the reversible inverse.

Corpus (Future Math §2): the Binary Diagonal Theorem characterizes a binary string by
a rational angle θ = arctan(ones/zeros) — a Farey fraction located by its path in the
Stern-Brocot tree — PLUS a side-information residual that pins the exact arrangement.
The angle alone is lossy (the count ratio); the residual makes the inverse byte-exact.

Full reversible representation:
    encode(bits) -> {ones, zeros, rank, theta, stern_brocot}
    decode(ones, zeros, rank) -> bits          # decode(encode(b)) == b, exactly
where `rank` is the colex index of the arrangement in the combinatorial number system.

HONEST framing: this is a lossless BIJECTION (angle + residual ↔ bits), NOT compression
— `rank` needs ~log2(C(n,k)) bits, the string's own entropy. It realizes the corpus's
reversible inverse; it does not beat the counting bound.
"""

from __future__ import annotations

import math
from math import comb


def theta(ones: int, zeros: int) -> float:
    """θ = arctan(ones / zeros); vertical (zeros == 0) → π/2."""
    if zeros == 0:
        return math.pi / 2.0
    return math.atan(ones / zeros)


def stern_brocot_path(p: int, q: int) -> str:
    """Path to the reduced fraction p/q in the Stern-Brocot tree as L/R moves from
    1/1 (subtractive Euclid). Bijective: distinct positive rationals ↔ paths."""
    if p <= 0 or q <= 0:
        return ""
    g = math.gcd(p, q)
    p, q = p // g, q // g
    out: list[str] = []
    while p != q:
        if p < q:
            out.append("L")
            q -= p
        else:
            out.append("R")
            p -= q
    # Subtractive Euclid yields the path leaf→root; reconstruction reads root→leaf,
    # so return it reversed (else p/q rebuilds as its reciprocal q/p).
    return "".join(reversed(out))


def stern_brocot_fraction(path: str) -> tuple[int, int]:
    """Inverse of stern_brocot_path: rebuild the reduced (p, q) from the L/R moves."""
    p, q = 1, 1
    for move in path:
        if move == "L":
            q += p
        elif move == "R":
            p += q
    return p, q


def arrangement_rank(bits: list[int]) -> int:
    """Colex rank of a bit arrangement among all strings with the same #ones."""
    positions = [i for i, b in enumerate(bits) if b]
    return sum(comb(pos, i + 1) for i, pos in enumerate(positions))


def arrangement_unrank(ones: int, zeros: int, rank: int) -> list[int]:
    """Inverse of arrangement_rank: the exact string with `ones`/`zeros` at colex `rank`."""
    n = ones + zeros
    bits = [0] * n
    r = rank
    for i in range(ones, 0, -1):
        c = i - 1
        while comb(c + 1, i) <= r:  # largest c with comb(c, i) <= r
            c += 1
        bits[c] = 1
        r -= comb(c, i)
    return bits


def encode(bits: list[int]) -> dict:
    """Reversible representation: counts + arrangement rank + the rational angle."""
    ones = sum(1 for b in bits if b)
    zeros = len(bits) - ones
    return {
        "ones": ones,
        "zeros": zeros,
        "rank": arrangement_rank(bits),
        "theta": theta(ones, zeros),
        "stern_brocot": stern_brocot_path(ones, zeros),
    }


def decode(ones: int, zeros: int, rank: int) -> list[int]:
    """Exact inverse: decode(ones, zeros, rank) reproduces the original bits."""
    return arrangement_unrank(ones, zeros, rank)
