"""Binary Diagonal reversible inverse — verification (Phase 43 Group B #7).

Proves decode(encode(bits)) == bits BYTE-EXACT exhaustively (every binary string up
to length 10), the Stern-Brocot rational↔path bijection, and the θ angle.
"""

import math
from math import gcd

from lumos_node import binary_diagonal as bd


def test_roundtrip_exhaustive():
    # Every binary string of length 0..10 must reconstruct exactly.
    for n in range(0, 11):
        for x in range(2 ** n):
            bits = [(x >> i) & 1 for i in range(n)]
            enc = bd.encode(bits)
            assert bd.decode(enc["ones"], enc["zeros"], enc["rank"]) == bits


def test_rank_unrank_inverse():
    bits = [1, 0, 1, 1, 0, 0, 1]
    enc = bd.encode(bits)
    assert enc["ones"] == 4 and enc["zeros"] == 3
    assert bd.arrangement_unrank(4, 3, enc["rank"]) == bits


def test_stern_brocot_bijective():
    for p in range(1, 16):
        for q in range(1, 16):
            if gcd(p, q) != 1:
                continue
            path = bd.stern_brocot_path(p, q)
            assert bd.stern_brocot_fraction(path) == (p, q)


def test_stern_brocot_reduces():
    # 4/6 and 2/3 are the same point on the tree (same angle).
    assert bd.stern_brocot_path(4, 6) == bd.stern_brocot_path(2, 3)


def test_theta():
    assert abs(bd.theta(1, 1) - math.pi / 4) < 1e-12
    assert abs(bd.theta(1, 0) - math.pi / 2) < 1e-12        # vertical
    assert abs(bd.theta(3, 4) - math.atan(3 / 4)) < 1e-12
    assert bd.theta(0, 5) == 0.0                            # all zeros → 0


def test_edge_all_ones_all_zeros():
    for bits in ([1, 1, 1, 1], [0, 0, 0], []):
        enc = bd.encode(bits)
        assert bd.decode(enc["ones"], enc["zeros"], enc["rank"]) == bits


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
