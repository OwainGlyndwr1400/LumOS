"""MCR-HDCU Decimal Phase Fold F₁₀ — verification (Phase 43 Group B #5, increment 1).

Builds the corpus-explicit residue fold ρ(d)=e^{i2π(d/b)}, F_b=∏ρ(dᵢ). Confirms
it is a lossy residue signature (Σ mod b) with modular binding — exactly the
"Residue Hyperdimensional Computing" the corpus names.
"""

import cmath
import math

from lumos_node import mcr_hdcu as m


def test_phasor_is_unit_and_spoke():
    assert abs(m.phasor(0) - 1.0) < 1e-12          # ρ(0) = 1
    assert abs(m.phasor(5, 10) - (-1.0)) < 1e-12   # ρ(5) on base 10 = e^{iπ} = -1
    for d in range(10):
        assert abs(abs(m.phasor(d)) - 1.0) < 1e-12  # every spoke is unit-modulus


def test_phasor_is_modular():
    # symbols are taken mod base
    assert abs(m.phasor(13, 10) - m.phasor(3, 10)) < 1e-12


def test_digits_of():
    assert m.digits_of(123) == [1, 2, 3]
    assert m.digits_of(0) == [0]
    assert m.digits_of(-405) == [4, 0, 5]
    assert m.digits_of(0b1011, 2) == [1, 0, 1, 1]


def test_fold_is_residue():
    # F_b collapses to Σ mod b
    assert m.fold_residue([1, 2, 3]) == 6
    assert m.fold_residue([9, 7]) == 6          # 16 mod 10
    assert m.fold_residue([5, 5, 5, 5]) == 0    # 20 mod 10
    # the angle and spoke readout agree with the residue
    assert abs(m.fold_angle([1, 2, 3]) - (m.TAU * 6 / 10)) < 1e-9
    assert m.fold_spoke([1, 2, 3]) == 6
    assert m.fold_spoke([5, 5, 5, 5]) == 0


def test_fold_is_lossy_by_design():
    # Different sequences with the same residue fold to the SAME phasor.
    z1 = m.encode_number(123)   # 1+2+3 = 6
    z2 = m.encode_number(97)    # 9+7 = 16 ≡ 6
    z3 = m.encode_number(60)    # 6+0 = 6
    assert abs(z1 - z2) < 1e-9
    assert abs(z1 - z3) < 1e-9
    assert m.similarity(z1, z2) > 1.0 - 1e-9


def test_bind_is_modular_product():
    # bind = complex product = modular add on the spoke wheel
    assert abs(m.bind(m.phasor(3), m.phasor(4)) - m.phasor(7)) < 1e-12
    assert abs(m.bind(m.phasor(6), m.phasor(7)) - m.phasor(3)) < 1e-12  # 13 mod 10
    assert abs(m.phase_fold([1, 2, 3]) - m.phasor(6)) < 1e-12


def test_bundle_superposition():
    # bundling identical phasors returns that phasor (unit resultant)
    z = m.phasor(2)
    assert abs(m.bundle([z, z, z]) - z) < 1e-12
    # opposite spokes cancel to ~0
    assert abs(m.bundle([m.phasor(0), m.phasor(5)])) < 1e-9


def test_similarity_bounds():
    assert abs(m.similarity(m.phasor(3), m.phasor(3)) - 1.0) < 1e-12
    assert m.similarity(m.phasor(0), m.phasor(5)) < -1.0 + 1e-9   # antipodal spokes
    # base-12 quarter turn → orthogonal → ~0
    assert abs(m.similarity(m.phasor(0, 12), m.phasor(3, 12))) < 1e-9


def test_encode_number_matches_digit_fold():
    assert abs(m.encode_number(2026) - m.phase_fold([2, 0, 2, 6])) < 1e-12


# ── High-D phasor HDC (increment 3): z=exp(jφx), golden angle ────────────────

def _rand_real_vec(dim, rng):
    return [rng.uniform(0.0, 1.0) for _ in range(dim)]


def test_hd_golden_angle():
    assert abs(m.GOLDEN_ANGLE - math.pi * (3.0 - 5.0 ** 0.5)) < 1e-12
    assert abs(math.degrees(m.GOLDEN_ANGLE) - 137.50776) < 1e-3


def test_hd_encode_unit_phasors():
    for zk in m.encode_hd([0.0, 0.5, 1.0, -2.3, 42.0]):
        assert abs(abs(zk) - 1.0) < 1e-12


def test_hd_bind_adds_phase():
    # bind(exp(jφ·0.3), exp(jφ·0.4)) == exp(jφ·0.7)
    a, b = m.encode_hd([0.3]), m.encode_hd([0.4])
    assert abs(m.bind_hd(a, b)[0] - m.encode_hd([0.7])[0]) < 1e-12


def test_hd_unbind_is_exact_inverse():
    import random
    rng = random.Random(1)
    v = m.encode_hd(_rand_real_vec(128, rng))
    key = m.encode_hd(_rand_real_vec(128, rng))
    recovered = m.unbind_hd(m.bind_hd(v, key), key)
    assert max(abs(rk - vk) for rk, vk in zip(recovered, v)) < 1e-9   # lossless


def test_hd_self_similarity_is_one():
    z = m.encode_hd([0.1, 0.2, 0.3, 0.4])
    assert abs(m.similarity_hd(z, z) - 1.0) < 1e-12


def test_hd_associative_recall():
    import random
    rng = random.Random(42)
    dim, n = 256, 3
    # Spread inputs so the golden-angle phasors wrap the full circle → near-
    # orthogonal encodings (the regime HDC is designed for; narrow inputs give
    # correlated vectors and a weak margin — real but not a codec flaw).
    def vec():
        return m.encode_hd([rng.uniform(0.0, 100.0) for _ in range(dim)])

    keys = [vec() for _ in range(n)]
    vals = [vec() for _ in range(n)]
    memory = m.bundle_hd([m.bind_hd(keys[i], vals[i]) for i in range(n)])
    # Recall value 0 by unbinding with key 0 — must be the clear most-similar value.
    recall = m.unbind_hd(memory, keys[0])
    sims = [m.similarity_hd(recall, vals[j]) for j in range(n)]
    assert sims[0] == max(sims)                       # correct value wins
    assert sims[0] > 0.3                              # strong recall signal
    assert sims[0] > max(sims[1], sims[2]) + 0.2      # clearly above crosstalk


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
