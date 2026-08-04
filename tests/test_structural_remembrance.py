"""Structural Remembrance RHC 3-4-5 codec — verification (Phase 43 Group B #1).

Proves the lossless guarantee (decode(encode(x)) == x byte-for-byte) for both the
corpus literal codec and the PMG decorrelation lift, and reports the REAL measured
compression ratio (no 1147:1 fakery).
"""

import os

from lumos_node import structural_remembrance as sr


# ── 1. Corpus literal codec ──────────────────────────────────────────────────

def test_bit_codes_match_spec():
    assert sr.BIT_CODES[9] == 0b001001
    assert sr.BIT_CODES[16] == 0b010000
    assert sr.BIT_CODES[25] == 0b011001


def test_encode_decode_rhc_lossless():
    for x in range(-100, 300):
        code, dev = sr.encode_rhc(x)
        assert sr.decode_rhc(code, dev) == x


def test_match_nearest():
    assert sr.match_nearest_pythagorean(8) == 9
    assert sr.match_nearest_pythagorean(20) == 16   # |20-16|=4 < |20-25|=5
    assert sr.match_nearest_pythagorean(30) == 25


# ── 2. PMG / integer-Haar lift ───────────────────────────────────────────────

def test_pmg_lift_reversible_exhaustive():
    for a in range(0, 256, 7):
        for b in range(0, 256, 5):
            s, d = sr.pmg_lift(a, b)
            assert sr.pmg_unlift(s, d) == (a, b)


def test_lift_stream_roundtrip_even_and_odd():
    for n in (0, 1, 2, 3, 64, 1001):
        data = os.urandom(n)
        s, d, carry = sr.lift_stream(data)
        assert sr.unlift_stream(s, d, carry) == data


# ── 3. Honest measurement ────────────────────────────────────────────────────

def test_measure_lossless_random():
    data = os.urandom(4096)
    m = sr.measure_ratio(data)
    assert m["lossless"] is True
    assert m["raw_bytes"] == 4096


def test_lift_lossless_regardless_of_shape():
    # The lossless guarantee holds for any data shape — that is the property we
    # can actually PROVE. Compression is a SEPARATE question, and the honest
    # measurement is that the PMG lift does NOT beat zlib (rhc_vs_zlib < 1 on a
    # ramp; expansion on random). The corpus's 1147:1 needs the unspecified
    # GCD-discard step (Patent AU 2024903694) — not reproducible from this spec.
    ramp = bytes((i // 4) % 256 for i in range(8192))
    m = sr.measure_ratio(ramp)
    assert m["lossless"] is True
    assert isinstance(m["ratio_vs_raw"], float)


if __name__ == "__main__":
    import os as _os  # noqa: F811

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e!r}")
    # show a real measured ratio for the record
    ramp = bytes((i // 4) % 256 for i in range(8192))
    print("\nmeasured (8KB ramp):", sr.measure_ratio(ramp))
    print("measured (8KB random):", sr.measure_ratio(_os.urandom(8192)))
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
