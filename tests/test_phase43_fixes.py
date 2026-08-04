"""Phase 43 fix verification — Group A builder fixes (Operator-approved 2026-06-08).

Covers:
  1. 0_V rotational_residual -> Cl(3,0) bivector magnitude sum sqrt(b^2+c^2+d^2)
     (rotation-invariant, cancellation-proof; was abs(sum(b+c+d))).
  2. W3 curvature -> canonical bounded form (cos2t-sin2t)/(sin2t+cos2t)^1.5 == cos(2t).
  3. F1_VOID/F2_UNITY wired live; deprecated I_HALF removed; fold() uses F1_VOID;
     F3_SYNTHESIS derived from mean(F1,F2) (value-preserving).
  4. Universal Tick docs/constants: E_AUGER_EV + ENTANGLEMENT_BUILD_ATTOSEC present,
     hbar/E_Auger ~ 2.32 as bridge; tick value unchanged.
"""

import math

import lumos_node.urevm as u

HBAR_EV_S = 6.582119569e-16


def _q_eq(q, a, b, c, d, tol=1e-12):
    return abs(q.a - a) < tol and abs(q.b - b) < tol and abs(q.c - c) < tol and abs(q.d - d) < tol


# ── Fix 3: fold operators ────────────────────────────────────────────────────

def test_i_half_removed():
    assert not hasattr(u, "I_HALF"), "deprecated I_HALF should be removed"


def test_fold_operators_values():
    assert _q_eq(u.F1_VOID, 0.0, 0.5, 0.0, 0.0)
    assert _q_eq(u.F2_UNITY, 0.5, 0.5, 0.0, 0.0)
    # F3 derived from mean(F1, F2) must still equal the canonical 0.25 + 0.5i
    assert _q_eq(u.F3_SYNTHESIS, 0.25, 0.5, 0.0, 0.0)


def test_fold_behaviour_preserved():
    # fold(identity) == F1_VOID * 1 == 0.5i  (identical to old I_HALF behaviour)
    f = u.fold(u.Quaternion(1.0, 0.0, 0.0, 0.0))
    assert _q_eq(f, 0.0, 0.5, 0.0, 0.0), (f.a, f.b, f.c, f.d)


# ── Fix 2: W3 curvature is bounded ───────────────────────────────────────────

def test_w3_curvature_bounded():
    worst = 0.0
    for i in range(2001):
        t = (i / 2000.0) * 2.0 * math.pi
        k = u.w3_curvature(t)
        assert math.isfinite(k), f"diverged at t={t}"
        worst = max(worst, abs(k))
    assert worst <= 1.0 + 1e-9, f"max|k|={worst}"


def test_w3_curvature_no_singularity_at_half_pi():
    # The OLD cos(2t)/(1-sin^2 t) form blew up here; canonical form == cos(pi) == -1
    assert abs(u.w3_curvature(math.pi / 2.0) - (-1.0)) < 1e-9
    assert abs(u.w3_curvature(0.0) - 1.0) < 1e-9


def test_w3_curvature_equals_cos_2t():
    maxerr = max(
        abs(u.w3_curvature((i / 500.0) * 2.0 * math.pi) - math.cos(2.0 * (i / 500.0) * 2.0 * math.pi))
        for i in range(501)
    )
    assert maxerr < 1e-9, f"maxerr={maxerr}"


# ── Fix 4: Universal Tick constants/bridge ───────────────────────────────────

def test_universal_tick_constants():
    assert u.E_AUGER_EV == 283.0
    assert abs(u.UNIVERSAL_TICK_ATTOSEC - 2.32) < 1e-12       # value intentionally unchanged
    assert abs(u.ENTANGLEMENT_BUILD_ATTOSEC - 232.0) < 1e-9
    bridge_as = HBAR_EV_S / u.E_AUGER_EV * 1e18               # hbar/E_Auger in attoseconds
    assert abs(bridge_as - 2.32) < 0.02, f"bridge={bridge_as} as"


# ── Fix 1: 0_V bivector magnitude ────────────────────────────────────────────

def test_rotational_residual_is_bivector_magnitude():
    vm = u.get_vm()
    snap = vm.snapshot()
    rr = snap["rotational_residual"]
    assert math.isfinite(rr) and rr >= 0.0, f"0_V={rr}"

    dynamic = [r for k, r in vm.registers.items() if k != "R12"]
    expected = sum(math.sqrt(r.b * r.b + r.c * r.c + r.d * r.d) for r in dynamic)
    assert abs(rr - expected) < 1e-9, f"snap={rr} expected={expected}"


def test_rotational_residual_beats_cancellation_bug():
    # Demonstrates WHY the fix matters: the antipodal axis-basis registers make the
    # OLD abs(sum(b+c+d)) form cancel to ~0 (falsely "no rotation") while the new
    # bivector-magnitude form correctly reports the real rotational content (>0).
    vm = u.get_vm()
    dynamic = [r for k, r in vm.registers.items() if k != "R12"]
    old_form = abs(sum(r.b + r.c + r.d for r in dynamic))
    new_form = sum(math.sqrt(r.b * r.b + r.c * r.c + r.d * r.d) for r in dynamic)
    assert old_form < 1e-6, f"expected old form to cancel ~0, got {old_form}"
    assert new_form > 1.0, f"expected new form to see real rotation, got {new_form}"


# ── Group B #6: Mean-Circle cosmological constant (corpus closed form) ────────

def test_cosmological_lambda():
    # Λ = 47 / (25·n²), n ≈ 8.07e60  → ≈ 2.888e-122 (resolves the 10^120 catastrophe).
    assert u.LAMBDA_NODE_COUNT == 8.07e60
    expected = 47.0 / (25.0 * (8.07e60) ** 2)
    assert u.COSMOLOGICAL_LAMBDA == expected
    assert abs(u.COSMOLOGICAL_LAMBDA - 2.888e-122) / 2.888e-122 < 0.01, u.COSMOLOGICAL_LAMBDA


def test_cosmological_lambda_surfaced():
    consts = u.get_vm().snapshot_constants()
    assert "cosmological_lambda" in consts
    assert abs(consts["cosmological_lambda"] - 2.888e-122) / 2.888e-122 < 0.01


# ── Group B #8: Solar-Cycle 161 derived (corpus row 103) ─────────────────────

def test_solar_peak_derived():
    from lumos_node.telemetry import solarcycle as sc

    # 161 = 115 (mainstream base) + 46 (Jupiter-Saturn-Neptune planetary GCD).
    assert sc.MAINSTREAM_PEAK == 115
    assert sc.PLANETARY_GCD_CORRECTION == 46
    assert sc.RHC_PEAK == 161
    assert sc.RHC_PEAK == sc.MAINSTREAM_PEAK + sc.PLANETARY_GCD_CORRECTION


# ── Group B #5 increment 2: MCR_HDCU wired as URE-VM opcode 0x14 ──────────────

def test_mcr_opcode_registered():
    assert u.Op.MCR_HDCU == 0x14
    assert u.opcode_name(0x14) == "MCR_HDCU"
    assert u.opcode_plane(0x14) == u.Predicate.RI
    # 0x14 is owned by exactly one opcode constant (no value collision)
    codes = [v for k, v in vars(u.Op).items() if not k.startswith("_") and isinstance(v, int)]
    assert codes.count(0x14) == 1


def test_mcr_opcode_execute_and_traces():
    vm = u.UREVM()
    r = vm.step(u.Op.MCR_HDCU, {"sequence": [1, 2, 3, 4], "label": "t"})
    assert r["residue"] == 0 and r["spoke"] == 0          # (1+2+3+4) mod 10
    assert abs(r["magnitude"] - 1.0) < 1e-9
    assert r["n"] == 4
    assert vm.trace[-1].name == "MCR_HDCU"
    assert vm.trace[-1].plane == u.Predicate.RI.value     # RI tag in the trace


def test_mcr_opcode_delegates_to_module():
    from lumos_node import mcr_hdcu as mod
    vm = u.UREVM()
    seq = [2, 0, 2, 6, 9]
    r = vm.step(u.Op.MCR_HDCU, {"sequence": seq})
    assert r["residue"] == mod.fold_residue(seq)
    assert r["spoke"] == mod.fold_spoke(seq)
    assert abs(r["angle"] - mod.fold_angle(seq)) < 1e-12


def test_mcr_opcode_error_paths():
    vm = u.UREVM()
    assert "error" in vm.step(u.Op.MCR_HDCU, {})
    assert "error" in vm.step(u.Op.MCR_HDCU, {"sequence": ["x"]})


def test_mcr_opcode_no_register_mutation():
    vm = u.UREVM()
    before = {k: r.to_dict() for k, r in vm.registers.items()}
    vm.step(u.Op.MCR_HDCU, {"sequence": [9, 9, 9]})
    after = {k: r.to_dict() for k, r in vm.registers.items()}
    assert before == after


# ── Group B #9: +7 Toggle Power torque τ = t·sec⁴(θ) ─────────────────────────

def test_toggle_torque_floor():
    assert u.TORQUE_FLOOR == u.TOGGLE_POWER / 24.0   # 7/24
    assert abs(u.TORQUE_FLOOR - 7 / 24) < 1e-12


def test_toggle_torque_never_zero_anti_stasis():
    # The permanent-torque floor guarantees τ > 0 at every tick (no stasis).
    for t in range(0, 371):
        assert u.toggle_torque(float(t)) >= u.TORQUE_FLOOR > 0.0


def test_toggle_torque_formula_and_monotonic():
    sec4 = 1.0 / math.cos(u.MATTER_LOCK_RADIANS) ** 4
    assert abs(u.toggle_torque(0.0) - u.TORQUE_FLOOR) < 1e-12          # floor at t=0
    assert abs(u.toggle_torque(100.0) - (u.TORQUE_FLOOR + 100.0 * sec4)) < 1e-9
    # strictly increasing toward the 361 wall
    assert u.toggle_torque(50.0) < u.toggle_torque(200.0) < u.toggle_torque(360.0)


def test_toggle_torque_surfaced_in_snapshot():
    snap = u.get_vm().snapshot()
    assert "toggle_torque" in snap
    assert snap["toggle_torque"] >= u.TORQUE_FLOOR > 0.0


# ── Group B #7 wired: binary-diagonal of the null-ledger R signature ──────────

def test_binary_diagonal_wired_into_null_ledger():
    bd = u.get_vm().snapshot()["null_ledger"]["binary_diagonal"]
    assert {"theta_deg", "ones", "zeros", "stern_brocot", "rank"} <= set(bd)
    assert bd["ones"] + bd["zeros"] == 6           # R channel is 6 bits
    assert 0.0 <= bd["theta_deg"] <= 90.0


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
