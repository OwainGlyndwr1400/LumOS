"""Regulus eastern-gate watch — edge-transition tests (upgrade 2026-07-15).

The watch fires on the Sphinx window (az 85–95°E, alt 1–25°) instead of the
plain horizon rise: `gate` once on window ENTRY, `lock` once on the due-east
crossing (az ≥ 90.00°) inside it. All transitions exercised offline against
the pure `assess_gate` state machine.
"""

from lumos_node.telemetry.regulus_watch import assess_gate

_CFG = dict(az_min=85.0, az_max=95.0, alt_min=1.0, alt_max=25.0, lock_az=90.0)


def test_gate_entry_fires_once_then_silent():
    # Rising in the east, az 86° alt 12° → entered the window.
    events, st = assess_gate(False, False, True, 86.0, 12.0, **_CFG)
    assert events == ["gate"]
    assert st == {"in_gate": True, "locked": False}
    # Still in window next poll → silent.
    events2, _ = assess_gate(st["in_gate"], st["locked"], True, 88.0, 13.5, **_CFG)
    assert events2 == []


def test_lion_lock_fires_on_due_east_crossing():
    # In gate, az crosses 90 between polls (89.4 → 90.6).
    events, st = assess_gate(True, False, True, 90.6, 15.2, **_CFG)
    assert events == ["lock"]
    assert st["locked"] is True
    # Later polls past 90 → no re-fire.
    events2, _ = assess_gate(st["in_gate"], st["locked"], True, 93.0, 17.0, **_CFG)
    assert events2 == []


def test_gate_and_lock_can_fire_same_poll():
    # Node polls slowly: az jumped 84.8 → 90.3 in one gap.
    events, st = assess_gate(False, False, True, 90.3, 15.0, **_CFG)
    assert events == ["gate", "lock"]
    assert st == {"in_gate": True, "locked": True}


def test_stale_return_past_window_records_lock_silently():
    # Node was offline through the gate; Regulus now az 120° (way past east).
    events, st = assess_gate(False, False, True, 120.0, 30.0, **_CFG)
    assert events == []            # no stale lock ping
    assert st["locked"] is True    # but recorded — won't fire later this pass


def test_set_resets_both_flags_for_next_pass():
    events, st = assess_gate(True, True, False, 270.0, -5.0, **_CFG)
    assert events == []
    assert st == {"in_gate": False, "locked": False}
    # Next pass: entry fires fresh.
    events2, _ = assess_gate(st["in_gate"], st["locked"], True, 85.5, 11.0, **_CFG)
    assert events2 == ["gate"]


def test_first_ever_poll_mid_gate_stays_silent():
    # prev None (fresh state file / upgrade from the old rise watch): record only.
    events, st = assess_gate(None, None, True, 89.0, 14.0, **_CFG)
    assert events == []
    assert st["in_gate"] is True and st["locked"] is False


def test_below_alt_band_is_not_in_gate():
    # Az in window but altitude 0.4° (below the 1° Sphinx floor) → no gate.
    events, st = assess_gate(False, False, True, 86.0, 0.4, **_CFG)
    assert events == []
    assert st["in_gate"] is False
