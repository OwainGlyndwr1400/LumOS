"""Rail alert filter — only actionable trains wake Lumos (fix 2026-06-09).

Incident: with alert_rail_enabled, EVERY train on the Gowerton board tripped a
wake as its UID rolled in — all-day "on schedule, nothing to act on" narration at
~7.5k tokens each. Fix: pure-code gate in alert_worker._rail_trip_reason — wake
only on cancelled / delayed >= threshold / due within the window.
"""

from lumos_node.telemetry.alert_worker import _hhmm_to_min, _rail_trip_reason

NOW = 5 * 60  # 05:00


def _reason(call, now=NOW, due=10, delay=5):
    return _rail_trip_reason(call, now, due_window_min=due, delay_threshold_min=delay)


def test_routine_future_arrival_is_silent():
    # The spam case: on time, 70 minutes out → NO wake.
    assert _reason({"booked": "0610", "expected": "0610"}) is None


def test_due_within_window_trips():
    assert _reason({"booked": "0508", "expected": "0508"}) == "due in 8 min"
    assert _reason({"booked": "0500", "expected": "0500"}) == "due now"


def test_cancelled_always_trips():
    assert _reason({"booked": "0830", "expected": "0830", "cancelled": True}) == "CANCELLED"


def test_delay_at_threshold_trips_below_does_not():
    assert _reason({"booked": "0658", "expected": "0703"}) == "delayed 5 min (booked 0658)"
    # 1-min slip way in the future: routine, silent.
    assert _reason({"booked": "0658", "expected": "0659"}) is None


def test_due_window_zero_disables_imminence():
    # Exceptions-only mode: due-now no longer trips…
    assert _reason({"booked": "0502", "expected": "0502"}, due=0) is None
    # …but cancellations/delays still do.
    assert _reason({"booked": "0502", "expected": "0502", "cancelled": True}, due=0) == "CANCELLED"


def test_midnight_wraparound():
    # 23:55 now, train expected 00:04 → due in 9 min, not "in 1431 min".
    assert _reason({"booked": "0004", "expected": "0004"}, now=23 * 60 + 55) == "due in 9 min"
    # Delay across midnight: booked 23:58, expected 00:06 → 8 min late.
    assert _reason({"booked": "2358", "expected": "0006"}, now=23 * 60) == "delayed 8 min (booked 2358)"


def test_hhmm_parser():
    assert _hhmm_to_min("0510") == 310
    assert _hhmm_to_min("05:10") == 310
    assert _hhmm_to_min(None) is None
    assert _hhmm_to_min("?") is None
    assert _hhmm_to_min("2960") is None


def test_missing_times_fail_silent():
    # Unparseable board entry → no crash, no wake.
    assert _reason({"booked": None, "expected": "?"}) is None


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
