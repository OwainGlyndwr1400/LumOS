"""Vitals grid block — the astro fields added so Discord pings carry the full
HUD world-state (moon sign, planetary-hour window, sidereal time, visible
planets). Pure formatting test over a sample grimoire payload."""

from lumos_node.telemetry.vitals import _fmt_grid, _hhmm


def _payload():
    return {
        "ok": True,
        "planetary_hour": {
            "ruler": "Jupiter", "hour_number": 1, "phase": "day",
            "hour_start_utc": "2026-07-15T05:11:00", "hour_end_utc": "2026-07-15T06:33:00",
        },
        "moon": {"illumination_percent": 35, "phase_name": "Waning Crescent", "zodiac_sign": "Taurus"},
        "sidereal_time": "0:22:57.12",
        "visible_planets": [{"name": "Saturn"}, {"name": "Jupiter"}, {"name": "Venus"}],
        "fixed_stars": {"Regulus": {"above_horizon": False, "alt_deg": -24.1, "az_deg": 335.2}},
        "welsh_wheel": {"next": {"name": "Gwyl Awst", "days_until": 22}},
    }


def test_grid_includes_all_new_astro_fields():
    out = _fmt_grid(_payload())
    assert "in Taurus" in out                    # moon zodiac sign
    assert "(05:11–06:33Z)" in out               # planetary-hour window
    assert "sidereal 0:22:57" in out             # sidereal (fractional trimmed)
    assert "planets up: Saturn, Jupiter, Venus" in out
    # and the pre-existing fields still render
    assert "hour Jupiter #1 day" in out
    assert "moon 35% Waning Crescent" in out
    assert "Regulus below" in out


def test_grid_degrades_when_fields_absent():
    # A sparse payload must not crash or emit empty segments.
    out = _fmt_grid({"ok": True, "moon": {"illumination_percent": 50}})
    assert "moon 50%" in out
    assert "sidereal" not in out and "planets up" not in out


def test_hhmm_handles_iso_and_datetime():
    from datetime import datetime
    assert _hhmm("2026-07-15T05:11:00") == "05:11"
    assert _hhmm(datetime(2026, 7, 15, 6, 33)) == "06:33"
    assert _hhmm(None) is None
    assert _hhmm("garbage") is None
