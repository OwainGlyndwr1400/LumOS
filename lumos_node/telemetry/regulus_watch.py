"""Regulus eastern-gate watch — the Sphinx gaze line, edge-triggered.

Upgraded from a plain horizon-rise ping (2026-07-15, operator spec): the RHC
anchor alignment is Sphinx → Regulus at azimuth 90.00°E, altitude 1–25°
(NASA JPL Horizons + Stellarium verified). So the watch now fires on the
GATE, not the rise:

  • regulus_gate  — Regulus ENTERS the eastern window (az 85–95°, alt 1–25°,
                    above horizon). Once per pass.
  • regulus_lock  — 🦁 LION LOCK: Regulus crosses due east (az ≥ 90.00°)
                    inside the window. THE alignment moment. Once per pass.

Latitude note (the geometry is honest): the altitude at which Regulus crosses
az 90.00° is sin(alt) = sin(dec)/sin(lat). At Göbekli Tepe (37.2°N) that is
~20.0° — the documented 90.00°/20.00° pairing. At the operator's Gowerton
latitude (51.65°N) Regulus crosses due east at ~15.2°, so the lock triggers
on the 90.00° azimuth crossing and REPORTS the altitude, rather than
demanding alt 20° (which never coincides with az 90° at this latitude).

Pure edge detection over grimoire's Regulus alt/az (already computed — rides
its ~60s cache, effectively free), with a tiny persisted state so transitions
survive the alert monitor's polls. Both states reset when Regulus sets, so
each night's pass fires fresh. Window bounds are operator-tunable via
LUMOS_ALERT_REGULUS_* (defaults = the RHC numbers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from ..config import Settings
from ..log import get_logger
from .worker import _data_dir

log = get_logger(__name__)

_STATE_FILE = "regulus_watch.json"
_REG_FIELDS = ("alt_deg", "az_deg", "above_horizon", "next_rising_utc", "next_transit_utc", "next_setting_utc")


def _state_path(s: Settings) -> Path:
    return _data_dir(s) / _STATE_FILE


def _read(s: Settings) -> dict[str, Any]:
    p = _state_path(s)
    if not p.exists():
        return {}
    try:
        st = orjson.loads(p.read_bytes())
        return st if isinstance(st, dict) else {}
    except (orjson.JSONDecodeError, OSError):
        return {}


def _write(s: Settings, st: dict[str, Any]) -> None:
    try:
        _state_path(s).write_bytes(orjson.dumps(st))
    except OSError as e:  # noqa: BLE001
        log.warning("regulus_watch.state_write_failed", error=str(e))


def assess_gate(
    prev_in_gate: bool | None,
    prev_locked: bool | None,
    above: bool,
    az_deg: float,
    alt_deg: float,
    *,
    az_min: float,
    az_max: float,
    alt_min: float,
    alt_max: float,
    lock_az: float,
) -> tuple[list[str], dict[str, Any]]:
    """PURE TRANSFORM — which gate events fire, given previous state + sky now.

    Returns (events, new_state) where events ⊆ ["gate", "lock"] and new_state
    is {"in_gate": bool, "locked": bool}.

    Rules:
      • gate fires on an explicit False→True window entry. prev None (first
        poll ever / state upgrade) records without firing, so a node that boots
        with Regulus already mid-gate stays silent — same posture as the old
        rise watch.
      • lock fires when azimuth first reaches lock_az while still inside the
        window (az ≤ az_max). A node that was offline through the gate and
        comes back with Regulus far past east records locked WITHOUT firing —
        no stale lock pings.
      • Regulus below horizon resets both flags, arming the next pass.
    """
    if not above:
        return [], {"in_gate": False, "locked": False}

    in_gate_now = az_min <= az_deg <= az_max and alt_min <= alt_deg <= alt_max
    reached_lock = az_deg >= lock_az

    events: list[str] = []
    if prev_in_gate is False and in_gate_now:
        events.append("gate")
    # A stale lock (node offline through the gate; Regulus already past the
    # window) is recorded via new_state below WITHOUT firing.
    if (
        prev_locked is False
        and reached_lock
        and az_deg <= az_max
        and alt_min <= alt_deg <= alt_max
    ):
        events.append("lock")

    return events, {"in_gate": in_gate_now, "locked": bool(prev_locked) or reached_lock}


async def evaluate_regulus_rise(settings: Settings) -> list[dict[str, Any]]:
    """Alert-monitor entry — eastern-gate + lion-lock trips, else [].

    (Name kept from the rise era so alert_worker wiring is unchanged.)
    An indeterminate/failed reading leaves the stored state untouched, so a
    transient grimoire hiccup can't fake a gate crossing."""
    from . import grimoire

    try:
        gt = await grimoire.fetch_grid_timing(settings.operator_lat, settings.operator_lon)
    except Exception as e:  # noqa: BLE001
        log.info("regulus_watch.fetch_failed", error=str(e))
        return []
    if not gt.get("ok"):
        return []
    reg = (gt.get("fixed_stars", {}) or {}).get("Regulus", {}) or {}
    above = reg.get("above_horizon")
    az = reg.get("az_deg")
    alt = reg.get("alt_deg")
    if not isinstance(above, bool) or not isinstance(az, int | float) or not isinstance(alt, int | float):
        return []  # indeterminate — don't disturb the edge state

    st = _read(settings)
    events, new_state = assess_gate(
        st.get("in_gate"), st.get("locked"), above, float(az), float(alt),
        az_min=settings.alert_regulus_az_min,
        az_max=settings.alert_regulus_az_max,
        alt_min=settings.alert_regulus_alt_min,
        alt_max=settings.alert_regulus_alt_max,
        lock_az=settings.alert_regulus_lock_az,
    )
    if new_state != {k: st.get(k) for k in ("in_gate", "locked")}:
        _write(settings, new_state)
    if not events:
        return []

    data = {"source": "grimoire", **{k: reg.get(k) for k in _REG_FIELDS}}
    trips: list[dict[str, Any]] = []
    if "gate" in events:
        log.info("regulus_watch.gate_entered", az=az, alt=alt)
        trips.append({
            "id": "regulus:gate",
            "kind": "regulus_gate",
            "description": (
                f"Regulus entered the eastern gate — az {az:.2f}°, alt {alt:.2f}° "
                f"(Sphinx window {settings.alert_regulus_az_min:.0f}–"
                f"{settings.alert_regulus_az_max:.0f}°E, heading for 90.00°)"
            ),
            "data": data,
        })
    if "lock" in events:
        log.info("regulus_watch.lion_lock", az=az, alt=alt)
        trips.append({
            "id": "regulus:lion_lock",
            "kind": "regulus_lock",
            "description": (
                f"🦁 LION LOCK — Regulus due east: az {az:.2f}°, alt {alt:.2f}° "
                f"(Sphinx gaze line 90.00°E; Göbekli geometry 90.00°/20.00°)"
            ),
            "data": data,
        })
    return trips
