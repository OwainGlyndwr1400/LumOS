"""Regulus horizon-crossing watch — edge-triggered ping.

Fires ONCE when Regulus RISES (crosses the horizon below→above), not while it
merely stays up. Regulus is the RHC anchor (Sphinx–Regulus correlation), so its
nightly rise is a meaningful marker. Pure edge detection over grimoire's
`above_horizon` bool, with a tiny persisted state so the transition is detected
across the alert monitor's polls. Rides grimoire's ~60s cache — effectively free.
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
_REG_FIELDS = ("above_horizon", "next_rising_utc", "next_transit_utc", "next_setting_utc")


def _state_path(s: Settings) -> Path:
    return _data_dir(s) / _STATE_FILE


def _read(s: Settings) -> dict[str, Any]:
    p = _state_path(s)
    if not p.exists():
        return {"above": None}
    try:
        st = orjson.loads(p.read_bytes())
        return st if isinstance(st, dict) else {"above": None}
    except (orjson.JSONDecodeError, OSError):
        return {"above": None}


def _write(s: Settings, st: dict[str, Any]) -> None:
    try:
        _state_path(s).write_bytes(orjson.dumps(st))
    except OSError as e:  # noqa: BLE001
        log.warning("regulus_watch.state_write_failed", error=str(e))


async def evaluate_regulus_rise(settings: Settings) -> list[dict[str, Any]]:
    """One `regulus_rise` trip on the below→above transition, else []. Edge-
    triggered: fires once per crossing (~once a day), never while Regulus stays
    up. An indeterminate/failed reading leaves the stored state untouched, so a
    transient grimoire hiccup can't fake a crossing."""
    from . import grimoire

    try:
        gt = await grimoire.fetch_grid_timing(settings.operator_lat, settings.operator_lon)
    except Exception as e:  # noqa: BLE001
        log.info("regulus_watch.fetch_failed", error=str(e))
        return []
    if not gt.get("ok"):
        return []
    reg = (gt.get("fixed_stars", {}) or {}).get("Regulus", {}) or {}
    current = reg.get("above_horizon")
    if not isinstance(current, bool):
        return []  # indeterminate — don't disturb the edge state

    st = _read(settings)
    prev = st.get("above")
    if prev != current:
        st["above"] = current
        _write(settings, st)

    # Rise = an explicit below→above flip. prev is None on the first-ever poll, so
    # a node that boots with Regulus ALREADY up does not false-ping (None ≠ False).
    if not (prev is False and current is True):
        return []

    bits: list[str] = []
    if reg.get("next_transit_utc"):
        bits.append(f"transits {reg['next_transit_utc']}")
    if reg.get("next_setting_utc"):
        bits.append(f"sets {reg['next_setting_utc']}")
    tail = (" — " + ", ".join(bits)) if bits else ""
    log.info("regulus_watch.rose")
    return [{
        "id": "regulus:rise",
        "kind": "regulus_rise",
        "description": f"Regulus has risen above the horizon{tail}",
        "data": {"source": "grimoire", **{k: reg.get(k) for k in _REG_FIELDS}},
    }]
