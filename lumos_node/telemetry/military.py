"""Military aircraft classification over the shared adsb.lol feed.

Replicates the Osiris `classifyFlight` military test — NOT ICAO hex ranges.
A decoded adsb.lol aircraft is military if ANY of three signals hold:
  1. dbFlags bit 0 set  (adsb.lol DB tags it military)
  2. type_code ∈ MILITARY_INDICATORS  (known military airframe codes)
  3. callsign matches a US-mil call-prefix pattern (RCH/KING/DUKE/...)

Consumes telemetry.adsb.fetch_adsb_raw, so it shares one upstream call + TTL
slot with gpsjam (both derive from the same fetch). Keyless — no OpenSky
OAuth2 / quota dependency.
"""

from __future__ import annotations

import re
import time

from ..config import get_settings
from ..log import get_logger
from . import adsb


log = get_logger(__name__)


# Known military airframe type codes (Osiris MILITARY_INDICATORS set).
_MILITARY_TYPE_CODES: frozenset[str] = frozenset({
    "C17", "C5M", "C130", "C30J", "KC10", "KC46", "KC35",
    "E3CF", "E3TF", "E8A", "B1B", "B2", "B52",
    "F16", "F15", "F18", "F22", "F35", "A10", "F117",
    "RC135", "E6B", "P8A", "P3", "MQ9", "RQ4", "U2", "EP3", "RC12",
    "V22", "CH47", "UH60", "AH64", "AH1Z", "MV22",
    "EUFI", "RFAL", "TORD", "TYP", "GR4",
})

# US military callsign prefixes (Osiris regex).
_MIL_CALLSIGN_RE = re.compile(r"^(RCH|KING|DUKE|EVAC|JAKE|REACH|CONVOY)\d", re.IGNORECASE)

_NM_PER_KM = 0.539957


def _is_military(ac: dict) -> bool:
    """Osiris three-signal military test."""
    if ac.get("db_flags", 0) & 1:
        return True
    tc = (ac.get("type_code") or "").upper()
    if tc and tc in _MILITARY_TYPE_CODES:
        return True
    cs = ac.get("callsign") or ""
    if cs and _MIL_CALLSIGN_RE.match(cs):
        return True
    return False


async def fetch_military_aircraft(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 370.0,
) -> dict:
    """Military aircraft within radius of (lat, lon), via the shared adsb feed.

    radius_km defaults to ~370 km (≈200 nm). Falls back to operator location
    when lat/lon omitted. Returns {ok, count, aircraft:[mil only], center}.
    No own cache key — rides adsb.fetch_adsb_raw's cache; classification is cheap.
    """
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon
    radius_nm = radius_km * _NM_PER_KM

    raw = await adsb.fetch_adsb_raw(lat, lon, radius_nm=radius_nm)
    if not raw.get("ok"):
        return {
            "ok": False,
            "error": raw.get("error", "adsb fetch failed"),
            "count": 0,
            "aircraft": [],
            "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "fetched_at_unix": int(time.time()),
        }

    mil = [a for a in raw["aircraft"] if _is_military(a)]
    # Surface why-flagged tags for transparency in tool output.
    for a in mil:
        reasons = []
        if a.get("db_flags", 0) & 1:
            reasons.append("db_flag")
        tc = (a.get("type_code") or "").upper()
        if tc in _MILITARY_TYPE_CODES:
            reasons.append(f"type:{tc}")
        if a.get("callsign") and _MIL_CALLSIGN_RE.match(a["callsign"]):
            reasons.append("callsign")
        a["military_signals"] = reasons

    return {
        "ok": True,
        "count": len(mil),
        "total_aircraft_scanned": raw["count"],
        "aircraft": mil,
        "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "fetched_at_unix": int(time.time()),
    }
