"""Civilian aircraft classification over the shared adsb.lol feed.

Mirrors military.py but for the CIVILIAN categories the operator alerts on —
commercial airliners, general-aviation 'private', and private/business jets.
Reuses the Osiris classifyFlight type-code sets. Military craft are excluded
here (military.py owns those); both derive from one adsb fetch + TTL slot.

Keyless. Per-category gating lives in the caller (alert_worker); this module
fetches, drops military, classifies, and trims to an EXACT km radius (the
adsb.lol /dist endpoint only bounds roughly) with a distance attached.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Iterable

from ..config import get_settings
from ..log import get_logger
from . import adsb
from .military import _is_military


log = get_logger(__name__)

_NM_PER_KM = 0.539957

# Private/business jets (Osiris PRIVATE_JET_TYPES).
_PRIVATE_JET_TYPES: frozenset[str] = frozenset({
    "G150", "G200", "G280", "GLEX", "G500", "G550", "G600", "G650", "G700",
    "GLF2", "GLF3", "GLF4", "GLF5", "GLF6", "GL5T", "GL7T", "GV", "GIV",
    "CL30", "CL35", "CL60", "BD70", "BD10",
    "C25A", "C25B", "C25C", "C500", "C510", "C525", "C550", "C560", "C56X",
    "C680", "C700", "C750",
    "E35L", "E50P", "E55P", "E545", "E550",
    "FA50", "FA7X", "FA8X", "F900", "F2TH",
    "LJ35", "LJ40", "LJ45", "LJ60", "LJ70", "LJ75",
    "PC12", "PC24", "TBM7", "TBM8", "TBM9",
    "PRM1", "SF50", "EA50", "VLJ",
})

# Known commercial airliner type codes (Osiris's explicit non-private list).
_AIRLINER_TYPES: frozenset[str] = frozenset({
    "A319", "A320", "A321", "A332", "A333", "A339", "A343", "A359", "A388",
    "B737", "B738", "B739", "B38M", "B39M", "B752", "B753", "B763", "B764",
    "B772", "B77L", "B77W", "B788", "B789", "B78X",
    "E170", "E175", "E190", "E195", "CRJ7", "CRJ9", "AT43", "AT72", "DH8D",
})

# ICAO airline callsign: 3-letter operator code + flight number (e.g. BAW123).
_AIRLINE_CODE_RE = re.compile(r"^[A-Z]{3}\d")

CIVILIAN_CATEGORIES: frozenset[str] = frozenset({"commercial", "private", "jet"})


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def classify_civilian(ac: dict) -> str:
    """Osiris civilian classification → 'jet' | 'commercial' | 'private'.

    Caller has already excluded military. Default is 'commercial'; a private-jet
    airframe is 'jet'; anything with no airline callsign AND a non-airliner type
    code is 'private' (general aviation).
    """
    tc = (ac.get("type_code") or "").upper()
    cs = (ac.get("callsign") or "").strip().upper()
    if tc in _PRIVATE_JET_TYPES:
        return "jet"
    has_airline = bool(_AIRLINE_CODE_RE.match(cs))
    if not has_airline and tc and tc not in _AIRLINER_TYPES:
        return "private"
    return "commercial"


async def fetch_civilian_aircraft(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 44.0,
    categories: Iterable[str] = ("jet",),
) -> dict:
    """Civilian aircraft within an EXACT km radius of (lat, lon), classified and
    filtered to `categories` ⊆ {commercial, private, jet}. Military excluded.

    Rides adsb.fetch_adsb_raw's cache. Returns
    {ok, count, by_category, aircraft:[...+category+distance_km], center}.
    """
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon
    wanted = {c for c in categories} & CIVILIAN_CATEGORIES
    radius_nm = radius_km * _NM_PER_KM

    raw = await adsb.fetch_adsb_raw(lat, lon, radius_nm=radius_nm)
    if not raw.get("ok"):
        return {
            "ok": False, "error": raw.get("error", "adsb fetch failed"),
            "count": 0, "by_category": {}, "aircraft": [],
            "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "fetched_at_unix": int(time.time()),
        }

    out: list[dict[str, Any]] = []
    by_cat: dict[str, int] = {}
    for a in raw["aircraft"]:
        if _is_military(a):
            continue
        if a.get("lat") is None or a.get("lon") is None:
            continue
        d = _haversine_km(lat, lon, a["lat"], a["lon"])
        if d > radius_km:  # honor the EXACT env radius (/dist only bounds roughly)
            continue
        cat = classify_civilian(a)
        if cat not in wanted:
            continue
        out.append({**a, "category": cat, "distance_km": round(d, 1)})
        by_cat[cat] = by_cat.get(cat, 0) + 1

    out.sort(key=lambda a: a["distance_km"])
    return {
        "ok": True,
        "count": len(out),
        "by_category": by_cat,
        "aircraft": out,
        "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "fetched_at_unix": int(time.time()),
    }
