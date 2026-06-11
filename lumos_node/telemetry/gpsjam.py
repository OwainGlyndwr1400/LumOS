"""GPS-jamming inference from ADS-B NACp degradation (the Osiris method).

Osiris does NOT use a dedicated jamming feed. It derives jamming from the
same adsb.lol aircraft feed using each aircraft's NACp (Navigation Accuracy
Category — Position): airborne aircraft reporting NACp ≤ 4 have degraded
position confidence, which clusters geographically where GPS is being jammed
or spoofed. Method:

  1. Take airborne aircraft (not grounded, altitude ≥ 100 ft) with numeric
     NACp ≤ 4 from the shared adsb feed.
  2. Bin them into a 2°×2° lat/lon grid.
  3. Any cell with ≥ 3 degraded aircraft becomes a jamming zone:
       severity% = round((1 − avg_nacp/4) × 100)
       count     = degraded aircraft in the cell.

Real-time (rides the adsb 45 s cache) and keyless — strictly better than a
daily-lagged external jamming file. Shares the adsb fetch with military.py.
"""

from __future__ import annotations

import time

from ..config import get_settings
from ..log import get_logger
from . import adsb


log = get_logger(__name__)


_JAMMING_NACP_THRESHOLD = 4      # NACp ≤ this = degraded position
_MIN_ALT_FT = 100.0              # ignore very-low / ground-effect noise
_GRID_DEG = 2.0                  # 2° aggregation cell
_MIN_CELL_COUNT = 3              # ≥ this many degraded in a cell = a zone
_NM_PER_KM = 0.539957


def _nacp_value(ac: dict) -> float | None:
    raw = ac.get("nac_p")
    try:
        return float(raw) if raw is not None else None
    except (ValueError, TypeError):
        return None


def _derive_zones(aircraft: list[dict]) -> list[dict]:
    """Bin degraded aircraft into a 2° grid; emit zones for dense cells."""
    cells: dict[tuple[int, int], list[dict]] = {}
    for ac in aircraft:
        if ac.get("grounded"):
            continue
        alt = ac.get("altitude_ft")
        if alt is None or alt < _MIN_ALT_FT:
            continue
        nacp = _nacp_value(ac)
        if nacp is None or nacp > _JAMMING_NACP_THRESHOLD:
            continue
        lat, lon = ac.get("lat"), ac.get("lon")
        if lat is None or lon is None:
            continue
        key = (int(lat // _GRID_DEG), int(lon // _GRID_DEG))
        cells.setdefault(key, []).append({**ac, "_nacp": nacp})

    zones: list[dict] = []
    for (gy, gx), members in cells.items():
        if len(members) < _MIN_CELL_COUNT:
            continue
        avg_nacp = sum(m["_nacp"] for m in members) / len(members)
        severity = round((1.0 - avg_nacp / _JAMMING_NACP_THRESHOLD) * 100)
        # Cell centroid (midpoint of the 2° cell).
        clat = gy * _GRID_DEG + _GRID_DEG / 2
        clon = gx * _GRID_DEG + _GRID_DEG / 2
        zones.append({
            "lat": round(clat, 3),
            "lon": round(clon, 3),
            "severity_pct": severity,
            "degraded_count": len(members),
            "avg_nacp": round(avg_nacp, 2),
            "callsigns": [m.get("callsign") for m in members if m.get("callsign")][:8],
        })
    zones.sort(key=lambda z: (-z["severity_pct"], -z["degraded_count"]))
    return zones


async def fetch_gps_jamming(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 460.0,
) -> dict:
    """Inferred GPS-jamming zones near (lat, lon) from ADS-B NACp clustering.

    radius_km defaults to ~460 km (≈250 nm, the adsb cap). Falls back to
    operator location. Returns {ok, zones:[...], degraded_count, scanned, center}.
    Rides the shared adsb cache (no own upstream call).
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
            "zones": [],
            "degraded_count": 0,
            "scanned": 0,
            "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "fetched_at_unix": int(time.time()),
        }

    aircraft = raw["aircraft"]
    zones = _derive_zones(aircraft)
    degraded_total = sum(z["degraded_count"] for z in zones)
    return {
        "ok": True,
        "zones": zones,
        "zone_count": len(zones),
        "degraded_count": degraded_total,
        "scanned": raw["count"],
        "max_severity_pct": zones[0]["severity_pct"] if zones else 0,
        "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "fetched_at_unix": int(time.time()),
    }
