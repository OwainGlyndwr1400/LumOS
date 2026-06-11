"""Satellites overhead — SatNOGS TLE (Osiris-matched source) + skyfield geometry.

Source: SatNOGS DB API (https://db.satnogs.org/api/tle/?format=json) — the same
keyless TLE feed the Osiris board uses. Returns one object per tracked satellite
as {tle0: name, tle1: line1, tle2: line2}.

Propagation + observer geometry via skyfield (wraps sgp4, does the TEME →
topocentric az/elevation transform and ground sub-point correctly). For each
satellite we compute its elevation above the operator's horizon NOW; those above
`min_elevation` are "overhead." Mission classification follows Osiris's
NORAD-name-keyword approach (ISS / Starlink / GPS / military / weather / ...).

Performance: SatNOGS returns thousands of TLEs. The TLE set is cached 6 h (they
update ~daily); the per-query "who's overhead now" propagation runs in a thread
(asyncio.to_thread) so it never blocks the event loop, and its result is cached
60 s (satellites move on a minutes timescale).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache


log = get_logger(__name__)


_SATNOGS_TLE = "https://db.satnogs.org/api/tle/?format=json"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

# Mission classification by NORAD name keyword (Osiris-style). First match wins.
_MISSION_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ISS", "ZARYA", "TIANGONG", "CSS (", "TIANHE"), "station"),
    (("STARLINK",), "comms_constellation"),
    (("ONEWEB",), "comms_constellation"),
    (("IRIDIUM", "GLOBALSTAR", "ORBCOMM", "INTELSAT", "INMARSAT", "SES"), "comms"),
    (("GPS", "NAVSTAR", "GLONASS", "GALILEO", "BEIDOU", "QZS", "IRNSS"), "navigation"),
    (("NOAA", "METEOR", "GOES", "METOP", "HIMAWARI", "FENGYUN", "ELEKTRO"), "weather"),
    (("COSMOS", "USA ", "NROL", "YAOGAN", "OFEQ", "KOSMOS", "SHIYAN", "GAOFEN"), "military_recon"),
    (("SENTINEL", "LANDSAT", "TERRA", "AQUA", "WORLDVIEW", "PLANET", "DOVE", "SKYSAT", "ICEYE"), "earth_obs"),
    (("HUBBLE", "TESS", "CHEOPS", "JWST", "XMM", "INTEGRAL"), "science"),
)


def _classify(name: str) -> str:
    up = name.upper()
    for kws, label in _MISSION_KEYWORDS:
        if any(k in up for k in kws):
            return label
    return "other"


async def fetch_tle() -> list[dict[str, str]]:
    """SatNOGS TLE set, cached 6 h. Returns [{name, line1, line2}, ...].

    Caches the empty/failure path too so a SatNOGS outage doesn't re-hammer.
    """
    cached = tcache.get("tle")
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(_SATNOGS_TLE, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.info("satellites.tle_fetch_failed", error=str(e))
        tcache.put("tle", [], ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("tle", 21600))
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data if isinstance(data, list) else []:
        name = (item.get("tle0") or "").strip()
        l1 = (item.get("tle1") or "").strip()
        l2 = (item.get("tle2") or "").strip()
        if name and l1 and l2 and name not in seen:
            seen.add(name)
            out.append({"name": name, "line1": l1, "line2": l2})
    log.info("satellites.tle_loaded", count=len(out))
    tcache.put("tle", out, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("tle", 21600))
    return out


def _compute_overhead(
    tles: list[dict[str, str]],
    lat: float,
    lon: float,
    min_elevation: float,
    limit: int,
) -> list[dict[str, Any]]:
    """SYNC (runs in a thread) — propagate every TLE to NOW, keep those above
    `min_elevation` over the observer. Builds the skyfield objects here so the
    heavy work stays off the event loop.

    Per-satellite errors (decayed orbits, malformed TLE) are skipped — sgp4
    raises on some objects and we never want one bad TLE to kill the sweep.
    """
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale(builtin=True)  # offline — no leap-second download
    t = ts.now()
    observer = wgs84.latlon(lat, lon)

    results: list[dict[str, Any]] = []
    for tle in tles:
        try:
            sat = EarthSatellite(tle["line1"], tle["line2"], tle["name"], ts)
            alt, az, dist = (sat - observer).at(t).altaz()
            elev = alt.degrees
            if elev < min_elevation:
                continue
            sub = wgs84.subpoint(sat.at(t))
            results.append({
                "name": tle["name"],
                "mission": _classify(tle["name"]),
                "elevation_deg": round(elev, 1),
                "azimuth_deg": round(az.degrees, 1),
                "range_km": round(dist.km, 1),
                "sub_lat": round(sub.latitude.degrees, 3),
                "sub_lon": round(sub.longitude.degrees, 3),
                "altitude_km": round(sub.elevation.km, 1),
            })
        except Exception:  # noqa: BLE001 — one bad TLE must not kill the sweep
            continue

    results.sort(key=lambda s: -s["elevation_deg"])  # highest in sky first
    return results[:limit]


async def fetch_satellites_overhead(
    lat: float | None = None,
    lon: float | None = None,
    min_elevation: float = 10.0,
    limit: int = 30,
) -> dict[str, Any]:
    """Satellites currently above the observer's horizon (elevation ≥ min).

    Falls back to operator location. Returns {ok, count, satellites:[...],
    scanned, center, fetched_at}. Rides the 6 h TLE cache + 60 s result cache.
    """
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    cache_key = f"sat_passes_{lat:.3f}_{lon:.3f}_{min_elevation:.0f}_{limit}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    tles = await fetch_tle()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not tles:
        result = {
            "ok": False,
            "error": "no TLE data available (SatNOGS)",
            "count": 0,
            "satellites": [],
            "scanned": 0,
            "center": {"lat": lat, "lon": lon, "min_elevation_deg": min_elevation},
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("sat_passes", 60))
        return result

    try:
        overhead = await asyncio.to_thread(
            _compute_overhead, tles, lat, lon, min_elevation, limit
        )
    except Exception as e:  # noqa: BLE001 — propagation sweep failure
        log.warning("satellites.propagation_failed", error=str(e))
        result = {
            "ok": False,
            "error": f"propagation failed: {e}",
            "count": 0,
            "satellites": [],
            "scanned": len(tles),
            "center": {"lat": lat, "lon": lon, "min_elevation_deg": min_elevation},
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("sat_passes", 60))
        return result

    result = {
        "ok": True,
        "count": len(overhead),
        "satellites": overhead,
        "scanned": len(tles),
        "center": {"lat": lat, "lon": lon, "min_elevation_deg": min_elevation},
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("sat_passes", 60))
    return result
