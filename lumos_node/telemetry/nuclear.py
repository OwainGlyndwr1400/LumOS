"""Nuclear facilities nearby — bundled curated dataset + haversine filter.

Batch 4 (last Aether Scope data source). Unlike the other intel feeds this has
NO upstream: there is no clean free live feed for nuclear-facility locations, so
we ship a curated, web-verified PRIS-style snapshot (telemetry/data/
nuclear_facilities.json) and answer proximity queries locally. Facility
locations are public knowledge (IAEA PRIS / Wikipedia); the dataset is for
situational awareness, not targeting.

Pattern note: this is the only purely-static source, so the failure mode isn't
"upstream outage" but "packaging slip" — a missing/malformed data file degrades
to {ok: False} rather than crashing. Loads once (lru_cache for the process
lifetime), haversine-filters to a radius around the operator (or an explicit
point), returns nearest-first with distance + compass bearing. {ok, ...}
contract like the other intel fetchers.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache


log = get_logger(__name__)

_DATA_PATH = Path(__file__).parent / "data" / "nuclear_facilities.json"


@lru_cache(maxsize=1)
def _load_facilities() -> tuple[dict[str, Any], ...]:
    """Load + validate the bundled dataset once (cached for process lifetime —
    the file is static). Returns () if the file is missing/malformed so a
    packaging slip degrades gracefully instead of crashing the tool. Tuple (not
    list) so lru_cache returns an immutable, safely-shared value."""
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("nuclear.dataset_load_failed", error=str(e), path=str(_DATA_PATH))
        return ()
    rows = raw.get("facilities", []) if isinstance(raw, dict) else raw
    out: list[dict[str, Any]] = []
    for f in rows:
        try:
            lat = float(f["lat"])
            lon = float(f["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        out.append(
            {
                "name": f.get("name", "unknown"),
                "country": f.get("country", ""),
                "lat": lat,
                "lon": lon,
                "type": f.get("type", "power"),
                "status": f.get("status", "unknown"),
                "operator": f.get("operator", ""),
            }
        )
    log.info("nuclear.dataset_loaded", count=len(out))
    return tuple(out)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088  # mean Earth radius (km)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


_COMPASS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)


def _compass(bearing: float) -> str:
    return _COMPASS[int((bearing + 11.25) % 360 // 22.5)]


async def fetch_nuclear_facilities(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 300.0,
    limit: int = 25,
) -> dict[str, Any]:
    """Nuclear facilities within radius_km of a point (operator default).

    Returns {ok, count, facilities:[...nearest-first w/ distance_km, bearing],
    center, total_in_db, fetched_at}. Local compute over the static dataset; the
    24 h cache entry is mostly cosmetic (matches the documented plan) but keeps
    the {source → quota} report uniform with the live feeds.
    """
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    cache_key = f"nuclear_{lat:.3f}_{lon:.3f}_{radius_km:.0f}_{limit}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    facilities = _load_facilities()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if not facilities:
        result = {
            "ok": False,
            "error": "nuclear-facilities dataset unavailable (missing/empty)",
            "count": 0,
            "facilities": [],
            "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "total_in_db": 0,
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("nuclear", 86400))
        return result

    near: list[dict[str, Any]] = []
    for f in facilities:
        d = _haversine_km(lat, lon, f["lat"], f["lon"])
        if d <= radius_km:
            b = _bearing_deg(lat, lon, f["lat"], f["lon"])
            near.append(
                {**f, "distance_km": round(d, 1), "bearing_deg": round(b), "bearing": _compass(b)}
            )
    near.sort(key=lambda x: x["distance_km"])
    near = near[:limit]

    result = {
        "ok": True,
        "count": len(near),
        "facilities": near,
        "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "total_in_db": len(facilities),
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("nuclear", 86400))
    return result
