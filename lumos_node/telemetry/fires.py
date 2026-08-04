"""NASA FIRMS active-fire hotspots — satellite fire detection near a point.

Sources (keyless 24 h global CSVs, same files the Osiris board pulls):
  MODIS C6.1:      /data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv
  VIIRS S-NPP C2:  /data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv

Each row is one satellite-detected thermal anomaly (fire pixel) in the last
24 h: position, acquisition time, instrument, confidence, and FRP (fire
radiative power, MW). The global files are a few MB / tens of thousands of
rows; we parse once (30 min cache — FIRMS NRT updates on that order), then
per-query filtering is a cheap in-memory bbox prefilter + haversine.

Confidence is normalized across instruments to low/nominal/high (MODIS uses
0-100 ints, VIIRS uses l/n/h letters). The alert path drops "low" — MODIS
low-confidence pixels are frequently sun-glint/warm-surface false positives.

Alert dedup: hotspots are clustered to a ~1.1 km grid cell (2-decimal degree
rounding) so one burning field = one identity, not a new ping per satellite
pass wobble. A source-level cooldown (alert_fire_cooldown_minutes) throttles
the kind as a whole, mirroring the rail/aircraft/severe-wx pattern.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import Settings
from ..log import get_logger
from . import cache as tcache

log = get_logger(__name__)

_FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/data/active_fire"
_FIRMS_URLS = (
    f"{_FIRMS_BASE}/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv",
    f"{_FIRMS_BASE}/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_24h.csv",
)
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)  # global CSVs are a few MB

_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _bearing_compass(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return _COMPASS[int((deg + 22.5) // 45) % 8]


def _norm_confidence(raw: str) -> str:
    """MODIS 0-100 int OR VIIRS l/n/h letter → low | nominal | high."""
    v = raw.strip().lower()
    if v in {"l", "low"}:
        return "low"
    if v in {"n", "nominal"}:
        return "nominal"
    if v in {"h", "high"}:
        return "high"
    try:
        n = int(float(v))
    except ValueError:
        return "nominal"
    if n < 30:
        return "low"
    if n < 80:
        return "nominal"
    return "high"


def parse_firms_csv(text: str) -> list[dict[str, Any]]:
    """PURE TRANSFORM — one FIRMS CSV → hotspot dicts. Header-driven so the
    MODIS and VIIRS column layouts both parse; malformed rows are skipped."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    try:
        i_lat = header.index("latitude")
        i_lon = header.index("longitude")
    except ValueError:
        return []

    def col(name: str) -> int | None:
        return header.index(name) if name in header else None

    i_date, i_time = col("acq_date"), col("acq_time")
    i_sat, i_instr = col("satellite"), col("instrument")
    i_conf, i_frp, i_dn = col("confidence"), col("frp"), col("daynight")

    out: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) <= i_lon:
            continue
        try:
            lat = float(parts[i_lat])
            lon = float(parts[i_lon])
        except ValueError:
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        try:
            frp = float(parts[i_frp]) if i_frp is not None and parts[i_frp] else None
        except (ValueError, IndexError):
            frp = None
        hhmm = parts[i_time].zfill(4) if i_time is not None and len(parts) > i_time else ""
        out.append({
            "lat": lat,
            "lon": lon,
            "acq_date": parts[i_date] if i_date is not None and len(parts) > i_date else None,
            "acq_time_utc": f"{hhmm[:2]}:{hhmm[2:]}" if len(hhmm) == 4 else None,
            "satellite": parts[i_sat] if i_sat is not None and len(parts) > i_sat else None,
            "instrument": parts[i_instr] if i_instr is not None and len(parts) > i_instr else None,
            "confidence": _norm_confidence(parts[i_conf]) if i_conf is not None and len(parts) > i_conf else "nominal",
            "frp_mw": frp,
            "daynight": parts[i_dn].strip() if i_dn is not None and len(parts) > i_dn else None,
        })
    return out


async def _fetch_fires_raw() -> list[dict[str, Any]]:
    """Both global 24 h files, parsed + concatenated, cached 30 min.
    Sources fail independently; total failure caches [] so an outage doesn't
    re-download multi-MB files every poll."""
    cached = tcache.get("firms_raw")
    if cached is not None:
        return cached
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for url in _FIRMS_URLS:
            try:
                r = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
                r.raise_for_status()
                rows.extend(parse_firms_csv(r.text))
            except (httpx.HTTPError, ValueError) as e:
                log.info("fires.fetch_failed", url=url, error=str(e))
    log.info("fires.raw_loaded", hotspots=len(rows))
    tcache.put("firms_raw", rows)
    return rows


async def fetch_fires_nearby(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 50.0,
    limit: int = 25,
    min_confidence: str = "nominal",
) -> dict[str, Any]:
    """Fire hotspots within radius of a point, nearest first.

    Falls back to operator location. {ok, count, fires:[...], center, fetched_at}.
    Rides the 30 min raw cache; the filtered result caches 5 min per location.
    """
    from ..config import get_settings

    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    rank = {"low": 0, "nominal": 1, "high": 2}
    min_rank = rank.get(min_confidence.lower(), 1)
    cache_key = f"fires_{lat:.2f}_{lon:.2f}_{radius_km:.0f}_{min_rank}_{limit}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    raw = await _fetch_fires_raw()
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")

    # Cheap bbox prefilter, then exact haversine on survivors only.
    dlat = radius_km / 111.0
    coslat = math.cos(math.radians(lat)) or 1e-6
    dlon = radius_km / (111.0 * abs(coslat))
    hits: list[dict[str, Any]] = []
    for h in raw:
        if abs(h["lat"] - lat) > dlat or abs(h["lon"] - lon) > dlon:
            continue
        if rank.get(h["confidence"], 1) < min_rank:
            continue
        d = _haversine_km(lat, lon, h["lat"], h["lon"])
        if d > radius_km:
            continue
        hits.append({
            **h,
            "distance_km": round(d, 1),
            "bearing": _bearing_compass(lat, lon, h["lat"], h["lon"]),
        })
    hits.sort(key=lambda x: x["distance_km"])

    result = {
        "ok": True,
        "count": len(hits),
        "fires": hits[:limit],
        "scanned": len(raw),
        "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("fires", 300))
    return result


# ── Alert-monitor integration ────────────────────────────────────────────────


def cluster_cell(lat: float, lon: float) -> str:
    """~1.1 km dedup cell — one burning area = one alert identity."""
    return f"{round(lat, 2):.2f}_{round(lon, 2):.2f}"


async def evaluate_fires(settings: Settings) -> list[dict[str, Any]]:
    """Alert-monitor entry — hotspot trips within alert_fire_radius_km.
    Low-confidence pixels never wake; per-cell ids dedup via the monitor's
    identity cooldown; the source cooldown throttles the kind as a whole."""
    res = await fetch_fires_nearby(
        settings.operator_lat, settings.operator_lon,
        radius_km=settings.alert_fire_radius_km,
        limit=50,
        min_confidence="nominal",
    )
    if not res.get("ok"):
        return []
    trips: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for h in res.get("fires", []):
        cell = cluster_cell(h["lat"], h["lon"])
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        frp_txt = f", FRP {h['frp_mw']:.0f} MW" if h.get("frp_mw") else ""
        when = f" at {h['acq_time_utc']} UTC" if h.get("acq_time_utc") else ""
        trips.append({
            "id": f"fire:{cell}",
            "kind": "wildfire",
            "description": (
                f"Satellite fire hotspot ~{h['distance_km']:.0f} km {h['bearing']} "
                f"({h.get('instrument') or 'satellite'}, {h['confidence']} confidence"
                f"{frp_txt}{when})"
            ),
            "data": h,
        })
    if trips:
        log.info("fires.tripped", n=len(trips))
    return trips
