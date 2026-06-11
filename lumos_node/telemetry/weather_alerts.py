"""Severe-weather alerting — two independent free / no-key sources.

(1) OFFICIAL Met Office warnings via MeteoAlarm's UK JSON feed (CAP v1.2). Each
    warning carries a polygon, so we trip when the operator sits inside the
    warning area (distance 0 = "over you") or within alert_severe_wx_radius_km
    of its nearest edge — with an areaDesc name-match fallback (operator_regions)
    for any warning lacking a polygon.
(2) Open-Meteo point-forecast WATCH: trip when the next-N-hours forecast AT the
    operator's exact coords crosses a wind-gust / rain / snow / thunderstorm
    threshold. Precise + tunable, but NOT an official warning.

Both surface as alert-monitor trip dicts of kind 'severe_weather'
({id, kind, description, data}) via evaluate_severe_weather().
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import Settings
from ..log import get_logger
from . import cache as tcache


log = get_logger(__name__)

_METEOALARM_UK = "https://feeds.meteoalarm.org/api/v1/warnings/feeds-united-kingdom"
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_THUNDER_CODES = frozenset({95, 96, 99})  # WMO thunderstorm codes


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_polygon(poly: str) -> list[tuple[float, float]]:
    """CAP polygon 'lat,lon lat,lon ...' → [(lat, lon), ...]."""
    pts: list[tuple[float, float]] = []
    for token in (poly or "").split():
        if "," not in token:
            continue
        parts = token.split(",")
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    return pts


def _point_in_polygon(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    """Even-odd ray cast (horizontal ray in +lon at the point's latitude).
    poly is [(lat, lon), ...]."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lon_i = poly[i]
        lat_j, lon_j = poly[j]
        if (lat_i > lat) != (lat_j > lat):
            lon_at = lon_i + (lat - lat_i) / ((lat_j - lat_i) or 1e-12) * (lon_j - lon_i)
            if lon < lon_at:
                inside = not inside
        j = i
    return inside


def _polygon_distance_km(lat: float, lon: float, poly: list[tuple[float, float]]) -> float:
    """0.0 if the point is inside, else the min haversine to any vertex. Coarse
    but adequate for a county-scale warning polygon."""
    if not poly:
        return math.inf
    if _point_in_polygon(lat, lon, poly):
        return 0.0
    return min(_haversine_km(lat, lon, la, lo) for la, lo in poly)


def _awareness_from_event(event: str) -> str:
    e = (event or "").lower()
    if "red" in e:
        return "red"
    if "amber" in e or "orange" in e:
        return "amber"
    if "yellow" in e:
        return "yellow"
    return "unknown"


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


async def fetch_metoffice_warnings(
    lat: float, lon: float, radius_km: float, regions: str = ""
) -> dict[str, Any]:
    """Official Met Office warnings (via MeteoAlarm UK JSON) whose area covers /
    is within radius_km of (lat, lon). Returns {ok, warnings:[trip dicts]}.
    Cached 10 min (the whole UK feed, keyed once)."""
    cached = tcache.get("metoffice_uk_warnings")
    if cached is None:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    _METEOALARM_UK, timeout=_TIMEOUT, headers={"Accept": "application/json"}
                )
                r.raise_for_status()
                cached = r.json()
        except (httpx.HTTPError, ValueError) as e:
            log.info("wx.metoffice_fetch_failed", error=str(e))
            return {"ok": False, "error": str(e), "warnings": []}
        tcache.put("metoffice_uk_warnings", cached, ttl_seconds=600)

    region_set = {x.strip().lower() for x in (regions or "").split(",") if x.strip()}
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for w in (cached.get("warnings") or []):
        alert = w.get("alert") or {}
        ident = alert.get("identifier") or ""
        for info in (alert.get("info") or []):
            expires_dt = _parse_iso(info.get("expires"))
            if expires_dt is not None and expires_dt < now:
                continue  # already lapsed
            event = info.get("event") or "weather warning"
            best_dist = math.inf
            area_name: str | None = None
            matched_region = False
            for ar in (info.get("area") or []):
                desc = ar.get("areaDesc") or ""
                poly = _parse_polygon(ar.get("polygon") or "")
                if poly:
                    d = _polygon_distance_km(lat, lon, poly)
                    if d < best_dist:
                        best_dist, area_name = d, desc
                if region_set and desc and desc.strip().lower() in region_set:
                    matched_region = True
                    area_name = area_name or desc
            within = best_dist <= radius_km
            if not (within or matched_region):
                continue
            if best_dist == 0.0:
                dtxt = " (over you)"
            elif best_dist == math.inf:
                dtxt = ""
            else:
                dtxt = f" ~{best_dist:.0f} km"
            out.append({
                "id": f"wxwarn:{ident or event}:{area_name or '?'}",
                "kind": "severe_weather",
                "description": (
                    f"{event} for {area_name or 'your area'}{dtxt}"
                    f" — until {info.get('expires') or '?'}"
                ),
                "data": {
                    "source": "metoffice",
                    "event": event,
                    "level": _awareness_from_event(event),
                    "severity": info.get("severity"),
                    "onset": info.get("onset"),
                    "expires": info.get("expires"),
                    "area": area_name,
                    "headline": info.get("headline"),
                    "distance_km": None if best_dist == math.inf else round(best_dist, 1),
                },
            })
    return {"ok": True, "warnings": out}


async def fetch_weather_watch(
    lat: float, lon: float, hours: int,
    gust_mph: float, precip_mm: float, snow_cm: float,
) -> dict[str, Any]:
    """Open-Meteo point-forecast watch over the next `hours` at (lat, lon).
    Returns {ok, hits:[{kind, when, value, unit, threshold}]} — at most one hit
    per condition kind (the earliest crossing). Cached 15 min."""
    hours = max(1, min(48, int(hours)))
    cache_key = f"wx_watch_{lat:.3f}_{lon:.3f}_{hours}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_gusts_10m,precipitation,snowfall,weather_code",
        "wind_speed_unit": "mph",
        "forecast_hours": hours,
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(_OPEN_METEO_URL, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.info("wx.watch_fetch_failed", error=str(e))
        result = {"ok": False, "error": str(e), "hits": []}
        tcache.put(cache_key, result, ttl_seconds=900)
        return result

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    gusts = hourly.get("wind_gusts_10m") or []
    precip = hourly.get("precipitation") or []
    snow = hourly.get("snowfall") or []      # cm in Open-Meteo
    codes = hourly.get("weather_code") or []

    def _at(arr: list, i: int) -> Any:
        return arr[i] if i < len(arr) else None

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in range(min(len(times), hours)):
        t = times[i]
        g, p, sn, c = _at(gusts, i), _at(precip, i), _at(snow, i), _at(codes, i)
        if g is not None and g >= gust_mph and "gust" not in seen:
            hits.append({"kind": "gust", "when": t, "value": g, "unit": "mph", "threshold": gust_mph})
            seen.add("gust")
        if p is not None and p >= precip_mm and "rain" not in seen:
            hits.append({"kind": "rain", "when": t, "value": p, "unit": "mm/h", "threshold": precip_mm})
            seen.add("rain")
        if sn is not None and sn >= snow_cm and "snow" not in seen:
            hits.append({"kind": "snow", "when": t, "value": sn, "unit": "cm/h", "threshold": snow_cm})
            seen.add("snow")
        if c is not None and int(c) in _THUNDER_CODES and "thunder" not in seen:
            hits.append({"kind": "thunder", "when": t, "value": int(c), "unit": "wmo", "threshold": "95/96/99"})
            seen.add("thunder")

    result = {"ok": True, "hits": hits, "window_hours": hours}
    tcache.put(cache_key, result, ttl_seconds=900)
    return result


_WATCH_LABEL = {
    "gust": "High wind gusts",
    "rain": "Heavy rain",
    "snow": "Snow",
    "thunder": "Thunderstorm",
}


async def evaluate_severe_weather(settings: Settings, lat: float, lon: float) -> list[dict[str, Any]]:
    """Combined severe-weather trips from both enabled sources, as alert-monitor
    trip dicts (kind 'severe_weather'). Each source fails independently."""
    trips: list[dict[str, Any]] = []

    if settings.alert_metoffice_warnings_enabled:
        try:
            mw = await fetch_metoffice_warnings(
                lat, lon, settings.alert_severe_wx_radius_km, settings.operator_regions
            )
            trips.extend(mw.get("warnings", []))
        except Exception as e:  # noqa: BLE001
            log.info("wx.metoffice_eval_failed", error=str(e))

    if settings.alert_weather_watch_enabled:
        try:
            ww = await fetch_weather_watch(
                lat, lon,
                settings.weather_watch_hours,
                settings.weather_watch_gust_mph,
                settings.weather_watch_precip_mm,
                settings.weather_watch_snow_cm,
            )
            for h in ww.get("hits", []):
                trips.append({
                    "id": f"wxwatch:{h['kind']}",
                    "kind": "severe_weather",
                    "description": (
                        f"{_WATCH_LABEL.get(h['kind'], h['kind'])} forecast: "
                        f"{h['value']} {h['unit']} at {h['when']} (≥ {h['threshold']})"
                    ),
                    "data": {"source": "open-meteo", **h},
                })
        except Exception as e:  # noqa: BLE001
            log.info("wx.watch_eval_failed", error=str(e))

    return trips
