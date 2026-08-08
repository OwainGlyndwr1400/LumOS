"""Global-intel tools (Aether Scope Batch 1) — Lumos queries the same feeds
the Osiris board pulls: military aircraft + GPS jamming (shared adsb.lol),
OSINT news (Telegram + RSS), and derived conflict indicators.

All read-only. Each opens its own httpx client where needed and falls back to
the operator's configured lat/lon when location args are omitted (same pattern
as aircraft_overhead in telemetry_tools.py).
"""

from __future__ import annotations

import httpx

from ..log import get_logger
from ..telemetry import (
    conflict,
    fires,
    gdacs,
    gpsjam,
    grimoire,
    maritime,
    military,
    news,
    nuclear,
    outages,
    satellites,
)
from . import register

log = get_logger(__name__)


@register(
    name="military_aircraft_overhead",
    description=(
        "Military aircraft currently transponding within a radius of a point, "
        "via adsb.lol (keyless). Call when the operator asks about military "
        "flights / unusual air activity near them or a named place. Classifies "
        "military by adsb.lol DB flag, known airframe type codes, and US-mil "
        "callsign prefixes (RCH/REACH/etc). Defaults to operator location when "
        "lat/lon omitted. Returns count + per-aircraft callsign, type, altitude, "
        "heading, and which signal flagged it military."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "radius_km": {"type": "number", "default": 370.0, "description": "Search radius km (max ~460)."},
        },
        "required": [],
    },
)
async def military_aircraft_overhead(
    lat: float | None = None, lon: float | None = None, radius_km: float = 370.0
) -> dict:
    return await military.fetch_military_aircraft(lat=lat, lon=lon, radius_km=radius_km)


@register(
    name="gps_jamming_status",
    description=(
        "Inferred GPS-jamming zones near a point, derived from ADS-B navigation "
        "accuracy (NACp) degradation clustering — the same method the Osiris "
        "board uses. Call when the operator asks about GPS jamming/spoofing or "
        "navigation interference. Clusters of aircraft reporting degraded "
        "position confidence (NACp<=4) indicate jamming. Cross-reference with "
        "geomagnetic data: a Kp storm also degrades GPS, so high jamming during "
        "a solar storm may be space weather, not terrestrial. Defaults to "
        "operator location. Returns jamming zones with severity %, degraded "
        "aircraft count, and affected callsigns."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "radius_km": {"type": "number", "default": 460.0, "description": "Search radius km (max ~460)."},
        },
        "required": [],
    },
)
async def gps_jamming_status(
    lat: float | None = None, lon: float | None = None, radius_km: float = 460.0
) -> dict:
    return await gpsjam.fetch_gps_jamming(lat=lat, lon=lon, radius_km=radius_km)


@register(
    name="get_news_feed",
    description=(
        "Current OSINT news headlines from public Telegram channels "
        "(OSINTtechnical, Faytuks, Liveuamap, CyberKnow) with RSS fallback "
        "(BBC World, Al Jazeera). Call when the operator asks what's happening "
        "in the world / breaking news / OSINT. Each item has title, source, "
        "published time, link, and a risk score. Returns recent items sorted "
        "newest first."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50, "description": "Max items."},
        },
        "required": [],
    },
)
async def get_news_feed(limit: int = 20) -> dict:
    return await news.fetch_news(limit=limit)


@register(
    name="get_conflict_status",
    description=(
        "Conflict / war indicators derived from world news — headlines filtered "
        "by a conflict lexicon (strike, missile, troops, airstrike, etc.) and "
        "scored for severity. Call when the operator asks about conflict, war, "
        "geopolitical escalation, or 'is anything kicking off'. Returns an "
        "overall conflict score and the hottest items."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
async def get_conflict_status() -> dict:
    async with httpx.AsyncClient() as client:
        return await conflict.fetch_conflict_indicators(client)


@register(
    name="satellites_overhead",
    description=(
        "Satellites currently passing above a location's horizon, from the FULL "
        "CelesTrak active catalog (~11k objects incl. every Starlink, ISS, OneWeb; "
        "SatNOGS fallback) propagated in real time. Call when the operator asks what "
        "satellites / spacecraft are overhead, or about satellite movement above "
        "them. Defaults to operator location. Returns satellites above the horizon "
        "(elevation >= min) with name, NORAD id + international designator, owner "
        "COUNTRY, launch date, object type, mission (station/navigation/comms/"
        "military/weather/earth_obs/... — classified by NORAD NAME, a heuristic, "
        "NOT confirmed mission), elevation + azimuth, range km, and "
        "ground sub-point. Highest-in-sky first. First call after startup takes a "
        "few seconds (propagating the full catalog)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "min_elevation": {
                "type": "number", "default": 10.0,
                "description": "Min elevation angle in degrees (10 = comfortably above horizon).",
            },
            "limit": {"type": "integer", "default": 30, "minimum": 1, "maximum": 100, "description": "Max satellites."},
        },
        "required": [],
    },
)
async def satellites_overhead(
    lat: float | None = None, lon: float | None = None,
    min_elevation: float = 10.0, limit: int = 30,
) -> dict:
    return await satellites.fetch_satellites_overhead(
        lat=lat, lon=lon, min_elevation=min_elevation, limit=limit
    )


@register(
    name="ships_nearby",
    description=(
        "Live ships / vessels near a location via aisstream.io AIS. Call when the "
        "operator asks about ships, vessels, or maritime traffic near them or a "
        "coast/sea. Defaults to operator location (South Wales → Bristol Channel / "
        "Celtic Sea). Returns each vessel's name, MMSI, position, course, speed "
        "(knots), and navigation status. Takes ~4-5s (live AIS collection window)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "radius_km": {"type": "number", "default": 80.0, "description": "Search radius km."},
        },
        "required": [],
    },
)
async def ships_nearby(
    lat: float | None = None, lon: float | None = None, radius_km: float = 80.0
) -> dict:
    return await maritime.fetch_ships_bbox(lat=lat, lon=lon, radius_km=radius_km)


@register(
    name="grid_timing",
    description=(
        "Gnostic grid-timing snapshot — the operator's astro-timing node, "
        "computed locally with ephem. Call when the operator asks about the "
        "current planetary hour, the moon (phase / illumination / zodiac sign), "
        "fixed stars (Regulus, Spica, Aldebaran, Antares, Sirius — alt/az + "
        "above-horizon + next rise/transit/set), visible planets, sidereal time, "
        "or sunrise/noon/sunset. Regulus position is first-class — the Sphinx–"
        "Regulus correlation anchors the RHC framework, so 'is Regulus up?' is a "
        "real answerable question here. Defaults to operator location (South "
        "Wales). Returns the planetary hour + ruler glyph + harmonic tone (Hz), "
        "moon, solar events, sidereal time, fixed stars, and visible bodies. Set "
        "include_table=true for the full 24-hour planetary-hour schedule (bulky)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "include_table": {
                "type": "boolean", "default": False,
                "description": "Include the full 24-hour planetary-hour table (verbose).",
            },
        },
        "required": [],
    },
)
async def grid_timing(
    lat: float | None = None, lon: float | None = None, include_table: bool = False
) -> dict:
    return await grimoire.fetch_grid_timing(lat=lat, lon=lon, include_table=include_table)


@register(
    name="nuclear_facilities_nearby",
    description=(
        "Nuclear facilities within a radius of a point, from a curated bundled "
        "dataset (power reactors, enrichment/reprocessing, research, weapons, "
        "naval — IAEA PRIS / public sources). Call when the operator asks about "
        "nuclear plants / reactors / facilities near them or a named place, or "
        "wants to cross-reference a location against nuclear sites. Defaults to "
        "operator location (South Wales → nearest is Hinkley Point across the "
        "Bristol Channel). Returns each facility's name, country, type, status, "
        "operator, and distance + compass bearing from the center, nearest first. "
        "Local/static data (no live status) — pair with the news/conflict feed "
        "for situational context."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "radius_km": {"type": "number", "default": 300.0, "description": "Search radius km."},
            "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 100, "description": "Max facilities."},
        },
        "required": [],
    },
)
async def nuclear_facilities_nearby(
    lat: float | None = None, lon: float | None = None,
    radius_km: float = 300.0, limit: int = 25,
) -> dict:
    return await nuclear.fetch_nuclear_facilities(lat=lat, lon=lon, radius_km=radius_km, limit=limit)


@register(
    name="disaster_alerts",
    description=(
        "Current global natural-disaster events from GDACS (UN/EC Global Disaster "
        "Alert & Coordination System) — earthquakes, tropical cyclones, floods, "
        "volcanoes, wildfires, droughts, tsunamis. Each carries a Green/Orange/Red "
        "severity, affected country, and coordinates. Call when the operator asks "
        "about disasters, hazards, or major events happening in the world right now, "
        "or wants the severity-scored complement to the raw natural-events count. "
        "Keyless GDACS RSS. Returns events sorted by severity (red first)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "min_level": {
                "type": "string", "enum": ["green", "orange", "red"], "default": "green",
                "description": "Minimum alert level to include.",
            },
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
        },
        "required": [],
    },
)
async def disaster_alerts(min_level: str = "green", limit: int = 20) -> dict:
    snap = await gdacs.fetch_gdacs_events()
    if not snap.get("ok"):
        return snap
    rank = {"green": 1, "orange": 2, "red": 3}.get(min_level.lower(), 1)
    events = [e for e in snap["events"] if e["level_rank"] >= rank][:limit]
    return {**snap, "events": events, "count": len(events), "min_level": min_level}


@register(
    name="fires_nearby",
    description=(
        "NASA FIRMS satellite-detected active fire hotspots within a radius of a "
        "point (last 24 h, MODIS + VIIRS thermal anomalies — keyless). Call when the "
        "operator asks about fires / wildfires near them or a named place. Each "
        "hotspot has distance + compass bearing, acquisition time, instrument, "
        "confidence (low/nominal/high), and fire radiative power (MW). Defaults to "
        "operator location. Note: a hotspot is any thermal anomaly (can be industrial "
        "flares/agricultural burns), not only wildfires."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "radius_km": {"type": "number", "default": 50.0, "description": "Search radius km."},
            "min_confidence": {
                "type": "string", "enum": ["low", "nominal", "high"], "default": "nominal",
                "description": "Minimum detection confidence.",
            },
            "limit": {"type": "integer", "default": 25, "minimum": 1, "maximum": 100},
        },
        "required": [],
    },
)
async def fires_nearby(
    lat: float | None = None, lon: float | None = None,
    radius_km: float = 50.0, min_confidence: str = "nominal", limit: int = 25,
) -> dict:
    return await fires.fetch_fires_nearby(
        lat=lat, lon=lon, radius_km=radius_km, min_confidence=min_confidence, limit=limit
    )


@register(
    name="conflict_zones_nearby",
    description=(
        "Active geocoded conflict theatres (Ukraine, Gaza, Sudan, Yemen, Sahel, "
        "Taiwan strait, etc.), each enriched with live event counts from world news "
        "and its distance + representative headlines. Call when the operator asks "
        "which conflicts are active, where fighting is happening, or the nearest "
        "active theatre to them. Complements get_conflict_status (which scores the "
        "overall news picture) by placing the hotspots on the map. Zones sorted by "
        "distance from the operator."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
async def conflict_zones_nearby() -> dict:
    return await conflict.fetch_conflict_zones()


@register(
    name="internet_outages",
    description=(
        "Recent country-level internet outages / connectivity blackouts from IODA "
        "(Georgia Tech Internet Outage Detection & Analysis — keyless). Call when the "
        "operator asks whether a country has gone offline, about internet shutdowns, "
        "or wants a connectivity-blackout signal (often tied to coups, war, cable "
        "cuts, or state-ordered shutdowns). Returns outage events with country, IODA "
        "severity/score, start time and duration, highest-score first."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hours": {"type": "integer", "default": 24, "minimum": 1, "maximum": 168, "description": "Look-back window (hours)."},
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 200},
        },
        "required": [],
    },
)
async def internet_outages(hours: int = 24, limit: int = 50) -> dict:
    snap = await outages.fetch_outages(hours=hours, limit=limit)
    if snap.get("ok"):
        snap = {**snap, "outages": snap["outages"][:limit], "count": min(snap["count"], limit)}
    return snap


@register(
    name="satellite_imagery_coverage",
    description=(
        "Recent Sentinel satellite scenes (radar SAR first, optical fallback) "
        "covering a point — which imagery passes have captured an area, from the "
        "Element84 Earth Search + Copernicus STAC catalogs (keyless). Call when the "
        "operator asks what satellite imagery covers their area, when a location was "
        "last imaged, or about SAR/radar coverage. Defaults to operator location. "
        "Returns scenes with platform, instrument, acquisition datetime, SAR mode / "
        "cloud cover, newest first."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
            "radius_deg": {"type": "number", "default": 1.0, "description": "Half-box size in degrees (~111 km/deg)."},
            "days": {"type": "integer", "default": 30, "minimum": 1, "maximum": 365, "description": "Look-back window (days)."},
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
        },
        "required": [],
    },
)
async def satellite_imagery_coverage(
    lat: float | None = None, lon: float | None = None,
    radius_deg: float = 1.0, days: int = 30, limit: int = 10,
) -> dict:
    return await satellites.fetch_sar_scenes(
        lat=lat, lon=lon, radius_deg=radius_deg, days=days, limit=limit
    )
