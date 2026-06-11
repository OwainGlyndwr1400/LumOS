"""Global-intel tools (Aether Scope Batch 1) — Lumos queries the same feeds
the Osiris board pulls: military aircraft + GPS jamming (shared adsb.lol),
OSINT news (Telegram + RSS), and derived conflict indicators.

All read-only. Each opens its own httpx client where needed and falls back to
the operator's configured lat/lon when location args are omitted (same pattern
as aircraft_overhead in telemetry_tools.py).
"""

from __future__ import annotations

import httpx

from . import register
from ..config import get_settings
from ..log import get_logger
from ..telemetry import conflict, gpsjam, grimoire, maritime, military, news, nuclear, satellites


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
        "Satellites currently passing above a location's horizon, from SatNOGS "
        "TLE data propagated in real time. Call when the operator asks what "
        "satellites / spacecraft are overhead, or about satellite movement above "
        "them. Defaults to operator location. Returns satellites above the "
        "horizon (elevation >= min) with name, mission type (station/navigation/"
        "comms/military_recon/weather/earth_obs/...), elevation + azimuth in the "
        "sky, range km, and ground sub-point. Highest-in-sky first. First call "
        "after startup takes a few seconds (propagating the full catalog)."
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
