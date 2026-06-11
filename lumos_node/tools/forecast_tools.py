"""Anticipatory-forecast tool (Phase 39) — read-only look-ahead.

Passive/observe-only, so an autonomous wake AND the dawn briefing can both use
it. Wraps telemetry.forecast.build_forecast (celestial look-ahead + upcoming
military-recon satellite passes + NOAA Kp 3-day forecast).
"""

from __future__ import annotations

from . import register
from ..log import get_logger
from ..telemetry import forecast


log = get_logger(__name__)


@register(
    name="get_forecast",
    description=(
        "Anticipatory forecast — what's ABOUT to happen near a location over the "
        "next several hours, not just the current state. Returns: upcoming "
        "military-recon satellite passes (culmination time + peak elevation + "
        "bearing — the 'someone's watching overhead soon' heads-up); the NOAA "
        "3-day geomagnetic Kp forecast (peak predicted Kp + time, so you can warn "
        "BEFORE a storm lands — bio-impact look-ahead); and a celestial look-ahead "
        "(when Regulus next transits/rises/sets — the RHC anchor — plus next "
        "sunrise/sunset and when the current planetary hour ends). Call when the "
        "operator asks what's coming up / later today / tonight / the next pass / "
        "the look-ahead. Defaults to operator location when lat/lon omitted."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
        },
        "required": [],
    },
)
async def get_forecast(lat: float | None = None, lon: float | None = None) -> dict:
    return await forecast.build_forecast(lat, lon)


@register(
    name="get_weather",
    description=(
        "Current local surface weather at a location via Open-Meteo (free, no "
        "key): temperature + feels-like (°C), wind speed + gusts (mph) and "
        "direction, precipitation, humidity, pressure, cloud cover, and a "
        "plain-language sky condition. Call when the operator asks about the "
        "weather, temperature, wind, rain, or how it is outside RIGHT NOW. "
        "Defaults to operator location when lat/lon omitted."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
        },
        "required": [],
    },
)
async def get_weather(lat: float | None = None, lon: float | None = None) -> dict:
    return await forecast.fetch_local_weather(lat, lon)


@register(
    name="get_weather_warnings",
    description=(
        "Severe-weather warnings near a location: (1) OFFICIAL Met Office "
        "yellow/amber/red warnings (via MeteoAlarm — wind, rain, snow, ice, "
        "thunderstorm, fog) whose area covers or is near you, and (2) a precise "
        "Open-Meteo forecast watch for high gusts / heavy rain / snow / "
        "thunderstorms at your exact coords in the next several hours. Call when "
        "the operator asks about weather warnings, storms, severe/extreme "
        "weather, an amber/red alert, or whether anything dangerous is coming. "
        "Defaults to operator location when lat/lon omitted."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude (omit for operator default)."},
            "lon": {"type": "number", "description": "Longitude (omit for operator default)."},
        },
        "required": [],
    },
)
async def get_weather_warnings(lat: float | None = None, lon: float | None = None) -> dict:
    from ..config import get_settings
    from ..telemetry import weather_alerts

    s = get_settings()
    if lat is None or lon is None:
        lat, lon = s.operator_lat, s.operator_lon
    official = await weather_alerts.fetch_metoffice_warnings(
        lat, lon, s.alert_severe_wx_radius_km, s.operator_regions
    )
    watch = await weather_alerts.fetch_weather_watch(
        lat, lon, s.weather_watch_hours,
        s.weather_watch_gust_mph, s.weather_watch_precip_mm, s.weather_watch_snow_cm,
    )
    return {
        "ok": official.get("ok", False) or watch.get("ok", False),
        "official_warnings": official.get("warnings", []),
        "forecast_watch": watch.get("hits", []),
        "center": {"lat": lat, "lon": lon, "radius_km": s.alert_severe_wx_radius_km},
    }
