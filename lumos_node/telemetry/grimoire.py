"""Grid timing — Gnostic astro node (planetary hours, moon, fixed stars).

This is the operator's "Gnostic Grimoire" timing script lifted in-process: same
ephem math, no CLI / no file logging / no print. The standalone script on the
Desktop (Astro Timings/Gnostic Grimoire v3.2 ultimate.py) still runs as-is for
manual rituals; this module gives Lumos a clean callable that RETURNS the same
structured snapshot the script's json_data produced, so Lumos can reason over
planetary hours, moon phase, and fixed-star positions on demand.

Why this and not subprocess:
  - The script prints human-formatted text and writes log files as a side
    effect — neither is useful to a tool that needs structured data back.
  - Running .py via subprocess couples us to cwd + argparse + the Gnostic_Logs
    write path. Porting the computation is cleaner and side-effect-free.

Shape follows satellites.py (the other pure-local-compute telemetry): the ephem
work is synchronous and CPU-light but we run it in asyncio.to_thread so it never
touches the event loop, cache the result ~60 s (planetary hours are hour-long
windows; a minute of staleness is irrelevant), and return the {ok, ...} contract.

RHC relevance: Regulus alt/az is first-class here — the Sphinx–Regulus
correlation is the framework's physical anchor, so "is Regulus up right now?"
is a real operator question this answers with ephemeris, not a guess.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from math import degrees
from typing import Any
from zoneinfo import ZoneInfo

import ephem

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache


log = get_logger(__name__)


# ── Constants (ported verbatim from the Grimoire script) ─────────────────────
DEFAULT_TZ = "Europe/London"
DEFAULT_LOCATION_NAME = "South Wales"

PLANETS = ["Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon"]

# Chaldean order → weekday ruler (Mon=0). Drives the planetary-hour sequence.
WEEKDAY_RULERS = {0: "Moon", 1: "Mars", 2: "Mercury", 3: "Jupiter", 4: "Venus", 5: "Saturn", 6: "Sun"}

PLANET_GLYPHS = {
    "Saturn": "♄", "Jupiter": "♃", "Mars": "♂", "Sun": "☉",
    "Venus": "♀", "Mercury": "☿", "Moon": "☽", "unknown": "?",
}

ZODIAC_SIGNS = [
    ("Aries", "♈"), ("Taurus", "♉"), ("Gemini", "♊"), ("Cancer", "♋"),
    ("Leo", "♌"), ("Virgo", "♍"), ("Libra", "♎"), ("Scorpio", "♏"),
    ("Sagittarius", "♐"), ("Capricorn", "♑"), ("Aquarius", "♒"), ("Pisces", "♓"),
]

# ── Welsh Wheel of the Year (Phase 44) ───────────────────────────────────────
# The four solar festivals (Alban …) are computed from real solstice/equinox
# ephemeris; the four cross-quarter fire festivals sit on their traditional
# calendar dates. Y Gwir yn Erbyn y Byd.
_CROSS_QUARTERS = [
    (2, 1, "Gŵyl Forwyn", "Imbolc — first stirring of spring"),
    (5, 1, "Calan Mai", "Beltane — summer's door opens"),
    (8, 1, "Gŵyl Awst", "Lughnasadh — first harvest"),
    (11, 1, "Calan Gaeaf", "Samhain — winter's door, the thin night"),
]
_SOLAR_FESTIVALS = {
    # month of the event → (Welsh name, meaning)
    3: ("Alban Eilir", "spring equinox — light of the earth"),
    6: ("Alban Hefin", "summer solstice — light of the shore"),
    9: ("Alban Elfed", "autumn equinox — light of the water"),
    12: ("Alban Arthan", "winter solstice — light of Arthur"),
}


def _welsh_wheel(now_utc: datetime, tz: ZoneInfo) -> dict[str, Any]:
    """Next festivals on the Welsh wheel: real ephemeris for the Alban quarter
    days (solstice/equinox), traditional dates for the cross-quarters."""
    now_local = now_utc.astimezone(tz)
    candidates: list[tuple[datetime, str, str]] = []

    # Solar quarter days — pyephem gives the actual instants.
    for fn in (ephem.next_solstice, ephem.next_equinox):
        try:
            ev = fn(now_utc).datetime().replace(tzinfo=timezone.utc)
        except Exception:  # noqa: BLE001 — ephemeris edge; skip the entry
            continue
        name, meaning = _SOLAR_FESTIVALS.get(ev.month, ("?", ""))
        candidates.append((ev.astimezone(tz), name, meaning))

    # Cross-quarters — next occurrence of each traditional date.
    for month, day, name, meaning in _CROSS_QUARTERS:
        dt = now_local.replace(
            month=month, day=day, hour=0, minute=0, second=0, microsecond=0
        )
        if dt <= now_local:
            dt = dt.replace(year=dt.year + 1)
        candidates.append((dt, name, meaning))

    candidates.sort(key=lambda c: c[0])
    upcoming = [
        {
            "name": name,
            "meaning": meaning,
            "date_local": dt.isoformat(),
            "days_until": max(0, (dt - now_local).days),
        }
        for dt, name, meaning in candidates[:3]
    ]
    return {"next": upcoming[0] if upcoming else None, "upcoming": upcoming}


# Fixed stars the operator tracks. Regulus is central (Sphinx–Regulus / RHC).
FIXED_STARS = {
    "Regulus": "Regulus",
    "Spica": "Spica",
    "Aldebaran": "Aldebaran",
    "Antares": "Antares",
    "Sirius": "Sirius",
}

# Operator's planetary harmonic tones (Hz) — their custom mapping, kept as-is.
PLANETARY_TONES_HZ = {
    "Saturn": 147.0, "Jupiter": 183.0, "Mars": 105.0, "Sun": 126.0,
    "Venus": 216.0, "Mercury": 192.0, "Moon": 174.0, "unknown": 0.0,
}


@dataclass
class _SolarTimes:
    sunrise_utc: datetime | None
    noon_utc: datetime | None
    sunset_utc: datetime | None
    next_sunrise_utc: datetime | None


# ── ephem helpers (ported) ───────────────────────────────────────────────────
def _build_observer(lat: float, lon: float, date_utc: datetime | None = None) -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.pressure = 0
    obs.horizon = "-0:34"
    if date_utc is not None:
        obs.date = date_utc
    return obs


def _safe_dt(value: Any) -> datetime:
    """ephem.Date-or-datetime → timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return value.datetime().replace(tzinfo=timezone.utc)


def _moon_phase_percent(date_utc: datetime) -> float:
    moon = ephem.Moon()
    moon.compute(date_utc)
    return round(float(moon.phase), 1)


def _moon_age_days(date_utc: datetime) -> float:
    prev_new = ephem.previous_new_moon(date_utc)
    delta = _safe_dt(date_utc) - _safe_dt(prev_new)
    return round(delta.total_seconds() / 86400.0, 2)


def _moon_phase_name(moon_age_days: float) -> str:
    """Waxing/waning from age (more accurate than pure illumination bands)."""
    synodic_month = 29.53058867
    age = moon_age_days % synodic_month
    if age < 1.5 or age > synodic_month - 1.5:
        return "New Moon"
    elif age < 6.5:
        return "Waxing Crescent"
    elif age < 8.5:
        return "First Quarter"
    elif age < 13.5:
        return "Waxing Gibbous"
    elif age < 16.5:
        return "Full Moon"
    elif age < 21.5:
        return "Waning Gibbous"
    elif age < 23.5:
        return "Last Quarter"
    return "Waning Crescent"


def _sidereal_time(date_utc: datetime, lat: float, lon: float) -> str:
    return str(_build_observer(lat, lon, date_utc).sidereal_time())


def _sun_times(date_utc: datetime, lat: float, lon: float, tz: ZoneInfo) -> _SolarTimes:
    """Sunrise/noon/sunset anchored to the LOCAL calendar date (avoids the
    post-sunset bug where rising/setting span two solar days)."""
    sun = ephem.Sun()
    try:
        local_date = date_utc.astimezone(tz).date()
        local_noon = datetime.combine(local_date, time(12, 0, 0), tzinfo=tz)
        noon_utc_guess = local_noon.astimezone(timezone.utc)

        obs = _build_observer(lat, lon, noon_utc_guess)
        sunrise_utc = _safe_dt(obs.previous_rising(sun, use_center=True))
        sunset_utc = _safe_dt(obs.next_setting(sun, use_center=True))
        noon_utc = _safe_dt(obs.next_transit(sun))

        obs_next = _build_observer(lat, lon, sunset_utc + timedelta(seconds=1))
        next_sunrise_utc = _safe_dt(obs_next.next_rising(sun, use_center=True))
        return _SolarTimes(sunrise_utc, noon_utc, sunset_utc, next_sunrise_utc)
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        return _SolarTimes(None, None, None, None)


def _planetary_hour(now_utc: datetime, solar: _SolarTimes) -> dict[str, Any]:
    """Current planetary hour (Chaldean), its ruler, glyph, tone + window."""
    unknown = {
        "phase": "unknown", "hour_number": 0, "ruler": "unknown",
        "glyph": PLANET_GLYPHS["unknown"], "harmonic_tone_hz": 0.0,
        "hour_start_utc": None, "hour_end_utc": None,
    }
    if not solar.sunrise_utc or not solar.sunset_utc or not solar.next_sunrise_utc:
        return unknown

    sunrise_utc, sunset_utc, next_sunrise_utc = (
        solar.sunrise_utc, solar.sunset_utc, solar.next_sunrise_utc
    )
    day_hour_len = (sunset_utc - sunrise_utc).total_seconds() / 12
    night_hour_len = (next_sunrise_utc - sunset_utc).total_seconds() / 12

    ruler_of_day = WEEKDAY_RULERS.get(sunrise_utc.weekday(), "unknown")
    if ruler_of_day == "unknown":
        return unknown

    ruler_index = PLANETS.index(ruler_of_day)
    hourly_sequence = [PLANETS[(ruler_index + i) % 7] for i in range(24)]

    if sunrise_utc <= now_utc < sunset_utc:
        phase = "day"
        idx = min(int((now_utc - sunrise_utc).total_seconds() // day_hour_len), 11)
        hour_start = sunrise_utc + timedelta(seconds=idx * day_hour_len)
        hour_end = sunrise_utc + timedelta(seconds=(idx + 1) * day_hour_len)
    else:
        phase = "night"
        if now_utc >= sunset_utc:
            idx = min(max(int((now_utc - sunset_utc).total_seconds() // night_hour_len) + 12, 12), 23)
            hour_start = sunset_utc + timedelta(seconds=(idx - 12) * night_hour_len)
            hour_end = sunset_utc + timedelta(seconds=(idx - 11) * night_hour_len)
        else:
            # Before today's sunrise → use the previous night's tail safely.
            previous_sunset = sunset_utc - timedelta(days=1)
            idx = min(max(int((now_utc - previous_sunset).total_seconds() // night_hour_len) + 12, 12), 23)
            hour_start = previous_sunset + timedelta(seconds=(idx - 12) * night_hour_len)
            hour_end = previous_sunset + timedelta(seconds=(idx - 11) * night_hour_len)

    ruler = hourly_sequence[idx]
    return {
        "phase": phase,
        "hour_number": (idx % 12) + 1,
        "ruler": ruler,
        "glyph": PLANET_GLYPHS.get(ruler, "?"),
        "harmonic_tone_hz": PLANETARY_TONES_HZ.get(ruler, 0.0),
        "hour_start_utc": hour_start,
        "hour_end_utc": hour_end,
    }


def _planetary_hour_table(solar: _SolarTimes, tz: ZoneInfo) -> list[dict]:
    rows: list[dict] = []
    if not solar.sunrise_utc or not solar.sunset_utc or not solar.next_sunrise_utc:
        return rows
    day_hour_len = (solar.sunset_utc - solar.sunrise_utc).total_seconds() / 12
    night_hour_len = (solar.next_sunrise_utc - solar.sunset_utc).total_seconds() / 12
    ruler_index = PLANETS.index(WEEKDAY_RULERS[solar.sunrise_utc.weekday()])
    hourly_sequence = [PLANETS[(ruler_index + i) % 7] for i in range(24)]
    for i in range(24):
        if i < 12:
            start = solar.sunrise_utc + timedelta(seconds=i * day_hour_len)
            end = solar.sunrise_utc + timedelta(seconds=(i + 1) * day_hour_len)
            phase = "day"
        else:
            start = solar.sunset_utc + timedelta(seconds=(i - 12) * night_hour_len)
            end = solar.sunset_utc + timedelta(seconds=(i - 11) * night_hour_len)
            phase = "night"
        ruler = hourly_sequence[i]
        rows.append({
            "index_24": i + 1,
            "hour_number": (i % 12) + 1,
            "phase": phase,
            "ruler": ruler,
            "glyph": PLANET_GLYPHS.get(ruler, "?"),
            "tone_hz": PLANETARY_TONES_HZ.get(ruler, 0.0),
            "start_local": start.astimezone(tz).strftime("%H:%M:%S"),
            "end_local": end.astimezone(tz).strftime("%H:%M:%S"),
        })
    return rows


def _fixed_star(name: str, when_utc: datetime, lat: float, lon: float) -> dict[str, Any]:
    obs = _build_observer(lat, lon, when_utc)
    star = ephem.star(name)
    star.compute(obs)
    out = {
        "name": name,
        "alt_deg": round(degrees(star.alt), 2),
        "az_deg": round(degrees(star.az), 2),
        "mag": getattr(star, "mag", None),
        "ra": str(star.a_ra),
        "dec": str(star.a_dec),
        "above_horizon": bool(star.alt > 0),
        "next_rising_utc": None,
        "next_transit_utc": None,
        "next_setting_utc": None,
    }
    # Rise/transit/set can each fail (circumpolar / never-up) — each is its own try.
    try:
        out["next_rising_utc"] = _safe_dt(obs.next_rising(star, use_center=True)).isoformat()
    except Exception:  # noqa: BLE001
        pass
    try:
        out["next_transit_utc"] = _safe_dt(obs.next_transit(star)).isoformat()
    except Exception:  # noqa: BLE001
        pass
    try:
        out["next_setting_utc"] = _safe_dt(obs.next_setting(star, use_center=True)).isoformat()
    except Exception:  # noqa: BLE001
        pass
    return out


def _visible_planets(date_utc: datetime, lat: float, lon: float) -> list[dict]:
    visible = []
    obs = _build_observer(lat, lon, date_utc)
    for name in ["Saturn", "Jupiter", "Mars", "Venus", "Mercury", "Moon"]:
        body = getattr(ephem, name)()
        body.compute(obs)
        if body.alt > 0:
            visible.append({
                "name": name,
                "glyph": PLANET_GLYPHS.get(name, "?"),
                "alt_deg": round(degrees(body.alt), 2),
                "az_deg": round(degrees(body.az), 2),
                "ra": str(body.a_ra),
                "dec": str(body.a_dec),
            })
    return visible


def _ecliptic_longitude(body_name: str, date_utc: datetime) -> float:
    body = getattr(ephem, body_name)()
    body.compute(date_utc)
    return round(degrees(ephem.Ecliptic(body).lon) % 360, 2)


def _zodiac_from_longitude(lon_deg: float) -> tuple[str, str]:
    return ZODIAC_SIGNS[int((lon_deg % 360) // 30)]


def _iso_local(dt: datetime | None, tz: ZoneInfo) -> str | None:
    return None if dt is None else dt.astimezone(tz).isoformat()


# ── Synchronous compute (runs in a thread) ───────────────────────────────────
def compute_grid_timing(
    lat: float,
    lon: float,
    tz_name: str = DEFAULT_TZ,
    location_name: str = DEFAULT_LOCATION_NAME,
    include_table: bool = False,
) -> dict[str, Any]:
    """SYNC — the full Grimoire snapshot as a structured dict (the script's
    json_data, minus print/logging). Builds ephem objects here; meant for
    asyncio.to_thread. `include_table` adds the 24-hour planetary-hour schedule
    (bulky — off by default to keep tool payloads lean)."""
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    now_utc = now_local.astimezone(timezone.utc)

    solar = _sun_times(now_utc, lat, lon, tz)
    p_hour = _planetary_hour(now_utc, solar)

    moon_age = _moon_age_days(now_utc)
    moon_lon = _ecliptic_longitude("Moon", now_utc)
    moon_sign, moon_glyph = _zodiac_from_longitude(moon_lon)

    fixed_stars = {
        label: _fixed_star(ephem_name, now_utc, lat, lon)
        for label, ephem_name in FIXED_STARS.items()
    }

    planetary_hour = {
        "phase": p_hour["phase"],
        "hour_number": p_hour["hour_number"],
        "ruler": p_hour["ruler"],
        "glyph": p_hour["glyph"],
        "harmonic_tone_hz": p_hour["harmonic_tone_hz"],
        "hour_start_local": _iso_local(p_hour["hour_start_utc"], tz),
        "hour_end_local": _iso_local(p_hour["hour_end_utc"], tz),
    }
    if include_table:
        planetary_hour["table_24h"] = _planetary_hour_table(solar, tz)

    return {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_local": now_local.isoformat(),
        "location": {"name": location_name, "lat": lat, "lon": lon, "timezone_name": tz_name},
        "solar": {
            "sunrise_local": _iso_local(solar.sunrise_utc, tz),
            "noon_local": _iso_local(solar.noon_utc, tz),
            "sunset_local": _iso_local(solar.sunset_utc, tz),
            "next_sunrise_local": _iso_local(solar.next_sunrise_utc, tz),
        },
        "planetary_hour": planetary_hour,
        "moon": {
            "illumination_percent": _moon_phase_percent(now_utc),
            "phase_name": _moon_phase_name(moon_age),
            "age_days": moon_age,
            "ecliptic_longitude_deg": moon_lon,
            "zodiac_sign": moon_sign,
            "zodiac_glyph": moon_glyph,
        },
        "sidereal_time": _sidereal_time(now_utc, lat, lon),
        "visible_planets": _visible_planets(now_utc, lat, lon),
        "fixed_stars": fixed_stars,
        "welsh_wheel": _welsh_wheel(now_utc, tz),
    }


# ── Async wrapper (cache + to_thread + operator-location fallback) ───────────
async def fetch_grid_timing(
    lat: float | None = None,
    lon: float | None = None,
    tz_name: str = DEFAULT_TZ,
    include_table: bool = False,
) -> dict[str, Any]:
    """Grimoire grid-timing snapshot for a location (operator default).

    Returns {ok, ...grid timing fields..., fetched_at} or {ok: False, error}.
    Cached ~60 s under (lat, lon, table?) — planetary hours are hour-long
    windows, so a minute of staleness never matters.
    """
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    cache_key = f"grid_timing_{lat:.4f}_{lon:.4f}_{int(include_table)}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        snapshot = await asyncio.to_thread(
            compute_grid_timing, lat, lon, tz_name, DEFAULT_LOCATION_NAME, include_table
        )
    except Exception as e:  # noqa: BLE001 — ephem failure must not crash the tool
        log.warning("grimoire.compute_failed", error=str(e))
        result = {
            "ok": False,
            "error": f"grid-timing computation failed: {e}",
            "center": {"lat": lat, "lon": lon},
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("grid_timing", 60))
        return result

    result = {"ok": True, **snapshot, "fetched_at": fetched_at}
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("grid_timing", 60))
    return result
