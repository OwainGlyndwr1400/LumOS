"""Shared ADS-B fetcher (adsb.lol) — feeds BOTH military classification and
GPS-jamming inference from a single keyless upstream call.

Why a shared module: the Osiris reference derives two intel layers from ONE
adsb.lol fetch — military aircraft (classifyFlight) and GPS jamming (NACp
clustering). Replicating that here means `military.fetch_military_aircraft`
and `gpsjam.fetch_gps_jamming` both call `fetch_adsb_raw`, so a "what's
military + any jamming near me" check is one upstream request, and the TTL
cache absorbs the second consumer for free.

Source: adsb.lol v2 — https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{nm}
  - Keyless. No OAuth2, no quota bucket (unlike OpenSky).
  - The /dist endpoint caps at 250 nm radius; larger requests are clamped.
    Global coverage (the Phase-4 globe) needs multiple region calls; local
    operator-area intel (alert monitor + tools) fits in one 250 nm call.

Aircraft object fields used (adsb.lol v2 `ac[]`):
  hex, flight (callsign), t (type code), r (registration), lat, lon,
  alt_baro, gs (ground speed kt), track (heading), squawk, dbFlags (bitmask;
  bit 0 = military per adsb.lol DB), nac_p (Navigation Accuracy Category —
  Position; low values = degraded GPS), alt_baro == "ground" when grounded.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..log import get_logger
from . import cache as tcache


log = get_logger(__name__)


_ADSB_BASE = "https://api.adsb.lol/v2"
_TIMEOUT = httpx.Timeout(12.0, connect=5.0)
_MAX_DIST_NM = 250  # adsb.lol /dist endpoint hard cap
# Spoofed UA for politeness + to avoid bare-client 403s (matches Osiris's
# stealthFetch intent; NOT a security mechanism).
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _decode_aircraft(ac: dict[str, Any]) -> dict[str, Any]:
    """Normalize one adsb.lol `ac` entry into our flat aircraft dict.

    `alt_baro` is an int (feet) when airborne, the string "ground" when on the
    ground — we surface both a numeric `altitude_ft` (None when grounded) and a
    `grounded` bool. Velocity stays in knots (adsb.lol native unit).
    """
    alt_raw = ac.get("alt_baro")
    grounded = alt_raw == "ground"
    altitude_ft: float | None
    try:
        altitude_ft = None if grounded or alt_raw is None else float(alt_raw)
    except (ValueError, TypeError):
        altitude_ft = None

    callsign = ac.get("flight")
    if isinstance(callsign, str):
        callsign = callsign.strip() or None

    def _num(v: Any) -> float | None:
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    return {
        "hex": (ac.get("hex") or "").strip().lower() or None,
        "callsign": callsign,
        "type_code": (ac.get("t") or "").strip() or None,
        "registration": (ac.get("r") or "").strip() or None,
        "lat": _num(ac.get("lat")),
        "lon": _num(ac.get("lon")),
        "altitude_ft": altitude_ft,
        "speed_kts": _num(ac.get("gs")),
        "heading_deg": _num(ac.get("track")),
        "squawk": (ac.get("squawk") or "").strip() or None,
        "db_flags": int(ac.get("dbFlags") or 0),
        "nac_p": ac.get("nac_p"),  # kept raw; may be int or None
        "grounded": grounded,
    }


async def fetch_adsb_raw(
    lat: float,
    lon: float,
    radius_nm: float = 250.0,
) -> dict[str, Any]:
    """Fetch + decode aircraft within `radius_nm` of (lat, lon) from adsb.lol.

    Returns {ok, count, aircraft: [decoded...], center, fetched_at_unix}.
    Cached under the shared "adsb" TTL bucket so military + gpsjam consumers
    reuse one fetch. Caches the failure path too (never re-hammers on outage).
    """
    radius_nm = max(1.0, min(_MAX_DIST_NM, float(radius_nm)))
    cache_key = f"adsb_{lat:.4f}_{lon:.4f}_{radius_nm:.0f}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{_ADSB_BASE}/lat/{lat:.5f}/lon/{lon:.5f}/dist/{radius_nm:.0f}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.info("adsb.fetch_failed", url=url, error=str(e))
        result = {
            "ok": False,
            "error": str(e),
            "count": 0,
            "aircraft": [],
            "center": {"lat": lat, "lon": lon, "radius_nm": radius_nm},
            "fetched_at_unix": int(time.time()),
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("adsb", 45))
        return result

    raw = payload.get("ac") or []
    aircraft = [_decode_aircraft(a) for a in raw]
    # Keep only entries with a position fix (needed by every consumer).
    aircraft = [a for a in aircraft if a["lat"] is not None and a["lon"] is not None]
    result = {
        "ok": True,
        "count": len(aircraft),
        "aircraft": aircraft,
        "center": {"lat": lat, "lon": lon, "radius_nm": radius_nm},
        "fetched_at_unix": int(time.time()),
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("adsb", 45))
    return result
