"""Maritime / AIS vessel tracking via aisstream.io (Osiris-matched source).

aisstream.io is a live WebSocket AIS feed (the same source the Osiris board
uses). Unlike the REST fetchers, this opens a websocket, subscribes to a
bounding box, collects position reports for a few seconds, dedups by MMSI,
then closes. Free keyed tier; the key goes in LUMOS_AISSTREAM_KEY.

Protocol:
  connect wss://stream.aisstream.io/v0/stream
  send {"APIKey": key, "BoundingBoxes": [[[swLat,swLon],[neLat,neLon]]],
        "FilterMessageTypes": ["PositionReport"]}
  recv PositionReport messages:
    MetaData: {MMSI, ShipName, time_utc}
    Message.PositionReport: {UserID(MMSI), Latitude, Longitude, Cog, Sog,
                             TrueHeading, NavigationalStatus}

Collect window ~4 s (long enough to catch transmitting vessels in the box,
short enough to return promptly), result cached 60 s per bbox.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache


log = get_logger(__name__)


_AIS_WS = "wss://stream.aisstream.io/v0/stream"
_COLLECT_SECONDS = 4.0
_CONNECT_TIMEOUT = 8.0
_NM_PER_KM = 0.539957  # unused but kept for parity with other geo modules


def _bbox(lat: float, lon: float, radius_km: float) -> list[list[list[float]]]:
    """aisstream BoundingBoxes format: [[[swLat,swLon],[neLat,neLon]]].
    Latitude degree ≈ 111 km; longitude scaled by cos(lat)."""
    dlat = radius_km / 111.0
    cos_lat = max(0.05, math.cos(math.radians(lat)))
    dlon = radius_km / (111.0 * cos_lat)
    sw = [max(-90.0, lat - dlat), max(-180.0, lon - dlon)]
    ne = [min(90.0, lat + dlat), min(180.0, lon + dlon)]
    return [[sw, ne]]


# Common AIS navigational-status codes → human label (subset).
_NAV_STATUS = {
    0: "under way (engine)", 1: "at anchor", 2: "not under command",
    3: "restricted manoeuvrability", 4: "constrained by draught",
    5: "moored", 6: "aground", 7: "fishing", 8: "under way (sailing)",
}


async def fetch_ships_bbox(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 80.0,
) -> dict[str, Any]:
    """Live vessels within radius of (lat, lon) via aisstream.io.

    Falls back to operator location. Returns {ok, count, vessels:[...], center,
    fetched_at}. Caches the failure path (no key / connect error) too.
    """
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    cache_key = f"ais_{lat:.3f}_{lon:.3f}_{radius_km:.0f}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    key = (settings.aisstream_key or "").strip()
    if not key:
        result = {
            "ok": False,
            "error": "no aisstream key (set LUMOS_AISSTREAM_KEY)",
            "count": 0,
            "vessels": [],
            "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("ais", 60))
        return result

    subscription = {
        "APIKey": key,
        "BoundingBoxes": _bbox(lat, lon, radius_km),
        "FilterMessageTypes": ["PositionReport"],
    }

    vessels: dict[int, dict[str, Any]] = {}
    error: str | None = None
    try:
        from websockets.asyncio.client import connect

        async with connect(_AIS_WS, open_timeout=_CONNECT_TIMEOUT) as ws:
            await ws.send(json.dumps(subscription))
            deadline = time.monotonic() + _COLLECT_SECONDS
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if data.get("MessageType") != "PositionReport":
                    # aisstream sends an error message if the key/sub is bad.
                    if "error" in str(data).lower() and not vessels:
                        error = str(data)[:200]
                    continue
                pr = (data.get("Message") or {}).get("PositionReport") or {}
                meta = data.get("MetaData") or {}
                mmsi = pr.get("UserID") or meta.get("MMSI")
                if mmsi is None:
                    continue
                vessels[int(mmsi)] = {
                    "mmsi": int(mmsi),
                    "name": (meta.get("ShipName") or "").strip() or None,
                    "lat": pr.get("Latitude") if pr.get("Latitude") is not None else meta.get("latitude"),
                    "lon": pr.get("Longitude") if pr.get("Longitude") is not None else meta.get("longitude"),
                    "course_deg": pr.get("Cog"),
                    "speed_kts": pr.get("Sog"),
                    "heading_deg": pr.get("TrueHeading"),
                    "nav_status": _NAV_STATUS.get(pr.get("NavigationalStatus"), None),
                }
    except Exception as e:  # noqa: BLE001 — connect/timeout/protocol errors
        log.info("maritime.ws_failed", error=str(e))
        error = error or str(e)

    if not vessels and error:
        result = {
            "ok": False,
            "error": error,
            "count": 0,
            "vessels": [],
            "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("ais", 60))
        return result

    out = sorted(vessels.values(), key=lambda v: (v.get("speed_kts") or 0), reverse=True)
    result = {
        "ok": True,
        "count": len(out),
        "vessels": out,
        "center": {"lat": lat, "lon": lon, "radius_km": radius_km},
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("ais", 60))
    return result


# ── Persistent AIS monitor (Phase 3 alert layer) ─────────────────────────────
# The 4-second snapshot above is perfect for the on-demand `ships_nearby` tool,
# but it can't classify NAVAL vessels: AIS ship TYPE arrives only in the
# ShipStaticData message, broadcast ~every 6 minutes (positions come every few
# seconds). So the alert monitor instead holds ONE aisstream connection open and
# accumulates a rolling per-MMSI cache (position + nav_status + ship type + name)
# over time — making both the naval-type and anomalous-behaviour filters
# reliable, with live positions. This is additive; fetch_ships_bbox is untouched.

_AIS_STALE_SECONDS = 600.0  # drop a vessel not heard from in 10 min

# AIS ship-type codes that read as naval / military / enforcement presence.
# 35 = Military operations; 55 = Law enforcement. (Warships often run dark or
# spoof merchant types — type 35 means the vessel is DECLARING itself military,
# which is exactly the signal worth flagging.)
_NAVAL_SHIP_TYPES = frozenset({35, 55})

# Navigational-status codes that read as anomalous (distress / impaired / odd).
_ANOMALOUS_NAV_CODES = frozenset({2, 3, 6})  # not under command, restricted, aground

# Module-global rolling cache: mmsi -> {mmsi, name, lat, lon, course_deg,
# speed_kts, heading_deg, nav_status, nav_status_code, ship_type, last_seen}.
_vessel_cache: dict[int, dict[str, Any]] = {}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _ingest_ais(raw: str) -> None:
    """Fold one aisstream frame into the vessel cache (PositionReport updates
    position/nav; ShipStaticData updates type/name)."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return
    meta = data.get("MetaData") or {}
    mmsi = meta.get("MMSI")
    if mmsi is None:
        return
    mmsi = int(mmsi)
    v = _vessel_cache.setdefault(mmsi, {"mmsi": mmsi})
    v["last_seen"] = time.time()
    name = (meta.get("ShipName") or "").strip()
    if name:
        v["name"] = name

    msg = data.get("Message") or {}
    mtype = data.get("MessageType")
    if mtype == "PositionReport":
        pr = msg.get("PositionReport") or {}
        if pr.get("Latitude") is not None:
            v["lat"] = pr["Latitude"]
        if pr.get("Longitude") is not None:
            v["lon"] = pr["Longitude"]
        v["course_deg"] = pr.get("Cog")
        v["speed_kts"] = pr.get("Sog")
        v["heading_deg"] = pr.get("TrueHeading")
        v["nav_status_code"] = pr.get("NavigationalStatus")
        v["nav_status"] = _NAV_STATUS.get(pr.get("NavigationalStatus"))
    elif mtype == "ShipStaticData":
        sd = msg.get("ShipStaticData") or {}
        if sd.get("Type") is not None:
            v["ship_type"] = sd["Type"]
        sname = (sd.get("Name") or "").strip()
        if sname:
            v["name"] = sname


def _evict_stale() -> None:
    cutoff = time.time() - _AIS_STALE_SECONDS
    stale = [m for m, v in _vessel_cache.items() if v.get("last_seen", 0.0) < cutoff]
    for m in stale:
        _vessel_cache.pop(m, None)


def snapshot_vessels(lat: float, lon: float, radius_km: float) -> list[dict[str, Any]]:
    """Vessels from the persistent cache within radius of (lat, lon), each
    annotated with distance_km, bearing-free is_naval + is_anomalous flags.
    Evicts stale entries on read. Nearest-first."""
    _evict_stale()
    out: list[dict[str, Any]] = []
    for v in _vessel_cache.values():
        vlat, vlon = v.get("lat"), v.get("lon")
        if vlat is None or vlon is None:
            continue
        d = _haversine_km(lat, lon, vlat, vlon)
        if d > radius_km:
            continue
        out.append(
            {
                **v,
                "distance_km": round(d, 1),
                "is_naval": v.get("ship_type") in _NAVAL_SHIP_TYPES,
                "is_anomalous": v.get("nav_status_code") in _ANOMALOUS_NAV_CODES,
            }
        )
    out.sort(key=lambda x: x["distance_km"])
    return out


async def ais_monitor_loop(lat: float, lon: float, radius_km: float) -> None:
    """Persistent aisstream connection feeding _vessel_cache. Subscribes to
    PositionReport + ShipStaticData for the bbox; reconnects on drop with
    exponential backoff. Self-cancellable. No-op without a key."""
    settings = get_settings()
    key = (settings.aisstream_key or "").strip()
    if not key:
        log.info("maritime.ais_monitor.no_key")
        return

    sub = json.dumps(
        {
            "APIKey": key,
            "BoundingBoxes": _bbox(lat, lon, radius_km),
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        }
    )
    backoff = 2.0
    log.info("maritime.ais_monitor.started", radius_km=radius_km)
    while True:
        healthy = False  # did this connection actually carry frames for a while?
        try:
            from websockets.asyncio.client import connect

            async with connect(_AIS_WS, open_timeout=_CONNECT_TIMEOUT) as ws:
                await ws.send(sub)
                started = time.monotonic()
                log.info("maritime.ais_monitor.connected")
                async for raw in ws:
                    _ingest_ais(raw)
                    if not healthy and time.monotonic() - started > 30.0:
                        healthy = True  # proven good — a later drop may retry fast
            # NB: a CLEAN server close ends the `async for` WITHOUT raising, so
            # we fall through here (not into except). Both paths must pace the
            # reconnect below — otherwise an accept-then-clean-close key
            # rejection would hot-loop the endpoint with zero delay.
        except asyncio.CancelledError:
            log.info("maritime.ais_monitor.cancelled")
            raise
        except Exception as e:  # noqa: BLE001 — drop/timeout/protocol error
            log.info("maritime.ais_monitor.disconnected", error=str(e))

        # Single pacing point for EVERY reconnect (clean close OR error). Reset
        # backoff only when the connection proved healthy (frames flowed > 30 s);
        # an accept-then-fail (bad/over-quota key) keeps backing off, so it can
        # never spin or hammer aisstream.
        if healthy:
            backoff = 2.0
        try:
            await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            raise
        if not healthy:
            backoff = min(backoff * 2.0, 60.0)
