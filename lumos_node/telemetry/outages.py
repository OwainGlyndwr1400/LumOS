"""Internet outage detection — country-level connectivity blackouts (IODA).

Source: Georgia Tech IODA (Internet Outage Detection & Analysis) v2 API —
https://api.ioda.inetintel.cc.gatech.edu/v2/ — keyless. IODA fuses BGP,
active probing, and telescope data to flag when a country (or region) loses
connectivity. Ported from the Osiris `radar` route.

A national internet blackout is a strong world-pulse / geopolitical signal
(coup, war, state-ordered shutdown, cable cut). This module surfaces recent
country-level outage events; the alert path is OPTIONAL and default-off
(operator flag), edge-triggered per event id so one blackout pings once.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import Settings
from ..log import get_logger
from . import cache as tcache

log = get_logger(__name__)

_IODA_BASE = "https://api.ioda.inetintel.cc.gatech.edu/v2"
_TIMEOUT = httpx.Timeout(12.0, connect=5.0)


def parse_ioda_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    """PURE TRANSFORM — IODA outages/events JSON → compact outage dicts."""
    events = data.get("data") or [] if isinstance(data, dict) else []
    out: list[dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        loc = e.get("location") or ""
        code = loc.split("/")[1] if "/" in loc else loc
        if not code:
            continue
        start = e.get("start")
        out.append({
            "id": f"{code}-{start}",
            "country_code": code,
            "score": e.get("score") or 0,
            "severity": e.get("severity") or "unknown",
            "start_unix": start,
            "duration_s": e.get("duration") or 0,
            "datasource": (e.get("datasource") or "").replace("_", " ") or None,
        })
    out.sort(key=lambda x: -(x["score"] or 0))
    return out


async def fetch_outages(hours: int = 24, limit: int = 200) -> dict[str, Any]:
    """Recent country-level internet outages (last `hours`), cached 10 min.
    {ok, count, outages:[...], fetched_at}. Failure path cached too."""
    cache_key = f"outages_{hours}_{limit}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached
    now = int(time.time())
    frm = now - hours * 3600
    url = f"{_IODA_BASE}/outages/events?from={frm}&until={now}&entityType=country&limit={limit}"
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=_TIMEOUT, headers={"Accept": "application/json"})
            r.raise_for_status()
            outages = parse_ioda_events(r.json())
    except (httpx.HTTPError, ValueError) as e:
        log.info("outages.fetch_failed", error=str(e))
        result = {"ok": False, "count": 0, "outages": [], "fetched_at": fetched_at}
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("outages", 600))
        return result
    result = {"ok": True, "count": len(outages), "outages": outages, "fetched_at": fetched_at}
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("outages", 600))
    return result


# ── Alert-monitor integration (edge-triggered, optional) ─────────────────────


def fresh_outage_trips(
    outages: list[dict[str, Any]],
    seen: set[str],
    watch_codes: set[str] | None,
    min_score: float,
) -> list[dict[str, Any]]:
    """PURE TRANSFORM — outages worth a wake given the seen-set.

    watch_codes empty/None → any country qualifies; otherwise only those codes.
    Trips once per new event id at/above min_score."""
    trips: list[dict[str, Any]] = []
    for o in outages:
        eid = str(o["id"])
        if eid in seen:
            continue
        if (o.get("score") or 0) < min_score:
            continue
        if watch_codes and o["country_code"] not in watch_codes:
            continue
        trips.append({
            "id": f"outage:{eid}",
            "kind": "internet_outage",
            "description": (
                f"Internet outage detected in {o['country_code']} "
                f"(IODA severity {o['severity']}, score {o['score']})"
            ),
            "data": o,
        })
    return trips


async def evaluate_outages(settings: Settings) -> list[dict[str, Any]]:
    """Alert-monitor entry — internet-outage trips, else []. Edge-triggered via
    an in-module seen-set (process-lifetime; a restart re-evaluates once)."""
    snap = await fetch_outages()
    if not snap.get("ok"):
        return []
    watch = {c.strip().upper() for c in (settings.alert_outage_countries or "").split(",") if c.strip()}
    trips = fresh_outage_trips(
        snap["outages"], _seen_outages, watch or None, settings.alert_outage_min_score
    )
    for t in trips:
        _seen_outages.add(t["data"]["id"])
    if len(_seen_outages) > 1000:
        _seen_outages.clear()
    if trips:
        log.info("outages.tripped", n=len(trips))
    return trips


_seen_outages: set[str] = set()
