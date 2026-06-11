"""Rail / station-board telemetry via the Realtime Trains NG API (data.rtt.io).

Free personal API. TWO-STEP auth: a long-life REFRESH token (LUMOS_RTT_TOKEN,
from api-portal.rtt.io) is exchanged at /api/get_access_token for a short-life
ACCESS token (cached here until validUntil), which authorizes data calls.

Station boards: GET /rtt/location?code=<namespace>:<CRS>  (e.g. gb-nr:GWN).
We keep only services that CALL (stop) at the station — temporalData.displayAs
== "CALL" (vs "PASS") — which is the operator's ask: most trains pass through
without stopping → no trigger; a stopping train → one wake.

Direction (Llanelli west / Swansea east) is inferred for the Gowerton line.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import get_settings
from ..log import get_logger


log = get_logger(__name__)

_BASE = "https://data.rtt.io"

# Module-cached short-life access token (from the refresh-token exchange).
_access_token: str | None = None
_access_expiry: float = 0.0  # unix seconds; refresh ~60s before this


async def _get_access_token(client: httpx.AsyncClient, refresh: str) -> str | None:
    """Exchange the long-life refresh token for a short-life access token,
    cached until ~60s before its validUntil. Returns None on failure."""
    global _access_token, _access_expiry
    if _access_token and time.time() < _access_expiry:
        return _access_token
    try:
        r = await client.get(
            f"{_BASE}/api/get_access_token",
            headers={"Authorization": f"Bearer {refresh}"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        log.info("rail.auth_failed", error=str(e))
        return None
    _access_token = data.get("token")
    vu = data.get("validUntil")
    try:
        _access_expiry = datetime.fromisoformat(str(vu).replace("Z", "+00:00")).timestamp() - 60
    except Exception:  # noqa: BLE001
        _access_expiry = time.time() + 300.0  # fallback: 5 min
    return _access_token


# West-of-Gowerton destinations → westbound (toward Llanelli); else eastbound (Swansea).
_WEST_HINTS = (
    "llanelli", "carmarthen", "milford", "pembroke", "fishguard",
    "haverfordwest", "whitland", "tenby", "kilgetty", "gowerton",
)


def _direction(origin: str | None, dest: str | None) -> str:
    """Gowerton-line services run Swansea (E) ⇄ West Wales (W). Infer which way
    from the destination, falling back to the origin."""
    d = (dest or "").lower()
    if any(h in d for h in _WEST_HINTS):
        return "Llanelli (W)"
    if "swansea" in d or "cardiff" in d or "london" in d or "bristol" in d:
        return "Swansea (E)"
    return "Llanelli (W)" if "swansea" in (origin or "").lower() else "Swansea (E)"


def _hhmm(iso: str | None) -> str | None:
    """'2026-06-07T00:22:00' → '00:22'."""
    if iso and len(iso) >= 16 and iso[10] == "T":
        return iso[11:16]
    return iso


async def fetch_station_calls(code: str | None = None) -> dict[str, Any]:
    """Live board; returns only services that CALL (stop) at the station.

    Returns {ok, code, station, count, calls:[{uid, dest, origin, booked,
    expected, platform, operator, direction, cancelled}], fetched_at}. Never
    raises — auth/network errors return ok=False so the alert loop continues.
    """
    s = get_settings()
    code = (code or s.rail_station_code or "").strip()
    refresh = (s.rtt_token or "").strip()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not (code and refresh):
        return {"ok": False, "error": "no rtt_token or station code", "code": code,
                "station": code, "calls": [], "count": 0, "fetched_at": fetched_at}

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            access = await _get_access_token(client, refresh)
            if not access:
                return {"ok": False, "error": "access-token exchange failed", "code": code,
                        "station": code, "calls": [], "count": 0, "fetched_at": fetched_at}
            r = await client.get(
                f"{_BASE}/rtt/location",
                params={"code": code},
                headers={"Authorization": f"Bearer {access}"},
            )
            if r.status_code == 204:  # valid query, no services
                return {"ok": True, "code": code, "station": code,
                        "count": 0, "calls": [], "fetched_at": fetched_at}
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        log.info("rail.fetch_failed", error=str(e), code=code)
        return {"ok": False, "error": str(e), "code": code,
                "station": code, "calls": [], "count": 0, "fetched_at": fetched_at}

    station_name = (((data.get("query") or {}).get("location") or {}).get("description")) or code
    calls: list[dict[str, Any]] = []
    for svc in (data.get("services") or []):
        td = svc.get("temporalData") or {}
        if td.get("displayAs") != "CALL":  # PASS / cancelled-pass → not a stop
            continue
        arr = td.get("arrival") or {}
        dep = td.get("departure") or {}
        meta = svc.get("scheduleMetadata") or {}
        loc = svc.get("locationMetadata") or {}
        dest = (((svc.get("destination") or [{}])[0].get("location")) or {}).get("description")
        origin = (((svc.get("origin") or [{}])[0].get("location")) or {}).get("description")
        calls.append({
            "uid": meta.get("uniqueIdentity"),
            "dest": dest,
            "origin": origin,
            # Terminating services have no departure — fall back to arrival time.
            "booked": _hhmm(dep.get("scheduleAdvertised") or arr.get("scheduleAdvertised")),
            "expected": _hhmm(dep.get("realtimeForecast") or dep.get("scheduleAdvertised")
                              or arr.get("realtimeActual") or arr.get("scheduleAdvertised")),
            "platform": (loc.get("platform") or {}).get("actual"),
            "operator": (meta.get("operator") or {}).get("name"),
            "direction": _direction(origin, dest),
            "cancelled": bool(dep.get("isCancelled") or arr.get("isCancelled")),
        })
    calls.sort(key=lambda c: c.get("expected") or c.get("booked") or "")
    return {"ok": True, "code": code, "station": station_name,
            "count": len(calls), "calls": calls, "fetched_at": fetched_at}
