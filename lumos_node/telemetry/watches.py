"""Custom watches (Phase 39) — operator-defined alert rules beyond the fixed
home patch. "Lumos, keep an eye on the Taiwan Strait."

Each watch targets a (lat, lon, radius_km) and one or more SOURCES. The alert
monitor evaluates every enabled watch each poll alongside the built-in home
thresholds, and watch trips flow through the SAME dedup / cooldown / daily-cap /
wake fan-out (HUD + Discord). Trip ids are prefixed `watch:<id>:<entity>` so
dedup is per-watch-per-entity (the same vessel can trip a home alert AND a watch
independently).

Coverage:
  * military_air / gps_jamming / recon_satellite — query their feeds per-request
    with the watch's bbox, so they work ANYWHERE on Earth.
  * vessel — reads the persistent AIS cache, which only accumulates ships inside
    the AIS subscription bbox (around the operator). So vessel watches see local
    waters well and far oceans not at all. (Documented for the operator.)

Management is via the operator-only `manage_watch` tool — deliberately NOT a
passive tool, so an autonomous wake can never reconfigure its own monitoring.
Autonomy ends at speaking; changing what Lumos watches is an operator act.

Persistence: data/watches.json (a JSON list). Writes are atomic (temp + replace)
so a concurrent read from the alert loop never sees a torn file.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings
from ..log import get_logger
from . import gpsjam, maritime, military, satellites
from .worker import _data_dir


log = get_logger(__name__)

_WATCHES_FILE = "watches.json"
_MAX_WATCHES = 50  # generous; keeps a runaway loop from spawning unbounded polls

VALID_SOURCES: frozenset[str] = frozenset(
    {"military_air", "gps_jamming", "recon_satellite", "vessel"}
)
# adsb.lol caps a single query near ~460 km (250 nm); clamp the FETCH radius there
# even if the watch radius is larger (the operator is told).
_ADSB_MAX_RADIUS_KM = 460.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _watches_path(settings: Settings) -> Path:
    return _data_dir(settings) / _WATCHES_FILE


def load_watches(settings: Settings) -> list[dict[str, Any]]:
    """All watches (enabled + disabled). Never raises — a missing/corrupt file
    reads as an empty list so the alert loop simply finds nothing to add."""
    p = _watches_path(settings)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        log.warning("watches.load_failed", path=str(p))
        return []
    return data if isinstance(data, list) else []


def save_watches(settings: Settings, watches: list[dict[str, Any]]) -> None:
    """Atomic write (temp + os.replace) so the alert loop never reads a torn file."""
    p = _watches_path(settings)
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(watches, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, p)
    except OSError as e:
        log.warning("watches.save_failed", error=str(e))


def list_watches(settings: Settings) -> list[dict[str, Any]]:
    return load_watches(settings)


def add_watch(
    settings: Settings,
    *,
    label: str,
    lat: float,
    lon: float,
    radius_km: float = 150.0,
    sources: list[str] | None = None,
    sat_min_elevation_deg: float = 30.0,
) -> dict[str, Any]:
    """Create + persist a watch. Raises ValueError on bad input (the tool turns
    that into a clean message for Lumos)."""
    if not label or not str(label).strip():
        raise ValueError("a watch needs a label, e.g. 'Taiwan Strait'")
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lat/lon out of range: {lat}, {lon}")
    if radius_km <= 0 or radius_km > 5000:
        raise ValueError("radius_km must be between 0 and 5000")
    src = [s for s in (sources or []) if s in VALID_SOURCES]
    if not src:
        raise ValueError(
            f"pick at least one valid source from {sorted(VALID_SOURCES)}"
        )
    watches = load_watches(settings)
    if len(watches) >= _MAX_WATCHES:
        raise ValueError(f"watch limit reached ({_MAX_WATCHES}); remove one first")
    watch = {
        "id": "w_" + uuid.uuid4().hex[:8],
        "label": str(label).strip(),
        "lat": round(float(lat), 5),
        "lon": round(float(lon), 5),
        "radius_km": round(float(radius_km), 1),
        "sources": src,
        "sat_min_elevation_deg": round(float(sat_min_elevation_deg), 1),
        "enabled": True,
        "created_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    watches.append(watch)
    save_watches(settings, watches)
    log.info("watches.added", id=watch["id"], label=watch["label"], sources=src)
    return watch


def remove_watch(settings: Settings, watch_id: str) -> bool:
    watches = load_watches(settings)
    kept = [w for w in watches if w.get("id") != watch_id]
    if len(kept) == len(watches):
        return False
    save_watches(settings, kept)
    log.info("watches.removed", id=watch_id)
    return True


def set_enabled(settings: Settings, watch_id: str, enabled: bool) -> bool:
    watches = load_watches(settings)
    hit = False
    for w in watches:
        if w.get("id") == watch_id:
            w["enabled"] = bool(enabled)
            hit = True
            break
    if hit:
        save_watches(settings, watches)
        log.info("watches.toggled", id=watch_id, enabled=enabled)
    return hit


async def evaluate_watches(settings: Settings) -> list[dict[str, Any]]:
    """Trips from every ENABLED watch, in the same shape as alert_worker's home
    trips ({id, kind, description, data}). Each source failure is isolated — one
    dead feed for one watch never sinks the rest."""
    watches = [w for w in load_watches(settings) if w.get("enabled", True)]
    if not watches:
        return []
    trips: list[dict[str, Any]] = []

    for w in watches:
        wid = w.get("id", "?")
        label = w.get("label") or wid
        try:
            lat = float(w["lat"])
            lon = float(w["lon"])
            radius = float(w.get("radius_km", 150.0))
        except (KeyError, TypeError, ValueError):
            log.info("watch.bad_geometry", id=wid)
            continue
        sources = w.get("sources") or []
        pfx = f"watch:{wid}:"

        if "military_air" in sources:
            try:
                mil = await military.fetch_military_aircraft(
                    lat=lat, lon=lon, radius_km=min(radius, _ADSB_MAX_RADIUS_KM)
                )
                if mil.get("ok"):
                    for ac in mil.get("aircraft", []):
                        hexid = ac.get("hex") or ac.get("callsign") or "?"
                        cs = ac.get("callsign") or hexid
                        tc = ac.get("type_code") or "?"
                        trips.append({
                            "id": f"{pfx}mil:{hexid}", "kind": "military_air",
                            "description": (
                                f"[{label}] Military aircraft {cs} ({tc}) within "
                                f"{min(radius, _ADSB_MAX_RADIUS_KM):.0f} km"
                            ),
                            "data": {**ac, "watch": label, "watch_id": wid},
                        })
            except Exception as e:  # noqa: BLE001
                log.info("watch.mil_failed", id=wid, error=str(e))

        if "gps_jamming" in sources:
            try:
                gps = await gpsjam.fetch_gps_jamming(
                    lat=lat, lon=lon, radius_km=min(radius, _ADSB_MAX_RADIUS_KM)
                )
                if gps.get("ok"):
                    for z in gps.get("zones", []):
                        d = _haversine_km(lat, lon, z["lat"], z["lon"])
                        if d <= radius:
                            trips.append({
                                "id": f"{pfx}gps:{z['lat']:.1f}_{z['lon']:.1f}",
                                "kind": "gps_jamming",
                                "description": (
                                    f"[{label}] GPS-jamming zone {z['severity_pct']}% "
                                    f"severity, {z['degraded_count']} aircraft, ~{d:.0f} km"
                                ),
                                "data": {**z, "distance_km": round(d, 1),
                                         "watch": label, "watch_id": wid},
                            })
            except Exception as e:  # noqa: BLE001
                log.info("watch.gps_failed", id=wid, error=str(e))

        if "recon_satellite" in sources:
            try:
                min_el = float(w.get("sat_min_elevation_deg", 30.0))
                sats = await satellites.fetch_satellites_overhead(
                    lat=lat, lon=lon, min_elevation=min_el, limit=50
                )
                if sats.get("ok"):
                    for st in sats.get("satellites", []):
                        if st.get("mission") == "military_recon":
                            trips.append({
                                "id": f"{pfx}sat:{st['name']}", "kind": "recon_satellite",
                                "description": (
                                    f"[{label}] Recon satellite {st['name']} overhead at "
                                    f"{st['elevation_deg']:.0f}° elevation"
                                ),
                                "data": {**st, "watch": label, "watch_id": wid},
                            })
            except Exception as e:  # noqa: BLE001
                log.info("watch.sat_failed", id=wid, error=str(e))

        if "vessel" in sources:
            try:
                for v in maritime.snapshot_vessels(lat, lon, radius):
                    if v.get("is_naval") or v.get("is_anomalous"):
                        if v.get("is_naval"):
                            tag = "naval/military"
                        else:
                            ns = v.get("nav_status") or f"nav code {v.get('nav_status_code')}"
                            tag = f"anomalous ({ns})"
                        nm = v.get("name") or f"MMSI {v['mmsi']}"
                        trips.append({
                            "id": f"{pfx}ship:{v['mmsi']}", "kind": "vessel",
                            "description": (
                                f"[{label}] Vessel {nm} — {tag} — ~{v['distance_km']:.0f} km"
                            ),
                            "data": {**v, "watch": label, "watch_id": wid},
                        })
            except Exception as e:  # noqa: BLE001
                log.info("watch.ship_failed", id=wid, error=str(e))

    return trips
