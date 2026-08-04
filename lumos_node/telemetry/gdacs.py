"""GDACS global disaster alerts — severity-scored multi-hazard events.

Source: GDACS RSS (https://www.gdacs.org/xml/rss.xml) — the UN/EC Global
Disaster Alert and Coordination System. Keyless. Each item is one disaster
event (earthquake, tropical cyclone, flood, volcano, wildfire, drought,
tsunami) carrying a Green/Orange/Red alert level, affected country, and a
georss point. This is the severity-scored complement to EONET's raw
active-events count (cosmic.py): EONET says "how much is happening",
GDACS says "WHICH events matter and how badly".

Parsing is stdlib-only (xml.etree.ElementTree), matching news.py. The alert
path is EDGE-TRIGGERED like regulus_watch: a persisted seen-map keyed by
GDACS event id fires once per NEW event at/above the configured level, and
once more only if an event ESCALATES (orange → red). A multi-day red cyclone
therefore pings exactly once, not every cooldown expiry.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx
import orjson

from ..config import Settings
from ..log import get_logger
from . import cache as tcache
from .worker import _data_dir

log = get_logger(__name__)

_GDACS_RSS = "https://www.gdacs.org/xml/rss.xml"
_TIMEOUT = httpx.Timeout(12.0, connect=5.0)

_GDACS_NS = "{http://www.gdacs.org}"
_GEORSS_NS = "{http://www.georss.org/georss}"

_STATE_FILE = "gdacs_watch.json"
_SEEN_CAP = 500  # bound the persisted seen-map

_TYPE_LABELS = {
    "EQ": "Earthquake",
    "TC": "Tropical cyclone",
    "FL": "Flood",
    "VO": "Volcano",
    "WF": "Wildfire",
    "DR": "Drought",
    "TS": "Tsunami",
}

_LEVEL_RANK = {"green": 1, "orange": 2, "red": 3}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""


def parse_gdacs_xml(xml_text: str) -> list[dict[str, Any]]:
    """PURE TRANSFORM — GDACS RSS XML → compact event dicts. No network.

    Malformed items are skipped; a malformed document returns []. Split out
    from the fetch so tests can feed canned XML (repo tests are offline).

    XXE/billion-laughs hardening without a defusedxml dependency: both attack
    classes require a DTD (<!DOCTYPE/<!ENTITY). Legitimate RSS never carries
    one, so any document declaring a DTD is rejected before parsing — the same
    forbid_dtd posture defusedxml enforces.
    """
    head = xml_text[:4096]
    if "<!DOCTYPE" in head or "<!ENTITY" in head:
        log.warning("gdacs.dtd_rejected")
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        event_id = _text(item, f"{_GDACS_NS}eventid")
        level = _text(item, f"{_GDACS_NS}alertlevel").lower()
        if not event_id or level not in _LEVEL_RANK:
            continue
        etype = _text(item, f"{_GDACS_NS}eventtype").upper()
        lat = lon = None
        point = _text(item, f"{_GEORSS_NS}point")
        if point:
            parts = point.split()
            if len(parts) == 2:
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                except ValueError:
                    lat = lon = None
        out.append({
            "event_id": event_id,
            "episode_id": _text(item, f"{_GDACS_NS}episodeid") or None,
            "type": etype,
            "type_label": _TYPE_LABELS.get(etype, etype or "Event"),
            "level": level,
            "level_rank": _LEVEL_RANK[level],
            "country": _text(item, f"{_GDACS_NS}country") or None,
            "title": _text(item, "title"),
            "lat": lat,
            "lon": lon,
            "severity_text": _text(item, f"{_GDACS_NS}severity") or None,
            "population_text": _text(item, f"{_GDACS_NS}population") or None,
            "from_date": _text(item, f"{_GDACS_NS}fromdate") or None,
            "to_date": _text(item, f"{_GDACS_NS}todate") or None,
            "pub_date": _text(item, "pubDate") or None,
            "link": _text(item, "link") or None,
        })
    return out


async def fetch_gdacs_events() -> dict[str, Any]:
    """Current GDACS disaster events, cached 15 min. {ok, count, events}.

    Caches the failure path too so a GDACS outage doesn't re-hammer.
    """
    cached = tcache.get("gdacs")
    if cached is not None:
        return cached
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(_GDACS_RSS, timeout=_TIMEOUT)
            r.raise_for_status()
            events = parse_gdacs_xml(r.text)
    except (httpx.HTTPError, ValueError) as e:
        log.info("gdacs.fetch_failed", error=str(e))
        result = {"ok": False, "count": 0, "events": [], "fetched_at": fetched_at}
        tcache.put("gdacs", result)
        return result
    events.sort(key=lambda e: (-e["level_rank"], e["type"]))
    result = {"ok": True, "count": len(events), "events": events, "fetched_at": fetched_at}
    tcache.put("gdacs", result)
    return result


# ── Alert-monitor integration (edge-triggered, regulus_watch idiom) ──────────


def _state_path(s: Settings) -> Path:
    return _data_dir(s) / _STATE_FILE


def _read_state(s: Settings) -> dict[str, Any]:
    p = _state_path(s)
    if not p.exists():
        return {"seen": {}}
    try:
        st = orjson.loads(p.read_bytes())
        return st if isinstance(st, dict) and isinstance(st.get("seen"), dict) else {"seen": {}}
    except (orjson.JSONDecodeError, OSError):
        return {"seen": {}}


def _write_state(s: Settings, st: dict[str, Any]) -> None:
    try:
        _state_path(s).write_bytes(orjson.dumps(st))
    except OSError as e:
        log.warning("gdacs.state_write_failed", error=str(e))


def fresh_disaster_trips(
    events: list[dict[str, Any]],
    seen: dict[str, int],
    min_rank: int,
    operator_lat: float,
    operator_lon: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """PURE TRANSFORM — which events deserve a wake, given the seen-map.

    Trips on a NEW event id at/above min_rank, or a rank ESCALATION of a known
    one. Returns (trips, updated_seen). Updated seen records every current
    event (even below-threshold ones) so a later escalation is detected.
    """
    trips: list[dict[str, Any]] = []
    updated = dict(seen)
    for ev in events:
        eid = str(ev["event_id"])
        rank = int(ev["level_rank"])
        prev = updated.get(eid)
        updated[eid] = max(rank, prev or 0)
        if rank < min_rank or (prev is not None and prev >= rank):
            continue
        escalated = prev is not None and rank > prev
        where = ev.get("country") or "location n/a"
        dist_txt = ""
        if ev.get("lat") is not None and ev.get("lon") is not None:
            d = _haversine_km(operator_lat, operator_lon, ev["lat"], ev["lon"])
            dist_txt = f", ~{d:.0f} km away"
        verb = "escalated to" if escalated else "alert —"
        trips.append({
            "id": f"gdacs:{eid}:{rank}",
            "kind": "disaster",
            "description": (
                f"GDACS {ev['level'].upper()} {verb} {ev['type_label']}: "
                f"{ev.get('title') or eid} ({where}{dist_txt})"
            ),
            "data": ev,
        })
    if len(updated) > _SEEN_CAP:
        # Keep the most recently observed ids (dict preserves insertion order;
        # re-inserting current events last keeps them).
        updated = dict(list(updated.items())[-_SEEN_CAP:])
    return trips, updated


async def evaluate_gdacs(settings: Settings) -> list[dict[str, Any]]:
    """Alert-monitor entry — edge-triggered disaster trips, else []."""
    snap = await fetch_gdacs_events()
    if not snap.get("ok"):
        return []
    min_rank = _LEVEL_RANK.get(settings.alert_gdacs_min_level.lower(), 3)
    st = _read_state(settings)
    trips, updated = fresh_disaster_trips(
        snap["events"], st.get("seen", {}), min_rank,
        settings.operator_lat, settings.operator_lon,
    )
    if updated != st.get("seen"):
        _write_state(settings, {"seen": updated})
    if trips:
        log.info("gdacs.tripped", n=len(trips))
    return trips
