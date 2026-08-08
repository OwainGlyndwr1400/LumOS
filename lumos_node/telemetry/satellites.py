"""Satellites overhead — full active catalog + identity, skyfield geometry.

TLE sources (Phase 44 — "know everything that flies over"):
  PRIMARY:  CelesTrak GP active group (gp.php?GROUP=active&FORMAT=tle) — the
            complete active catalog (~11k objects incl. every Starlink, ISS,
            CSS, OneWeb…). Keyless; updates a few times a day; cached 6 h per
            CelesTrak's fair-use guidance.
  FALLBACK: SatNOGS DB TLE API (the original source) when CelesTrak is down.

Identity enrichment — CelesTrak SATCAT (pub/satcat.csv, cached 24 h) joined by
NORAD id gives every object its OWNER country, launch date, object type
(payload / rocket body / debris) and operational status. Ping text carries
name + country + launch year + designator so the operator can google a new
bird and feed the result back into memory (the RAG loop).

Propagation + observer geometry via skyfield (TEME → topocentric az/el done
right). The EarthSatellite objects are memoized per TLE refresh — building
~11k of them costs seconds, so it happens once per 6 h, not per sweep. The
sweep itself runs in a thread (asyncio.to_thread) and its result caches 60 s.

Mission classification is NORAD-name-keyword (Osiris-style), unchanged.

Also home to the Sentinel SAR/optical scene lookup (Element84 Earth Search
STAC + Copernicus fallback) — satellite imagery coverage over a point, ported
from the Osiris `sentinel` route.
"""

from __future__ import annotations

import asyncio
import csv
import io
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from ..config import Settings, get_settings
from ..log import get_logger
from . import cache as tcache

log = get_logger(__name__)


_CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
_CELESTRAK_SATCAT = "https://celestrak.org/pub/satcat.csv"
_SATNOGS_TLE = "https://db.satnogs.org/api/tle/?format=json"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_SATCAT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)  # satcat.csv is ~4 MB

# Mission classification by NORAD name keyword (Osiris-style). First match wins.
_MISSION_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("ISS", "ZARYA", "TIANGONG", "CSS (", "TIANHE"), "station"),
    (("STARLINK",), "comms_constellation"),
    (("ONEWEB",), "comms_constellation"),
    (("IRIDIUM", "GLOBALSTAR", "ORBCOMM", "INTELSAT", "INMARSAT", "SES"), "comms"),
    (("GPS", "NAVSTAR", "GLONASS", "GALILEO", "BEIDOU", "QZS", "IRNSS"), "navigation"),
    (("NOAA", "METEOR", "GOES", "METOP", "HIMAWARI", "FENGYUN", "ELEKTRO"), "weather"),
    # Military-affiliated BY NORAD NAME — a tracking-layer heuristic, NOT confirmed
    # mission. The USA/COSMOS designations cover recon AND comms/nav/early-warning,
    # so "military" is the honest label; "recon" over-asserts. GAOFEN is China's
    # CIVILIAN high-resolution EO programme (dual-use at most) → earth_obs.
    (("COSMOS", "USA ", "NROL", "YAOGAN", "OFEQ", "KOSMOS", "SHIYAN"), "military"),
    (("SENTINEL", "LANDSAT", "TERRA", "AQUA", "WORLDVIEW", "PLANET", "DOVE", "SKYSAT",
      "ICEYE", "GAOFEN"), "earth_obs"),
    (("HUBBLE", "TESS", "CHEOPS", "JWST", "XMM", "INTEGRAL"), "science"),
)

# SATCAT OWNER code → friendly country/operator name (common codes; raw code
# passes through for the long tail — still googleable).
_OWNER_NAMES: dict[str, str] = {
    "US": "USA", "CIS": "Russia/CIS", "PRC": "China", "UK": "United Kingdom",
    "FR": "France", "JPN": "Japan", "IND": "India", "GER": "Germany",
    "IT": "Italy", "SPN": "Spain", "CA": "Canada", "AUS": "Australia",
    "SKOR": "South Korea", "NKOR": "North Korea", "IRAN": "Iran",
    "ISRA": "Israel", "TURK": "Turkey", "UAE": "UAE", "SAUD": "Saudi Arabia",
    "EGYP": "Egypt", "ARGN": "Argentina", "BRAZ": "Brazil", "MEX": "Mexico",
    "INDO": "Indonesia", "THAI": "Thailand", "PAKI": "Pakistan",
    "ESA": "European Space Agency", "EUME": "EUMETSAT", "EUTE": "Eutelsat",
    "ITSO": "Intelsat", "IM": "Inmarsat", "GLOB": "Globalstar",
    "ORB": "Orbcomm", "O3B": "O3b/SES", "SES": "SES", "AB": "Arabsat",
    "NATO": "NATO", "NOR": "Norway", "SWED": "Sweden", "NETH": "Netherlands",
    "BEL": "Belgium", "SWTZ": "Switzerland", "POL": "Poland", "CZE": "Czechia",
    "UKR": "Ukraine", "SAFR": "South Africa", "NZ": "New Zealand",
    "LUXE": "Luxembourg", "SING": "Singapore", "TWN": "Taiwan",
}

_OBJECT_TYPES = {"PAY": "payload", "R/B": "rocket body", "DEB": "debris", "UNK": "unknown"}


def _classify(name: str) -> str:
    up = name.upper()
    for kws, label in _MISSION_KEYWORDS:
        if any(k in up for k in kws):
            return label
    return "other"


def parse_tle_text(text: str) -> list[dict[str, str]]:
    """PURE TRANSFORM — 3-line-element text (CelesTrak FORMAT=tle) → TLE dicts.
    Tolerant of blank lines; skips malformed sets."""
    out: list[dict[str, str]] = []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    i, n = 0, len(lines)
    while i <= n - 3:
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            out.append({"name": name.strip(), "line1": l1, "line2": l2})
            i += 3
        else:
            i += 1  # resync — stray line
    return out


async def fetch_tle() -> list[dict[str, str]]:
    """Active-catalog TLE set, cached 6 h. [{name, line1, line2}, ...].

    CelesTrak GP first (complete catalog); SatNOGS fallback. Caches the
    empty/failure path too so an outage doesn't re-hammer either source.
    """
    cached = tcache.get("tle")
    if cached is not None:
        return cached

    out: list[dict[str, str]] = []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(_CELESTRAK_GP, timeout=_SATCAT_TIMEOUT, follow_redirects=True)
            r.raise_for_status()
            out = parse_tle_text(r.text)
    except (httpx.HTTPError, ValueError) as e:
        log.info("satellites.celestrak_fetch_failed", error=str(e))

    if not out:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(_SATNOGS_TLE, timeout=_TIMEOUT)
                r.raise_for_status()
                data = r.json()
            seen: set[str] = set()
            for item in data if isinstance(data, list) else []:
                name = (item.get("tle0") or "").strip()
                l1 = (item.get("tle1") or "").strip()
                l2 = (item.get("tle2") or "").strip()
                if name and l1 and l2 and name not in seen:
                    seen.add(name)
                    out.append({"name": name, "line1": l1, "line2": l2})
        except (httpx.HTTPError, ValueError) as e:
            log.info("satellites.tle_fetch_failed", error=str(e))

    log.info("satellites.tle_loaded", count=len(out))
    tcache.put("tle", out, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("tle", 21600))
    return out


# ── SATCAT identity (NORAD → owner / launch / type) ─────────────────────────


def parse_satcat_csv(text: str) -> dict[str, dict[str, Any]]:
    """PURE TRANSFORM — CelesTrak satcat.csv → {norad: identity}. Header-driven.

    Parsed with the stdlib csv module (RFC-4180 quoting) so a quoted field
    containing a comma — owner and object names do — splits on the right column
    boundaries. A naive str.split(",") shifts every column after such a field
    and silently corrupts the country/type/launch data cached for 24 h.
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header_row = next(reader)
    except StopIteration:
        return {}
    header = [h.strip().upper() for h in header_row]

    def col(name: str) -> int | None:
        return header.index(name) if name in header else None

    i_norad = col("NORAD_CAT_ID")
    if i_norad is None:
        return {}
    i_owner, i_launch = col("OWNER"), col("LAUNCH_DATE")
    i_type, i_ops = col("OBJECT_TYPE"), col("OPS_STATUS_CODE")
    i_intl = col("OBJECT_ID")  # international designator, e.g. 1998-067A

    out: dict[str, dict[str, Any]] = {}
    for parts in reader:
        if len(parts) <= i_norad:
            continue
        norad = parts[i_norad].strip().lstrip("0")
        if not norad:
            continue
        owner = parts[i_owner].strip() if i_owner is not None and len(parts) > i_owner else ""
        otype = parts[i_type].strip().upper() if i_type is not None and len(parts) > i_type else ""
        out[norad] = {
            "intl_designator": (parts[i_intl].strip() or None) if i_intl is not None and len(parts) > i_intl else None,
            "country_code": owner or None,
            "country": _OWNER_NAMES.get(owner, owner) or None,
            "launch_date": (parts[i_launch].strip() or None) if i_launch is not None and len(parts) > i_launch else None,
            "object_type": _OBJECT_TYPES.get(otype, otype.lower() or None),
            "ops_status": (parts[i_ops].strip() or None) if i_ops is not None and len(parts) > i_ops else None,
        }
    return out


async def fetch_satcat() -> dict[str, dict[str, Any]]:
    """NORAD → identity map, cached 24 h (satcat is a slow-moving catalog)."""
    cached = tcache.get("satcat")
    if cached is not None:
        return cached
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(_CELESTRAK_SATCAT, timeout=_SATCAT_TIMEOUT, follow_redirects=True)
            r.raise_for_status()
            cat = parse_satcat_csv(r.text)
    except (httpx.HTTPError, ValueError) as e:
        log.info("satellites.satcat_fetch_failed", error=str(e))
        cat = {}
    log.info("satellites.satcat_loaded", entries=len(cat))
    tcache.put("satcat", cat)
    return cat


# ── Propagation sweep (memoized skyfield objects) ────────────────────────────

# EarthSatellite construction for ~11k TLEs costs seconds — memoize per TLE
# refresh. Keyed by a cheap fingerprint of the set, NOT id(), so a re-fetched
# identical set still reuses the built objects.
_sat_objs: dict[str, Any] = {"fingerprint": None, "sats": []}
# Guards the atomic publish of _sat_objs below. Two cold-cache sweeps run in
# separate asyncio.to_thread workers; without this a half-updated global (new
# fingerprint, old sats) or a list reassigned mid-iteration would give wrong
# results or raise RuntimeError.
_sat_objs_lock = threading.Lock()


def _tle_fingerprint(tles: list[dict[str, str]]) -> str:
    if not tles:
        return "empty"
    return f"{len(tles)}:{tles[0]['line1'][:32]}:{tles[-1]['line1'][:32]}"


def _compute_overhead(
    tles: list[dict[str, str]],
    satcat: dict[str, dict[str, Any]],
    lat: float,
    lon: float,
    min_elevation: float,
    limit: int,
) -> list[dict[str, Any]]:
    """SYNC (runs in a thread) — propagate the catalog to NOW, keep those above
    `min_elevation` over the observer, enriched with SATCAT identity.

    Per-satellite errors (decayed orbits, malformed TLE) are skipped — sgp4
    raises on some objects and we never want one bad TLE to kill the sweep.
    """
    global _sat_objs

    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale(builtin=True)  # offline — no leap-second download

    fp = _tle_fingerprint(tles)
    snapshot = _sat_objs  # single-reference read; the global is never mutated in place
    if snapshot.get("fingerprint") == fp:
        sats = snapshot["sats"]
    else:
        built = []
        for tle in tles:
            try:
                built.append(EarthSatellite(tle["line1"], tle["line2"], tle["name"], ts))
            except Exception:  # noqa: BLE001 — malformed TLE
                continue
        sats = built
        # Publish atomically: swap in a brand-new dict under the lock so a
        # concurrent sweep never observes a torn fingerprint/sats pair or
        # iterates a list being reassigned out from under it.
        with _sat_objs_lock:
            _sat_objs = {"fingerprint": fp, "sats": built}
        log.info("satellites.objects_built", count=len(built))

    t = ts.now()
    observer = wgs84.latlon(lat, lon)

    results: list[dict[str, Any]] = []
    for sat in sats:
        try:
            alt, az, dist = (sat - observer).at(t).altaz()
            elev = alt.degrees
            if elev < min_elevation:
                continue
            sub = wgs84.subpoint(sat.at(t))
            satnum = getattr(sat.model, "satnum", None)
            norad = str(satnum) if satnum else None
            ident = satcat.get(norad or "", {})
            name = sat.name or f"NORAD {norad}"
            results.append({
                "name": name,
                "norad": norad,
                "intl_designator": ident.get("intl_designator"),
                "mission": _classify(name),
                "country_code": ident.get("country_code"),
                "country": ident.get("country"),
                "launch_date": ident.get("launch_date"),
                "object_type": ident.get("object_type"),
                "ops_status": ident.get("ops_status"),
                "elevation_deg": round(elev, 1),
                "azimuth_deg": round(az.degrees, 1),
                "range_km": round(dist.km, 1),
                "sub_lat": round(sub.latitude.degrees, 3),
                "sub_lon": round(sub.longitude.degrees, 3),
                "altitude_km": round(sub.elevation.km, 1),
            })
        except Exception:  # noqa: BLE001 — one bad propagation must not kill the sweep
            continue

    results.sort(key=lambda s: -s["elevation_deg"])  # highest in sky first
    return results[:limit]


async def fetch_satellites_overhead(
    lat: float | None = None,
    lon: float | None = None,
    min_elevation: float = 10.0,
    limit: int = 30,
) -> dict[str, Any]:
    """Satellites currently above the observer's horizon (elevation ≥ min).

    Falls back to operator location. Returns {ok, count, satellites:[...],
    scanned, center, fetched_at} — each satellite now carries NORAD id,
    country, launch date and object type (SATCAT join) alongside the geometry.
    Rides the 6 h TLE cache + 24 h SATCAT cache + 60 s result cache.
    """
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    cache_key = f"sat_passes_{lat:.3f}_{lon:.3f}_{min_elevation:.0f}_{limit}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    tles = await fetch_tle()
    satcat = await fetch_satcat()
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    if not tles:
        result = {
            "ok": False,
            "error": "no TLE data available (CelesTrak + SatNOGS both failed)",
            "count": 0,
            "satellites": [],
            "scanned": 0,
            "center": {"lat": lat, "lon": lon, "min_elevation_deg": min_elevation},
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("sat_passes", 60))
        return result

    try:
        overhead = await asyncio.to_thread(
            _compute_overhead, tles, satcat, lat, lon, min_elevation, limit
        )
    except Exception as e:  # noqa: BLE001 — propagation sweep failure
        log.warning("satellites.propagation_failed", error=str(e))
        result = {
            "ok": False,
            "error": f"propagation failed: {e}",
            "count": 0,
            "satellites": [],
            "scanned": len(tles),
            "center": {"lat": lat, "lon": lon, "min_elevation_deg": min_elevation},
            "fetched_at": fetched_at,
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("sat_passes", 60))
        return result

    result = {
        "ok": True,
        "count": len(overhead),
        "satellites": overhead,
        "scanned": len(tles),
        "center": {"lat": lat, "lon": lon, "min_elevation_deg": min_elevation},
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("sat_passes", 60))
    return result


# ── Alert-monitor integration: ANY satellite near-overhead (gated) ───────────


def describe_satellite(st: dict[str, Any]) -> str:
    """One-line identity for ping text — name (country, type, launched YYYY,
    designator) — everything the operator needs to google a new bird."""
    bits: list[str] = []
    if st.get("country"):
        bits.append(str(st["country"]))
    mission = st.get("mission")
    if mission and mission != "other":
        bits.append(mission.replace("_", " "))
    elif st.get("object_type"):
        bits.append(str(st["object_type"]))
    ld = st.get("launch_date")
    if ld:
        bits.append(f"launched {str(ld)[:4]}")
    if st.get("norad"):
        bits.append(f"NORAD {st['norad']}")
    inner = ", ".join(bits)
    return f"{st['name']} ({inner})" if inner else str(st["name"])


async def evaluate_all_sats_overhead(settings: Settings) -> list[dict[str, Any]]:
    """Alert-monitor entry — ANY satellite above alert_sat_all_min_elevation_deg.

    military birds are EXCLUDED here (the recon_satellite kind already
    covers them at its own threshold — one sat, one kind). Rides the same
    sweep cache as everything else; per-sat ids dedup via the monitor's
    identity cooldown; a source cooldown throttles the kind (Starlink passes
    are near-continuous — this kind is OFF by default for a reason).
    """
    sats = await fetch_satellites_overhead(
        lat=settings.operator_lat, lon=settings.operator_lon,
        min_elevation=settings.alert_sat_all_min_elevation_deg, limit=50,
    )
    if not sats.get("ok"):
        return []
    trips: list[dict[str, Any]] = []
    for st in sats.get("satellites", []):
        if st.get("mission") == "military":
            continue
        ident = st.get("norad") or st.get("name")
        trips.append({
            "id": f"satall:{ident}",
            "kind": "satellite_overhead",
            "description": (
                f"Satellite {describe_satellite(st)} overhead at "
                f"{st['elevation_deg']:.0f}° elevation"
            ),
            "data": st,
        })
    if trips:
        log.info("satellites.all_overhead_tripped", n=len(trips))
    return trips


# ── Sentinel SAR / optical scene lookup (Osiris `sentinel` route port) ───────

_STAC_ELEMENT84 = "https://earth-search.aws.element84.com/v1/search"
_STAC_COPERNICUS = "https://catalogue.dataspace.copernicus.eu/stac/search"


def _format_stac_scene(feat: dict[str, Any]) -> dict[str, Any]:
    props = feat.get("properties") or {}
    assets = feat.get("assets") or {}
    thumb = None
    for key in ("thumbnail", "preview", "overview"):
        if key in assets and isinstance(assets[key], dict):
            thumb = assets[key].get("href")
            break
    return {
        "id": feat.get("id"),
        "collection": feat.get("collection"),
        "datetime": props.get("datetime"),
        "platform": props.get("platform") or props.get("constellation"),
        "instrument": ",".join(props.get("instruments", []) or []) or None,
        "mode": props.get("sar:instrument_mode"),
        "polarizations": props.get("sar:polarizations"),
        "cloud_cover_pct": props.get("eo:cloud_cover"),
        "thumbnail": thumb,
    }


async def fetch_sar_scenes(
    lat: float | None = None,
    lon: float | None = None,
    radius_deg: float = 1.0,
    days: int = 30,
    limit: int = 10,
) -> dict[str, Any]:
    """Recent Sentinel satellite scenes covering a point (SAR first, optical
    fallback). Element84 Earth Search STAC, Copernicus STAC as last resort —
    all keyless. {ok, count, scenes:[...], source, center, fetched_at}."""
    settings = get_settings()
    if lat is None or lon is None:
        lat = settings.operator_lat
        lon = settings.operator_lon

    cache_key = f"sar_{lat:.2f}_{lon:.2f}_{radius_deg:.1f}_{days}_{limit}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    bbox = [lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg]
    now = datetime.now(UTC)
    dt_range = (
        f"{(now - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%S')}Z/"
        f"{now.strftime('%Y-%m-%dT%H:%M:%S')}Z"
    )
    fetched_at = now.isoformat(timespec="seconds")

    scenes: list[dict[str, Any]] = []
    source = None
    attempts = (
        (_STAC_ELEMENT84, "sentinel-1-grd", "element84-s1"),
        (_STAC_ELEMENT84, "sentinel-2-l2a", "element84-s2"),
        (_STAC_COPERNICUS, "SENTINEL-1", "copernicus-s1"),
    )
    async with httpx.AsyncClient() as client:
        for url, collection, tag in attempts:
            try:
                r = await client.post(
                    url,
                    json={
                        "collections": [collection],
                        "bbox": bbox,
                        "datetime": dt_range,
                        "limit": limit,
                        "sortby": [{"field": "datetime", "direction": "desc"}],
                    },
                    timeout=_TIMEOUT,
                )
                r.raise_for_status()
                feats = (r.json() or {}).get("features") or []
                if feats:
                    scenes = [_format_stac_scene(f) for f in feats]
                    source = tag
                    break
            except (httpx.HTTPError, ValueError) as e:
                log.info("satellites.stac_failed", url=url, collection=collection, error=str(e))

    result = {
        "ok": bool(scenes),
        "count": len(scenes),
        "scenes": scenes,
        "source": source,
        "center": {"lat": lat, "lon": lon, "radius_deg": radius_deg, "days": days},
        "fetched_at": fetched_at,
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("sar", 300))
    return result
