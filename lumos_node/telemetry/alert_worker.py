"""Alert monitor (Phase 3) — event-driven threshold wakes.

Generalizes the cosmic-trigger scaffold (telemetry/worker.py) to the full intel
layer: polls each source on a cadence, evaluates NUMERIC thresholds in PURE CODE
(no LLM), and on a fresh trip wakes Lumos via autonomy.trigger_autonomous_turn
with ONLY the tripped events as context. Autonomy ends at speaking.

Design (locked 2026-05-29):
  • Event-driven, not a timed dump: the poll + threshold check is tokenless code;
    the LLM is invoked only on a trip, and sees only what tripped.
  • Per-(source, identity) dedup: a given aircraft hex / ship MMSI / satellite /
    GPS zone re-alerts only after `alert_cooldown_minutes`; a daily cap bounds
    total wakes; a new distinct entity is a new alert.
  • Bundled wake: all FRESH trips in one poll cycle become ONE wake ("here's
    what's around"), not N separate pings.
  • Gated: runs when alert_monitor_enabled; only WAKES when autonomy_enabled.
    alert_monitor_enabled + autonomy OFF = DRY RUN (logs what WOULD trip without
    pinging) — handy for tuning thresholds first.
  • Ships ride the persistent AIS cache (maritime.ais_monitor_loop), started as a
    child task here, so the naval-type + anomalous filters are reliable.

Thresholds (Erydir's locked values, from config):
  Kp/flare/quake/NEO (reuse cosmic) · mil-air ≤40 mi · ships naval|anomalous
  ≤50 mi · GPS zone ≤150 km · mil-recon sat ≥60° elevation.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from ..config import Settings, get_settings
from ..events import EventBus
from ..log import get_logger
from . import cosmic, gpsjam, maritime, military, satellites
from .worker import (
    _chat_idle_seconds,
    _data_dir,
    _evaluate_thresholds as _evaluate_cosmic,
    _today_iso,
)


log = get_logger(__name__)

_LOG_FILE = "alert_events.jsonl"
_STATE_FILE = "alert_state.json"
_LOG_CAP = 1000


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _state_path(s: Settings) -> Path:
    return _data_dir(s) / _STATE_FILE


def _log_path(s: Settings) -> Path:
    return _data_dir(s) / _LOG_FILE


def _read_state(s: Settings) -> dict[str, Any]:
    p = _state_path(s)
    base = {"identities": {}, "fires_today": 0, "day_iso": ""}
    if not p.exists():
        return base
    try:
        st = orjson.loads(p.read_bytes())
        for k, v in base.items():
            st.setdefault(k, v)
        return st
    except (orjson.JSONDecodeError, OSError):
        return base


def _write_state(s: Settings, st: dict[str, Any]) -> None:
    try:
        _state_path(s).write_bytes(orjson.dumps(st, option=orjson.OPT_INDENT_2))
    except OSError as e:
        log.warning("alert.state_write_failed", error=str(e))


def _append_log(s: Settings, entry: dict[str, Any]) -> None:
    p = _log_path(s)
    try:
        with p.open("ab") as f:
            f.write(orjson.dumps(entry))
            f.write(b"\n")
    except OSError as e:
        log.warning("alert.log_write_failed", error=str(e))
        return
    try:
        with p.open("rb") as f:
            lines = f.readlines()
        if len(lines) > _LOG_CAP * 2:
            with p.open("wb") as f:
                f.writelines(lines[-_LOG_CAP:])
    except OSError:
        pass


async def _evaluate_alerts(settings: Settings) -> list[dict[str, Any]]:
    """Gather current threshold trips across all sources. Each trip is
    {id, kind, description, data}. `id` is the dedup identity (per aircraft /
    vessel / satellite / GPS zone / cosmic kind)."""
    lat, lon = settings.operator_lat, settings.operator_lon
    trips: list[dict[str, Any]] = []

    # ── Cosmic (Kp / flare / quake / NEO) — reuse the cosmic evaluator verbatim.
    try:
        snap = await cosmic.snapshot_all()
        for ev in _evaluate_cosmic(snap, settings):
            trips.append(
                {"id": f"cosmic:{ev['kind']}", "kind": ev["kind"],
                 "description": ev["description"], "data": ev}
            )
    except Exception as e:  # noqa: BLE001
        log.info("alert.cosmic_failed", error=str(e))

    # ── Military aircraft within radius.
    try:
        mil = await military.fetch_military_aircraft(
            lat=lat, lon=lon, radius_km=settings.alert_military_air_radius_km
        )
        if mil.get("ok"):
            for ac in mil.get("aircraft", []):
                hexid = ac.get("hex") or ac.get("callsign") or "?"
                cs = ac.get("callsign") or hexid
                tc = ac.get("type_code") or "?"
                trips.append(
                    {"id": f"mil:{hexid}", "kind": "military_air",
                     "description": (
                         f"Military aircraft {cs} ({tc}) within "
                         f"{settings.alert_military_air_radius_km:.0f} km"
                     ),
                     "data": ac}
                )
    except Exception as e:  # noqa: BLE001
        log.info("alert.mil_failed", error=str(e))

    # ── Civilian aircraft within radius — per-category (commercial/private/jet)
    # toggles; military is handled above. Each category is independently gateable
    # because commercial overhead is near-constant in a populated area.
    air_cats = {
        c
        for c, on in (
            ("commercial", settings.alert_aircraft_commercial),
            ("private", settings.alert_aircraft_private),
            ("jet", settings.alert_aircraft_jet),
        )
        if on
    }
    if air_cats:
        try:
            from . import aircraft as _aircraft
            civ = await _aircraft.fetch_civilian_aircraft(
                lat=lat, lon=lon,
                radius_km=settings.alert_aircraft_radius_km,
                categories=air_cats,
            )
            if civ.get("ok"):
                _air_label = {"commercial": "Commercial", "private": "Private", "jet": "Private jet"}
                for ac in civ.get("aircraft", []):
                    hexid = ac.get("hex") or ac.get("callsign") or "?"
                    cs = ac.get("callsign") or hexid
                    cat = ac.get("category", "aircraft")
                    tc = ac.get("type_code") or "?"
                    dist = ac.get("distance_km")
                    dtxt = f" ~{dist:.0f} km" if dist is not None else ""
                    trips.append(
                        {"id": f"air:{hexid}", "kind": "aircraft",
                         "description": f"{_air_label.get(cat, cat)} aircraft {cs} ({tc}){dtxt}",
                         "data": ac}
                    )
        except Exception as e:  # noqa: BLE001
            log.info("alert.aircraft_failed", error=str(e))

    # ── GPS-jamming zones whose centroid is within the alert radius.
    try:
        gps = await gpsjam.fetch_gps_jamming(lat=lat, lon=lon)
        if gps.get("ok"):
            for z in gps.get("zones", []):
                d = _haversine_km(lat, lon, z["lat"], z["lon"])
                if d <= settings.alert_gps_jam_radius_km:
                    trips.append(
                        {"id": f"gps:{z['lat']:.1f}_{z['lon']:.1f}", "kind": "gps_jamming",
                         "description": (
                             f"GPS-jamming zone {z['severity_pct']}% severity, "
                             f"{z['degraded_count']} aircraft, ~{d:.0f} km away"
                         ),
                         "data": {**z, "distance_km": round(d, 1)}}
                    )
    except Exception as e:  # noqa: BLE001
        log.info("alert.gps_failed", error=str(e))

    # ── Military-recon satellites at high elevation (near-overhead pass).
    try:
        sats = await satellites.fetch_satellites_overhead(
            lat=lat, lon=lon, min_elevation=settings.alert_sat_min_elevation_deg, limit=50
        )
        if sats.get("ok"):
            for st in sats.get("satellites", []):
                if st.get("mission") == "military_recon":
                    trips.append(
                        {"id": f"sat:{st['name']}", "kind": "recon_satellite",
                         "description": (
                             f"Recon satellite {st['name']} overhead at "
                             f"{st['elevation_deg']:.0f}° elevation"
                         ),
                         "data": st}
                    )
    except Exception as e:  # noqa: BLE001
        log.info("alert.sat_failed", error=str(e))

    # ── Ships — persistent AIS cache; naval-type OR anomalous-behaviour within radius.
    try:
        for v in maritime.snapshot_vessels(lat, lon, settings.alert_ship_radius_km):
            if v.get("is_naval") or v.get("is_anomalous"):
                if v.get("is_naval"):
                    tag = "naval/military"
                else:
                    ns = v.get("nav_status") or f"nav code {v.get('nav_status_code')}"
                    tag = f"anomalous ({ns})"
                nm = v.get("name") or f"MMSI {v['mmsi']}"
                trips.append(
                    {"id": f"ship:{v['mmsi']}", "kind": "vessel",
                     "description": f"Vessel {nm} — {tag} — ~{v['distance_km']:.0f} km",
                     "data": v}
                )
    except Exception as e:  # noqa: BLE001
        log.info("alert.ship_failed", error=str(e))

    # ── Rail: a train CALLING (stopping) at the operator's home station (Gowerton).
    # Gated by alert_rail_enabled so there's no rtt.io call unless it's on + keyed.
    if settings.alert_rail_enabled:
        try:
            from . import rail
            board = await rail.fetch_station_calls()
            if board.get("ok"):
                stn = board.get("station") or board.get("code")
                lt = time.localtime()
                now_min = lt.tm_hour * 60 + lt.tm_min
                for t in board.get("calls", []):
                    # Wake ONLY on actionable trains: cancelled / delayed / due
                    # within the window. The board's routine future arrivals used
                    # to trip one wake EACH as their UIDs rolled in — all-day
                    # "on schedule, nothing to act on" spam.
                    reason = _rail_trip_reason(
                        t, now_min,
                        due_window_min=settings.alert_rail_due_minutes,
                        delay_threshold_min=settings.alert_rail_delay_minutes,
                    )
                    if reason is None:
                        continue
                    when = t.get("expected") or t.get("booked") or "?"
                    plat = f", plat {t['platform']}" if t.get("platform") else ""
                    direction = f" [{t['direction']}]" if t.get("direction") else ""
                    trips.append(
                        {"id": f"rail:{t['uid']}", "kind": "train",
                         "description": (
                             f"Train to {t.get('dest')} calling at {stn} "
                             f"~{when}{plat}{direction} — {reason}"
                         ),
                         "data": {**t, "trip_reason": reason}}
                    )
        except Exception as e:  # noqa: BLE001
            log.info("alert.rail_failed", error=str(e))

    # ── Severe weather: official Met Office warnings (MeteoAlarm, polygon/region)
    # + Open-Meteo point-forecast watch. Each source has its own on/off; both
    # surface as kind 'severe_weather' and share the severe-wx source cooldown.
    if settings.alert_metoffice_warnings_enabled or settings.alert_weather_watch_enabled:
        try:
            from . import weather_alerts
            trips.extend(await weather_alerts.evaluate_severe_weather(settings, lat, lon))
        except Exception as e:  # noqa: BLE001
            log.info("alert.severe_wx_failed", error=str(e))

    # ── Regulus horizon-crossing (edge-triggered): ONE ping when Regulus rises
    # below→above. Cheap (rides grimoire's ~60s cache); the watcher owns its own
    # transition state so it never re-pings while the star merely stays up.
    if settings.alert_regulus_rise_enabled:
        try:
            from . import regulus_watch
            trips.extend(await regulus_watch.evaluate_regulus_rise(settings))
        except Exception as e:  # noqa: BLE001
            log.info("alert.regulus_failed", error=str(e))

    # ── Operator-defined custom watches (Phase 39) — same dedup/cooldown/wake
    # path. Trip ids are prefixed 'watch:<id>:' so a watch fires independently of
    # the home thresholds (one feed failing for one watch never sinks the rest).
    try:
        from . import watches as _watches
        trips.extend(await _watches.evaluate_watches(settings))
    except Exception as e:  # noqa: BLE001
        log.info("alert.watches_failed", error=str(e))

    return trips


def _hhmm_to_min(s: str | None) -> int | None:
    """'0510' / '05:10' → minutes-of-day; None if missing/unparseable."""
    if not s:
        return None
    s = s.replace(":", "").strip()
    if len(s) != 4 or not s.isdigit():
        return None
    h, m = int(s[:2]), int(s[2:])
    if h > 23 or m > 59:
        return None
    return h * 60 + m


def _rail_trip_reason(
    call: dict[str, Any],
    now_min: int,
    *,
    due_window_min: int,
    delay_threshold_min: int,
) -> str | None:
    """Why this train deserves a WAKE — or None for a routine on-time arrival.

    Actionable = cancelled, delayed >= threshold, or actually due within the
    window. Pure code (per this module's design: thresholds are tokenless);
    everything else stays silent instead of burning a ~7.5k-token wake to
    narrate "on schedule, nothing to act on".
    """
    if call.get("cancelled"):
        return "CANCELLED"
    exp = _hhmm_to_min(call.get("expected"))
    booked = _hhmm_to_min(call.get("booked"))
    if exp is not None and booked is not None:
        late = ((exp - booked + 720) % 1440) - 720  # signed, midnight-safe
        if late >= delay_threshold_min:
            return f"delayed {late} min (booked {call.get('booked')})"
    if exp is not None and due_window_min > 0:
        until = (exp - now_min) % 1440  # minutes until expected call
        if until <= due_window_min:
            return "due now" if until <= 1 else f"due in {until} min"
    return None


def _global_eligible(settings: Settings, state: dict[str, Any]) -> tuple[bool, str]:
    """Daily-cap + chat-active gates (per-identity cooldown handled separately)."""
    today = _today_iso()
    if (
        state.get("day_iso") == today
        and int(state.get("fires_today") or 0) >= settings.alert_daily_cap
    ):
        return False, "daily_cap"
    skip = settings.alert_skip_if_chat_active_minutes * 60
    if skip > 0:
        idle = _chat_idle_seconds(settings)
        if idle < skip:
            # Don't self-suppress: an autonomous wake's own append_turn bumps
            # identity_events.jsonl mtime, which would otherwise read as "chat
            # active" for the next 1-2 polls. If the recent write IS our own
            # last wake (within a few seconds), it's not the operator chatting.
            last_wake = float(state.get("last_wake_unix") or 0.0)
            activity_mtime = time.time() - idle
            if abs(activity_mtime - last_wake) > 10.0:
                return False, "chat_active"
    return True, ""


# Governor (Phase 39): these safety/security trips always wake regardless of the
# node's coherence — the Nephilim gate only holds NON-critical kinds.
_CRITICAL_KINDS = frozenset({"military_air", "gps_jamming"})


async def _poll_once(settings: Settings, bus: EventBus) -> None:
    trips = await _evaluate_alerts(settings)
    # Heartbeat: the monitor is otherwise silent on a quiet cycle, so log one
    # line per poll showing it's alive + how many vessels the persistent AIS
    # feed has cached (climbs as ships broadcast). Lets the operator SEE it
    # watching. (Raise alert_poll_interval_seconds if the cadence feels noisy.)
    log.info("alert.poll", trips=len(trips), vessels_cached=len(maritime._vessel_cache))
    if not trips:
        return

    state = _read_state(settings)
    now = time.time()
    cooldown = settings.alert_cooldown_minutes * 60
    identities: dict[str, Any] = state.get("identities", {})
    fresh = [t for t in trips if (now - float(identities.get(t["id"], 0.0))) >= cooldown]

    audit: dict[str, Any] = {
        "ts_unix": now,
        "ts_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trips": [{"id": t["id"], "kind": t["kind"]} for t in trips],
        "fresh": [t["id"] for t in fresh],
        "woke": False,
    }

    # Rail source-level cooldown: trains AS A WHOLE wake at most once per
    # alert_rail_cooldown_minutes, regardless of how many distinct services
    # become due/delayed. Without this a busy board drip-wakes all evening —
    # each train is individually deduped, but the per-identity cooldown can't
    # see "it's ALL trains, again". 0 = off (per-train dedup only).
    rail_cd = settings.alert_rail_cooldown_minutes * 60
    if rail_cd > 0 and any(t["kind"] == "train" for t in fresh):
        last_rail = float(state.get("last_rail_wake_unix") or 0.0)
        if now - last_rail < rail_cd:
            held_rail = [t["id"] for t in fresh if t["kind"] == "train"]
            fresh = [t for t in fresh if t["kind"] != "train"]
            audit["rail_held"] = held_rail
            log.info(
                "alert.rail_cooldown",
                held=held_rail,
                mins_left=round((rail_cd - (now - last_rail)) / 60, 1),
            )

    # Aircraft source-level cooldown — same idea as rail. A civilian sky over a
    # populated area is busy; without this a steady stream of distinct hexes
    # would drip-wake even with per-aircraft dedup. 0 = off.
    air_cd = settings.alert_aircraft_cooldown_minutes * 60
    if air_cd > 0 and any(t["kind"] == "aircraft" for t in fresh):
        last_air = float(state.get("last_aircraft_wake_unix") or 0.0)
        if now - last_air < air_cd:
            held_air = [t["id"] for t in fresh if t["kind"] == "aircraft"]
            fresh = [t for t in fresh if t["kind"] != "aircraft"]
            audit["aircraft_held"] = held_air
            log.info(
                "alert.aircraft_cooldown",
                held=held_air,
                mins_left=round((air_cd - (now - last_air)) / 60, 1),
            )

    # Severe-weather source cooldown — warnings/watch shouldn't re-narrate every
    # poll while a multi-hour warning is active. Default 60 min. 0 = off.
    wx_cd = settings.alert_severe_wx_cooldown_minutes * 60
    if wx_cd > 0 and any(t["kind"] == "severe_weather" for t in fresh):
        last_wx = float(state.get("last_severe_weather_wake_unix") or 0.0)
        if now - last_wx < wx_cd:
            held_wx = [t["id"] for t in fresh if t["kind"] == "severe_weather"]
            fresh = [t for t in fresh if t["kind"] != "severe_weather"]
            audit["severe_wx_held"] = held_wx
            log.info(
                "alert.severe_wx_cooldown",
                held=held_wx,
                mins_left=round((wx_cd - (now - last_wx)) / 60, 1),
            )

    if not fresh:
        audit["skipped_reason"] = "all_on_cooldown"
        _append_log(settings, audit)
        return

    eligible, reason = _global_eligible(settings, state)
    if not eligible:
        audit["skipped_reason"] = reason
        _append_log(settings, audit)
        log.info("alert.skipped", reason=reason, fresh=[t["id"] for t in fresh])
        return

    # Force dry-run when autonomy is off OR the operator location is unset
    # (0,0 = null-island sentinel; waking against it would describe a wrong-
    # region world). Dry-run logs what WOULD wake and does NOT consume the dedup
    # (so trips still fire once the gate clears).
    coords_unset = settings.operator_lat == 0.0 and settings.operator_lon == 0.0
    if not settings.autonomy_enabled or coords_unset:
        reason = "no_operator_location" if coords_unset else "autonomy_disabled_dry_run"
        audit["skipped_reason"] = reason
        _append_log(settings, audit)
        log.info("alert.dry_run", reason=reason, would_wake=[t["id"] for t in fresh])
        return

    # Phase 39 — Nephilim/SILR governor. Hold NON-critical wakes when the node's
    # own coherence is below the floor; critical safety trips always pass. Off by
    # default; when on but not enforcing, only LOGS what it would hold (mirroring
    # the autonomy dry-run above). A held wake does NOT consume the dedup, so it
    # fires once coherence recovers.
    if settings.nephilim_wake_gate_enabled:
        from .. import nephilim
        from ..autonomy import get_autonomous_session

        coherence = (get_autonomous_session().last_nephilim or {}).get("coherence")
        critical = any(t["kind"] in _CRITICAL_KINDS for t in fresh)
        allow, gate_reason = nephilim.wake_gate(
            coherence, floor=settings.nephilim_coherence_floor, critical=critical
        )
        if not allow:
            audit["governor"] = {
                "coherence": coherence,
                "reason": gate_reason,
                "enforced": settings.nephilim_wake_gate_enforce,
            }
            if settings.nephilim_wake_gate_enforce:
                audit["skipped_reason"] = gate_reason
                _append_log(settings, audit)
                log.info(
                    "alert.governor_suppressed",
                    reason=gate_reason, coherence=coherence,
                    held=[t["id"] for t in fresh],
                )
                return
            log.info(
                "alert.governor_dry_run",
                reason=gate_reason, coherence=coherence,
                would_hold=[t["id"] for t in fresh],
            )

    # Phase 39 — PQI (Prime-Qualified Intent). Hold NON-critical wakes unless the
    # URE-VM clock is on a Pendinium prime (p≡1 mod 12 — the prime-spiral set);
    # critical trips bypass. Off by default; dry-run logs; a held wake does NOT
    # consume the dedup, so it fires on a later prime tick.
    if settings.pqi_wake_gate_enabled:
        from ..urevm import PENDINIUM_PRIMES, get_vm

        cp = int(get_vm().cycle_position)
        w = settings.pqi_window
        pqi_critical = any(t["kind"] in _CRITICAL_KINDS for t in fresh)
        on_prime = any(abs(cp - p) <= w for p in PENDINIUM_PRIMES)
        if not pqi_critical and not on_prime:
            audit["pqi"] = {"cycle_position": cp, "window": w, "enforced": settings.pqi_wake_gate_enforce}
            if settings.pqi_wake_gate_enforce:
                audit["skipped_reason"] = "pqi_not_prime_tick"
                _append_log(settings, audit)
                log.info("alert.pqi_held", cycle_position=cp, held=[t["id"] for t in fresh])
                return
            log.info("alert.pqi_dry_run", cycle_position=cp, would_hold=[t["id"] for t in fresh])

    # Bundle all fresh trips into ONE wake.
    kinds = sorted({t["kind"] for t in fresh})
    summary = (
        f"{len(fresh)} alert{'s' if len(fresh) != 1 else ''} near you: "
        + "; ".join(t["description"] for t in fresh[:5])
    )
    trigger = {
        "kinds": kinds,
        "summary": summary,
        "events": [
            {"kind": t["kind"], "description": t["description"], "data": t["data"]}
            for t in fresh
        ],
    }

    from ..autonomy import trigger_autonomous_turn
    try:
        spoken = await trigger_autonomous_turn(trigger, bus)
        # trigger_autonomous_turn swallows turn errors internally (a wake
        # failure must never crash the worker) and returns "" — e.g. when
        # LM Studio is unreachable mid-reload. An empty wake means the
        # operator NEVER HEARD the alert: do NOT consume the dedup or the
        # daily cap, so the same trip retries on the next poll instead of
        # being silently eaten by its own cooldown.
        if not spoken.strip():
            audit["skipped_reason"] = "wake_turn_failed_will_retry"
            _append_log(settings, audit)
            log.warning(
                "alert.wake_failed_will_retry",
                kinds=sorted({t["kind"] for t in fresh}),
                trips=[t["id"] for t in fresh],
            )
            return
        audit["woke"] = True
    except Exception as e:  # noqa: BLE001
        audit["wake_error"] = str(e)
        _append_log(settings, audit)
        log.warning("alert.wake_failed", error=str(e))
        return

    # Consume dedup + daily cap; prune identities older than a day to bound size.
    today = _today_iso()
    for t in fresh:
        identities[t["id"]] = now
    identities = {k: v for k, v in identities.items() if now - float(v) < 86400.0}
    state["identities"] = identities
    state["fires_today"] = (
        int(state.get("fires_today") or 0) + 1 if state.get("day_iso") == today else 1
    )
    state["day_iso"] = today
    # Stamp the wake time (post-wake) so the chat-active gate can tell our own
    # append_turn write apart from a real operator turn on the next poll.
    state["last_wake_unix"] = time.time()
    # Stamp the last rail wake so the source-level rail cooldown can throttle the
    # whole train source (not just per-service) over the following polls.
    if any(t["kind"] == "train" for t in fresh):
        state["last_rail_wake_unix"] = now
    if any(t["kind"] == "aircraft" for t in fresh):
        state["last_aircraft_wake_unix"] = now
    if any(t["kind"] == "severe_weather" for t in fresh):
        state["last_severe_weather_wake_unix"] = now
    _write_state(settings, state)
    _append_log(settings, audit)
    log.info("alert.woke", kinds=kinds, n=len(fresh))


async def alert_monitor_loop(bus: EventBus) -> None:
    """Background loop. Starts the persistent AIS monitor (ships), then polls all
    sources every `alert_poll_interval_seconds`, waking on fresh trips. Self-
    cancellable. No-op when alert_monitor_enabled is False."""
    settings = get_settings()
    if not settings.alert_monitor_enabled:
        log.info("alert.monitor.disabled")
        return

    interval = max(30, settings.alert_poll_interval_seconds)
    if settings.operator_lat == 0.0 and settings.operator_lon == 0.0:
        log.warning(
            "alert.monitor.no_operator_location",
            note="operator_lat/lon unset (0,0) — geo alerts would evaluate null island; "
            "set LUMOS_OPERATOR_LAT/LON. Running dry-run only until configured.",
        )
    log.info("alert.monitor.started", interval_s=interval, autonomy=settings.autonomy_enabled)

    # Persistent AIS cache child task (ships). Best-effort; no key → no-op.
    ais_task = asyncio.create_task(
        maritime.ais_monitor_loop(
            settings.operator_lat, settings.operator_lon, settings.alert_ship_radius_km
        )
    )
    try:
        while True:
            try:
                await asyncio.sleep(interval)
                await _poll_once(settings, bus)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("alert.monitor.iter_failed", error=str(e))
    finally:
        ais_task.cancel()
        try:
            await ais_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
