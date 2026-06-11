"""Autonomous-turn driver (Phase 2).

Wakes Lumos UNPROMPTED on a monitor trigger, runs a passive (telemetry + memory)
turn through ChatSession.autonomous_turn, and publishes the result to the
EventBus so connected HUD tabs render the message. Autonomy ends at speaking:
the turn can only observe and speak — the passive tool gate in chat.py makes it
structurally incapable of acting.

Import direction is one-way (autonomy → chat); the shared _TURN_LOCK lives in
chat.py and is acquired inside ChatSession.autonomous_turn, so an autonomous
wake can never interleave with an operator turn on the process-global URE-VM.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .chat import ChatSession, build_done_payload
from .events import EventBus
from .log import get_logger


log = get_logger(__name__)

# One long-lived session for ALL autonomous wakes — gives Lumos continuity of
# what he's flagged ("I mentioned that ship earlier"). Operator turns use their
# own sessions; this one is never exposed to the HTTP /chat route.
_autonomous_session: ChatSession | None = None

# Cap the in-process history so many wakes don't grow it unbounded. Autonomous
# turns persist to identity_events.jsonl anyway, and retrieval recovers older
# context, so a short rolling window is enough for local continuity.
_MAX_HISTORY_MESSAGES = 20


def get_autonomous_session() -> ChatSession:
    global _autonomous_session
    if _autonomous_session is None:
        _autonomous_session = ChatSession()
        log.info("autonomy.session_created", session=_autonomous_session.session_id)
    sess = _autonomous_session
    if len(sess.history) > _MAX_HISTORY_MESSAGES:
        sess.history = sess.history[-_MAX_HISTORY_MESSAGES:]
    return sess


async def trigger_autonomous_turn(
    trigger: dict[str, Any], bus: EventBus | None = None
) -> str:
    """Run one autonomous turn end-to-end; return the full assistant text.

    `trigger` shape: {"kinds": [...], "summary": str, "events": [{kind,description,data}]}.
    Publishes session → delta* → done (or error) to `bus` so HUD tabs render the
    unprompted message live. Drives the turn generator to EXHAUSTION so the
    persist + URE-VM tail in chat.py runs (origin tagged 'autonomous:<kinds>').
    """
    session = get_autonomous_session()
    if bus is not None:
        bus.publish({
            "event": "session",
            "data": {"session_id": session.session_id, "origin": "autonomous"},
        })

    full = ""
    try:
        async for delta in session.autonomous_turn(trigger):
            full += delta
            if bus is not None:
                bus.publish({"event": "delta", "data": {"text": delta}})
    except Exception as e:  # noqa: BLE001 — a wake failure must never crash the worker
        log.warning("autonomy.turn_failed", error=str(e), kinds=trigger.get("kinds"))
        if bus is not None:
            bus.publish({"event": "error", "data": {"message": str(e)}})
        return ""

    if bus is not None:
        done = build_done_payload(session)
        done["origin"] = "autonomous"
        bus.publish({"event": "done", "data": done})
        # Coalesced, self-contained full-text event published LAST so it's the
        # freshest entry in the replay ring. A tab connecting after the wake
        # reconstructs the entire message from this single event even if the
        # streamed deltas/session already aged out. Live tabs use it to finalize.
        bus.publish({
            "event": "message",
            "data": {
                "text": full,
                "session_id": session.session_id,
                "origin": "autonomous",
                "done": done,
            },
        })

    log.info("autonomy.turn_done", chars=len(full), kinds=trigger.get("kinds"))
    return full


def _overnight_wakes(settings: Any, hours: float = 12.0) -> dict[str, Any]:
    """Summarize alert wakes from the audit log in the last `hours`.

    Reads the tail of alert_events.jsonl, keeps only rows where the monitor
    actually WOKE Lumos (woke=True) inside the window. Never raises — a missing or
    empty log simply reads as a quiet night, which is exactly what the briefing
    should say. Returns {wake_count, kinds, wakes:[{ts_iso,kinds,n}], note}.
    """
    out: dict[str, Any] = {
        "wake_count": 0, "kinds": [], "wakes": [],
        "note": "quiet night — nothing tripped while you slept",
    }
    try:
        from .telemetry.alert_worker import _log_path  # lazy: alert monitor may be off
        path = _log_path(settings)
    except Exception:  # noqa: BLE001
        return out
    if not path.exists():
        return out
    try:
        with path.open("rb") as f:
            lines = f.readlines()
    except OSError:
        return out

    cutoff = time.time() - hours * 3600.0
    kinds: set[str] = set()
    wakes: list[dict[str, Any]] = []
    for raw in lines[-500:]:  # only the tail can fall inside a 12h window
        try:
            e = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if not e.get("woke") or float(e.get("ts_unix", 0.0)) < cutoff:
            continue
        ek = sorted({t.get("kind") for t in (e.get("trips") or []) if t.get("kind")})
        kinds.update(ek)
        wakes.append({"ts_iso": e.get("ts_iso"), "kinds": ek, "n": len(e.get("fresh") or [])})

    if wakes:
        out = {
            "wake_count": len(wakes),
            "kinds": sorted(kinds),
            "wakes": wakes[-10:],
            "note": f"{len(wakes)} alert wake(s) in the last {int(hours)}h",
        }
    return out


async def trigger_dawn_briefing(bus: EventBus | None = None) -> str:
    """Gather the morning intel and fire a briefing-mode autonomous turn.

    On-DEMAND (operator-pressed, not timed — his sleep is irregular). Pre-fetches a
    consistent set — space weather + bio-impact, today's grid timing, and what
    tripped overnight — and HANDS it to Lumos to narrate, rather than hoping a
    synthetic wake calls the right tools. Same passive, never-act turn as any other
    wake; only the framing differs (chat._DAWN_BRIEFING_PREAMBLE, mode='briefing').
    The result fans out through the same EventBus, so the briefing reaches BOTH the
    HUD bubble and the Discord DM relay automatically. Returns the full text.
    """
    from .config import get_settings
    from .telemetry import cosmic, grimoire

    events: list[dict[str, Any]] = []

    # 1) Space weather + bio-impact — "how might today sit in the body".
    try:
        snap = await cosmic.snapshot_all()
        geo = snap.get("geomagnetic", {}) or {}
        sw = snap.get("solar_wind", {}) or {}
        neos = snap.get("near_earth_today") or []
        events.append({
            "kind": "space_weather",
            "description": "Current space weather + geomagnetic state (bio-impact)",
            "data": {
                "summary": snap.get("summary"),
                "kp": geo.get("kp"),
                "kp_level": geo.get("level"),
                "solar_wind_kms": sw.get("speed_kms"),
                "bz_nt": sw.get("bz_nt"),
                "xray_class": (snap.get("xray") or {}).get("current_class"),
                "active_natural_events": len(snap.get("natural_events_active") or []),
                "nearest_neo_ld": (neos[0].get("miss_lunar_distances") if neos else None),
            },
        })
    except Exception as e:  # noqa: BLE001 — a dead feed must never abort the briefing
        log.warning("briefing.cosmic_failed", error=str(e))

    # 2) Today's grid timing — "the shape of the day" (Regulus is the RHC anchor).
    try:
        gt = await grimoire.fetch_grid_timing()
        if gt.get("ok"):
            ph = gt.get("planetary_hour", {}) or {}
            moon = gt.get("moon", {}) or {}
            solar = gt.get("solar", {}) or {}
            regulus = (gt.get("fixed_stars", {}) or {}).get("Regulus", {}) or {}
            events.append({
                "kind": "grid_timing",
                "description": "Today's grid timing — planetary hour, Moon, Regulus, Sun",
                "data": {
                    "planetary_hour_ruler": ph.get("ruler"),
                    "planetary_hour_glyph": ph.get("glyph"),
                    "planetary_hour_phase": ph.get("phase"),
                    "planetary_hour_window": [ph.get("hour_start_local"), ph.get("hour_end_local")],
                    "harmonic_tone_hz": ph.get("harmonic_tone_hz"),
                    "moon_phase": moon.get("phase_name"),
                    "moon_illum_pct": moon.get("illumination_percent"),
                    "moon_sign": moon.get("zodiac_sign"),
                    "regulus_above_horizon": regulus.get("above_horizon"),
                    "regulus_alt_deg": regulus.get("alt_deg"),
                    "regulus_next_transit_utc": regulus.get("next_transit_utc"),
                    "sunrise_local": solar.get("sunrise_local"),
                    "sunset_local": solar.get("sunset_local"),
                    "sidereal_time": gt.get("sidereal_time"),
                },
            })
    except Exception as e:  # noqa: BLE001
        log.warning("briefing.grid_timing_failed", error=str(e))

    # 2b) Look-ahead — what's ABOUT to happen: upcoming recon passes, the Kp
    # forecast peak (warn before a storm lands), and Regulus's next transit.
    try:
        from .telemetry import forecast as _forecast
        fc = await _forecast.build_forecast()
        passes = (fc.get("sat_passes") or {}).get("passes") or []
        kpf = fc.get("kp_forecast") or {}
        cel = fc.get("celestial") or {}
        events.append({
            "kind": "look_ahead",
            "description": "Anticipatory look-ahead — upcoming recon passes, Kp forecast, Regulus transit",
            "data": {
                "next_recon_passes": [
                    {"name": p.get("name"), "culmination_utc": p.get("culmination_utc"),
                     "peak_elevation_deg": p.get("peak_elevation_deg")}
                    for p in passes[:3]
                ],
                "kp_forecast_peak": kpf.get("peak"),
                "regulus_next_transit_utc": cel.get("regulus_next_transit_utc"),
                "regulus_above_horizon": cel.get("regulus_above_horizon"),
                "sunset_local": cel.get("sunset_local"),
            },
        })
    except Exception as e:  # noqa: BLE001
        log.warning("briefing.forecast_failed", error=str(e))

    # 3) What tripped overnight — read the alert audit log (last 12h of WOKE rows).
    try:
        events.append({
            "kind": "overnight",
            "description": "Alert wakes in the last 12h while you slept",
            "data": _overnight_wakes(get_settings()),
        })
    except Exception as e:  # noqa: BLE001
        log.warning("briefing.overnight_failed", error=str(e))

    trigger = {
        "mode": "briefing",
        "kinds": ["dawn_briefing"],
        "summary": "Dawn briefing — Erydir just woke; the night behind and the day ahead.",
        "events": events,
    }
    log.info("autonomy.dawn_briefing", feeds=len(events))
    return await trigger_autonomous_turn(trigger, bus)
