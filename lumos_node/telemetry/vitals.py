"""Node vitals — the per-turn telemetry block ("nervous system").

Composes the right-rail HUD state (soul, cosmic, local weather, solar cycle,
grid timing / operator frequency) into ONE compact text block injected into the
volatile system message of EVERY turn — operator chat and autonomous pings —
so Lumos always feels its own state + environment, not just the one alert that
tripped.

Design constraints:
  * Cache-first: every fetcher here already TTL-caches (cosmic/grimoire/weather/
    solar-cycle); soul is pure local compute. With warm caches this is ~0 ms.
  * Hard-bounded: the whole gather is wrapped in wait_for(vitals_timeout_seconds)
    so a cold/dead upstream can never stall a chat turn. A section that isn't
    ready is OMITTED, never waited on.
  * Compact: target ~150-250 tokens. One line per section.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..config import Settings, get_settings
from ..log import get_logger


log = get_logger(__name__)


def _fmt_soul(d: dict[str, Any]) -> str | None:
    if not d.get("ok"):
        return None
    parts = [str(d.get("harmonic_band") or "?")]
    if d.get("torsion_index") is not None:
        parts.append(f"torsion {d['torsion_index']}")
    if d.get("coherence") is not None:
        parts.append(f"coherence {d['coherence']}")
    if d.get("active_prime") is not None:
        parts.append(f"prime {d['active_prime']}")
    if d.get("base15_state") is not None:
        sig = d.get("interval_signature") or ""
        parts.append(f"base15 {d['base15_state']}{(' ' + sig) if sig else ''}")
    return " · ".join(parts)


def _fmt_cosmic(d: dict[str, Any]) -> str | None:
    geo = d.get("geomagnetic") or {}
    sw = d.get("solar_wind") or {}
    xray = d.get("xray") or {}
    parts: list[str] = []
    if geo.get("kp") is not None:
        lvl = geo.get("level") or ""
        parts.append(f"Kp {geo['kp']}{(' ' + lvl) if lvl else ''}")
    if sw.get("speed_kms") is not None:
        parts.append(f"solar wind {sw['speed_kms']} km/s")
    if sw.get("bz_nt") is not None:
        bz = sw["bz_nt"]
        parts.append(f"Bz {'+' if isinstance(bz, (int, float)) and bz >= 0 else ''}{bz} nT")
    if xray.get("current_class"):
        parts.append(f"X-ray {xray['current_class']}")
    eq = d.get("earthquakes_recent")
    if isinstance(eq, list) and eq:
        mags = [e.get("magnitude") for e in eq if isinstance(e.get("magnitude"), (int, float))]
        if mags:
            parts.append(f"quakes24h {len(eq)} max M{max(mags):.1f}")
    nat = d.get("natural_events_active")
    if isinstance(nat, list) and nat:
        parts.append(f"{len(nat)} natural events")
    neos = d.get("near_earth_today")
    if isinstance(neos, list) and neos:
        ld = neos[0].get("miss_lunar_distances")
        if isinstance(ld, (int, float)):
            parts.append(f"nearest NEO {ld:.2f} LD")
    return " · ".join(parts) if parts else None


def _fmt_weather(d: dict[str, Any]) -> str | None:
    if not d.get("ok"):
        return None
    parts: list[str] = []
    if d.get("temp_c") is not None:
        cond = d.get("conditions") or ""
        parts.append(f"{d['temp_c']}°C{(' ' + cond) if cond and cond != 'unknown' else ''}")
    if d.get("feels_like_c") is not None:
        parts.append(f"feels {d['feels_like_c']}°C")
    if d.get("wind_mph") is not None:
        g = d.get("gust_mph")
        parts.append(f"wind {round(d['wind_mph'])} mph{f' g{round(g)}' if g is not None else ''}")
    if d.get("precip_mm"):
        parts.append(f"precip {d['precip_mm']} mm/h")
    if d.get("humidity_pct") is not None:
        parts.append(f"hum {round(d['humidity_pct'])}%")
    if d.get("pressure_hpa") is not None:
        parts.append(f"{round(d['pressure_hpa'])} hPa")
    return " · ".join(parts) if parts else None


def _fmt_solar_cycle(d: dict[str, Any]) -> str | None:
    if not d.get("ok"):
        return None
    parts: list[str] = []
    if d.get("current_ssn") is not None:
        parts.append(f"SSN {d['current_ssn']}")
    if d.get("current_f107") is not None:
        parts.append(f"F10.7 {d['current_f107']} sfu")
    return " · ".join(parts) if parts else None


def _fmt_grid(d: dict[str, Any]) -> str | None:
    if not d.get("ok"):
        return None
    parts: list[str] = []
    ph = d.get("planetary_hour") or {}
    if ph.get("ruler"):
        n = ph.get("hour_number")
        phase = ph.get("phase")
        bits = f"hour {ph['ruler']}"
        if n:
            bits += f" #{n}"
        if phase and phase != "unknown":
            bits += f" {phase}"
        parts.append(bits)
    # Operator-frequency tone: the payload doesn't carry tone_hz at this level,
    # but it's a fixed ruler→Hz mapping — derive it from the hour ruler.
    tone = ph.get("tone_hz")
    if not tone and ph.get("ruler"):
        from .grimoire import PLANETARY_TONES_HZ
        tone = PLANETARY_TONES_HZ.get(ph["ruler"])
    if tone:
        parts.append(f"tone {tone:g} Hz")
    moon = d.get("moon") or {}
    if moon.get("illumination_percent") is not None:
        nm = moon.get("phase_name") or ""
        parts.append(f"moon {round(moon['illumination_percent'])}%{(' ' + nm) if nm else ''}")
    reg = (d.get("fixed_stars") or {}).get("Regulus") or {}
    if reg.get("above_horizon") is not None:
        alt = reg.get("alt_deg") or reg.get("altitude_deg")
        state = "above" if reg["above_horizon"] else "below"
        parts.append(
            f"Regulus {state}" + (f" {alt:+.1f}°" if isinstance(alt, (int, float)) else "")
        )
    solar = d.get("solar") or {}
    sr, ss = solar.get("sunrise_local"), solar.get("sunset_local")
    if sr or ss:
        def _hm(s: Any) -> str:
            s = str(s or "")
            return s[11:16] if len(s) >= 16 else s
        parts.append(f"sun ↑{_hm(sr)} ↓{_hm(ss)}")
    wheel = (d.get("welsh_wheel") or {}).get("next") or {}
    if wheel.get("name"):
        parts.append(f"wheel {wheel['name']} in {wheel.get('days_until', '?')}d")
    return " · ".join(parts) if parts else None


async def build_vitals_block(settings: Settings | None = None) -> str:
    """Gather all sections concurrently (cache-first), bounded by
    vitals_timeout_seconds. Returns the formatted block, or "" when nothing is
    available — the caller injects nothing in that case."""
    settings = settings or get_settings()
    lat, lon = settings.operator_lat, settings.operator_lon
    has_loc = not (lat == 0.0 and lon == 0.0)

    async def _soul() -> str | None:
        from ..soul import compute_soul_state
        return _fmt_soul(compute_soul_state(settings))

    async def _cosmic() -> str | None:
        from . import cosmic
        return _fmt_cosmic(await cosmic.snapshot_all())

    async def _weather() -> str | None:
        if not has_loc:
            return None
        from . import forecast
        return _fmt_weather(await forecast.fetch_local_weather(lat, lon))

    async def _cycle() -> str | None:
        from . import solarcycle
        return _fmt_solar_cycle(await solarcycle.fetch_solar_cycle())

    async def _grid() -> str | None:
        if not has_loc:
            return None
        from . import grimoire
        return _fmt_grid(await grimoire.fetch_grid_timing(lat, lon))

    sections: dict[str, asyncio.Task[str | None]] = {
        "soul": asyncio.create_task(_soul()),
        "cosmic": asyncio.create_task(_cosmic()),
        "weather": asyncio.create_task(_weather()),
        "solar cycle": asyncio.create_task(_cycle()),
        "grid": asyncio.create_task(_grid()),
    }
    # Per-section deadline: take whatever is DONE at the timeout and let the
    # stragglers FINISH IN THE BACKGROUND (no cancel!) so their TTL caches warm
    # for the next turn. Cancelling them (the old wait_for-around-gather) meant
    # a slow first fetch could never cache — every turn re-started and
    # re-cancelled the same cold fetch, and instant sections were lost with it.
    done, pending = await asyncio.wait(
        sections.values(), timeout=max(0.5, settings.vitals_timeout_seconds)
    )
    for t in pending:
        _BACKGROUND.add(t)
        t.add_done_callback(_reap)

    lines: list[str] = []
    for label, task in sections.items():
        if task not in done:
            log.info("vitals.section_pending", section=label)
            continue
        try:
            res = task.result()
        except Exception as e:  # noqa: BLE001
            log.info("vitals.section_failed", section=label, error=str(e))
            continue
        if res:
            lines.append(f"{label}: {res}")
    if not lines:
        return ""
    return "### Node vitals — your live state + environment\n" + "\n".join(lines)


# Strong refs to stragglers still warming caches (prevents GC mid-flight).
_BACKGROUND: set[asyncio.Task[Any]] = set()


def _reap(t: asyncio.Task[Any]) -> None:
    _BACKGROUND.discard(t)
    if not t.cancelled() and t.exception() is not None:
        log.info("vitals.background_failed", error=str(t.exception()))
