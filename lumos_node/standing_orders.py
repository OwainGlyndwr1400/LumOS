"""Standing Orders (Phase 45.5) — bounded, operator-signed proactive agency.

The one deliberate exception to "autonomy ends at speaking" — and it's safe
precisely because it is NOT the model deciding to act. A Standing Order is a
recipe the OPERATOR wrote and cryptographically signed; when its trigger fires,
plain deterministic code runs the recipe. Lumos-the-model still only observes
and speaks. This is a signed cron rule, not an autonomous decision.

Two independent locks, both required:
  1. AUTHENTICITY — the playbook carries an HMAC-SHA256 signature over its
     orders, keyed by LUMOS_STANDING_ORDERS_TOKEN. Edit the file without
     re-signing (`lumos orders sign`) and the whole playbook is refused. A
     dropped-in or tampered file cannot inject orders.
  2. CAPABILITY — a recipe may only use actions from a fixed whitelist
     (dossier / note / briefing / discord_dm / workspace_file). Any other
     action string is ignored, even in a validly-signed file. Deny-by-default.

Plus the usual gates: OFF unless LUMOS_STANDING_ORDERS_ENABLED, per-action daily
caps, and workspace writes validated against LUMOS_GIT_WORKSPACES (no commits,
no push — writing a file is the ceiling).

Wiring: fired from the alert worker's PURE-CODE path AFTER a wake has spoken —
the same doctrine-safe seam the enrichment queue uses. The wake speaks; then the
operator's signed orders run.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import time
from pathlib import Path
from typing import Any

import httpx
import orjson

from .config import Settings, get_settings
from .log import get_logger

log = get_logger(__name__)

# The complete action whitelist. Nothing outside this set can ever run, however
# the playbook is signed. Each maps to a handler in _ACTION_HANDLERS below.
ALLOWED_ACTIONS = frozenset(
    {"dossier", "note", "briefing", "discord_dm", "workspace_file", "rhc_snapshot"}
)

_NOTES_SUBDIR = "standing_orders"
_STATE_FILE = "standing_orders_state.json"
_DISCORD_MSG_MAX = 1900


# ── Signature (authenticity lock) ────────────────────────────────────────────

def _canonical(orders: list[dict[str, Any]]) -> bytes:
    """Stable byte serialization of the orders for HMAC (sorted keys)."""
    return orjson.dumps(orders, option=orjson.OPT_SORT_KEYS)


def compute_signature(orders: list[dict[str, Any]], token: str) -> str:
    """HMAC-SHA256 hex of the canonical orders, keyed by the operator token."""
    return hmac.new(token.encode("utf-8"), _canonical(orders), hashlib.sha256).hexdigest()


def verify_playbook(playbook: dict[str, Any], token: str) -> tuple[bool, str]:
    """PURE — is this playbook authentically signed by the token holder?
    Returns (ok, reason). Uses constant-time comparison."""
    if not token:
        return False, "no LUMOS_STANDING_ORDERS_TOKEN set — refusing to run unsigned orders"
    if not isinstance(playbook, dict):
        return False, "playbook is not an object"
    orders = playbook.get("orders")
    sig = playbook.get("signature")
    if not isinstance(orders, list):
        return False, "playbook has no 'orders' list"
    if not isinstance(sig, str) or not sig:
        return False, "playbook is unsigned (run `lumos orders sign`)"
    expected = compute_signature(orders, token)
    if not hmac.compare_digest(expected, sig):
        return False, "signature mismatch — playbook was edited without re-signing, or wrong token"
    return True, "ok"


def _playbook_path(s: Settings) -> Path:
    raw = s.standing_orders_path.strip()
    if raw:
        return Path(raw).expanduser()
    from .telemetry.worker import _data_dir
    return _data_dir(s) / "standing_orders.json"


def load_orders(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Load + verify the playbook. Returns the orders list, or [] if the file is
    absent, malformed, or fails the signature check (logged, never raised)."""
    settings = settings or get_settings()
    p = _playbook_path(settings)
    if not p.exists():
        return []
    try:
        playbook = orjson.loads(p.read_bytes())
    except (orjson.JSONDecodeError, OSError) as e:
        log.warning("orders.load_failed", error=str(e))
        return []
    ok, reason = verify_playbook(playbook, settings.standing_orders_token)
    if not ok:
        log.warning("orders.rejected", reason=reason)
        return []
    return [o for o in playbook["orders"] if isinstance(o, dict)]


# ── Trigger matching (pure) ──────────────────────────────────────────────────

def order_matches(order: dict[str, Any], trip: dict[str, Any]) -> bool:
    """PURE — does this order's `when` match the wake trip?

    `when` must name a `kind`; any other keys are matched (case-insensitive
    substring) against the trip's `data`. Missing/blank `when` never matches
    (fail-closed — a malformed order does nothing rather than firing on all)."""
    when = order.get("when")
    if not isinstance(when, dict) or not when.get("kind"):
        return False
    if trip.get("kind") != when["kind"]:
        return False
    data = trip.get("data") or {}
    for key, want in when.items():
        if key == "kind":
            continue
        have = data.get(key)
        if have is None:
            return False
        if str(want).lower() not in str(have).lower():
            return False
    return True


# ── Action handlers (the capability whitelist) ───────────────────────────────

async def _act_dossier(s: Settings, trip: dict[str, Any], step: dict[str, Any]) -> str:
    """Compile a markdown dossier: the triggering event + nearby world-state
    (satellites overhead, fire hotspots, active GDACS disasters). Each feed is
    best-effort so one dead source never sinks the dossier. Returns markdown."""
    lat, lon = s.operator_lat, s.operator_lon
    radius = float(step.get("radius_km", 500))
    lines = [
        f"# Dossier — {trip.get('kind', 'event')}",
        f"_compiled {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} · "
        f"trigger: {trip.get('description', trip.get('kind'))}_",
        "",
        "## Triggering event",
        "```",
        orjson.dumps(trip.get("data") or {}, option=orjson.OPT_INDENT_2 | 
        orjson.OPT_SERIALIZE_NUMPY).decode(),
        "```",
    ]
    # The four world-state feeds are independent I/O — fetch them CONCURRENTLY
    # (wall-clock = slowest feed, not the sum) and log any that fail BY NAME.
    # They used to run serially under contextlib.suppress(Exception), which made
    # an incomplete dossier indistinguishable from a quiet sky: "satellite feed
    # down" and "no satellites overhead" rendered identically. Still best-effort
    # — a failed feed drops its section and never sinks the dossier.
    from .telemetry import cosmic, fires, gdacs, satellites

    async def _feed(name: str, coro: Any) -> Any:
        try:
            return await coro
        except Exception as e:  # noqa: BLE001 — best-effort feed; log + drop its section
            log.warning(
                "orders.dossier_feed_failed", feed=name, error=f"{type(e).__name__}: {e}"
            )
            return None

    snap, sats, f, g = await asyncio.gather(
        _feed("cosmic", cosmic.snapshot_all()),
        _feed(
            "satellites",
            satellites.fetch_satellites_overhead(lat, lon, min_elevation=30, limit=8),
        ),
        _feed("fires", fires.fetch_fires_nearby(lat, lon, radius_km=radius, limit=10)),
        _feed("gdacs", gdacs.fetch_gdacs_events()),
    )

    # Formatting below stays suppress-guarded per section (a markdown builder
    # must never raise); FETCH failures were already logged by name above.
    # Space weather + recent seismic leads — the context that matters most for
    # flare/quake orders.
    with contextlib.suppress(Exception):
        if snap is not None:
            geo = snap.get("geomagnetic") or {}
            sw = snap.get("solar_wind") or {}
            xr = snap.get("xray") or {}
            cx: list[str] = []
            if geo.get("kp") is not None:
                cx.append(f"Kp {geo['kp']}")
            if sw.get("speed_kms") is not None:
                cx.append(f"solar wind {sw['speed_kms']} km/s")
            if sw.get("bz_nt") is not None:
                cx.append(f"Bz {sw['bz_nt']} nT")
            if xr.get("current_class"):
                cx.append(f"X-ray {xr['current_class']}")
            eqs = snap.get("earthquakes_recent") or []
            if cx or eqs:
                lines += ["", "## Space weather & recent seismic"]
                if cx:
                    lines.append("- " + " · ".join(cx))
                for e in eqs[:6]:
                    lines.append(f"- M{e.get('magnitude', '?')} {e.get('place', '')}".rstrip())
    with contextlib.suppress(Exception):
        if sats and sats.get("ok") and sats.get("satellites"):
            lines += ["", "## Satellites overhead (≥30°)"]
            lines += [f"- {satellites.describe_satellite(x)} — {x['elevation_deg']:.0f}°" for x in sats["satellites"]]
    with contextlib.suppress(Exception):
        if f and f.get("ok") and f.get("fires"):
            lines += ["", f"## Fire hotspots within {radius:.0f} km"]
            lines += [f"- ~{x['distance_km']:.0f} km {x['bearing']} ({x['confidence']}, {x.get('instrument','')})" for x in f["fires"]]
    with contextlib.suppress(Exception):
        if g and g.get("ok") and g.get("events"):
            top = [e for e in g["events"] if e.get("level_rank", 0) >= 2][:8]
            if top:
                lines += ["", "## Active GDACS disasters (orange+)"]
                lines += [f"- {e['level'].upper()} {e['type_label']} — {e.get('country') or '?'}: {e.get('title','')}" for e in top]
    return "\n".join(lines)


def _notes_dir(s: Settings) -> Path:
    from .telemetry.worker import _data_dir
    d = _data_dir(s) / _NOTES_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _act_note(s: Settings, trip: dict[str, Any], step: dict[str, Any], ctx: dict[str, Any]) -> str:
    text = str(step.get("text") or trip.get("description") or trip.get("kind") or "")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{stamp}] ({trip.get('kind')}) {text}\n"
    with contextlib.suppress(OSError), (_notes_dir(s) / "notes.log").open("a", encoding="utf-8") as f:
        f.write(line)
    return line.strip()


async def _act_briefing(s: Settings, trip: dict[str, Any], step: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Append the prepared material to a dated briefing file the operator reads
    (does NOT re-fire the LLM dawn briefing — just stages content for it)."""
    body = ctx.get("dossier") or str(step.get("text") or trip.get("description") or "")
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with contextlib.suppress(OSError), (_notes_dir(s) / f"briefing-{day}.md").open("a", encoding="utf-8") as f:
        f.write(f"\n---\n{body}\n")
    return f"staged into briefing-{day}.md"


async def _act_discord_dm(s: Settings, trip: dict[str, Any], step: dict[str, Any], ctx: dict[str, Any]) -> str:
    token, op = s.discord_token, s.discord_operator_id
    if not token or not op:
        return "discord skipped (no token / operator id)"
    prefix = str(step.get("prefix") or "").strip()
    body = ctx.get("dossier") or str(trip.get("description") or trip.get("kind"))
    text = f"{prefix}\n{body}" if prefix else body
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://discord.com/api/v10/users/@me/channels",
                headers={"Authorization": f"Bot {token}"}, json={"recipient_id": str(op)},
            )
            r.raise_for_status()
            ch = r.json()["id"]
            for i in range(0, len(text), _DISCORD_MSG_MAX):
                m = await client.post(
                    f"https://discord.com/api/v10/channels/{ch}/messages",
                    headers={"Authorization": f"Bot {token}"}, json={"content": text[i:i + _DISCORD_MSG_MAX]},
                )
                m.raise_for_status()
        return "DM sent to operator"
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("orders.discord_failed", error=str(e))
        return f"discord failed: {e}"


def _safe_workspace_target(s: Settings, workspace: str, rel: str) -> Path | None:
    """Validate `workspace` is an approved git workspace and `rel` stays inside
    it (no traversal). Returns the resolved target path, or None if unsafe."""
    from .tools.git_tools import _check_repo_path
    try:
        repo = _check_repo_path(workspace)
    except PermissionError:
        return None
    target = (repo / rel).resolve()
    try:
        target.relative_to(repo)
    except ValueError:
        return None  # path traversal outside the workspace
    return target


async def _act_workspace_file(s: Settings, trip: dict[str, Any], step: dict[str, Any], ctx: dict[str, Any]) -> str:
    workspace = str(step.get("workspace") or "")
    rel = str(step.get("path") or "").replace("{ts}", time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    rel = rel.replace("{kind}", str(trip.get("kind", "event")))
    if not workspace or not rel:
        return "workspace_file skipped (missing workspace/path)"
    target = _safe_workspace_target(s, workspace, rel)
    if target is None:
        return "workspace_file refused (path outside an approved git workspace)"
    body = ctx.get("dossier") or str(step.get("text") or trip.get("description") or "")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return f"wrote {target}"
    except OSError as e:
        return f"workspace_file failed: {e}"


# ── RHC seismic variables (operator's SeismicRisk model — logged for analysis) ──
# SeismicRisk = f(CMEIntensity × AtmosphericPressure × CrustalStress). This
# snapshots every MEASURABLE sub-variable at flare/quake time into a JSONL log so
# the qualitative f(A×B×C) can later be fit to real thresholds against a base
# rate (hits AND misses → an ROC curve). The two sub-variables with no feed
# (Schumann ELF amplitude, geodetic tectonic strain) are logged null + flagged —
# never faked, so it's explicit which sensors would complete the model.

_XRAY_LETTER_WM2 = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def parse_xray_flux(class_str: str | None) -> float | None:
    """GOES flare class → W/m². 'M2.3' → 2.3e-5, 'X1.0' → 1e-4. None if bad."""
    if not class_str:
        return None
    base = _XRAY_LETTER_WM2.get(class_str[0].upper())
    if base is None:
        return None
    try:
        mult = float(class_str[1:]) if len(class_str) > 1 and class_str[1:] else 1.0
    except ValueError:
        mult = 1.0
    return round(base * mult, 12)


def spring_tide_factor(illum_pct: float | None) -> float | None:
    """Lunar tidal proxy: 1.0 at new/full (spring tide) → 0.0 at quarter (neap).
    Proxy only — true tidal stress also needs lunar distance (perigee/apogee)."""
    if illum_pct is None:
        return None
    return round(abs(float(illum_pct) - 50.0) / 50.0, 3)


def _atmospheric_term(f107: Any, ssn: Any, iono_index: Any, schumann: dict | None) -> dict[str, Any]:
    """AtmosphericPressure term. Schumann amplitude filled from a configured
    feed when present + tagged with source/confidence; else null + flagged."""
    base = {"f107": f107, "ssn": ssn, "ionospheric_index": iono_index}
    if schumann and schumann.get("ok") and schumann.get("amplitude") is not None:
        return {
            **base,
            "schumann_amplitude": schumann["amplitude"],
            "schumann_frequency_hz": schumann.get("frequency_hz"),
            "schumann_source": schumann.get("source"),
            "schumann_confidence": schumann.get("confidence"),
        }
    return {
        **base,
        "schumann_amplitude": None,
        "_missing": (
            "schumann_amplitude — "
            + (schumann.get("confidence") if schumann else "no ELF (7.83 Hz) feed configured")
            + "; set LUMOS_SCHUMANN_URL to complete this term"
        ),
    }


def compute_rhc_variables(
    cosmic_snap: dict | None, cycle: dict | None, grid: dict | None,
    schumann: dict | None = None,
) -> dict[str, Any]:
    """PURE — the three RHC SeismicRisk terms from live telemetry payloads.
    Raw sub-components are logged alongside each composite index so the model's
    weights stay re-tunable from the data. Missing feeds → null + a _missing note.
    `schumann` (from telemetry.schumann.fetch_schumann) fills the AtmosphericPressure
    ELF term when configured; None/unreachable keeps it honestly null."""
    sw = (cosmic_snap or {}).get("solar_wind") or {}
    xr = (cosmic_snap or {}).get("xray") or {}
    geo = (cosmic_snap or {}).get("geomagnetic") or {}
    speed, bz = sw.get("speed_kms"), sw.get("bz_nt")
    bz_south = max(0.0, -float(bz)) if isinstance(bz, (int, float)) else None
    flux = parse_xray_flux(xr.get("current_class"))
    cme_index = None
    if isinstance(speed, (int, float)) and flux is not None:
        cme_index = round(float(speed) * flux * 1e5 * (1.0 + (bz_south or 0.0)), 4)

    f107, ssn = (cycle or {}).get("current_f107"), (cycle or {}).get("current_ssn")
    iono_index = (
        round(float(f107) + (float(ssn) if isinstance(ssn, (int, float)) else 0.0), 2)
        if isinstance(f107, (int, float)) else None
    )

    moon = (grid or {}).get("moon") or {}
    illum = moon.get("illumination_percent")
    reg = ((grid or {}).get("fixed_stars") or {}).get("Regulus") or {}
    eqs = (cosmic_snap or {}).get("earthquakes_recent") or []
    mags = [e.get("magnitude") for e in eqs if isinstance(e.get("magnitude"), (int, float))]

    return {
        "cme_intensity": {
            "solar_wind_kms": speed, "bz_nt": bz, "bz_south_coupling": bz_south,
            "xray_class": xr.get("current_class"), "xray_flux_wm2": flux, "index": cme_index,
        },
        "atmospheric_pressure": _atmospheric_term(f107, ssn, iono_index, schumann),
        "crustal_stress": {
            "lunar_illum_pct": illum, "lunar_phase": moon.get("phase_name"),
            "lunar_sign": moon.get("zodiac_sign"), "spring_tide_factor": spring_tide_factor(illum),
            "regulus_alt_deg": reg.get("alt_deg"), "regulus_az_deg": reg.get("az_deg"),
            "recent_quake_count": len(eqs),
            "recent_quake_max_mag": round(max(mags), 1) if mags else None,
            "tectonic_strain": None,
            "_missing": "tectonic_strain — no geodetic/GPS strain feed; regional accumulated strain not measured",
        },
        "kp": geo.get("kp"),
    }


async def _act_rhc_snapshot(s: Settings, trip: dict[str, Any], step: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Log the RHC SeismicRisk inputs at trigger time → rhc_seismic_log.jsonl
    (one flat row per flare/quake, hits AND misses — the analysis dataset)."""
    from .telemetry import cosmic, grimoire, solarcycle

    snap = cycle = grid = schumann = None
    with contextlib.suppress(Exception):
        snap = await cosmic.snapshot_all()
    with contextlib.suppress(Exception):
        cycle = await solarcycle.fetch_solar_cycle()
    with contextlib.suppress(Exception):
        grid = await grimoire.fetch_grid_timing(s.operator_lat, s.operator_lon)
    with contextlib.suppress(Exception):
        from .telemetry import schumann as _schumann
        schumann = await _schumann.fetch_schumann()
    rhc = compute_rhc_variables(snap, cycle, grid, schumann)
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger_kind": trip.get("kind"),
        "trigger": trip.get("description"),
        **rhc,
    }
    ctx["rhc"] = row
    with contextlib.suppress(OSError), (_notes_dir(s) / "rhc_seismic_log.jsonl").open("ab") as f:
        f.write(orjson.dumps(row, option=orjson.OPT_SERIALIZE_NUMPY))
        f.write(b"\n")
    c, cr = rhc["cme_intensity"], rhc["crustal_stress"]
    return (
        f"RHC logged — CME idx {c['index']} (SW {c['solar_wind_kms']}km/s, Bz {c['bz_nt']}, "
        f"{c['xray_class']}) · tide {cr['spring_tide_factor']} · Regulus alt {cr['regulus_alt_deg']}° · "
        f"{cr['recent_quake_count']} recent quakes"
    )


# ── Execution ────────────────────────────────────────────────────────────────

def _read_state(s: Settings) -> dict[str, Any]:
    from .telemetry.worker import _data_dir
    p = _data_dir(s) / _STATE_FILE
    base = {"day_iso": "", "counts": {}}
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
    from .telemetry.worker import _data_dir
    with contextlib.suppress(OSError):
        (_data_dir(s) / _STATE_FILE).write_bytes(orjson.dumps(st))


async def execute_order(s: Settings, order: dict[str, Any], trip: dict[str, Any]) -> list[dict[str, str]]:
    """Run one matched order's recipe. Unknown actions are skipped (whitelist).
    `dossier` output is shared via ctx so later steps (dm/file/briefing) reuse
    it instead of recompiling. Returns a per-step result log."""
    steps = order.get("do")
    if not isinstance(steps, list):
        return []
    ctx: dict[str, Any] = {}
    results: list[dict[str, str]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if action not in ALLOWED_ACTIONS:
            results.append({"action": str(action), "result": "REFUSED (not in whitelist)"})
            log.warning("orders.action_refused", action=str(action), order=order.get("name"))
            continue
        try:
            if action == "dossier":
                ctx["dossier"] = await _act_dossier(s, trip, step)
                out = f"compiled ({len(ctx['dossier'])} chars)"
            else:
                out = await _ACTION_HANDLERS[action](s, trip, step, ctx)
        except Exception as e:  # noqa: BLE001 — one bad step must not sink the order
            out = f"error: {e}"
            log.warning("orders.step_failed", action=action, error=str(e))
        results.append({"action": str(action), "result": out})
    return results


_ACTION_HANDLERS = {
    "note": _act_note,
    "briefing": _act_briefing,
    "discord_dm": _act_discord_dm,
    "workspace_file": _act_workspace_file,
    "rhc_snapshot": _act_rhc_snapshot,
}


async def fire_orders(trips: list[dict[str, Any]], settings: Settings | None = None) -> int:
    """Entry point (called from the alert worker after a wake). Runs every
    matching signed order for the tripped events, respecting the per-action
    daily cap. Returns the number of orders executed. Never raises."""
    settings = settings or get_settings()
    if not settings.standing_orders_enabled:
        return 0
    orders = load_orders(settings)
    if not orders:
        return 0

    from .telemetry.worker import _today_iso
    state = _read_state(settings)
    today = _today_iso()
    if state.get("day_iso") != today:
        state = {"day_iso": today, "counts": {}}
    counts: dict[str, int] = state["counts"]
    cap = settings.standing_orders_daily_cap

    fired = 0
    for trip in trips:
        for order in orders:
            name = str(order.get("name", "unnamed"))
            if not order_matches(order, trip):
                continue
            if int(counts.get(name, 0)) >= cap:
                log.info("orders.capped", order=name)
                continue
            results = await execute_order(settings, order, trip)
            counts[name] = int(counts.get(name, 0)) + 1
            fired += 1
            log.info("orders.fired", order=name, kind=trip.get("kind"),
                     steps=[r["action"] for r in results])
    state["counts"] = counts
    state["day_iso"] = today
    _write_state(settings, state)
    return fired
