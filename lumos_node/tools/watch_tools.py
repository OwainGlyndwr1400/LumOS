"""Custom-watch management (Phase 39) — OPERATOR-ONLY (deliberately NOT passive).

`manage_watch` lets the operator say "Lumos, keep an eye on the Taiwan Strait"
and have Lumos create a standing alert rule that the monitor checks every poll,
firing through the same wake path (HUD + Discord) as the built-in home alerts.

This tool is excluded from the autonomous passive set (see tool_router.py): an
unprompted wake can observe and speak, but must never reconfigure its own
monitoring. Changing what Lumos watches is an operator act, by design.
"""

from __future__ import annotations

from . import register
from ..config import get_settings
from ..log import get_logger
from ..telemetry import watches


log = get_logger(__name__)


@register(
    name="manage_watch",
    description=(
        "Create, list, or remove a custom WATCH — a standing alert rule for a place "
        "ANYWHERE on Earth. Use when the operator says things like 'keep an eye on "
        "the Taiwan Strait', 'ping me if a warship comes near Milford Haven', 'list "
        "my watches', or 'stop watching X'. A watch fires through the same alert → "
        "wake path as the built-in home alerts (HUD bubble + Discord DM), with the "
        "same cooldown and daily cap. SOURCES: military_air, gps_jamming, and "
        "recon_satellite query their feeds per-request so they work anywhere; vessel "
        "reads the live AIS cache, which only covers waters near the operator (say so "
        "if asked to watch a far-off sea). For action='add' you must supply lat/lon — "
        "use the coordinates of the named place from your own knowledge, and if you're "
        "unsure of a location, ASK the operator rather than guessing. Returns the "
        "created watch (with its id), the list, or the removal result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "remove", "enable", "disable"],
                "description": "What to do. 'list' shows all watches with their ids.",
            },
            "label": {
                "type": "string",
                "description": "Human name for the watch, e.g. 'Taiwan Strait' (action=add).",
            },
            "lat": {
                "type": "number",
                "description": "Latitude of the place to watch (action=add). Supply from your knowledge of the named location.",
            },
            "lon": {
                "type": "number",
                "description": "Longitude of the place to watch (action=add).",
            },
            "radius_km": {
                "type": "number",
                "default": 150.0,
                "description": "Watch radius in km (action=add). Default 150.",
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["military_air", "gps_jamming", "recon_satellite", "vessel"],
                },
                "description": "Which feeds to watch (action=add). At least one.",
            },
            "sat_min_elevation_deg": {
                "type": "number",
                "default": 30.0,
                "description": "For the recon_satellite source: minimum overhead elevation (deg) to trip (action=add).",
            },
            "watch_id": {
                "type": "string",
                "description": "The watch id (action=remove/enable/disable). Get it from action=list.",
            },
        },
        "required": ["action"],
    },
)
async def manage_watch(
    action: str,
    label: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 150.0,
    sources: list[str] | None = None,
    sat_min_elevation_deg: float = 30.0,
    watch_id: str | None = None,
) -> dict:
    settings = get_settings()
    act = (action or "").strip().lower()

    if act == "list":
        return {"ok": True, "watches": watches.list_watches(settings)}

    if act == "add":
        if lat is None or lon is None:
            return {
                "ok": False,
                "error": "add needs lat and lon — supply the coordinates of the place, or ask the operator if unsure",
            }
        try:
            w = watches.add_watch(
                settings,
                label=label or "",
                lat=lat,
                lon=lon,
                radius_km=radius_km,
                sources=sources,
                sat_min_elevation_deg=sat_min_elevation_deg,
            )
        except (ValueError, TypeError) as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "added": w}

    if act == "remove":
        if not watch_id:
            return {"ok": False, "error": "remove needs watch_id — call action=list to find it"}
        return {"ok": watches.remove_watch(settings, watch_id), "removed": watch_id}

    if act in ("enable", "disable"):
        if not watch_id:
            return {"ok": False, "error": f"{act} needs watch_id — call action=list to find it"}
        ok = watches.set_enabled(settings, watch_id, act == "enable")
        return {"ok": ok, "watch_id": watch_id, "enabled": act == "enable"}

    return {
        "ok": False,
        "error": f"unknown action '{action}' — use add | list | remove | enable | disable",
    }
