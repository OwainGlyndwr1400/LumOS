"""Schumann ELF amplitude — the RHC AtmosphericPressure sub-variable.

Honest by construction. There is NO universal keyless JSON API for live Schumann
resonance: the well-known "live" sites (Tomsk mirrors, EarthWave) publish
spectrogram IMAGES, not numbers, and single-station provenance is contentious.
So this module is OPT-IN and self-describing:

  • LUMOS_SCHUMANN_URL unset  → returns {ok: False, amplitude: None} — the RHC
    log keeps schumann_amplitude null + flagged, never fabricated.
  • LUMOS_SCHUMANN_URL set to a JSON endpoint you trust → the parser pulls an
    amplitude (+ frequency/power if present) and every logged row carries a
    `source` + `confidence` tag so the dataset is transparent about what it is.

The parser is pure + tested; the fetch is best-effort and fails to a clean null
(a dead/blocking feed must never break a flare-triggered Standing Order).
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache

log = get_logger(__name__)

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# Field names commonly used for amplitude / frequency across ELF JSON feeds.
_AMP_KEYS = ("amplitude", "amp", "power", "value", "sr_amplitude", "a1")
_FREQ_KEYS = ("frequency", "freq", "f0", "peak_frequency", "sr_frequency")


def _dig(data: dict[str, Any], dotted: str) -> Any:
    """Resolve a possibly-dotted key path ('data.amp') against a nested dict."""
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first_num(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = data.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                continue
    return None


def parse_schumann(data: Any, amp_key: str = "", freq_key: str = "") -> dict[str, Any]:
    """PURE — a JSON payload → {ok, amplitude, frequency_hz, power}. Explicit
    _key overrides win; otherwise auto-detect common field names. Non-dict or
    amplitude-less payloads yield ok:False (so the caller logs a clean null)."""
    if not isinstance(data, dict):
        return {"ok": False, "amplitude": None, "frequency_hz": None, "power": None}
    amp = _dig(data, amp_key) if amp_key else None
    if amp is None:
        amp = _first_num(data, _AMP_KEYS)
    elif isinstance(amp, str):
        try:
            amp = float(amp)
        except ValueError:
            amp = None
    freq = _dig(data, freq_key) if freq_key else None
    if freq is None:
        freq = _first_num(data, _FREQ_KEYS)
    if not isinstance(amp, (int, float)):
        return {"ok": False, "amplitude": None, "frequency_hz": None, "power": None}
    return {
        "ok": True,
        "amplitude": round(float(amp), 4),
        "frequency_hz": round(float(freq), 4) if isinstance(freq, (int, float)) else None,
        "power": _first_num(data, ("power", "p1", "sr_power")),
    }


async def fetch_schumann() -> dict[str, Any]:
    """Live Schumann reading from LUMOS_SCHUMANN_URL, cached 5 min. Returns
    {ok, amplitude, frequency_hz, power, source, confidence}. Unconfigured or
    unreachable → ok:False with amplitude None (the honest-null path)."""
    settings = get_settings()
    url = settings.schumann_url.strip()
    if not url:
        return {"ok": False, "amplitude": None, "source": None,
                "confidence": "no feed configured (LUMOS_SCHUMANN_URL unset)"}

    cached = tcache.get("schumann")
    if cached is not None:
        return cached

    result: dict[str, Any]
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(url, headers={"Accept": "application/json"}, follow_redirects=True)
            r.raise_for_status()
            data = r.json()
        parsed = parse_schumann(data, settings.schumann_amp_key.strip(), settings.schumann_freq_key.strip())
        result = {
            **parsed,
            "source": url,
            "confidence": "single-station-proxy" if parsed["ok"] else "feed returned no amplitude",
        }
    except (httpx.HTTPError, ValueError) as e:
        log.info("schumann.fetch_failed", error=str(e))
        result = {"ok": False, "amplitude": None, "source": url, "confidence": f"fetch failed: {e}"}

    tcache.put("schumann", result)
    return result
