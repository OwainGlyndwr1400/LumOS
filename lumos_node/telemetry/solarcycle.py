"""Solar Cycle 25 telemetry — NOAA/SWPC observed sunspot indices vs the RHC
Jupiter-Saturn-GCD peak prediction.

The 2019 NOAA/NASA SC25 Prediction Panel called a weak cycle peaking near a
smoothed sunspot number (SSN) of ~115. The RHC frame predicts the SC25 peak at
~161, from the Jupiter-Saturn synodic GCD resonance. NOAA's own observed record
now puts the realised SC25 peak at 160.9 (smoothed, 2024-10) -- so this panel
just reads the live scoreboard, no hand-entry.

Source: services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json
Monthly rows: time-tag, ssn (monthly mean), smoothed_ssn (13-month smoothed;
-1.0 = not yet computable, needs +/-6 months of data), f10.7, ...
SC25 began at solar minimum ~2019-12.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..log import get_logger

log = get_logger(__name__)

_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
_SC25_START = "2019-12"  # SC25 solar minimum (ISO year-month; lexical compare is safe)

# Predictions overlaid on the live observed record.
MAINSTREAM_PEAK = 115  # NOAA/NASA SC25 Prediction Panel consensus (2019)
# RHC SC25 peak = mainstream base cycle + planetary-GCD correction. Per the RHC
# theorem index (Future Math 01_theorem_index row 103, Paper 12): 161 = 115 (base)
# + 46 (planetary GCD), where +46 is the constructive-interference correction from
# the Jupiter-Saturn-Neptune conjunction GCD. NOAA's observed SC25 peak landed at
# 160.9 (smoothed, 2024-10) → RHC miss 0.1 vs mainstream miss 45.9.
PLANETARY_GCD_CORRECTION = 46  # Jupiter-Saturn-Neptune synodic-GCD constructive interference
RHC_PEAK = MAINSTREAM_PEAK + PLANETARY_GCD_CORRECTION  # = 161

# SC25 indices update monthly; a 6h cache is plenty and is kind to NOAA.
_TTL = 6 * 3600.0
_cache: tuple[float, dict[str, Any]] | None = None


def _num(v: Any) -> float | None:
    """Coerce a NOAA field to float, treating the -1 sentinel (and anything
    non-numeric) as missing. smoothed_ssn carries -1 until the 13-month window
    can be computed (~6 months after the fact)."""
    if isinstance(v, (int, float)) and v >= 0:
        return float(v)
    return None


async def fetch_solar_cycle(*, force: bool = False) -> dict[str, Any]:
    """Pull NOAA's observed SC25 indices; return current state + smoothed peak,
    scored against the RHC (161) vs mainstream (115) predictions. Cached 6h.
    Fail-soft: always returns {ok: bool, ...} so a NOAA blip can't break the HUD."""
    global _cache
    if not force and _cache is not None and (time.time() - _cache[0]) < _TTL:
        return _cache[1]
    try:
        # This file is NOAA's FULL observed record back to 1749 — multi-MB. A
        # 12s timeout failed routinely on cold fetches (HUD showed "loading…"
        # for 5-10 min until a retry landed). It's 6h-cached, so a one-off
        # slow download is fine — give it room.
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.get(_URL)
            resp.raise_for_status()
            rows = resp.json()
    except Exception as e:  # noqa: BLE001 — network/parse, fail soft
        log.info("solarcycle.fetch_failed", error=str(e))
        return {"ok": False, "error": str(e)}

    sc25 = [r for r in rows if str(r.get("time-tag", "")) >= _SC25_START]
    if not sc25:
        return {"ok": False, "error": "no SC25 rows in NOAA record"}

    latest = sc25[-1]
    current_ssn = _num(latest.get("ssn"))
    # The peak + 'current smoothed' must use only computed (>=0) smoothed values;
    # the trailing ~6 months are -1 sentinels.
    smoothed = [(r, _num(r.get("smoothed_ssn"))) for r in sc25]
    smoothed = [(r, s) for r, s in smoothed if s is not None]
    peak_row, peak_val = max(smoothed, key=lambda rs: rs[1]) if smoothed else (None, None)
    last_sm_row, last_sm_val = smoothed[-1] if smoothed else (None, None)

    def _miss(pred: int) -> float | None:
        return round(abs(peak_val - pred), 1) if peak_val is not None else None

    rhc_miss = _miss(RHC_PEAK)
    main_miss = _miss(MAINSTREAM_PEAK)

    # Declining limb once the latest monthly SSN sits below the smoothed peak.
    phase = "rising"
    if peak_val is not None and current_ssn is not None and current_ssn < peak_val:
        phase = "declining"

    result: dict[str, Any] = {
        "ok": True,
        "source": "NOAA SWPC · observed-solar-cycle-indices",
        "current_month": latest.get("time-tag"),
        "current_ssn": current_ssn,
        "current_f107": _num(latest.get("f10.7")),
        "smoothed_month": last_sm_row.get("time-tag") if last_sm_row else None,
        "smoothed_ssn": last_sm_val,
        "peak_month": peak_row.get("time-tag") if peak_row else None,
        "peak_ssn": peak_val,
        "phase": phase,
        "rhc_prediction": RHC_PEAK,
        "rhc_miss": rhc_miss,
        "mainstream_prediction": MAINSTREAM_PEAK,
        "mainstream_miss": main_miss,
        # which call landed closer to the realised peak
        "rhc_wins": rhc_miss is not None and main_miss is not None and rhc_miss <= main_miss,
    }
    _cache = (time.time(), result)
    return result
