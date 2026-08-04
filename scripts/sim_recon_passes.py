"""One-off: count military_recon passes by peak-elevation threshold over next 24h.

Uses the same SatNOGS feed and classification keywords as
lumos_node.telemetry.satellites, then runs a 24h propagation per satellite to
isolate passes (continuous arcs above 0 deg) and record each pass's peak
elevation. Aggregates pass counts at thresholds [55, 60, 65, 70, 75, 80].
"""
from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from datetime import timedelta

import httpx

from lumos_node.config import get_settings
from lumos_node.telemetry.satellites import _MISSION_KEYWORDS, _classify, fetch_tle


THRESHOLDS = [55, 60, 65, 70, 75, 80]


async def main() -> None:
    settings = get_settings()
    lat = settings.operator_lat
    lon = settings.operator_lon
    print(f"observer: lat={lat:.4f} lon={lon:.4f}")

    tles = await fetch_tle()
    print(f"tle set: {len(tles)} satellites")
    if not tles:
        print("ABORT: no TLEs (SatNOGS unreachable / cache empty)")
        sys.exit(1)

    recon = [t for t in tles if _classify(t["name"]) == "military_recon"]
    print(f"military_recon subset: {len(recon)} satellites")
    if not recon:
        print("ABORT: classifier produced 0 recon sats")
        sys.exit(1)

    # Show a sample so we can sanity-check the classification
    print("sample recon names:")
    for t in recon[:10]:
        print(f"  - {t['name']}")

    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale(builtin=True)
    observer = wgs84.latlon(lat, lon)

    # 24h window starting now, sample every 30 s. 30s catches all >55 deg
    # culminations cleanly (LEO recon sats spend several min above 55 deg).
    t0 = ts.now()
    # ts.tt_jd works in TT days; +1 day = 1.0
    step_s = 30.0
    n_steps = int((24 * 3600) / step_s) + 1
    # Build a time vector
    import numpy as np
    seconds = np.arange(n_steps) * step_s
    times = ts.tt_jd(t0.tt + seconds / 86400.0)

    # For each sat: compute elevation array, find passes (above-horizon arcs),
    # peak elevation per pass. Count by threshold.
    threshold_counts: dict[int, int] = {th: 0 for th in THRESHOLDS}
    # Also collect details per pass for high-elevation report
    high_passes: list[tuple[float, str, float]] = []  # (peak_deg, name, hours_from_now)
    sats_with_any_pass = 0
    sats_errored = 0

    for tle in recon:
        try:
            sat = EarthSatellite(tle["line1"], tle["line2"], tle["name"], ts)
            topocentric = (sat - observer).at(times)
            alt, az, dist = topocentric.altaz()
            elev = alt.degrees  # numpy array
        except Exception as e:
            sats_errored += 1
            continue

        # A "pass" = contiguous run of elev > 0. Find run starts/ends.
        above = elev > 0
        if not above.any():
            continue
        # Find transition indices
        diff = np.diff(above.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        # Handle case where we start already above
        if above[0]:
            starts = np.concatenate(([0], starts))
        if above[-1]:
            ends = np.concatenate((ends, [len(above)]))

        had_pass = False
        for s, e in zip(starts, ends):
            peak = float(elev[s:e].max())
            if peak <= 0:
                continue
            had_pass = True
            for th in THRESHOLDS:
                if peak >= th:
                    threshold_counts[th] += 1
            if peak >= 55:
                # hours from start of window for peak
                peak_idx = s + int(np.argmax(elev[s:e]))
                hours = peak_idx * step_s / 3600.0
                high_passes.append((peak, tle["name"], hours))

        if had_pass:
            sats_with_any_pass += 1

    print(f"\nsats with any pass above horizon in 24h: {sats_with_any_pass}")
    print(f"sats errored during propagation: {sats_errored}")

    print("\n=== passes per 24h by peak-elevation threshold ===")
    print(f"{'threshold':>10}  {'passes/day':>10}")
    for th in THRESHOLDS:
        print(f"{th:>9}°  {threshold_counts[th]:>10}")

    if high_passes:
        high_passes.sort(reverse=True)
        print(f"\ntop high-elevation recon passes in next 24h (>=55 deg):")
        print(f"{'peak deg':>9}  {'h from now':>10}  name")
        for peak, name, h in high_passes[:25]:
            print(f"{peak:>9.1f}  {h:>10.2f}  {name}")
        print(f"\ntotal >=55 deg passes listed: {len(high_passes)}")


if __name__ == "__main__":
    asyncio.run(main())
