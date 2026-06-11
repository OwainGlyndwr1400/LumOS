import { useEffect, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";

// Snapshot shape mirrors lumos_node/telemetry/grimoire.py fetch_grid_timing().
// Kept loose (every field optional) so a degraded compute still renders — same
// posture as CosmicSection. Local time fields are ISO strings already formatted
// in the operator's tz by the backend, so we slice "HH:MM" rather than re-parse.
interface GrimoireSnapshot {
  ok?: boolean;
  planetary_hour?: {
    phase?: string;
    hour_number?: number;
    ruler?: string;
    glyph?: string;
    harmonic_tone_hz?: number;
    hour_start_local?: string | null;
    hour_end_local?: string | null;
  };
  moon?: {
    illumination_percent?: number;
    phase_name?: string;
    age_days?: number;
    ecliptic_longitude_deg?: number;
    zodiac_sign?: string;
    zodiac_glyph?: string;
  };
  sidereal_time?: string;
  solar?: {
    sunrise_local?: string | null;
    noon_local?: string | null;
    sunset_local?: string | null;
    next_sunrise_local?: string | null;
  };
  fixed_stars?: Record<
    string,
    {
      alt_deg?: number;
      az_deg?: number;
      above_horizon?: boolean;
      next_rising_utc?: string | null;
    }
  >;
  visible_planets?: Array<{ name?: string; glyph?: string; alt_deg?: number; az_deg?: number }>;
  welsh_wheel?: {
    next?: { name?: string; meaning?: string; date_local?: string; days_until?: number } | null;
    upcoming?: Array<{ name?: string; meaning?: string; date_local?: string; days_until?: number }>;
  };
  fetched_at?: string;
}

// Planetary hours are hour-long windows; refresh well inside that so the live
// "window" row and the current ruler stay accurate without hammering.
const REFRESH_MS = 60 * 1000; // 1 min — matches the 60 s server-side cache
const STALE_MS = 10 * 60 * 1000;

function fmtAge(fetchedAt?: string): string {
  if (!fetchedAt) return "—";
  const ts = Date.parse(fetchedAt);
  if (Number.isNaN(ts)) return "—";
  const ageSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (ageSec < 60) return `${ageSec}s ago`;
  if (ageSec < 3600) return `${Math.floor(ageSec / 60)}m ago`;
  return `${Math.floor(ageSec / 3600)}h ago`;
}

// Backend already localizes these ISO strings; just take the HH:MM slice.
function clock(iso?: string | null): string {
  if (!iso || iso.length < 16 || iso[10] !== "T") return "—";
  return iso.slice(11, 16);
}

export default function GrimoireSection({
  RowEl,
}: {
  RowEl: (props: { k: string; v: ReactNode }) => ReactElement;
}) {
  const [snap, setSnap] = useState<GrimoireSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const inflight = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (inflight.current) return;
      inflight.current = true;
      try {
        const r = await fetch("/api/grimoire/current");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = (await r.json()) as GrimoireSnapshot;
        if (!cancelled) {
          setSnap(data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "fetch failed");
      } finally {
        inflight.current = false;
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const id = window.setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  if (loading && !snap) {
    return <div className="text-muted">loading…</div>;
  }
  if ((error && !snap) || (snap && snap.ok === false)) {
    return <div className="text-muted">unavailable ({error ?? "compute failed"})</div>;
  }

  const ph = snap?.planetary_hour ?? {};
  const moon = snap?.moon ?? {};
  const sun = snap?.solar ?? {};
  const regulus = snap?.fixed_stars?.Regulus;
  const visible = snap?.visible_planets ?? [];
  const stale =
    snap?.fetched_at && Date.now() - Date.parse(snap.fetched_at) > STALE_MS;

  return (
    <>
      {ph.ruler && (
        <RowEl
          k="hour"
          v={
            <span title={`day-hour ${ph.hour_number ?? "?"}`}>
              <span className="text-fg">
                {ph.glyph ? `${ph.glyph} ` : ""}
                {ph.ruler}
              </span>
              <span className="text-muted">
                {" "}
                · #{ph.hour_number ?? "?"} {ph.phase ?? ""}
              </span>
            </span>
          }
        />
      )}
      {ph.harmonic_tone_hz != null && ph.harmonic_tone_hz > 0 && (
        <RowEl k="tone" v={`${ph.harmonic_tone_hz.toFixed(0)} Hz`} />
      )}
      {(ph.hour_start_local || ph.hour_end_local) && (
        <RowEl
          k="hour window"
          v={
            <span className="text-muted">
              {clock(ph.hour_start_local)} → {clock(ph.hour_end_local)}
            </span>
          }
        />
      )}
      {moon.illumination_percent != null && (
        <RowEl
          k="moon"
          v={
            <span title={moon.age_days != null ? `age ${moon.age_days}d` : ""}>
              {moon.illumination_percent.toFixed(0)}%
              {moon.phase_name ? (
                <span className="text-muted"> · {moon.phase_name}</span>
              ) : null}
            </span>
          }
        />
      )}
      {moon.zodiac_sign && (
        <RowEl
          k="moon sign"
          v={
            <span title={moon.ecliptic_longitude_deg != null ? `${moon.ecliptic_longitude_deg.toFixed(1)}° ecliptic` : ""}>
              {moon.zodiac_glyph ? `${moon.zodiac_glyph} ` : ""}
              {moon.zodiac_sign}
            </span>
          }
        />
      )}
      {regulus?.alt_deg != null && (
        <RowEl
          k="Regulus"
          v={
            <span
              className={regulus.above_horizon ? "text-accent" : "text-muted"}
              title={
                regulus.az_deg != null
                  ? `az ${regulus.az_deg.toFixed(0)}°${
                      !regulus.above_horizon && regulus.next_rising_utc
                        ? ` · rises ${clock(regulus.next_rising_utc)} UTC`
                        : ""
                    }`
                  : ""
              }
            >
              {regulus.alt_deg >= 0 ? "+" : ""}
              {regulus.alt_deg.toFixed(1)}° · {regulus.above_horizon ? "above" : "below"}
            </span>
          }
        />
      )}
      {(sun.sunrise_local || sun.sunset_local) && (
        <RowEl
          k="sun"
          v={
            <span className="text-muted">
              ↑{clock(sun.sunrise_local)} ↓{clock(sun.sunset_local)}
            </span>
          }
        />
      )}
      {snap?.sidereal_time && (
        <RowEl
          k="sidereal"
          v={<span className="text-fg">{snap.sidereal_time.split(".")[0]}</span>}
        />
      )}
      {visible.length > 0 && (
        <RowEl
          k="visible"
          v={
            <span title={visible.map((p) => p.name).join(", ")}>
              {visible.length} ·{" "}
              {visible.map((p) => p.glyph).filter(Boolean).join(" ")}
            </span>
          }
        />
      )}
      {snap?.welsh_wheel?.next?.name && (
        <RowEl
          k="wheel"
          v={
            <span
              className={
                (snap.welsh_wheel.next.days_until ?? 99) <= 3
                  ? "text-accent"
                  : "text-fg"
              }
              title={`${snap.welsh_wheel.next.meaning ?? ""} · ${
                snap.welsh_wheel.next.date_local?.slice(0, 16) ?? ""
              }${
                snap.welsh_wheel.upcoming && snap.welsh_wheel.upcoming.length > 1
                  ? ` · then ${snap.welsh_wheel.upcoming
                      .slice(1)
                      .map((f) => `${f.name} ${f.days_until}d`)
                      .join(", ")}`
                  : ""
              }`}
            >
              {snap.welsh_wheel.next.name}
              <span className="text-muted">
                {" "}
                · {snap.welsh_wheel.next.days_until}d
              </span>
            </span>
          }
        />
      )}
      <RowEl
        k="fetched"
        v={
          <span className="text-muted">
            {fmtAge(snap?.fetched_at)}
            {stale ? " ⚠" : ""}
          </span>
        }
      />
    </>
  );
}
