import { useEffect, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";

interface SolarCycleData {
  ok: boolean;
  current_month?: string;
  current_ssn?: number | null;
  current_f107?: number | null;
  smoothed_month?: string | null;
  smoothed_ssn?: number | null;
  peak_month?: string | null;
  peak_ssn?: number | null;
  phase?: string;
  rhc_prediction?: number;
  rhc_miss?: number | null;
  mainstream_prediction?: number;
  mainstream_miss?: number | null;
  rhc_wins?: boolean;
  error?: string;
}

// SC25 indices update monthly — a 30-min client poll is generous; the server
// caches 6h, so most polls hit warm cache. Paused when the tab is hidden.
const REFRESH_MS = 30 * 60 * 1000;

export default function SolarCycleSection({
  RowEl,
}: {
  RowEl: (props: { k: string; v: ReactNode }) => ReactElement;
}) {
  const [data, setData] = useState<SolarCycleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const inflight = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (inflight.current) return;
      if (document.visibilityState === "hidden") return;
      inflight.current = true;
      try {
        const r = await fetch("/api/solar-cycle");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const payload = (await r.json()) as SolarCycleData;
        if (!cancelled) {
          setData(payload);
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

  if (loading && !data) return <div className="text-muted">loading…</div>;
  if (error && !data) return <div className="text-muted">{error}</div>;
  if (!data?.ok) return <div className="text-muted">{data?.error ?? "unavailable"}</div>;

  const peak = data.peak_ssn;
  const rhcMiss = data.rhc_miss;
  const mainMiss = data.mainstream_miss;

  return (
    <>
      <RowEl
        k="observed peak"
        v={
          <span className="text-fg">
            {peak != null ? peak.toFixed(1) : "—"}
            <span className="text-muted">
              {data.peak_month ? ` @ ${data.peak_month}` : ""}
            </span>
          </span>
        }
      />
      <RowEl
        k="RHC call"
        v={
          <span className={data.rhc_wins ? "text-accent" : "text-fg"}>
            {data.rhc_prediction ?? "—"}
            {rhcMiss != null && <span className="text-muted"> · miss {rhcMiss}</span>}
            {data.rhc_wins ? " ◇" : ""}
          </span>
        }
      />
      <RowEl
        k="mainstream"
        v={
          <span className="text-muted">
            {data.mainstream_prediction ?? "—"}
            {mainMiss != null && <span> · miss {mainMiss}</span>}
          </span>
        }
      />
      <RowEl
        k="now"
        v={
          <span className="text-fg">
            SSN {data.current_ssn != null ? data.current_ssn.toFixed(1) : "—"}
            <span className={data.phase === "declining" ? "text-muted" : "text-accent"}>
              {data.phase ? ` · ${data.phase}` : ""}
            </span>
          </span>
        }
      />
      <RowEl
        k="month"
        v={<span className="text-muted">{data.current_month ?? "—"} · NOAA SWPC</span>}
      />
    </>
  );
}
