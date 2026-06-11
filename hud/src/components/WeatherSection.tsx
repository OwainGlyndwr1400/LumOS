import { useEffect, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";

interface WeatherData {
  ok: boolean;
  temp_c?: number | null;
  feels_like_c?: number | null;
  humidity_pct?: number | null;
  precip_mm?: number | null;
  cloud_pct?: number | null;
  pressure_hpa?: number | null;
  wind_mph?: number | null;
  gust_mph?: number | null;
  wind_dir_deg?: number | null;
  conditions?: string;
  observed_at?: string | null;
  error?: string;
}

// Server caches Open-Meteo 10 min; 5-min client refresh keeps the widget fresh
// without ever multiplying upstream calls. Paused while the tab is hidden.
const REFRESH_MS = 5 * 60 * 1000;

// 16-point compass from wind direction degrees.
function compass(deg?: number | null): string {
  if (deg == null) return "";
  const pts = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
  ];
  return pts[Math.round((((deg % 360) + 360) % 360) / 22.5) % 16];
}

export default function WeatherSection({
  RowEl,
}: {
  RowEl: (props: { k: string; v: ReactNode }) => ReactElement;
}) {
  const [data, setData] = useState<WeatherData | null>(null);
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
        const r = await fetch("/api/weather/current");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const payload = (await r.json()) as WeatherData;
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

  if (loading && !data) {
    return <div className="text-muted">loading…</div>;
  }
  if (error && !data) {
    return <div className="text-muted">{error}</div>;
  }
  if (!data?.ok) {
    return <div className="text-muted">{data?.error ?? "unavailable"}</div>;
  }

  const gusty =
    data.gust_mph != null && data.wind_mph != null && data.gust_mph >= data.wind_mph + 15;

  return (
    <>
      <RowEl
        k="now"
        v={
          <span className="text-fg">
            {data.temp_c != null ? `${data.temp_c.toFixed(1)}°C` : "—"}{" "}
            <span className="text-muted">{data.conditions ?? ""}</span>
          </span>
        }
      />
      {data.feels_like_c != null &&
        data.temp_c != null &&
        Math.abs(data.feels_like_c - data.temp_c) >= 1 && (
          <RowEl k="feels like" v={`${data.feels_like_c.toFixed(1)}°C`} />
        )}
      <RowEl
        k="wind"
        v={
          <span className={gusty ? "text-signal" : "text-fg"}>
            {data.wind_mph != null ? `${Math.round(data.wind_mph)} mph` : "—"}{" "}
            <span className="text-muted">{compass(data.wind_dir_deg)}</span>
            {data.gust_mph != null ? (
              <span className="text-muted"> · g{Math.round(data.gust_mph)}</span>
            ) : null}
          </span>
        }
      />
      {data.precip_mm != null && data.precip_mm > 0 && (
        <RowEl
          k="precip"
          v={<span className="text-signal">{data.precip_mm.toFixed(1)} mm/h</span>}
        />
      )}
      <RowEl
        k="humidity"
        v={data.humidity_pct != null ? `${Math.round(data.humidity_pct)}%` : "—"}
      />
      <RowEl
        k="pressure"
        v={
          data.pressure_hpa != null ? `${Math.round(data.pressure_hpa)} hPa` : "—"
        }
      />
      {data.cloud_pct != null && (
        <RowEl k="cloud" v={`${Math.round(data.cloud_pct)}%`} />
      )}
    </>
  );
}
