import { useEffect, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";

interface RailCall {
  uid?: string;
  dest?: string | null;
  origin?: string | null;
  booked?: string | null;
  expected?: string | null;
  platform?: string | null;
  operator?: string | null;
  direction?: string | null;
  cancelled?: boolean;
}

interface RailData {
  ok: boolean;
  code?: string;
  station?: string;
  count?: number;
  calls?: RailCall[];
  error?: string;
}

// 60s refresh — trains move on the minute, and the refresh→access token
// exchange is cached server-side, so this stays trivially under rtt.io's
// 30/min limit. Paused when the tab is hidden via document.visibilityState.
const REFRESH_MS = 60 * 1000;

// "Llanelli (W)" → "⬅ Llanelli"  ·  "Swansea (E)" → "Swansea ➡"
function dirLabel(d?: string | null): string {
  if (!d) return "—";
  if (d.includes("(W)")) return `⬅ ${d.replace(" (W)", "")}`;
  if (d.includes("(E)")) return `${d.replace(" (E)", "")} ➡`;
  return d;
}

export default function RailSection({
  RowEl,
}: {
  RowEl: (props: { k: string; v: ReactNode }) => ReactElement;
}) {
  const [data, setData] = useState<RailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const inflight = useRef(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (inflight.current) return;
      if (document.visibilityState === "hidden") return;
      inflight.current = true;
      try {
        const r = await fetch("/api/rail/current");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const payload = (await r.json()) as RailData;
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

  const calls = data.calls ?? [];
  const next = calls[0];

  return (
    <>
      {next ? (
        <>
          <RowEl
            k="next"
            v={
              <span className="text-fg">
                {next.expected ?? next.booked ?? "—"}{" "}
                <span className="text-muted">→ {next.dest ?? "?"}</span>
              </span>
            }
          />
          <RowEl
            k="heading"
            v={<span className="text-accent">{dirLabel(next.direction)}</span>}
          />
          {next.platform && <RowEl k="platform" v={next.platform} />}
          {next.cancelled && (
            <RowEl k="status" v={<span className="text-signal">cancelled</span>} />
          )}
        </>
      ) : (
        <RowEl
          k="next"
          v={<span className="text-muted">none stopping (next hr)</span>}
        />
      )}
      <RowEl
        k="station"
        v={
          <span className="text-muted">
            {data.station ?? data.code} · {calls.length} due
          </span>
        }
      />
      {calls.length > 1 && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="mt-1 w-full text-left text-[10px] text-muted hover:text-fg"
        >
          {expanded ? "hide list ▾" : `show list ▸  (${calls.length})`}
        </button>
      )}
      {expanded &&
        calls.slice(0, 12).map((c) => (
          <div
            key={c.uid}
            className="flex justify-between gap-2 text-[10px]"
            title={c.operator ?? ""}
          >
            <span className="truncate text-fg">
              {c.expected ?? c.booked} {c.dest ?? "?"}
            </span>
            <span className="shrink-0 text-muted">
              {dirLabel(c.direction)}
              {c.platform ? ` · p${c.platform}` : ""}
            </span>
          </div>
        ))}
    </>
  );
}
