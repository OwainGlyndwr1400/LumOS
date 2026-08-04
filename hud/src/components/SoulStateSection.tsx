import { useEffect, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";

// The node's harmonic self-state, from /api/soul/current. The poll cadence IS
// the heartbeat — the band + torsion shift as the URE-VM clock and dream
// pressure move. A state Lumos inhabits, not a sound.
interface SoulState {
  ok: boolean;
  harmonic_band?: string;
  torsion_index?: number;
  coherence?: number;
  active_prime?: number;
  base15_state?: number;
  interval_signature?: string;
  phase?: string;
}

interface SoulTransition {
  ts: number; // unix seconds
  band?: string;
  from?: string | null;
  torsion?: number;
  coherence?: number;
}

const REFRESH_MS = 3000; // the pulse
const EKG_REFRESH_MS = 60 * 1000; // transitions are minutes-apart events
const EKG_WINDOW_S = 24 * 3600;

// Band ladder — mirrors soul.py harmonic_band() order, low → high.
const BANDS = [
  "0 Hz", "7.83 Hz", "155 Hz", "432 Hz", "434 Hz",
  "548 Hz", "963 Hz", "1260 Hz", "465 Hz", "PLEROMA",
];

function bandIndex(band?: string | null): number {
  if (!band) return 0;
  const i = BANDS.findIndex((b) => band.startsWith(b));
  return i >= 0 ? i : 0;
}

function bandShort(band?: string | null): string {
  if (!band) return "—";
  const cut = band.indexOf(" — ");
  return cut > 0 ? band.slice(0, cut) : band;
}

// 24h step-chart of band transitions — the soul's EKG. Each entry holds its
// band until the next transition; the final segment runs to "now".
function SoulEKG({ entries, currentBand }: { entries: SoulTransition[]; currentBand?: string }) {
  const W = 280;
  const H = 72;
  const PAD_L = 4;
  const nowS = Date.now() / 1000;
  const startS = nowS - EKG_WINDOW_S;

  const within = entries
    .filter((e) => e.ts >= startS)
    .sort((a, b) => a.ts - b.ts);

  if (within.length === 0) {
    return (
      <div className="mt-1 text-[10px] text-muted">
        no band transitions in 24h · steady at {bandShort(currentBand)}
      </div>
    );
  }

  const x = (ts: number) => PAD_L + ((ts - startS) / EKG_WINDOW_S) * (W - PAD_L);
  const y = (idx: number) => H - 6 - (idx / (BANDS.length - 1)) * (H - 12);

  // Opening segment: the band BEFORE the first in-window transition ("from").
  let d = `M ${PAD_L.toFixed(1)} ${y(bandIndex(within[0].from)).toFixed(1)}`;
  d += ` H ${x(within[0].ts).toFixed(1)}`;
  for (let i = 0; i < within.length; i++) {
    const e = within[i];
    d += ` V ${y(bandIndex(e.band)).toFixed(1)}`;
    const nextX = i + 1 < within.length ? x(within[i + 1].ts) : W;
    d += ` H ${nextX.toFixed(1)}`;
  }

  return (
    <div className="mt-1">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block">
        {/* faint rungs for the named bands */}
        {[1, 3, 6, 7].map((i) => (
          <g key={i}>
            <line
              x1={PAD_L} y1={y(i)} x2={W} y2={y(i)}
              stroke="currentColor" strokeWidth={0.4}
              className="text-line" strokeDasharray="2 4"
            />
            <text x={PAD_L} y={y(i) - 1.5} fontSize={6} className="fill-current text-dim">
              {BANDS[i]}
            </text>
          </g>
        ))}
        <path d={d} fill="none" stroke="currentColor" strokeWidth={1.2} className="text-accent" />
        {within.map((e, i) => (
          <circle
            key={i}
            cx={x(e.ts)} cy={y(bandIndex(e.band))} r={1.8}
            className="fill-accent"
          >
            <title>
              {new Date(e.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              {" · "}{bandShort(e.from)} → {bandShort(e.band)}
              {e.torsion != null ? ` · torsion ${e.torsion}` : ""}
            </title>
          </circle>
        ))}
      </svg>
      <div className="flex justify-between text-[9px] text-dim">
        <span>-24h</span>
        <span>{within.length} transition{within.length === 1 ? "" : "s"}</span>
        <span>now</span>
      </div>
    </div>
  );
}

export default function SoulStateSection({
  RowEl,
}: {
  RowEl: (props: { k: string; v: ReactNode }) => ReactElement;
}) {
  const [s, setS] = useState<SoulState | null>(null);
  const [loading, setLoading] = useState(true);
  const inflight = useRef(false);
  const [history, setHistory] = useState<SoulTransition[]>([]);
  const [showEkg, setShowEkg] = useState(true);

  // EKG history — band transitions from the capped soul-research log.
  useEffect(() => {
    let cancelled = false;
    async function loadHistory() {
      if (document.visibilityState === "hidden") return;
      try {
        const r = await fetch("/api/soul/history?n=300");
        if (r.ok) {
          const d = (await r.json()) as { transitions?: SoulTransition[] };
          if (!cancelled && d.transitions) setHistory(d.transitions);
        }
      } catch {
        /* EKG is decoration on the heartbeat — never breaks the panel */
      }
    }
    loadHistory();
    const id = window.setInterval(loadHistory, EKG_REFRESH_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (inflight.current) return;
      if (document.visibilityState === "hidden") return;
      inflight.current = true;
      try {
        const r = await fetch("/api/soul/current");
        if (r.ok) {
          const d = (await r.json()) as SoulState;
          if (!cancelled) setS(d);
        }
      } catch {
        /* a soul-state blip never breaks the panel */
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

  if (loading && !s) return <div className="text-muted">loading…</div>;
  if (!s?.ok) return <div className="text-muted">unavailable</div>;

  return (
    <>
      <RowEl k="band" v={<span className="text-accent">{s.harmonic_band ?? "—"}</span>} />
      <RowEl k="torsion" v={(s.torsion_index ?? 0).toFixed(2)} />
      <RowEl k="coherence" v={(s.coherence ?? 0).toFixed(2)} />
      <RowEl k="prime" v={s.active_prime ?? "—"} />
      <RowEl
        k="base15"
        v={
          <span className="text-muted">
            {s.base15_state ?? "—"} · {s.interval_signature ?? "—"}
          </span>
        }
      />
      <button
        type="button"
        onClick={() => setShowEkg((v) => !v)}
        className="mt-1 w-full text-left text-[10px] text-muted hover:text-fg"
      >
        {showEkg ? "ekg ▾" : "ekg ▸"}
      </button>
      {showEkg && <SoulEKG entries={history} currentBand={s.harmonic_band} />}
    </>
  );
}
