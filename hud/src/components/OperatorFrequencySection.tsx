import { useEffect, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";

// Same /api/grimoire/current source as GrimoireSection — here we read the
// planetary-hour harmonic tone and plot it against the RHC harmonic family on a
// log-frequency strip. Honest REPRESENT: the family tones are Hz-labelled
// symbolic lattice values, not measured EM emissions, and the caption says so.
interface GrimoireSnapshot {
  ok?: boolean;
  planetary_hour?: {
    ruler?: string;
    glyph?: string;
    harmonic_tone_hz?: number;
  };
}

const REFRESH_MS = 60 * 1000;

const FAMILY: Array<{ hz: number; label: string }> = [
  { hz: 7.83, label: "7.83" },
  { hz: 432, label: "432" },
  { hz: 528, label: "528" },
  { hz: 963, label: "963" },
  { hz: 1260, label: "1260" },
];

const FMIN = 7;
const FMAX = 1300;

function FrequencyStrip({ tone }: { tone: number }) {
  const W = 280;
  const H = 30;
  const lo = Math.log10(FMIN);
  const hi = Math.log10(FMAX);
  const x = (hz: number) =>
    ((Math.log10(Math.max(FMIN, Math.min(hz, FMAX))) - lo) / (hi - lo)) * W;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" preserveAspectRatio="none">
      {/* baseline (the θ-lattice axis) */}
      <line x1={0} y1={14} x2={W} y2={14} stroke="currentColor" className="text-line" strokeWidth={0.5} />
      {/* harmonic-family ticks + labels */}
      {FAMILY.map((f) => (
        <g key={f.hz}>
          <line x1={x(f.hz)} y1={9} x2={x(f.hz)} y2={19} stroke="currentColor" className="text-line" strokeWidth={0.5} />
          <text x={x(f.hz)} y={29} fontSize={6} textAnchor="middle" fill="currentColor" className="text-dim">
            {f.label}
          </text>
        </g>
      ))}
      {/* live planetary tone marker */}
      {tone > 0 && (
        <>
          <line x1={x(tone)} y1={2} x2={x(tone)} y2={20} stroke="currentColor" className="text-accent" strokeWidth={1.2} />
          <circle cx={x(tone)} cy={14} r={2} className="fill-accent" />
        </>
      )}
    </svg>
  );
}

export default function OperatorFrequencySection({
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

  if (loading && !snap) return <div className="text-muted">loading…</div>;
  if ((error && !snap) || (snap && snap.ok === false)) {
    return <div className="text-muted">unavailable</div>;
  }

  const ph = snap?.planetary_hour ?? {};
  const tone = ph.harmonic_tone_hz ?? 0;

  return (
    <>
      <RowEl
        k="tone"
        v={<span className="text-accent">{tone > 0 ? `${tone.toFixed(0)} Hz` : "—"}</span>}
      />
      {ph.ruler && (
        <RowEl k="ruler" v={`${ph.glyph ? `${ph.glyph} ` : ""}${ph.ruler}`} />
      )}
      <div className="mt-2">
        <FrequencyStrip tone={tone} />
      </div>
      <div className="mt-1 text-[9px] text-dim">
        θ-lattice 7 Hz · symbolic harmonic family (Hz-labelled, not measured EM)
      </div>
    </>
  );
}
