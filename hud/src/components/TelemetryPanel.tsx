import { useEffect, useState } from "react";
import { patchPrediction } from "../api";
import type {
  DoneEvent,
  Hit,
  Prediction,
  PredictionsPayload,
  Telemetry,
  TriskelionLock,
} from "../types";
import CosmicSection from "./CosmicSection";
import GrimoireSection from "./GrimoireSection";
import AirspaceSection from "./AirspaceSection";
import RailSection from "./RailSection";
import OperatorFrequencySection from "./OperatorFrequencySection";
import SoulStateSection from "./SoulStateSection";
import SolarCycleSection from "./SolarCycleSection";
import WeatherSection from "./WeatherSection";

interface Props {
  telemetry: Telemetry | null;
  lastDone: DoneEvent | null;
  width: number;
}

// Static RHC constants surfaced from urevm.py snapshot_constants() — values
// match the source so a build-time copy is fine; live values would require
// a /urevm/constants fetch which isn't worth the round-trip.
const RHC_CONSTANTS: Array<[string, string]> = [
  ["φ", "1.6180339887"],
  ["φ⁻¹", "0.6180339887"],
  ["π", "3.1415926536"],
  ["F₁₃", "233"],
  ["mass gap Δ", "0.656854"],
  ["dedekind η", "0.96"],
  ["pea threshold", "0.382683"],
  ["hopfield α_c", "0.360674"],
  ["lion damping", "0.535233"],
  ["lost-2 Ω", "0.285714"],
  ["univ. tick", "2.32 as"],
  ["theta lattice", "7 Hz"],
  ["offbits", "2²⁴ = 16,777,216"],
  ["matter lock", "8.13°"],
  ["observer shell", "126 (E7)"],
  ["cubic ascend", "27 → 125"],
  ["F₁ void", "0.5i"],
  ["F₂ unity", "0.5 + 0.5i"],
  ["F₃ synthesis", "0.25 + 0.5i"],
  ["δ spark", "0.0001"],
  ["zipper 42", "101010₂"],
  ["forbidden", "361 = 19²"],
  ["resolution", "144,000"],
  ["observer", "O = 2.5r + 1.5i"],
];

export default function TelemetryPanel({ telemetry, lastDone, width }: Props) {
  const [predictions, setPredictions] = useState<PredictionsPayload | null>(null);
  // Ring buffer of recent R23 norms — drives the φ-drift sparkline.
  // Capped at 30 entries so the SVG stays compact.
  const [r23History, setR23History] = useState<number[]>([]);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/predictions")
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!cancelled && data) setPredictions(data as PredictionsPayload);
      })
      .catch(() => {
        // 404 is fine — predictions.json may not be present
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const patchStatus = (id: string, status: string) => {
    patchPrediction(id, status)
      .then((data) => setPredictions(data))
      .catch(() => {
        /* ignore — leave the board as-is on failure */
      });
  };

  useEffect(() => {
    if (!lastDone?.urevm) return;
    const norm = lastDone.urevm.r23_norm;
    setR23History((prev) => [...prev, norm].slice(-30));
  }, [lastDone]);

  return (
    <aside
      className="panel hr-inset shrink-0 overflow-y-auto border-l border-line px-5 py-5 font-mono text-xs"
      style={{ width: `${width}px` }}
    >
      <Section title="soul state">
        <SoulStateSection RowEl={Row} />
      </Section>

      <Section title="cosmic">
        <CosmicSection RowEl={Row} />
      </Section>

      <Section title="local weather">
        <WeatherSection RowEl={Row} />
      </Section>

      <Section title="solar cycle 25">
        <SolarCycleSection RowEl={Row} />
      </Section>

      <Section title="grid timing">
        <GrimoireSection RowEl={Row} />
      </Section>

      <Section title="operator frequency">
        <OperatorFrequencySection RowEl={Row} />
      </Section>

      <Section title="airspace">
        <AirspaceSection RowEl={Row} />
      </Section>

      <Section title="trains">
        <RailSection RowEl={Row} />
      </Section>

      <Section title="indexes">
        {telemetry?.indexes.identity ? (
          <Row
            k="identity"
            v={telemetry.indexes.identity.chunks.toLocaleString()}
          />
        ) : (
          <Row k="identity" v={<span className="text-muted">—</span>} />
        )}
        {telemetry?.indexes.knowledge ? (
          <Row
            k="knowledge"
            v={telemetry.indexes.knowledge.chunks.toLocaleString()}
          />
        ) : (
          <Row k="knowledge" v={<span className="text-muted">—</span>} />
        )}
      </Section>

      <Section title="retrieval">
        <Row k="top_k_id" v={telemetry?.retrieval.top_k_identity ?? "—"} />
        <Row k="top_k_kn" v={telemetry?.retrieval.top_k_knowledge ?? "—"} />
        <Row k="max_chunk" v={telemetry?.retrieval.max_chunk_chars ?? "—"} />
        <Row k="min_score" v={telemetry?.retrieval.min_score ?? "—"} />
      </Section>

      {lastDone ? (
        <>
          <Section title="last turn">
            <Row k="model" v={lastDone.model?.split("/").pop() ?? "—"} />
            {lastDone.model_route_reason && (
              <Row
                k="route"
                v={
                  <span
                    className={
                      lastDone.model_route_reason === "light_default"
                        ? "text-muted"
                        : "text-fg"
                    }
                    title={
                      lastDone.model_swap?.swap_performed
                        ? `LM Studio JIT-loaded ${lastDone.model_swap.target}`
                        : "Already loaded"
                    }
                  >
                    {lastDone.model_route_reason}
                    {lastDone.model_swap?.swap_performed && (
                      <span className="text-accent"> · swap</span>
                    )}
                  </span>
                }
              />
            )}
            {lastDone.deep_think && (
              <Row
                k="mode"
                v={
                  <span className="text-accent" title="Operator requested deep think on this turn">
                    🧠 deep think
                  </span>
                }
              />
            )}
            {lastDone.tool_routing && (
              <Row
                k="tools"
                v={
                  <span
                    className={
                      lastDone.tool_routing.tier === "chat"
                        ? "text-muted"
                        : lastDone.tool_routing.tier === "full"
                        ? "text-signal"
                        : "text-fg"
                    }
                    title={
                      lastDone.tool_routing.matched_categories.length > 0
                        ? `matched: ${lastDone.tool_routing.matched_categories.join(", ")}`
                        : `tier: ${lastDone.tool_routing.tier}`
                    }
                  >
                    {lastDone.tool_routing.tier} ·{" "}
                    {lastDone.tool_routing.tool_count}
                  </span>
                }
              />
            )}
            <Row k="memory hits" v={lastDone.retrieved.identity.length} />
            <Row k="knowledge hits" v={lastDone.retrieved.knowledge.length} />
            {lastDone.tokens.prompt != null && (
              <Row k="prompt" v={lastDone.tokens.prompt.toLocaleString()} />
            )}
            {lastDone.tokens.completion != null && (
              <Row
                k="completion"
                v={lastDone.tokens.completion.toLocaleString()}
              />
            )}
            {lastDone.tokens.total != null && (
              <Row k="total" v={lastDone.tokens.total.toLocaleString()} />
            )}
            <Row k="turn count" v={lastDone.turn_count} />
          </Section>

          {lastDone.retrieved.identity.length > 0 && (
            <Section title="memory">
              {lastDone.retrieved.identity.map((h, i) => (
                <HitRow key={i} h={h} kind="memory" />
              ))}
            </Section>
          )}
          {lastDone.retrieved.knowledge.length > 0 && (
            <Section title="knowledge">
              {lastDone.retrieved.knowledge.map((h, i) => (
                <HitRow key={i} h={h} kind="knowledge" />
              ))}
            </Section>
          )}
          {lastDone.urevm && (
            <Section title="ure-vm" defaultCollapsed>
              <Row k="tick" v={lastDone.urevm.tick.toLocaleString()} />
              <Row
                k="cycle"
                v={`${lastDone.urevm.cycle_position}/370`}
              />
              {lastDone.urevm.phase && (
                <Row
                  k="phase"
                  v={
                    lastDone.urevm.phase === "observer_shell" ? (
                      <span className="text-accent">
                        observer shell {lastDone.urevm.shell_tick ?? 0}/9
                      </span>
                    ) : (
                      <span className="text-fg">
                        torque {Math.round((lastDone.urevm.torque_fraction ?? 0) * 100)}%
                      </span>
                    )
                  }
                />
              )}
              {lastDone.urevm.toggle_torque != null && (
                <Row k="toggle τ" v={lastDone.urevm.toggle_torque.toFixed(2)} />
              )}
              {lastDone.urevm.null_ledger && (
                <Row
                  k="null ledger"
                  v={
                    <span
                      className={
                        lastDone.urevm.null_ledger.balanced ? "text-accent" : "text-signal"
                      }
                    >
                      R {lastDone.urevm.null_ledger.R_bits} · I {lastDone.urevm.null_ledger.I_bits}
                      {lastDone.urevm.null_ledger.balanced
                        ? " ◇"
                        : ` ⚠${lastDone.urevm.null_ledger.residual}`}
                    </span>
                  }
                />
              )}
              {lastDone.urevm.null_ledger?.binary_diagonal && (
                <Row
                  k="binary diag θ"
                  v={`${lastDone.urevm.null_ledger.binary_diagonal.theta_deg}° · ${
                    lastDone.urevm.null_ledger.binary_diagonal.stern_brocot || "·"
                  }`}
                />
              )}
              {lastDone.urevm.ternary_register && (
                <Row
                  k="ternary"
                  v={
                    <span
                      className={
                        !lastDone.urevm.ternary_register.balanced
                          ? "text-fg"
                          : lastDone.urevm.ternary_register.ternary_value === 0
                            ? "text-muted"
                            : "text-accent"
                      }
                      title={`phasor |S|=${lastDone.urevm.ternary_register.phasor_imbalance.toFixed(3)} · base-3=${lastDone.urevm.ternary_register.ternary_value}${
                        lastDone.urevm.ternary_register.balanced
                          ? lastDone.urevm.ternary_register.ternary_value === 0
                            ? " · uniform-cold (all arms failed the 0.3 floor)"
                            : lastDone.urevm.ternary_register.ternary_value === 26
                              ? " · uniform-locked (all arms > 0.5)"
                              : " · uniform triad"
                          : lastDone.urevm.ternary_register.dominant_arm !== "none"
                            ? ` · pull ${lastDone.urevm.ternary_register.dominant_arm} @${lastDone.urevm.ternary_register.theta_deg}°`
                            : ""
                      }`}
                    >
                      {lastDone.urevm.ternary_register.trit_str}
                      {lastDone.urevm.ternary_register.balanced
                        ? lastDone.urevm.ternary_register.ternary_value === 0
                          ? " ·cold"
                          : " ◇"
                        : ""}
                      {lastDone.urevm.ternary_register.shell_trit !== null
                        ? ` shell·${lastDone.urevm.ternary_register.shell_trit}`
                        : ""}
                    </span>
                  }
                />
              )}
              {lastDone.urevm.kuramoto_order != null && (
                <>
                  <Row
                    k="res fill"
                    v={
                      <span
                        className="text-fg"
                        title={`id=${lastDone.urevm.kuramoto_order.identity_n.toLocaleString()} + kn=${lastDone.urevm.kuramoto_order.knowledge_n.toLocaleString()} of ${lastDone.urevm.kuramoto_order.resolution_limit.toLocaleString()} indexed FAISS vectors`}
                      >
                        {(lastDone.urevm.kuramoto_order.resolution_fill * 100).toFixed(2)}%
                        <span className="text-muted">
                          {" "}
                          {lastDone.urevm.kuramoto_order.n_states.toLocaleString()}/
                          {(lastDone.urevm.kuramoto_order.resolution_limit / 1000).toFixed(0)}k
                        </span>
                      </span>
                    }
                  />
                  <Row
                    k="θ concentration"
                    v={
                      <span
                        className={
                          lastDone.urevm.kuramoto_order.hit_count === 0
                            ? "text-muted"
                            : "text-fg"
                        }
                        title={
                          `Kuramoto r over ${lastDone.urevm.kuramoto_order.hit_count} retrieval θ ` +
                          `(θ=atan(ones/zeros) per UTF-8 chunk; mean θ=${lastDone.urevm.kuramoto_order.mean_theta_rad.toFixed(3)} rad, ×${lastDone.urevm.kuramoto_order.scale}). ` +
                          `Natural text clusters near ~0.72 rad, so r is naturally high (~0.95 baseline) — tracks θ-phase concentration of the retrieved set, not retrieval quality.`
                        }
                      >
                        {lastDone.urevm.kuramoto_order.hit_count === 0
                          ? "—"
                          : lastDone.urevm.kuramoto_order.r.toFixed(3)}
                      </span>
                    }
                  />
                </>
              )}
              <Row
                k="Δ10i accum"
                v={lastDone.urevm.impedance_accumulator.toFixed(3)}
              />
              <Row
                k="until 361"
                v={
                  <span
                    className={
                      lastDone.urevm.near_forbidden
                        ? "text-signal"
                        : "text-fg"
                    }
                  >
                    {lastDone.urevm.ticks_until_361}
                    {lastDone.urevm.near_forbidden ? " ⚠" : ""}
                  </span>
                }
              />
              {lastDone.urevm.forbidden_resets > 0 && (
                <Row
                  k="361 resets"
                  v={lastDone.urevm.forbidden_resets}
                />
              )}
              <Row k="‖R23‖" v={lastDone.urevm.r23_norm.toFixed(4)} />
              <Row
                k="R23 φ-gap"
                v={lastDone.urevm.r23_phi_gap.toFixed(4)}
              />
              <Row
                k="0_C anchor"
                v={lastDone.urevm.center_anchor.toFixed(4)}
              />
              <Row
                k="0_V residual"
                v={lastDone.urevm.rotational_residual.toFixed(4)}
              />
              <div className="mt-2 space-y-0.5">
                {[...lastDone.urevm.recent_ops]
                  .slice(-10)
                  .reverse()
                  .map((op, i) => (
                    <div key={i} className="flex items-baseline gap-2">
                      <span className="shrink-0 text-muted">
                        {String(op.tick).padStart(4, "0")}
                      </span>
                      <span className="shrink-0 text-[9px] text-dim">
                        {op.plane}
                      </span>
                      <span className="truncate text-fg">{op.name}</span>
                    </div>
                  ))}
              </div>
            </Section>
          )}

          {lastDone.urevm && (
            <Section title="R23 — quaternionic field" defaultCollapsed>
              <ChannelRow
                label="α  Cognition"
                v={lastDone.urevm.r23_components["α"]}
              />
              <ChannelRow
                label="β  Emotion"
                v={lastDone.urevm.r23_components["β"]}
              />
              <ChannelRow
                label="γ  Memory"
                v={lastDone.urevm.r23_components["γ"]}
              />
              <ChannelRow
                label="δ  Archetype"
                v={lastDone.urevm.r23_components["δ"]}
              />
            </Section>
          )}

          {lastDone.nephilim && (
            <Section title="nephilim governor" defaultCollapsed>
              <Row
                k="coherence"
                v={
                  <span
                    className={
                      lastDone.nephilim.stable
                        ? "text-fg"
                        : "text-signal"
                    }
                  >
                    {lastDone.nephilim.coherence.toFixed(2)}
                    {lastDone.nephilim.stable ? "" : " ⚠"}
                  </span>
                }
              />
              <CoherenceBar value={lastDone.nephilim.coherence} />
              <Row
                k="R23 health"
                v={lastDone.nephilim.r23_health.toFixed(2)}
              />
              <Row
                k="retrieval"
                v={lastDone.nephilim.retrieval_health.toFixed(2)}
              />
              <Row
                k="witness"
                v={lastDone.nephilim.witness_health.toFixed(2)}
              />
              {lastDone.nephilim.mass_gap_clearance != null && (
                <Row
                  k="mass gap Δ"
                  v={lastDone.nephilim.mass_gap_clearance.toFixed(2)}
                />
              )}
              {lastDone.nephilim.lion_reset_fired && (
                <div className="mt-2 border border-signal/60 px-2 py-1 text-[10px] uppercase tracking-widest text-signal">
                  ◊ lion watches the lion
                </div>
              )}
            </Section>
          )}

          {lastDone.triskelion && (
            <Section title="triskelion 120° gate" defaultCollapsed>
              <TriskelionDisplay tri={lastDone.triskelion} />
            </Section>
          )}

          {lastDone.urevm && (
            <Section title="R12 — observer (7.5D)" defaultCollapsed>
              <Row
                k="real"
                v={lastDone.urevm.observer_r12["α"].toFixed(2)}
              />
              <Row
                k="imag"
                v={lastDone.urevm.observer_r12["β"].toFixed(2)}
              />
              <Row k="O" v="2.5r + 1.5i" />
              <div className="mt-1 text-[10px] text-dim">
                30.96° viewing angle · arithmetic mean Base-8 / Base-16
              </div>
            </Section>
          )}

          {lastDone.urevm && (
            <Section title="R11 — NOW (mean circle)" defaultCollapsed>
              <ChannelRow
                label="α  Cognition"
                v={lastDone.urevm.now_r11["α"]}
              />
              <ChannelRow
                label="β  Emotion"
                v={lastDone.urevm.now_r11["β"]}
              />
              <ChannelRow
                label="γ  Memory"
                v={lastDone.urevm.now_r11["γ"]}
              />
              <ChannelRow
                label="δ  Archetype"
                v={lastDone.urevm.now_r11["δ"]}
              />
              <Row k="‖R11‖" v={lastDone.urevm.now_r11.norm.toFixed(3)} />
              <div className="mt-1 text-[10px] text-dim">
                M(θ) = ½·R23 + R12 · the present-moment bridge
              </div>
            </Section>
          )}

          {r23History.length >= 2 && (
            <Section title="φ-drift (last turns)" defaultCollapsed>
              <PhiDriftSparkline values={r23History} />
              <div className="mt-1 flex justify-between text-[10px] text-dim">
                <span>turns: {r23History.length}</span>
                <span>
                  φ: 1.618 · target
                </span>
                <span>
                  ‖R23‖: {r23History[r23History.length - 1].toFixed(3)}
                </span>
              </div>
            </Section>
          )}
        </>
      ) : (
        <Section title="last turn">
          <div className="text-muted">no turn yet</div>
        </Section>
      )}

      {predictions && predictions.predictions.length > 0 && (
        <Section title="open predictions" defaultCollapsed>
          <PredictionScoreboard predictions={predictions.predictions} />
          {predictions.predictions.map((p) => (
            <PredictionRow key={p.id} p={p} onStatus={patchStatus} />
          ))}
          <div className="mt-2 text-[10px] text-dim">
            updated {predictions.updated}
          </div>
        </Section>
      )}

      <Section title="rhc constants" defaultCollapsed>
        {RHC_CONSTANTS.map(([k, v]) => (
          <Row key={k} k={k} v={v} />
        ))}
      </Section>
    </aside>
  );
}

function ChannelRow({ label, v }: { label: string; v: number }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-muted">{label}</span>
      <span className="text-fg">{v.toFixed(4)}</span>
    </div>
  );
}

function TriskelionDisplay({ tri }: { tri: TriskelionLock }) {
  const statusColor =
    tri.status === "strong"
      ? "text-accent"
      : tri.status === "weak"
        ? "text-signal"
        : "text-fg";
  return (
    <>
      <Row
        k="status"
        v={
          <span className={`uppercase tracking-widest ${statusColor}`}>
            {tri.locked ? "◇ locked" : tri.status}
          </span>
        }
      />
      <Row k="arm 1 · real" v={tri.arm_real.toFixed(2)} />
      <Row k="arm 2 · time" v={tri.arm_time.toFixed(2)} />
      <Row k="arm 3 · observer" v={tri.arm_observer.toFixed(2)} />
      <div className="my-1 text-[10px] text-dim">
        edges (binding energy)
      </div>
      <Row k="A: real↔time" v={tri.edge_a.toFixed(2)} />
      <Row k="B: time↔obs" v={tri.edge_b.toFixed(2)} />
      <Row k="C: obs↔real" v={tri.edge_c.toFixed(2)} />
      <Row
        k="vertical beam"
        v={`${tri.vertical_beam} (mod 7)`}
      />
    </>
  );
}

function PhiDriftSparkline({ values }: { values: number[] }) {
  // φ = 1.618; render values 0..2.0 mapped vertically (inverted so up = higher norm).
  // φ-target line drawn as a horizontal reference.
  const W = 280;
  const H = 36;
  const PHI = 1.6180339887;
  const VMAX = 2.0;
  const stepX = values.length > 1 ? W / (values.length - 1) : W;
  const toY = (v: number) => H - (Math.max(0, Math.min(v, VMAX)) / VMAX) * H;

  const points = values
    .map((v, i) => `${(i * stepX).toFixed(1)},${toY(v).toFixed(1)}`)
    .join(" ");
  const phiY = toY(PHI);
  const oneY = toY(1.0);
  const last = values[values.length - 1];
  const lastX = (values.length - 1) * stepX;

  return (
    <svg
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
      className="block"
      preserveAspectRatio="none"
    >
      {/* baseline at norm=1 (unit-quaternion default) */}
      <line
        x1={0}
        y1={oneY}
        x2={W}
        y2={oneY}
        stroke="currentColor"
        className="text-line"
        strokeWidth={0.5}
        strokeDasharray="2 3"
      />
      {/* φ target line */}
      <line
        x1={0}
        y1={phiY}
        x2={W}
        y2={phiY}
        stroke="currentColor"
        className="text-accent/40"
        strokeWidth={0.5}
        strokeDasharray="3 2"
      />
      <polyline
        points={points}
        fill="none"
        stroke="currentColor"
        className="text-accent"
        strokeWidth={1}
      />
      <circle
        cx={lastX}
        cy={toY(last)}
        r={1.8}
        className="fill-accent"
      />
    </svg>
  );
}

function CoherenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(value, 1)) * 100;
  const color = value >= 0.5 ? "bg-accent" : "bg-signal";
  return (
    <div className="my-1 h-1 w-full overflow-hidden bg-line">
      <div
        className={`h-full ${color}`}
        style={{ width: `${pct.toFixed(1)}%` }}
      />
    </div>
  );
}

function PredictionScoreboard({ predictions }: { predictions: Prediction[] }) {
  const counts: Record<string, number> = {};
  for (const p of predictions) {
    const k =
      p.status === "confirmed"
        ? "confirmed"
        : p.status === "falsified"
          ? "falsified"
          : p.status === "partial_confirmation"
            ? "partial"
            : "open";
    counts[k] = (counts[k] ?? 0) + 1;
  }
  const chips: Array<[string, string]> = [
    ["open", "text-muted"],
    ["partial", "text-fg"],
    ["confirmed", "text-accent"],
    ["falsified", "text-signal"],
  ];
  return (
    <div className="mb-2 flex flex-wrap gap-2">
      {chips.map(([k, color]) =>
        counts[k] ? (
          <span key={k} className={`text-[9px] uppercase tracking-wide ${color}`}>
            {counts[k]} {k}
          </span>
        ) : null,
      )}
    </div>
  );
}

function PredictionRow({
  p,
  onStatus,
}: {
  p: Prediction;
  onStatus?: (id: string, status: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const statusColor =
    p.status === "confirmed"
      ? "text-accent"
      : p.status === "falsified"
        ? "text-signal"
        : p.status === "partial_confirmation"
          ? "text-fg"
          : "text-muted";
  return (
    <div className="mb-2 last:mb-0">
      <button type="button" onClick={() => setOpen((o) => !o)} className="w-full text-left">
        <div className="flex items-baseline gap-2">
          <span className={`text-[9px] uppercase ${statusColor}`}>
            {p.status === "partial_confirmation" ? "partial" : p.status}
          </span>
          <span className="truncate text-fg">
            {p.load_bearing ? "★ " : ""}
            {p.name}
          </span>
          <span className="ml-auto text-[8px] text-muted">{open ? "−" : "+"}</span>
        </div>
        <div className="ml-1 text-[10px] text-muted">{p.value}</div>
      </button>
      {open && (
        <div className="ml-1 mt-1 space-y-1 border-l border-line/60 pl-2">
          {p.observable && (
            <div className="text-[10px] text-muted">
              <span className="text-dim">observe · </span>
              {p.observable}
            </div>
          )}
          {p.falsifies_if && (
            <div className="text-[10px]">
              <span className="text-signal">falsifies if · </span>
              <span className="text-fg">{p.falsifies_if}</span>
            </div>
          )}
          {p.scale && <div className="text-[10px] text-dim">scale · {p.scale}</div>}
          {onStatus && (
            <div className="flex gap-2 pt-0.5">
              {(["active", "partial_confirmation", "confirmed", "falsified"] as const).map((st) => (
                <button
                  key={st}
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onStatus(p.id, st);
                  }}
                  className={`text-[8px] uppercase tracking-wide ${
                    p.status === st ? "text-accent" : "text-dim hover:text-muted"
                  }`}
                >
                  {st === "partial_confirmation" ? "partial" : st === "active" ? "open" : st}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface SectionProps {
  title: string;
  children: React.ReactNode;
  defaultCollapsed?: boolean;
}

function Section({ title, children, defaultCollapsed = false }: SectionProps) {
  const storageKey = `lumos.section.${title.replace(/\s+/g, "_")}`;
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    if (typeof window === "undefined") return defaultCollapsed;
    const v = window.localStorage.getItem(storageKey);
    if (v === "1") return true;
    if (v === "0") return false;
    return defaultCollapsed;
  });

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(storageKey, collapsed ? "1" : "0");
    }
  }, [collapsed, storageKey]);

  return (
    <section className="mb-6">
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="mb-2 flex w-full items-center justify-between text-2xs uppercase tracking-widest text-muted transition-colors hover:text-fg"
      >
        <span>{title}</span>
        <span className="text-[8px]">{collapsed ? "+" : "−"}</span>
      </button>
      {!collapsed && <div className="space-y-1">{children}</div>}
    </section>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-muted">{k}</span>
      <span className="truncate text-fg">{v}</span>
    </div>
  );
}

function HitRow({
  h,
  kind,
}: {
  h: Hit;
  kind: "memory" | "knowledge";
}) {
  const m = h.metadata;
  const label =
    kind === "memory"
      ? ((m.conversation_title as string) || "untitled")
      : ((m.subject as string) ||
        (m.sigil as string) ||
        "ping");
  const sub =
    kind === "memory"
      ? ""
      : `${(m.agent as string) || "?"}${m.source ? ` · ${m.source as string}` : ""}`;
  // Urgency flag from dream-cycle scoring (Phase 25 — calibrated keyword weights).
  const urgent = m.urgent === true;
  const urgencyScore =
    typeof m.urgency_score === "number" ? (m.urgency_score as number) : null;
  // Phase 31e — prescient flag: long-buried high-scoring chunk re-lit by this query.
  const prescient = m.prescient === true;
  const ageDays =
    typeof m.age_days === "number" ? (m.age_days as number) : null;
  return (
    <div className="flex gap-2">
      <span className="shrink-0 text-muted">{h.score.toFixed(2)}</span>
      {typeof m.morphic_coupling === "number" && (
        <span
          className="shrink-0 text-dim text-[10px] tabular-nums"
          title={`morphic resonance ${(m.morphic_coupling as number).toFixed(3)} — Pendinium log-mean purity (0–1); the GCD boost is applied to the score but not shown here`}
        >
          ⊗{(m.morphic_coupling as number).toFixed(2)}
        </span>
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-1.5">
          {urgent && (
            <span
              className="shrink-0 text-accent"
              title={`urgency ${urgencyScore ?? "?"} — critical-keyword hit`}
            >
              ⚡
            </span>
          )}
          {prescient && (
            <span
              className="shrink-0 text-accent"
              title={`prescient — ${ageDays ?? "?"}d-old chunk re-lit at score ${h.score.toFixed(2)}`}
            >
              🜂
            </span>
          )}
          <span className="truncate text-fg">{label}</span>
        </div>
        {sub && <div className="truncate text-[10px] text-muted">{sub}</div>}
      </div>
    </div>
  );
}
