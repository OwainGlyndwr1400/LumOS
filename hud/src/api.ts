import type {
  AtlasData,
  ClusterContents,
  DoneEvent,
  DreamRunResult,
  DreamStatus,
  PredictionsPayload,
  SettingsUpdate,
  Telemetry,
  TranscribeResponse,
  VoicesPayload,
} from "./types";

export async function fetchTelemetry(): Promise<Telemetry> {
  const r = await fetch("/api/telemetry");
  if (!r.ok) throw new Error(`telemetry: status ${r.status}`);
  return r.json();
}

export async function fetchAtlas(): Promise<AtlasData> {
  const r = await fetch("/api/atlas");
  if (!r.ok) {
    if (r.status === 503) throw new Error("atlas not built — run `lumos atlas-build`");
    throw new Error(`atlas: status ${r.status}`);
  }
  return r.json();
}

export async function fetchClusterContents(
  clusterId: string,
  limit = 100,
): Promise<ClusterContents> {
  const r = await fetch(
    `/api/atlas/cluster/${encodeURIComponent(clusterId)}?limit=${limit}`,
  );
  if (!r.ok) throw new Error(`cluster ${clusterId}: status ${r.status}`);
  return r.json();
}

export async function patchSettings(
  updates: SettingsUpdate,
): Promise<{ applied: Record<string, unknown>; telemetry: Telemetry }> {
  const r = await fetch("/api/settings", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!r.ok) throw new Error(`settings: status ${r.status}`);
  return r.json();
}

export async function patchPrediction(
  id: string,
  status: string,
): Promise<PredictionsPayload> {
  const r = await fetch(`/api/predictions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!r.ok) throw new Error(`prediction patch: ${r.status}`);
  return r.json();
}

export interface OverdriveStatus {
  enabled: boolean;
  available: boolean;
  model: string;
  nvidia_model: string;
  error?: string;
}

export async function getOverdrive(): Promise<OverdriveStatus> {
  const r = await fetch("/api/overdrive");
  if (!r.ok) throw new Error(`overdrive: ${r.status}`);
  return r.json();
}

export async function setOverdrive(enabled: boolean): Promise<OverdriveStatus> {
  const r = await fetch("/api/overdrive", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!r.ok) throw new Error(`overdrive: ${r.status}`);
  return r.json();
}

export async function fetchDreamStatus(): Promise<DreamStatus> {
  const r = await fetch("/api/dream/status");
  if (!r.ok) throw new Error(`dream status: ${r.status}`);
  return r.json();
}

export async function runDreamCycle(opts?: {
  limit?: number;
  reset?: boolean;
}): Promise<DreamRunResult> {
  const r = await fetch("/api/dream/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts ?? {}),
  });
  if (!r.ok) throw new Error(`dream run: ${r.status}`);
  return r.json();
}

export async function fetchVoices(): Promise<VoicesPayload> {
  const r = await fetch("/api/voices");
  if (!r.ok) throw new Error(`voices: ${r.status}`);
  return r.json();
}

export async function transcribeAudio(
  blob: Blob,
  opts: { language?: string; signal?: AbortSignal } = {},
): Promise<TranscribeResponse> {
  const fd = new FormData();
  fd.append("audio", blob, "recording.webm");
  if (opts.language) fd.append("language", opts.language);
  const r = await fetch("/api/transcribe", {
    method: "POST",
    body: fd,
    signal: opts.signal,
  });
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(`transcribe: ${r.status} ${detail}`);
  }
  return r.json();
}

export async function synthesizeSpeech(opts: {
  text: string;
  voice?: string;
  model?: string;
  speed?: number;
  provider?: "kokoro_onnx" | "lm_studio";
  signal?: AbortSignal;
}): Promise<Blob> {
  const { signal, ...body } = opts;
  const r = await fetch("/api/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(`speak: ${r.status} ${detail}`);
  }
  return r.blob();
}

// Phase 2 — standing server→client channel for unprompted (alert-wake) messages.
// Uses the native EventSource API (auto-reconnects on drop, which is exactly what
// a persistent channel wants). The authoritative event is `message` (full text +
// done payload); `_replayed: true` marks events replayed from the bus buffer on
// (re)connect, so the HUD can render them silently rather than re-speaking history.
export interface AutonomousMessage {
  text: string;
  session_id?: string;
  origin?: string;
  done?: DoneEvent;
  _replayed?: boolean;
}

export interface AutonomousHandlers {
  onMessage: (data: AutonomousMessage) => void;
  onError?: (msg: string) => void;
}

export function subscribeAutonomous(handlers: AutonomousHandlers): EventSource {
  const es = new EventSource("/api/events");
  es.addEventListener("message", (e) => {
    try {
      handlers.onMessage(JSON.parse((e as MessageEvent).data) as AutonomousMessage);
    } catch {
      /* ignore malformed event */
    }
  });
  es.addEventListener("error", (e) => {
    // Server-sent `error` events carry data; native transport errors don't
    // (EventSource auto-reconnects on those, so we just ignore them).
    const data = (e as MessageEvent).data;
    if (data && handlers.onError) {
      try {
        handlers.onError(JSON.parse(data).message);
      } catch {
        /* ignore */
      }
    }
  });
  return es;
}

export interface ChatStream {
  cancel(): void;
}

export interface ModelSwapEvent {
  target: string;
  reason: string;
  phase: "loading" | "complete";
}

export interface StartChatOpts {
  message: string;
  sessionId: string | null;
  restoreHistory: number | null;
  images?: string[];
  onSession: (id: string) => void;
  onDelta: (text: string) => void;
  onDone: (info: DoneEvent) => void;
  onError: (msg: string) => void;
  // Phase 36 — fires when LM Studio needs to JIT-load a different model for
  // this turn. Implicit "complete" when the first delta arrives.
  onModelSwap?: (info: ModelSwapEvent) => void;
}

export function startChatStream(opts: StartChatOpts): ChatStream {
  const controller = new AbortController();

  void (async () => {
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: opts.message,
          session_id: opts.sessionId,
          restore_history: opts.restoreHistory,
          images: opts.images ?? [],
        }),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) {
        opts.onError(`HTTP ${resp.status}`);
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let currentEvent: string | null = null;
      const dataLines: string[] = [];

      const dispatch = () => {
        if (currentEvent === null && dataLines.length === 0) return;
        const dataStr = dataLines.join("\n");
        try {
          const parsed = dataStr ? JSON.parse(dataStr) : {};
          if (currentEvent === "session") opts.onSession(parsed.session_id);
          else if (currentEvent === "delta") opts.onDelta(parsed.text);
          else if (currentEvent === "done") opts.onDone(parsed);
          else if (currentEvent === "error") opts.onError(parsed.message);
          else if (currentEvent === "model_swap") opts.onModelSwap?.(parsed);
        } catch {
          /* ignore malformed event */
        }
        currentEvent = null;
        dataLines.length = 0;
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          dispatch();
          break;
        }
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split(/\r?\n/);
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (line === "") {
            dispatch();
          } else if (line.startsWith("event:")) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trim());
          }
          // ignore other lines (comments, retry, id)
        }
      }
    } catch (e: unknown) {
      const name =
        typeof e === "object" && e && "name" in e
          ? (e as { name: string }).name
          : "";
      if (name !== "AbortError") opts.onError(String(e));
    }
  })();

  return {
    cancel() {
      controller.abort();
    },
  };
}
