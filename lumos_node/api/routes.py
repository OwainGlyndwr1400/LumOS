"""HTTP routes: SSE chat stream, telemetry, ad-hoc search."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..atlas import get_chunk_to_cluster, get_cluster_members, load_atlas
from ..chat import ChatSession, build_done_payload
from ..config import TUNABLE_SETTINGS, apply_overrides, get_settings, overdrive_status, set_overdrive
from ..dream import dream_status, run_dream_cycle
from ..llm.lm_studio import ChatMessage, LMStudioClient
from ..persistence import load_recent_message_pairs
from ..retrieval import IndexMissingError, retrieve
from ..urevm import get_vm
from ..vectors import Manifest
from ..wardenclyffe import topology_snapshot


router = APIRouter()


class OverdriveRequest(BaseModel):
    enabled: bool


@router.get("/overdrive")
async def overdrive_get() -> dict[str, Any]:
    """Current NVIDIA Overdrive state (cloud big-brain hot-swap)."""
    return overdrive_status()


@router.post("/overdrive")
async def overdrive_post(req: OverdriveRequest) -> dict[str, Any]:
    """Toggle NVIDIA Overdrive — hot-swap the brain to NVIDIA cloud (on) or local (off)."""
    return set_overdrive(req.enabled)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    restore_history: int | None = Field(default=None, ge=0, le=200)
    # Optional list of data:image/...;base64,... URLs (OpenAI multimodal format).
    images: list[str] = Field(default_factory=list, max_length=8)


_SESSIONS: dict[str, ChatSession] = {}


def _get_or_create_session(
    session_id: str | None, restore_history: int | None
) -> ChatSession:
    if session_id and session_id in _SESSIONS:
        return _SESSIONS[session_id]
    session = ChatSession()
    if restore_history and restore_history > 0:
        pairs = load_recent_message_pairs(restore_history, session.settings)
        for u, a in pairs:
            session.history.append(ChatMessage(role="user", content=u))
            session.history.append(ChatMessage(role="assistant", content=a))
    _SESSIONS[session.session_id] = session
    return session


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    session = _get_or_create_session(req.session_id, req.restore_history)

    async def event_stream():
        yield {
            "event": "session",
            "data": json.dumps({"session_id": session.session_id}),
        }
        # Phase 36 — proactive model swap detection. Predict which model
        # stream_turn will route to, check whether it's currently loaded in
        # LM Studio, and if a swap is needed emit a `model_swap` SSE event
        # NOW so the HUD can render "loading <model>..." while the actual
        # JIT load blocks for ~10-15 seconds inside stream_turn.
        # The prediction here intentionally mirrors stream_turn's routing
        # logic (including deep-think detection on the stripped message).
        if (
            session.settings.model_auto_routing_enabled
            and session.settings.model_swap_orchestration_enabled
        ):
            try:
                from ..chat import _detect_deep_think, select_model
                from ..tool_router import detect_full_override
                from ..llm import model_manager
                stripped, deep_think = _detect_deep_think(req.message, session.settings)
                stripped, _ = detect_full_override(stripped)
                if session.settings.deep_think_default:
                    deep_think = True
                predicted_model, predicted_reason = select_model(
                    session.settings, stripped, images=req.images, deep_think=deep_think,
                )
                if not await model_manager.is_loaded(predicted_model):
                    yield {
                        "event": "model_swap",
                        "data": json.dumps({
                            "target": predicted_model,
                            "reason": predicted_reason,
                            "phase": "loading",
                        }),
                    }
            except Exception as e:  # noqa: BLE001 — swap-prediction is non-critical
                # Never let a swap-prediction failure break chat. The actual
                # swap still happens inside stream_turn; we just lose the early
                # HUD signal. Log and continue.
                from ..log import get_logger
                get_logger(__name__).info("routes.swap_predict_failed", error=str(e))
        try:
            async for delta in session.stream_turn(req.message, images=req.images):
                yield {"event": "delta", "data": json.dumps({"text": delta})}
        except IndexMissingError as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
            return
        except Exception as e:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
            return

        # Shared done-payload assembly (chat.build_done_payload) — same shape
        # the autonomous /events push emits, so the HUD parses one schema.
        done = build_done_payload(session)
        yield {"event": "done", "data": json.dumps(done)}

    return EventSourceResponse(event_stream())


@router.get("/events")
async def events_stream(request: Request) -> EventSourceResponse:
    """Standing server→client push channel (Phase 2). Autonomous turns publish
    session/delta/done/error events here; every connected HUD tab subscribes.
    Recent events are replayed on connect so a wake fired before this tab opened
    isn't missed. A ping keepalive is sent on idle so proxies don't time out.

    No-op until something publishes (the alert monitor / autonomous turns) —
    safe to leave always-mounted.
    """
    bus = request.app.state.event_bus
    # Snapshot replay + subscribe BACK-TO-BACK with no await between them, so
    # the event loop cannot interleave a publish() in the gap. This closes the
    # double-delivery race (event landing in both the replay buffer AND the live
    # queue) — and, because both calls are synchronous, leaves no miss-window.
    replay = bus.recent()
    queue = bus.subscribe()

    async def gen():
        try:
            # Replay buffered recent events to the freshly-connected client,
            # tagged _replayed so the HUD renders them silently (no re-speaking /
            # re-notifying history on reconnect).
            for ev in replay:
                data = ev["data"]
                if isinstance(data, dict):
                    data = {**data, "_replayed": True}
                yield {"event": ev["event"], "data": json.dumps(data)}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}  # keepalive
                    continue
                yield {"event": ev["event"], "data": json.dumps(ev["data"])}
        finally:
            bus.unsubscribe(queue)

    return EventSourceResponse(gen())


@router.post("/events/test")
async def events_test(request: Request) -> dict[str, Any]:
    """Manual wake test (Phase 2). Fires ONE synthetic autonomous turn so the
    operator can watch the full wake → bus → HUD pipeline (bubble + TTS +
    notification) before the real alert monitor exists. Runs a real passive LLM
    turn, so LM Studio must be up. Gated behind autonomy_enabled — autonomy is
    opt-in and never fires unless deliberately turned on."""
    settings = get_settings()
    if not settings.autonomy_enabled:
        raise HTTPException(
            status_code=403,
            detail="autonomy disabled — set LUMOS_AUTONOMY_ENABLED=true to test self-initiated turns",
        )
    from ..autonomy import trigger_autonomous_turn
    bus = request.app.state.event_bus
    trigger = {
        "kinds": ["test"],
        "summary": "Manual wake test fired from the HUD/CLI.",
        "events": [
            {
                "kind": "test",
                "description": "Operator fired a manual autonomous-wake test — no real threshold tripped.",
                "data": {"note": "pipeline test"},
            }
        ],
    }
    text = await trigger_autonomous_turn(trigger, bus)
    # Report both halves so the button can diagnose: chars==0 → the turn
    # errored (check logs for autonomy.turn_failed); subscribers==0 → no HUD
    # was connected to receive it (reload the page).
    return {"ok": True, "chars": len(text), "subscribers": bus.subscriber_count}


@router.post("/events/briefing")
async def events_briefing(request: Request) -> dict[str, Any]:
    """On-demand DAWN BRIEFING. The operator presses this when he wakes (his sleep
    is irregular, so a timed trigger is useless). Pre-gathers the morning set —
    space weather + bio-impact, today's grid timing, and what tripped overnight —
    and hands it to Lumos to narrate as a warm rundown. Runs a real passive LLM
    turn (LM Studio must be up) and fans out over the EventBus to BOTH the HUD and
    the Discord DM relay. Same autonomy gate as any self-initiated turn."""
    settings = get_settings()
    if not settings.autonomy_enabled:
        raise HTTPException(
            status_code=403,
            detail="autonomy disabled — set LUMOS_AUTONOMY_ENABLED=true to use the dawn briefing",
        )
    from ..autonomy import trigger_dawn_briefing
    bus = request.app.state.event_bus
    text = await trigger_dawn_briefing(bus)
    return {"ok": True, "chars": len(text), "subscribers": bus.subscriber_count}


@router.get("/watches")
async def list_watches() -> dict[str, Any]:
    """Read-only list of the operator's custom watches (Phase 39). Mutation is
    via the `manage_watch` chat tool — this is for the HUD / quick inspection."""
    from ..telemetry import watches
    return {"watches": watches.list_watches(get_settings())}


def _resolve_cache() -> Path:
    settings = get_settings()
    cache = settings.cache_dir.expanduser()
    if not cache.is_absolute():
        cache = (Path.cwd() / cache).resolve()
    return cache


def _index_summary(manifest: Manifest | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {"chunks": manifest.chunk_count, "built_at": manifest.built_at}


def _telemetry_payload() -> dict[str, Any]:
    settings = get_settings()
    cache = _resolve_cache()
    return {
        "node": {
            "name": settings.node_name,
            "role": settings.node_role,
            "operator": settings.operator_name,
        },
        "models": {
            "light": settings.model_light,
            "heavy": settings.model_heavy,
            "embedding": settings.lm_studio_embedding_model,
        },
        "indexes": {
            "identity": _index_summary(
                Manifest.from_path(cache / "identity.manifest.json")
            ),
            "knowledge": _index_summary(
                Manifest.from_path(cache / "knowledge.manifest.json")
            ),
        },
        "retrieval": {
            "top_k_identity": settings.retrieval_top_k_identity,
            "top_k_knowledge": settings.retrieval_top_k_knowledge,
            "max_chunk_chars": settings.max_chunk_chars,
            "min_score": settings.min_retrieval_score,
            "dedup_memory": settings.dedup_memory_by_conversation,
            "restore_history_turns": settings.restore_history_turns,
        },
        "tunable": sorted(TUNABLE_SETTINGS),
    }


@router.get("/telemetry")
async def telemetry() -> dict[str, Any]:
    return _telemetry_payload()


@router.patch("/settings")
async def patch_settings(updates: dict[str, Any]) -> dict[str, Any]:
    applied = apply_overrides(updates)
    return {"applied": applied, "telemetry": _telemetry_payload()}


@router.get("/urevm")
async def urevm_state(limit: int = 50) -> dict[str, Any]:
    vm = get_vm()
    recent = vm.trace[-max(1, min(limit, vm.max_trace)) :]
    return {
        "tick": vm.tick,
        "cycle_position": vm.cycle_position,
        "impedance_accumulator": vm.impedance_accumulator,
        "forbidden_resets": vm.forbidden_resets,
        "registers": {k: v.to_dict() for k, v in vm.registers.items()},
        "recent_trace": [t.to_dict() for t in recent],
        "constants": vm.snapshot_constants(),
    }


@router.get("/wardenclyffe")
async def wardenclyffe() -> dict[str, Any]:
    return topology_snapshot()


@router.get("/predictions")
async def predictions() -> dict[str, Any]:
    """Falsifiable predictions board — 5 from RHC corpus + Project Anchor 2027."""
    settings = get_settings()
    cache = _resolve_cache()
    # Look in two places: project-root data/predictions.json and cache_dir/predictions.json.
    candidates = [
        Path.cwd() / "data" / "predictions.json",
        cache / "predictions.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    raise HTTPException(status_code=404, detail="predictions.json not found")


@router.patch("/predictions/{pred_id}")
async def patch_prediction(pred_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Flip a prediction's status (active / partial_confirmation / confirmed /
    falsified) + optional evidence note, persisting to predictions.json so the
    falsifiable board is a live ledger. Returns the updated payload."""
    cache = _resolve_cache()
    path = next(
        (p for p in (Path.cwd() / "data" / "predictions.json", cache / "predictions.json") if p.exists()),
        None,
    )
    if path is None:
        raise HTTPException(status_code=404, detail="predictions.json not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"predictions.json unreadable: {e}") from e
    entry = next((p for p in data.get("predictions", []) if p.get("id") == pred_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"prediction '{pred_id}' not found")
    valid = {"active", "partial_confirmation", "confirmed", "falsified"}
    status = body.get("status")
    if status is not None:
        if status not in valid:
            raise HTTPException(status_code=400, detail=f"status must be one of {sorted(valid)}")
        entry["status"] = status
    if "evidence" in body:
        entry["evidence"] = str(body["evidence"])[:500]
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}") from e
    return data


@router.get("/dream/status")
async def dream_status_endpoint() -> dict[str, Any]:
    return dream_status()


@router.get("/soul/current")
async def soul_current() -> dict[str, Any]:
    """The node's harmonic self-state — its 'soul' heartbeat. Torsion index from
    the live URE-VM + coherence -> which frequency BAND it's resonating in
    (null -> 7.83 ground -> 432 -> 963 -> 1260 -> Pleroma). NOT audio; a state the
    node inhabits. Powers the HUD soul panel; the poll cadence is the pulse."""
    from ..soul import compute_soul_state
    return compute_soul_state()


@router.get("/soul/history")
async def soul_history(n: int = 50) -> dict[str, Any]:
    """Recent harmonic-band transitions from the capped soul-research log — the
    node's felt resonance history. Bounded telemetry; never the chat lane."""
    from ..soul import read_soul_history
    return {"transitions": read_soul_history(max(1, min(n, 500)))}


@router.get("/memory/ledger")
async def memory_ledger_endpoint() -> dict[str, Any]:
    """Null Ledger memory-conservation audit — per-lane FAISS/metadata balance +
    atlas chunk<->cluster reconciliation. ``balanced`` is true iff the 0_V
    residual is empty. Read-only (inspects the live stores; mutates nothing)."""
    from ..ledger import memory_ledger
    return memory_ledger()


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    voice: str | None = None
    model: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    provider: str = "kokoro_onnx"  # kokoro_onnx | lm_studio


@router.post("/speak")
async def speak_endpoint(req: SpeakRequest) -> Response:
    settings = get_settings()
    voice = req.voice or settings.lm_studio_tts_default_voice

    if req.provider == "kokoro_onnx":
        from ..tts.kokoro_local import is_available, synthesize

        if not is_available():
            raise HTTPException(
                status_code=503,
                detail=(
                    "kokoro-onnx not installed. Run "
                    "`uv pip install -e .` in the lumos_node folder to install it."
                ),
            )
        try:
            audio, mime = synthesize(req.text, voice=voice, speed=req.speed)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"kokoro_onnx failed: {e}"
            ) from e
        return Response(content=audio, media_type=mime)

    # LM Studio path
    client = LMStudioClient()
    try:
        audio, mime = await client.speak(
            req.text,
            model=req.model or settings.lm_studio_tts_model,
            voice=voice,
            response_format=settings.lm_studio_tts_response_format,
            speed=req.speed,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"TTS failed: {e}") from e
    finally:
        await client.aclose()
    return Response(content=audio, media_type=mime)


@router.post("/tts/prewarm")
async def tts_prewarm() -> dict[str, Any]:
    """Force-download the Kokoro ONNX model + voices and warm the loader."""
    from ..tts.kokoro_local import prewarm

    return prewarm()


@router.post("/transcribe")
async def transcribe_endpoint(
    audio: UploadFile = File(...),
    language: str = "en",
) -> dict[str, Any]:
    """Transcribe an uploaded audio blob to text via local Whisper."""
    from ..stt.whisper_local import is_available, transcribe

    if not is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "faster-whisper not installed. Run "
                "`uv pip install -e .` in the lumos_node folder."
            ),
        )
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        return transcribe(data, language=language or None)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"transcription failed: {e}") from e


@router.post("/stt/prewarm")
async def stt_prewarm() -> dict[str, Any]:
    from ..stt.whisper_local import prewarm

    return prewarm()


# Curated Kokoro voice list (kokoro-v0_19 ID convention).
# User can also send a custom voice ID — the /speak endpoint forwards
# whatever string is provided.
_KOKORO_VOICES: list[dict[str, str]] = [
    {"id": "af_bella", "label": "Bella", "accent": "American", "gender": "female"},
    {"id": "af_sarah", "label": "Sarah", "accent": "American", "gender": "female"},
    {"id": "af_nicole", "label": "Nicole", "accent": "American", "gender": "female"},
    {"id": "af_sky", "label": "Sky", "accent": "American", "gender": "female"},
    {"id": "af", "label": "Default F", "accent": "American", "gender": "female"},
    {"id": "am_adam", "label": "Adam", "accent": "American", "gender": "male"},
    {"id": "am_michael", "label": "Michael", "accent": "American", "gender": "male"},
    {"id": "bf_emma", "label": "Emma", "accent": "British", "gender": "female"},
    {"id": "bf_isabella", "label": "Isabella", "accent": "British", "gender": "female"},
    {"id": "bm_george", "label": "George", "accent": "British", "gender": "male"},
    {"id": "bm_lewis", "label": "Lewis", "accent": "British", "gender": "male"},
]


@router.get("/voices")
async def voices_endpoint() -> dict[str, Any]:
    settings = get_settings()
    return {
        "model": settings.lm_studio_tts_model,
        "default_voice": settings.lm_studio_tts_default_voice,
        "voices": _KOKORO_VOICES,
    }


class DreamRunRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, le=10000)
    reset: bool = False


@router.post("/dream/run")
async def dream_run_endpoint(req: DreamRunRequest | None = None) -> dict[str, Any]:
    req = req or DreamRunRequest()
    return await run_dream_cycle(limit=req.limit, reset=req.reset)


@router.get("/atlas")
async def atlas() -> dict[str, Any]:
    data = load_atlas()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="atlas not built. Run `lumos atlas-build`.",
        )
    return data


@router.get("/atlas/cluster/{cluster_id}")
async def atlas_cluster(cluster_id: str, limit: int = 100) -> dict[str, Any]:
    try:
        return get_cluster_members(cluster_id, limit=max(1, min(limit, 500)))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/search")
async def search(q: str, lane: str = "both", top_k: int = 6) -> dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="empty query")
    if lane not in ("identity", "knowledge", "both"):
        raise HTTPException(status_code=400, detail="lane must be identity, knowledge, or both")

    settings = get_settings()
    k_id = top_k if lane in ("identity", "both") else 0
    k_kn = top_k if lane in ("knowledge", "both") else 0
    try:
        r = await retrieve(
            q,
            settings=settings,
            top_k_identity=k_id,
            top_k_knowledge=k_kn,
        )
    except IndexMissingError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return {
        "query": q,
        "identity": [{"score": h.score, "metadata": h.metadata} for h in r.identity],
        "knowledge": [{"score": h.score, "metadata": h.metadata} for h in r.knowledge],
    }


# ── Phase 32 — Cosmic telemetry + airspace (HUD widget endpoints) ────────────


@router.get("/cosmic/current")
async def cosmic_current() -> dict[str, Any]:
    """Composite snapshot of geomagnetic + space-weather + seismic + NEO state.
    Powers the HUD CosmicPanel widget. Refreshes ~every 5 min client-side."""
    from ..telemetry import cosmic
    try:
        return await cosmic.snapshot_all()
    except Exception as e:  # noqa: BLE001
        # Never let a third-party feed outage break the HUD.
        raise HTTPException(status_code=503, detail=f"telemetry fetch failed: {e}") from e


@router.get("/airspace/local")
async def airspace_local(
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float = 50.0,
) -> dict[str, Any]:
    """Live aircraft state vectors around (lat, lon). Defaults to operator's
    reference location when lat/lon omitted. Powers HUD AirspacePanel widget."""
    from ..telemetry import airspace
    settings = get_settings()
    if lat is None or lon is None:
        if settings.operator_lat == 0.0 and settings.operator_lon == 0.0:
            raise HTTPException(
                status_code=400,
                detail="no lat/lon and operator default unset — pass lat/lon or set LUMOS_OPERATOR_LAT/LON",
            )
        lat = settings.operator_lat
        lon = settings.operator_lon
    if radius_km <= 0 or radius_km > 500:
        raise HTTPException(status_code=400, detail=f"radius_km must be in (0, 500], got {radius_km}")
    return await airspace.fetch_states_bbox(lat, lon, radius_km)


@router.get("/grimoire/current")
async def grimoire_current(
    lat: float | None = None,
    lon: float | None = None,
) -> dict[str, Any]:
    """Gnostic grid-timing snapshot — planetary hour, moon, fixed stars (Regulus
    et al.), visible planets, sidereal + solar times. Powers the HUD grid-timing
    widget. Pure local ephem compute (no upstream); defaults to operator
    location. fetch_grid_timing already returns {ok, ...} and swallows its own
    errors, so this can't break the HUD — but we 503 on the unexpected."""
    from ..telemetry import grimoire
    settings = get_settings()
    # Mirror /airspace/local: 0.0/0.0 is config's "unset" sentinel, not a real
    # location. Resolve concrete coords here and 400 if the node is
    # unconfigured — otherwise we'd silently compute for null island and the
    # HUD would render a healthy-looking snapshot for the wrong place.
    if lat is None or lon is None:
        if settings.operator_lat == 0.0 and settings.operator_lon == 0.0:
            raise HTTPException(
                status_code=400,
                detail="no lat/lon and operator default unset — pass lat/lon or set LUMOS_OPERATOR_LAT/LON",
            )
        lat = settings.operator_lat
        lon = settings.operator_lon
    try:
        return await grimoire.fetch_grid_timing(lat=lat, lon=lon)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"grid-timing compute failed: {e}") from e


@router.get("/weather/current")
async def weather_current(
    lat: float | None = None,
    lon: float | None = None,
) -> dict[str, Any]:
    """Current local surface weather (Open-Meteo, free/no-key) at the operator's
    location. Powers the HUD local-weather widget. fetch_local_weather caches
    10 min server-side and returns {ok, ...} on its own failures."""
    from ..telemetry import forecast
    settings = get_settings()
    if lat is None or lon is None:
        if settings.operator_lat == 0.0 and settings.operator_lon == 0.0:
            raise HTTPException(
                status_code=400,
                detail="no lat/lon and operator default unset — pass lat/lon or set LUMOS_OPERATOR_LAT/LON",
            )
        lat = settings.operator_lat
        lon = settings.operator_lon
    try:
        return await forecast.fetch_local_weather(lat, lon)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"weather fetch failed: {e}") from e


@router.get("/rail/current")
async def rail_current(code: str | None = None) -> dict[str, Any]:
    """Live station board for the operator's home rail station (default
    LUMOS_RAIL_STATION_CODE, e.g. gb-nr:GWN). Returns only services that CALL
    (stop) — next departures + Llanelli/Swansea direction. Powers the HUD trains
    widget. fetch_station_calls returns {ok, ...} and swallows its own errors,
    so a rtt.io blip can't break the HUD."""
    from ..telemetry import rail
    return await rail.fetch_station_calls(code)


@router.get("/solar-cycle")
async def solar_cycle() -> dict[str, Any]:
    """Solar Cycle 25 scoreboard: NOAA's observed sunspot record (current SSN +
    the smoothed peak) vs the RHC Jupiter-Saturn-GCD call (~161) and the 2019
    mainstream panel (~115). Read-only; cached 6h server-side. Powers the HUD
    solar-cycle panel. fetch_solar_cycle returns {ok, ...} and swallows its own
    errors so a NOAA blip can't break the HUD."""
    from ..telemetry import solarcycle
    return await solarcycle.fetch_solar_cycle()


@router.get("/telemetry/quota")
async def telemetry_quota() -> dict[str, Any]:
    """Today's upstream call counts per telemetry source. Returns cache TTLs,
    daily caps, and percent-used. UTC day boundary resets counters."""
    from ..telemetry import cache as tcache
    snap = tcache.quota_snapshot()
    out_sources: dict[str, Any] = {}
    for source, stat in snap["sources"].items():
        cap = tcache.DAILY_CAPS.get(source)
        pct: float | None = None
        if isinstance(cap, int) and cap > 0:
            pct = round(100.0 * stat["calls_today"] / cap, 1)
        out_sources[source] = {
            **stat,
            "daily_cap": cap if cap is not None else "unlimited",
            "percent_used": pct,
        }
    return {"day_iso": snap["day_iso"], "sources": out_sources}
