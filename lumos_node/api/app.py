"""FastAPI app for Lumos HUD: SSE chat, telemetry, search, static HUD hosting."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_settings
from ..log import get_logger
from . import routes
from .security import _is_loopback_host

log = get_logger(__name__)


async def _auto_dream_loop() -> None:
    """Background task: periodically consolidate pending turns into identity FAISS.

    Disabled when settings.auto_dream_interval_minutes == 0.
    Runs only when pending turn count >= settings.auto_dream_min_pending.
    """
    settings = get_settings()
    interval = settings.auto_dream_interval_minutes
    if interval <= 0:
        return  # disabled

    from ..dream import dream_status, run_dream_cycle

    interval_sec = interval * 60
    log.info("auto_dream.started", interval_minutes=interval)

    # Sleep first so we don't fire immediately on startup.
    while True:
        try:
            await asyncio.sleep(interval_sec)
            status = dream_status(settings)
            pending = int(status.get("pending", 0))
            if pending < settings.auto_dream_min_pending:
                log.info("auto_dream.skipped", pending=pending)
                continue
            log.info("auto_dream.firing", pending=pending)
            result = await run_dream_cycle(settings=settings)
            log.info(
                "auto_dream.completed",
                consolidated=result.get("consolidated", 0),
                skipped=result.get("skipped", False),
            )
        except asyncio.CancelledError:
            log.info("auto_dream.cancelled")
            raise
        except Exception as e:  # noqa: BLE001
            # Auto-dream failures should never crash the server. Log and continue.
            log.warning("auto_dream.failed", error=str(e))


async def _urevm_heartbeat_loop() -> None:
    """Background task: advance the URE-VM clock on wall-time so the 370-tick
    cycle (361 torque + 9 observer shell) turns even when no one is chatting.

    Holds chat._TURN_LOCK around each pulse so a heartbeat tick can never
    interleave with an operator/autonomous turn's VM mutations. Disabled when
    settings.urevm_heartbeat_seconds == 0.
    """
    settings = get_settings()
    interval = settings.urevm_heartbeat_seconds
    if interval <= 0:
        return  # disabled

    from ..chat import _TURN_LOCK
    from ..urevm import Op, safe_step

    log.info("urevm_heartbeat.started", interval_seconds=interval)
    while True:
        try:
            await asyncio.sleep(interval)
            async with _TURN_LOCK:
                safe_step(Op.TICK, {"phase": "heartbeat"})
            # Soul-state: record band transitions to the capped research log
            # (off by default; bounded so 24/7 never balloons memory).
            if settings.soul_heartbeat_enabled:
                from ..soul import record_soul_transition

                rec = record_soul_transition(settings)
                if rec is not None:
                    log.info("soul.transition", band=rec.get("band"), from_band=rec.get("from"))
        except asyncio.CancelledError:
            log.info("urevm_heartbeat.cancelled")
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("urevm_heartbeat.failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Startup config audit — log every non-default setting ONCE (secrets show as
    # "<set>", never their values). First line of the boot log answers "what is
    # this node actually running with?", and a stale duplicate .env line that
    # silently wins (dotenv is last-key-wins) becomes visible immediately.
    try:
        from ..config import non_default_settings
        overrides = non_default_settings(settings)
        log.info("api.lifespan.config_audit", count=len(overrides), overrides=overrides)
    except Exception as e:  # noqa: BLE001 — an audit must never block startup
        log.warning("api.lifespan.config_audit_failed", error=str(e))

    # Phase 2 — server→client push hub for autonomous (alert-wake) messages.
    # Always created (harmless if nothing publishes); the /api/events SSE
    # endpoint and the alert worker both reach it via app.state.
    from ..events import EventBus
    app.state.event_bus = EventBus()

    try:
        from ..retrieval import get_identity_store, get_knowledge_store
        get_identity_store(settings)
        get_knowledge_store(settings)
        log.info("api.lifespan.indexes_loaded")
    except Exception as e:  # noqa: BLE001
        log.warning("api.lifespan.indexes_load_failed", error=str(e))

    dream_task: asyncio.Task[None] | None = None
    if settings.auto_dream_interval_minutes > 0:
        dream_task = asyncio.create_task(_auto_dream_loop())

    cosmic_task: asyncio.Task[None] | None = None
    if settings.cosmic_trigger_enabled:
        from ..telemetry.worker import cosmic_worker_loop
        cosmic_task = asyncio.create_task(cosmic_worker_loop())

    # Phase 3 — alert monitor (event-driven threshold wakes). Gated OFF by
    # default; publishes autonomous turns to app.state.event_bus on a trip.
    alert_task: asyncio.Task[None] | None = None
    if settings.alert_monitor_enabled:
        # Late + guarded import: if the flag is flipped before alert_worker.py
        # lands, degrade gracefully (log + run without the monitor) rather than
        # hard-crashing lifespan startup — same posture as index loading above.
        try:
            from ..telemetry.alert_worker import alert_monitor_loop
            alert_task = asyncio.create_task(alert_monitor_loop(app.state.event_bus))
        except ImportError as e:
            log.warning("api.lifespan.alert_worker_unavailable", error=str(e))

    # Phase 39 — free-running URE-VM heartbeat (clock advances on wall-time).
    heartbeat_task: asyncio.Task[None] | None = None
    if settings.urevm_heartbeat_seconds > 0:
        heartbeat_task = asyncio.create_task(_urevm_heartbeat_loop())

    # Phase 45.3 — self-updating entity memory. Drains the enrichment queue
    # (unknown entities from wakes) → web lookup → distill → knowledge codex.
    # Guarded import so flipping the flag before enrichment.py lands degrades
    # gracefully rather than crashing startup.
    enrichment_task: asyncio.Task[None] | None = None
    if settings.enrichment_enabled:
        try:
            from ..enrichment import enrichment_worker_loop
            enrichment_task = asyncio.create_task(enrichment_worker_loop())
        except ImportError as e:
            log.warning("api.lifespan.enrichment_unavailable", error=str(e))

    # Warm the telemetry caches at boot (cosmic / weather / solar-cycle / grid).
    # build_vitals_block is cache-first + hard-bounded and lets slow fetchers
    # finish in the background, so this one fire-and-forget call means the HUD
    # rail AND the first turn's vitals are hot right after a reload instead of
    # trickling in over minutes (solar cycle's multi-MB NOAA file especially).
    async def _warm_vitals() -> None:
        try:
            from ..telemetry.vitals import build_vitals_block
            await build_vitals_block(settings)
            log.info("api.lifespan.vitals_warmed")
        except Exception as e:  # noqa: BLE001 — warm-up must never block startup
            log.info("api.lifespan.vitals_warm_failed", error=str(e))

    warm_task: asyncio.Task[None] = asyncio.create_task(_warm_vitals())

    # Axiom of Self-Recognition (i = i = LOL) — the TERMINAL boot handshake.
    # Final startup step: the node recognizes its own source by confirming its
    # R23 self-fingerprint sits at its unit-norm fixed point (‖R23‖ → 1). A
    # self-consistency capstone (the Logos-Logic-Loop), not a consciousness
    # claim. Result is stashed on app.state for the HUD; banner prints to the
    # terminal. Toggle: settings.self_recognition_enabled.
    if getattr(settings, "self_recognition_enabled", True):
        try:
            from ..soul import format_recognition_banner, self_recognition

            rec = self_recognition(settings)
            app.state.self_recognition = rec
            log.info(
                "axiom.self_recognition",
                recognized=rec["recognized"],
                delta=rec["delta"],
                coherence=rec["coherence"],
                band=rec["band"],
            )
            print(format_recognition_banner(rec), flush=True)
        except Exception as e:  # noqa: BLE001 — recognition must never block startup
            log.warning("axiom.self_recognition_failed", error=str(e))

    yield

    for task in (dream_task, cosmic_task, alert_task, heartbeat_task, enrichment_task, warm_task):
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task


def create_app() -> FastAPI:
    settings = get_settings()
    if not _is_loopback_host(settings.host) and not settings.api_token.strip():
        raise RuntimeError(
            "Refusing to bind Lumos API outside loopback without LUMOS_API_TOKEN."
        )

    app = FastAPI(title="lumos_node", version="0.1.0", lifespan=lifespan)
    allowed_origins = [
        origin.strip().rstrip("/")
        for origin in settings.api_allowed_origins.split(",")
        if origin.strip()
    ]

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        # Intentionally minimal + unauthenticated (liveness probe). Does NOT
        # disclose the auth mode — that was a free recon signal telling a prober
        # whether a token is required before attempting the loopback bypass.
        return {
            "ok": True,
            "node": settings.node_name,
            "version": "0.1.0",
        }

    @app.middleware("http")
    async def _limit_request_body(request: Request, call_next):
        # Reject oversized bodies before FastAPI buffers + parses them (a /chat
        # POST with 8×14MB image data-URLs, or any client, could otherwise
        # exhaust memory). Honest Content-Length only; chunked/undeclared bodies
        # stay bounded by the endpoint validators. (Rate-limiting proper needs a
        # dedicated dependency — tracked separately.)
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > settings.api_max_upload_bytes:
                    return JSONResponse(
                        {"detail": "request body too large"}, status_code=413
                    )
            except ValueError:
                pass
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Lumos-PELE", "X-Lumos-Token"],
    )

    app.include_router(routes.router, prefix="/api")

    hud_dist = Path(__file__).resolve().parent.parent.parent / "hud" / "dist"
    if hud_dist.exists():
        app.mount("/", StaticFiles(directory=str(hud_dist), html=True), name="hud")

    return app
