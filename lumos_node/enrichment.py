"""Self-updating entity memory (Phase 45.3) — close the google→RAG loop.

The operator's manual ritual: a wake names a new satellite / vessel / disaster,
they google it, paste the summary back, and next time the RAG already knows it.
This automates exactly that — WITHOUT breaking the "autonomy ends at speaking"
doctrine:

  • The WAKE never acts. When a wake fires, the alert worker (pure code, not the
    LLM turn) drops the tripped entities onto a queue. That's it — noting "look
    this up later", not looking it up.
  • A SEPARATE background worker (enrichment_worker_loop, like the cosmic
    worker) later pops the queue, does a web_search, distills it with the local
    brain into a few factual sentences, and appends it to entity_codex.jsonl.
  • The next ingest / dream consolidation folds the codex into the KNOWLEDGE
    lane, so the entity is now first-class memory — retrieved on the next pass.

So the autonomous turn still only observes + speaks; the outbound web lookup is
a deliberate, separately-gated, deny-by-default background chore the operator
switches on once (LUMOS_ENRICHMENT_ENABLED) instead of googling forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .config import Settings, get_settings
from .knowledge.dreams import KnowledgeChunk
from .log import get_logger
from .telemetry.worker import _chat_idle_seconds, _data_dir, _today_iso

log = get_logger(__name__)

_QUEUE_FILE = "queue.jsonl"
_SEEN_FILE = "seen.json"
_STATE_FILE = "worker_state.json"
_CODEX_FILE = "entity_codex.jsonl"
_SUBDIR = "enrichment"


@dataclass(frozen=True)
class EntityRef:
    """One enrichable entity pulled from a wake trip."""
    key: str          # dedup identity, e.g. "sat:25544"
    kind: str         # satellite | vessel | disaster
    name: str         # human name for the codex subject
    query: str        # the web_search query to run


def _enrich_dir(s: Settings) -> Path:
    d = _data_dir(s) / _SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def codex_path(s: Settings) -> Path:
    return _enrich_dir(s) / _CODEX_FILE


# ── Pure extraction: wake trips → enrichable entity refs ─────────────────────

def extract_entities(trips: list[dict[str, Any]]) -> list[EntityRef]:
    """PURE — pull enrichable entities from wake trips. Only durable, nameable
    objects worth a permanent codex entry (satellites / vessels / disasters);
    ephemeral kinds (aircraft, trains, weather) are skipped. Deduped by key."""
    out: list[EntityRef] = []
    seen: set[str] = set()
    for t in trips:
        data = t.get("data") or {}
        kind = t.get("kind")
        ref: EntityRef | None = None
        if kind in ("recon_satellite", "satellite_overhead"):
            norad = data.get("norad")
            name = data.get("name") or (f"NORAD {norad}" if norad else None)
            if name:
                key = f"sat:{norad or name}"
                country = data.get("country") or ""
                ref = EntityRef(key, "satellite", str(name),
                                f"{name} satellite {country} NORAD {norad or ''} purpose operator".strip())
        elif kind == "military_air":
            # Military aircraft are identifiable, durable airframes (like mil
            # sats) — worth a codex entry. Civilian 'aircraft' (private/jet/
            # commercial) stays skipped: ephemeral, not worth permanent memory.
            hexid = data.get("hex")
            cs = data.get("callsign")
            tc = data.get("type_code")
            name = cs or hexid
            if hexid or cs:
                ref = EntityRef(f"milair:{hexid or cs}", "military_aircraft", str(name),
                                f"{cs or ''} {tc or ''} military aircraft type operator unit".strip())
        elif kind == "vessel":
            mmsi = data.get("mmsi")
            name = data.get("name") or (f"MMSI {mmsi}" if mmsi else None)
            if mmsi or name:
                key = f"vessel:{mmsi or name}"
                ref = EntityRef(key, "vessel", str(name or f"MMSI {mmsi}"),
                                f"vessel {name or ''} MMSI {mmsi or ''} ship type flag".strip())
        elif kind == "disaster":
            eid = data.get("event_id")
            label = data.get("type_label") or "disaster"
            country = data.get("country") or ""
            title = data.get("title") or label
            if eid:
                ref = EntityRef(f"gdacs:{eid}", "disaster", str(title),
                                f"{label} {country} {title} disaster impact".strip())
        if ref and ref.key not in seen:
            seen.add(ref.key)
            out.append(ref)
    return out


# ── Queue + seen-set persistence ─────────────────────────────────────────────

def _read_seen(s: Settings) -> set[str]:
    p = _enrich_dir(s) / _SEEN_FILE
    if not p.exists():
        return set()
    try:
        data = orjson.loads(p.read_bytes())
        return set(data) if isinstance(data, list) else set()
    except (orjson.JSONDecodeError, OSError):
        return set()


def _write_seen(s: Settings, seen: set[str]) -> None:
    # Bound the seen-set so it can't grow without limit (keep most-recent).
    trimmed = list(seen)[-5000:]
    try:
        (_enrich_dir(s) / _SEEN_FILE).write_bytes(orjson.dumps(trimmed))
    except OSError as e:  # noqa: BLE001
        log.warning("enrichment.seen_write_failed", error=str(e))


def queue_entities(refs: list[EntityRef], settings: Settings | None = None) -> int:
    """Append NEW entity refs (not already seen) to the lookup queue. Marks them
    seen immediately so the same entity is never queued twice. Pure file I/O —
    safe to call from the alert worker's non-LLM code path. Returns #queued."""
    settings = settings or get_settings()
    if not refs:
        return 0
    seen = _read_seen(settings)
    fresh = [r for r in refs if r.key not in seen]
    if not fresh:
        return 0
    qpath = _enrich_dir(settings) / _QUEUE_FILE
    # Respect a max queue depth so a burst can't create an unbounded backlog.
    existing = 0
    if qpath.exists():
        try:
            existing = sum(1 for _ in qpath.open("rb"))
        except OSError:
            existing = 0
    room = max(0, settings.enrichment_max_queue - existing)
    to_add = fresh[:room]
    try:
        with qpath.open("ab") as f:
            for r in to_add:
                f.write(orjson.dumps({"key": r.key, "kind": r.kind, "name": r.name, "query": r.query}))
                f.write(b"\n")
    except OSError as e:  # noqa: BLE001
        log.warning("enrichment.queue_write_failed", error=str(e))
        return 0
    for r in fresh:            # mark ALL fresh seen (even if queue was full) so
        seen.add(r.key)        # we don't thrash re-queueing a dropped overflow
    _write_seen(settings, seen)
    if to_add:
        log.info("enrichment.queued", n=len(to_add), dropped=len(fresh) - len(to_add))
    return len(to_add)


def queue_from_trips(trips: list[dict[str, Any]], settings: Settings | None = None) -> int:
    """Convenience for the alert worker: extract + queue in one call."""
    return queue_entities(extract_entities(trips), settings)


def _pop_jobs(s: Settings, n: int) -> list[dict[str, Any]]:
    """Read up to n jobs from the head of the queue and rewrite the remainder."""
    qpath = _enrich_dir(s) / _QUEUE_FILE
    if not qpath.exists():
        return []
    try:
        lines = qpath.read_bytes().splitlines()
    except OSError:
        return []
    jobs, rest = lines[:n], lines[n:]
    try:
        qpath.write_bytes(b"\n".join(rest) + (b"\n" if rest else b""))
    except OSError as e:  # noqa: BLE001
        log.warning("enrichment.queue_rewrite_failed", error=str(e))
    out: list[dict[str, Any]] = []
    for ln in jobs:
        try:
            out.append(orjson.loads(ln))
        except orjson.JSONDecodeError:
            continue
    return out


def _requeue_jobs(s: Settings, jobs: list[dict[str, Any]]) -> None:
    """Append jobs back to the TAIL of the queue — used when a search backend fails
    transiently (e.g. rate-limited SearXNG), so the entity is retried later instead
    of being silently lost."""
    if not jobs:
        return
    qpath = _enrich_dir(s) / _QUEUE_FILE
    try:
        with qpath.open("ab") as f:
            for job in jobs:
                f.write(orjson.dumps(job))
                f.write(b"\n")
    except OSError as e:  # noqa: BLE001
        log.warning("enrichment.requeue_failed", error=str(e))


# ── Distillation + codex append ──────────────────────────────────────────────

def _format_results(results: list[dict[str, Any]], limit: int = 5) -> str:
    lines = []
    for r in results[:limit]:
        title = (r.get("title") or "").strip()
        snip = (r.get("snippet") or "").strip()
        if title or snip:
            lines.append(f"- {title}: {snip}")
    return "\n".join(lines)


_DISTILL_SYSTEM = (
    "You are an intelligence archivist. Given web search results about an entity, "
    "write 3-4 factual, self-contained sentences describing what it is: its type, "
    "operator/owner/country, purpose, and any notable facts. No preamble, no "
    "hedging, no 'according to' — just the distilled facts. If the results are "
    "unclear or empty, say so in one sentence."
)


async def _distill(
    ref: EntityRef, results: list[dict[str, Any]], model: str, max_tokens: int,
) -> str | None:
    """Distill search results into 3-4 factual sentences. Returns None when the
    model comes back empty (e.g. a thinking model burning the whole completion
    budget inside its reasoning block) — the caller must file NOTHING then."""
    from .llm.lm_studio import ChatMessage, LMStudioClient

    formatted = _format_results(results)
    if not formatted:
        return f"{ref.name}: no web information found (searched but results were empty)."
    client = LMStudioClient()
    try:
        msg = await client.chat(
            model,
            [
                ChatMessage(role="system", content=_DISTILL_SYSTEM),
                ChatMessage(role="user", content=f"Entity: {ref.name}\n\nSearch results:\n{formatted}"),
            ],
            temperature=0.2,
            # The operator's app-wide completion budget (LUMOS_AUTONOMOUS_MAX_TOKENS,
            # same as forge uses) — NOT a hardcoded cap. Thinking models spend
            # hundreds of tokens reasoning before the visible answer; a small
            # fixed cap made every distill come back empty.
            max_tokens=max_tokens,
        )
        content = str(msg.get("content") or "")
        # If a template change ever leaks the <think> block into content, keep
        # only the visible answer after it (same failure class as 2026-07-09).
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        return content.strip() or None
    finally:
        await client.aclose()


def _append_codex(s: Settings, ref: EntityRef, summary: str, sources: list[str]) -> None:
    entry = {
        "key": ref.key,
        "kind": ref.kind,
        "name": ref.name,
        "summary": summary,
        "sources": sources[:5],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        with codex_path(s).open("ab") as f:
            f.write(orjson.dumps(entry))
            f.write(b"\n")
    except OSError as e:  # noqa: BLE001
        log.warning("enrichment.codex_write_failed", error=str(e))


# ── Knowledge-lane bridge: entity_codex.jsonl → KnowledgeChunk ───────────────

def _entity_chunk(key: str, kind: str, name: str, summary: str) -> KnowledgeChunk:
    """Build the KnowledgeChunk for one enriched entity. Shared by the ingest
    bridge (rebuild path) and the live hot-append, so both produce byte-identical
    chunks (same chunk_id) — no duplication if an entity is later re-derived."""
    text = f"{name} — {summary}"
    cid = hashlib.sha256(f"{key}|{text}".encode()).hexdigest()[:16]
    return KnowledgeChunk(
        chunk_id=cid, ping_id=f"entity:{key}", sigil="", agent="entity_codex",
        urgency_score=0, urgency_weight=0,
        source=f"geosentinel_enrichment:{kind}", subject=name,
        seed=summary[:200], fragment_count=0, text=text,
    )


def iter_entity_codex_chunks(path: Path) -> Iterator[KnowledgeChunk]:
    """Yield KnowledgeChunk from entity_codex.jsonl so a full ingest rebuild
    folds enriched entities into the knowledge lane (the CANONICAL path). Day to
    day the worker hot-appends live; this is what a from-scratch rebuild reads."""
    if not path.exists():
        return
    try:
        raw = path.read_bytes().splitlines()
    except OSError:
        return
    for ln in raw:
        try:
            e = orjson.loads(ln)
        except orjson.JSONDecodeError:
            continue
        summary = (e.get("summary") or "").strip()
        name = (e.get("name") or "").strip()
        if not summary or not name:
            continue
        yield _entity_chunk(str(e.get("key") or name), str(e.get("kind", "entity")), name, summary)


async def _hot_append_knowledge(chunk: KnowledgeChunk, settings: Settings) -> bool:
    """Append ONE chunk to the live on-disk knowledge FAISS + metadata — NO full
    rebuild (a full re-embed of the lane: ~half an hour on today's ~60k-chunk
    corpus-only store, hours on the old 600k-chunk dream_pings one — either way
    absurd for one chunk). Mirrors fish.py's hot-append:
    embed the chunk, load the store, add, save. Retrievable at the next app
    start; entity_codex.jsonl remains the canonical copy a rebuild re-derives.
    Best-effort — a failure just means "waits for the next rebuild", never fatal.
    """
    try:
        from .ingest import KNOWLEDGE_INDEX, KNOWLEDGE_META
        from .llm.lm_studio import LMStudioClient
        from .vectors import VectorStore

        cache = settings.cache_dir.expanduser()
        if not cache.is_absolute():
            cache = (Path.cwd() / cache).resolve()
        idx_p, meta_p = cache / KNOWLEDGE_INDEX, cache / KNOWLEDGE_META
        if not (idx_p.exists() and meta_p.exists()):
            return False
        client = LMStudioClient()
        try:
            vecs = await client.embed([chunk.text], model=settings.lm_studio_embedding_model)
        finally:
            await client.aclose()
        # load+add+save rewrites the ENTIRE index (GBs on a large store), so run
        # it in a worker THREAD — doing it inline froze the event loop for
        # seconds per entity, dropping the AIS websocket and stalling HUD/SSE
        # (the socket.send() storm). Same fix class as forge_verify.
        def _load_add_save() -> None:
            store = VectorStore.load(idx_p, meta_p)
            store.add(vecs, [chunk.to_metadata()])
            store.save(idx_p, meta_p)

        await asyncio.to_thread(_load_add_save)
        return True
    except Exception as e:  # noqa: BLE001 — hot-append is best-effort; file is canonical
        log.warning("enrichment.hot_append_failed", error=str(e))
        return False


def codex_signature(s: Settings) -> tuple[int, float]:
    """(size, mtime) of the codex, folded into the ingest manifest so new
    enriched entities invalidate freshness and the next ingest picks them up."""
    p = codex_path(s)
    if not p.exists():
        return (0, 0.0)
    try:
        st = p.stat()
        return (st.st_size, st.st_mtime)
    except OSError:
        return (0, 0.0)


# ── Background worker ────────────────────────────────────────────────────────

def _read_state(s: Settings) -> dict[str, Any]:
    p = _enrich_dir(s) / _STATE_FILE
    base = {"day_iso": "", "done_today": 0}
    if not p.exists():
        return base
    try:
        st = orjson.loads(p.read_bytes())
        for k, v in base.items():
            st.setdefault(k, v)
        return st
    except (orjson.JSONDecodeError, OSError):
        return base


def _write_state(s: Settings, st: dict[str, Any]) -> None:
    with contextlib.suppress(OSError):
        (_enrich_dir(s) / _STATE_FILE).write_bytes(orjson.dumps(st))


async def _process_one(s: Settings, job: dict[str, Any], model: str) -> str:
    """Enrich one entity. Returns:
      'filed' — summarised + indexed (counts against the daily cap);
      'retry' — the SEARCH backend failed transiently (rate-limited/errored) —
                caller re-queues so we try again later once it recovers;
      'drop'  — bad job, or search worked but yielded nothing usable — discard."""
    from .tools.web_tools import web_search

    ref = EntityRef(
        key=str(job.get("key", "")), kind=str(job.get("kind", "entity")),
        name=str(job.get("name", "")), query=str(job.get("query", "")),
    )
    if not ref.query:
        return "drop"
    try:
        res = await web_search(ref.query, max_results=5)
    except Exception as e:  # noqa: BLE001
        log.info("enrichment.search_failed", key=ref.key, error=str(e))
        return "retry"
    if res.get("error"):
        # Every backend failed (e.g. SearXNG/DDG rate-limited/blocked). This is
        # transient — re-queue rather than lose the entity, and let the loop back off.
        log.info("enrichment.search_error", key=ref.key, error=str(res.get("error"))[:120])
        return "retry"
    results = res.get("results") or []
    summary = await _distill(ref, results, model, s.autonomous_max_tokens)
    if summary is None:
        # Search worked but nothing usable — file NOTHING (no codex/index junk).
        # Not transient, so don't retry forever; drop it.
        log.warning("enrichment.distill_empty", key=ref.key, name=ref.name)
        return "drop"
    sources = [r.get("url", "") for r in results if r.get("url")]
    _append_codex(s, ref, summary, sources)          # 1) canonical file (rebuild path)
    chunk = _entity_chunk(ref.key, ref.kind, ref.name, summary)
    appended = await _hot_append_knowledge(chunk, s)  # 2) live index — no re-ingest
    log.info("enrichment.filed", key=ref.key, name=ref.name, hot_appended=appended)
    return "filed"


async def enrichment_worker_loop() -> None:
    """Background loop — drains the enrichment queue, gated + capped + polite.
    No-op when disabled. Started from the API lifespan like the cosmic worker."""
    settings = get_settings()
    if not settings.enrichment_enabled:
        log.info("enrichment.disabled")
        return
    interval = max(30, settings.enrichment_poll_interval_seconds)
    max_backoff = 16.0
    backoff = 1.0  # grows when the search backend keeps failing; resets on any win
    log.info("enrichment.started", interval_s=interval, daily_cap=settings.enrichment_daily_cap)
    while True:
        try:
            await asyncio.sleep(interval * backoff)
            settings = get_settings()
            # Politeness gates: skip while the operator is actively chatting, and
            # respect the daily cap so a big backlog can't hammer web + brain.
            skip = settings.enrichment_skip_if_chat_active_minutes * 60
            if skip > 0 and _chat_idle_seconds(settings) < skip:
                continue
            state = _read_state(settings)
            today = _today_iso()
            if state.get("day_iso") != today:
                state = {"day_iso": today, "done_today": 0}
            if int(state.get("done_today", 0)) >= settings.enrichment_daily_cap:
                continue
            jobs = _pop_jobs(settings, settings.enrichment_batch_size)
            if not jobs:
                continue
            model = settings.model_light
            done = 0
            retry_jobs: list[dict[str, Any]] = []
            for job in jobs:
                status = await _process_one(settings, job, model)
                if status == "filed":
                    done += 1
                elif status == "retry":
                    retry_jobs.append(job)  # transient search failure — try later
                # "drop" → discard silently
            if retry_jobs:
                _requeue_jobs(settings, retry_jobs)
            # Back off when the WHOLE batch failed transiently (search backend
            # blocked) so we stop hammering it; reset the instant anything succeeds.
            # Only 'filed' jobs count against the daily cap, so failures are free.
            if done > 0:
                backoff = 1.0
            elif retry_jobs:
                backoff = min(max_backoff, backoff * 2)
                log.info(
                    "enrichment.backoff", multiplier=backoff,
                    next_wait_s=int(interval * backoff),
                )
            state["done_today"] = int(state.get("done_today", 0)) + done
            state["day_iso"] = today
            _write_state(settings, state)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a bad job must never kill the loop
            log.warning("enrichment.iter_failed", error=str(e))
