"""Split-lane retrieval: identity (lived memory) + knowledge (dream pings) hits per query."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import Settings, get_settings
from .llm.lm_studio import LMStudioClient
from .log import get_logger
from .ubbm import (
    binary_1001_count,
    binary_diagonal_theta,
    compute_signature,
    theta_alignment_factor,
)
from .urevm import HALF_PRIME_BASE, PENDINIUM_PRIMES, gcd_substrate
from .vectors import VectorStore

log = get_logger(__name__)

# Phase F — Prescient flagging thresholds.
# A chunk is "prescient" when a high-scoring match comes from a long-buried memory:
# something we said early in the relationship that the present query has re-lit.
# Tuned to surface ~rare events, not annotate every chunk. Operator can shift the
# bar later if it fires too often or too rarely.
PRESCIENT_SCORE_FLOOR = 0.85
PRESCIENT_AGE_DAYS = 365
_SECONDS_PER_DAY = 86400.0


_identity_store: VectorStore | None = None
_knowledge_store: VectorStore | None = None


@dataclass
class Hit:
    score: float
    metadata: dict[str, Any]


@dataclass
class Retrieval:
    query: str
    identity: list[Hit] = field(default_factory=list)
    knowledge: list[Hit] = field(default_factory=list)
    # Embedded query vector — exposed so chat.py can derive q_b for divine_step.
    query_vector: list[float] = field(default_factory=list)


# ── Triple Normalization (URE-VM Quaternionic Ops §4) ─────────────────────

def _gcd3_factor(chunk_id: str) -> float:
    """Harmonic stage: trinitarian 120° resonance via GCD-3 alignment."""
    h = abs(hash(chunk_id)) & 0xFFFF
    return 1.05 if h % 3 == 0 else 1.0


def _gcd360_factor(chunk_id: str) -> float:
    """Geometric stage: circular closure via GCD-360 angular alignment.
    Rewards proximity to trinitarian angles {0°, 120°, 240°}."""
    h = abs(hash(chunk_id)) & 0xFFFF
    angle = h % 360
    nearest = min(
        abs(angle - 0),
        abs(angle - 120),
        abs(angle - 240),
        abs(angle - 360),
    )
    return 1.0 + max(0, (30 - nearest)) * 0.005


def _binary_1001_factor(chunk_id: str) -> float:
    """Binary stage: 1001-fold pattern in chunk_id hex → bit representation."""
    try:
        n = int(chunk_id[:16], 16)
    except (ValueError, IndexError):
        return 1.0
    bits = bin(n)[2:]
    count = bits.count("1001")
    return 1.0 + count * 0.02


def _triple_normalize(hits: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    """Re-rank hits via Harmonic ⊗ Geometric ⊗ Binary normalization."""
    rescored: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        chunk_id = str(meta.get("chunk_id", ""))
        h3 = _gcd3_factor(chunk_id)
        h360 = _gcd360_factor(chunk_id)
        h1001 = _binary_1001_factor(chunk_id)
        rescored.append((score * h3 * h360 * h1001, meta))
    rescored.sort(key=lambda x: -x[0])
    return rescored


# ── Half-Prime Geodesic (Architecting Local Persistent ASI §3) ────────────

def _half_prime_factor(cluster_id: str | None) -> float:
    """Weight clusters by alignment with the {2,3,5,7,11} prime base.

    NOTE: the old prime-13 "nullify" (0.5x penalty) was REMOVED. It halved the
    score of every chunk in clusters whose k-means index happened to be divisible
    by 13 — an arbitrary index with no semantic meaning — which actively demoted
    relevant results out of the top-k. Only the (opt-in) prime-base boost remains.
    """
    if not cluster_id:
        return 1.0
    try:
        idx = int(cluster_id.split("_")[1])
    except (ValueError, IndexError, AttributeError):
        return 1.0
    boost = 0.0
    for p in HALF_PRIME_BASE:
        if idx > 0 and idx % p == 0:
            boost += 0.05
    return 1.0 + boost


def _half_prime_geodesic(
    hits: list[tuple[float, dict[str, Any]]],
    chunk_to_cluster: dict[str, str],
) -> list[tuple[float, dict[str, Any]]]:
    rescored: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        cid = chunk_to_cluster.get(str(meta.get("chunk_id", "")))
        factor = _half_prime_factor(cid)
        rescored.append((score * factor, meta))
    rescored.sort(key=lambda x: -x[0])
    return rescored


class IndexMissingError(RuntimeError):
    pass


def _resolve_cache(settings: Settings) -> Path:
    cache = settings.cache_dir.expanduser()
    if not cache.is_absolute():
        cache = (Path.cwd() / cache).resolve()
    return cache


def _load_lane(cache: Path, prefix: str) -> VectorStore:
    idx = cache / f"{prefix}.faiss"
    meta = cache / f"{prefix}.jsonl"
    if not (idx.exists() and meta.exists()):
        raise IndexMissingError(
            f"{prefix} index not built at {idx}. Run `lumos ingest` first."
        )
    return VectorStore.load(idx, meta)


# Serializes the one-time lazy load of each lane. retrieve() now runs these
# getters in a worker thread, so two concurrent cold retrievals could otherwise
# both read a multi-GB index off disk at once. Double-checked so the warm path
# stays lock-free.
_store_load_lock = threading.Lock()


def get_identity_store(settings: Settings | None = None) -> VectorStore:
    global _identity_store
    if _identity_store is None:
        with _store_load_lock:
            if _identity_store is None:
                settings = settings or get_settings()
                _identity_store = _load_lane(_resolve_cache(settings), "identity")
    return _identity_store


def get_knowledge_store(settings: Settings | None = None) -> VectorStore:
    global _knowledge_store
    if _knowledge_store is None:
        with _store_load_lock:
            if _knowledge_store is None:
                settings = settings or get_settings()
                _knowledge_store = _load_lane(_resolve_cache(settings), "knowledge")
    return _knowledge_store


def reload_stores() -> None:
    global _identity_store, _knowledge_store
    _identity_store = None
    _knowledge_store = None


# Circuit breaker: NVIDIA rotates/retires hosted rerank models (the old one now
# 410s). On a PERMANENT failure (4xx) we disable reranking for the process and log
# ONCE — retrying a gone endpoint every turn just spams the log and wastes a call.
# Transient failures (5xx / timeout / 429) do NOT trip it. Reset by restarting.
_rerank_disabled = False


def _rerank_url(settings: Settings) -> str:
    """Hosted NeMo Retriever reranking endpoint.

    Reranking lives on ai.api.nvidia.com/v1/retrieval/<model-slug>/reranking —
    a different host+path from the OpenAI-compatible chat base
    (integrate.api.nvidia.com/v1). Honour an explicit nvidia_rerank_url override;
    otherwise derive the model-specific path (dots in the slug → underscores,
    e.g. nvidia/llama-3.2-... → nvidia/llama-3_2-...).
    """
    override = (settings.nvidia_rerank_url or "").strip()
    if override:
        return override
    slug = settings.nvidia_rerank_model.replace(".", "_")
    return f"https://ai.api.nvidia.com/v1/retrieval/{slug}/reranking"


async def _nvidia_rerank(query: str, candidates: list, settings: Settings) -> list:
    """Reorder (score, metadata) candidates most-relevant-first using NVIDIA's
    NeMo Retriever reranking cross-encoder. Keeps the original cosine scores —
    only the ORDER changes. Returns the input unchanged on any error (reranking
    must never break a turn)."""
    global _rerank_disabled
    if _rerank_disabled or len(candidates) < 2:
        return candidates
    passages = [{"text": (m.get("text") or "")[:2000]} for _s, m in candidates]
    payload = {
        "model": settings.nvidia_rerank_model,
        "query": {"text": query[:2000]},
        "passages": passages,
        "truncate": "END",
    }
    url = _rerank_url(settings)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.nvidia_api_key.strip()}"},
                json=payload,
            )
            r.raise_for_status()
            rankings = (r.json() or {}).get("rankings") or []
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (400, 401, 403, 404, 405, 410, 422):
            # Permanent — model/endpoint gone or key rejected. Trip the breaker so
            # we stop hammering (and log-spamming) a URL that will never recover.
            _rerank_disabled = True
            log.warning(
                "retrieval.nvidia_rerank_disabled",
                status=code, url=url,
                hint="rerank model/endpoint gone or rejected — disabled for this run; "
                     "set a current LUMOS_NVIDIA_RERANK_MODEL (build.nvidia.com) or "
                     "LUMOS_NVIDIA_RERANK_ENABLED=false",
            )
        else:
            log.info("retrieval.nvidia_rerank_transient", status=code, url=url)
        return candidates
    except Exception as e:  # noqa: BLE001 — best-effort; fall back to cosine order
        log.info("retrieval.nvidia_rerank_failed", error=str(e), url=url)
        return candidates
    order = [
        rk["index"]
        for rk in rankings
        if isinstance(rk, dict) and isinstance(rk.get("index"), int)
    ]
    if not order:
        return candidates
    seen: set[int] = set()
    reordered = []
    for i in order:
        if 0 <= i < len(candidates) and i not in seen:
            reordered.append(candidates[i])
            seen.add(i)
    reordered += [c for i, c in enumerate(candidates) if i not in seen]
    return reordered


async def retrieve(
    query: str,
    *,
    settings: Settings | None = None,
    top_k_identity: int | None = None,
    top_k_knowledge: int | None = None,
    _recursion_depth_remaining: int | None = None,
) -> Retrieval:
    """Run the 5-phase retrieval pipeline. When `retrieval_recursion_depth > 0`,
    after the first pass we re-query using the top identity hit's text and
    merge the additional hits (Phase 36 — Rocchio-style relevance feedback,
    pattern borrowed from Paper 2's `JointMemoryBridge.search(recursion_depth)`).

    `_recursion_depth_remaining` is internal — callers should leave it unset
    so the setting drives recursion. Each recursive call decrements it.
    """
    settings = settings or get_settings()
    if _recursion_depth_remaining is None:
        _recursion_depth_remaining = settings.retrieval_recursion_depth
    # Offload the (potentially multi-GB) lazy index load to a worker thread so a
    # cold load can't freeze the event loop that serves the SSE heartbeat and the
    # alert-monitor loop. Warm calls return the cached store near-instantly.
    identity_store = await asyncio.to_thread(get_identity_store, settings)
    knowledge_store = await asyncio.to_thread(get_knowledge_store, settings)

    client = LMStudioClient()
    try:
        vectors = await client.embed([query], model=settings.lm_studio_embedding_model)
    finally:
        await client.aclose()
    vec = vectors[0]

    k_id = top_k_identity if top_k_identity is not None else settings.retrieval_top_k_identity
    k_kn = (
        top_k_knowledge if top_k_knowledge is not None else settings.retrieval_top_k_knowledge
    )
    min_score = settings.min_retrieval_score

    # Phase A — raw cosine similarity from FAISS. The search is CPU-bound C code
    # over a large index; run it off the event loop too (every chat turn hits this).
    raw_identity = (
        await asyncio.to_thread(identity_store.search, vec, k_id) if k_id > 0 else []
    )
    raw_knowledge = (
        await asyncio.to_thread(knowledge_store.search, vec, k_kn) if k_kn > 0 else []
    )

    # Phase B — Yang-Mills Mass Gap impedance floor: reject computationally-
    # frictionless noise (similarity < 0.657 = Δ = √32 - 5).
    survived_identity = [(s, m) for s, m in raw_identity if s >= min_score]
    survived_knowledge = [(s, m) for s, m in raw_knowledge if s >= min_score]

    # Phase B.5 — NVIDIA NeMo Retriever reranker (opt-in, off by default). Reorder
    # the mass-gap survivors by a true cross-encoder relevance score. One /ranking
    # call per lane; falls back to cosine order on any error, never breaks a turn.
    if settings.nvidia_rerank_enabled and settings.nvidia_api_key.strip():
        survived_identity = await _nvidia_rerank(query, survived_identity, settings)
        survived_knowledge = await _nvidia_rerank(query, survived_knowledge, settings)

    # Phases C/D/E — esoteric re-rank (Triple Normalization, Half-Prime Geodesic,
    # UBBM θ-alignment). OFF by default (esoteric_rerank_enabled): these key off
    # chunk-id hashes, arbitrary k-means cluster indices, and UTF-8 bit density —
    # not semantic relevance — so they reorder the floor's survivors by numerology
    # and can bury the true best match. When disabled, survivors pass straight
    # through in cosine order (Phase A/B only).
    if settings.esoteric_rerank_enabled:
        # Phase C — Triple Normalization (Harmonic GCD-3 ⊗ Geometric GCD-360 ⊗ Binary 1001).
        normed_identity = _triple_normalize(survived_identity)
        normed_knowledge = _triple_normalize(survived_knowledge)

        # Phase D — Half-Prime Geodesic cluster scoring.
        from .atlas import get_chunk_to_cluster

        cluster_map = get_chunk_to_cluster(settings)
        geodesic_identity = _half_prime_geodesic(normed_identity, cluster_map)
        geodesic_knowledge = _half_prime_geodesic(normed_knowledge, cluster_map)

        # Phase E — UBBM θ-alignment re-rank + signature attachment.
        # Boost chunks whose Binary Diagonal angle is closest to the query's, and
        # attach the full UBBM signature per hit under "ubbm_signature".
        query_theta = binary_diagonal_theta(query)
        aligned_identity = _ubbm_align(geodesic_identity, query_theta, list(vec))
        aligned_knowledge = _ubbm_align(geodesic_knowledge, query_theta, list(vec))
    else:
        # Pure-cosine path — the floor's survivors, already in cosine-descending
        # order from FAISS. No id/cluster/bit-density perturbation applied.
        aligned_identity = survived_identity
        aligned_knowledge = survived_knowledge

    # Phase E.5 — Morphic-Resonance Coupling (Phase 40, default-OFF). Anchored
    # hits use log-mean (<=1.0) × Pendinium-GCD (>=1.0); anchorless hits use the
    # stitch-1001 GCD only (lm=1.0, gcd_factor>=1.0, strictly >1.0 when lambda>0).
    # Provably a 1.0 no-op ONLY when disabled. Score can move up or down vs the
    # pre-E.5 order, but only re-orders survivors of the Phase B mass-gap floor —
    # it never resurrects a sub-floor hit.
    if settings.morphic_resonance_enabled:
        _mlam = settings.morphic_resonance_lambda
        aligned_identity = _morphic_align(aligned_identity, query, _mlam)
        aligned_knowledge = _morphic_align(aligned_knowledge, query, _mlam)

    # Phase F — prescient flagging: surface long-buried high-scoring memories.
    # Knowledge chunks lack a reliable ingest timestamp, so this is a no-op there
    # in practice; identity chunks get age_days and (when both thresholds met)
    # prescient=True attached to their metadata.
    now_ts = time.time()
    flagged_identity = _flag_prescient(aligned_identity, now_ts)
    flagged_knowledge = _flag_prescient(aligned_knowledge, now_ts)

    result = Retrieval(
        query=query,
        query_vector=list(vec) if isinstance(vec, list) else list(vec),
        identity=[Hit(score=s, metadata=m) for s, m in flagged_identity],
        knowledge=[Hit(score=s, metadata=m) for s, m in flagged_knowledge],
    )

    # Phase 36 — Rocchio-style 1-hop expansion. Take the top identity hit's
    # text as the next query, re-run the full pipeline, merge dedup-by-chunk_id.
    # Surfaces 2-hop semantic neighbors the original query alone wouldn't find.
    # Cost: +1 LM Studio embedding + 1 FAISS lookup per recursion level.
    if _recursion_depth_remaining > 0 and result.identity:
        next_query_text = (result.identity[0].metadata.get("text") or "").strip()
        # Cap to first ~200 chars so the next embedding stays focused on the
        # top hit's lede rather than wandering into long-form tail content.
        next_query_text = next_query_text[:200]
        if next_query_text and next_query_text.lower() != query.strip().lower():
            hop = await retrieve(
                next_query_text,
                settings=settings,
                top_k_identity=top_k_identity,
                top_k_knowledge=top_k_knowledge,
                _recursion_depth_remaining=_recursion_depth_remaining - 1,
            )
            seen = {h.metadata.get("chunk_id", "") for h in result.identity + result.knowledge}
            new_id = [h for h in hop.identity if h.metadata.get("chunk_id", "") not in seen]
            new_kn = [h for h in hop.knowledge if h.metadata.get("chunk_id", "") not in seen]
            result = Retrieval(
                query=query,
                query_vector=result.query_vector,
                identity=sorted(list(result.identity) + new_id, key=lambda h: -h.score),
                knowledge=sorted(list(result.knowledge) + new_kn, key=lambda h: -h.score),
            )

    return result


def _ubbm_align(
    hits: list[tuple[float, dict[str, Any]]],
    query_theta: float,
    query_vec: list[float],
) -> list[tuple[float, dict[str, Any]]]:
    """Phase E re-rank: apply θ-alignment factor + attach UBBM signature.

    Composes multiplicatively with existing score. Signatures land in the
    chunk's metadata under "ubbm_signature" so the HUD/clients can inspect.
    """
    rescored: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        chunk_text = str(meta.get("text", ""))
        sig = compute_signature(chunk_text, embedding=query_vec)
        chunk_theta = sig["theta"]
        factor = theta_alignment_factor(query_theta, chunk_theta)
        new_meta = {**meta, "ubbm_signature": sig}
        rescored.append((score * factor, new_meta))
    rescored.sort(key=lambda x: -x[0])
    return rescored


def _morphic_coupling_weight(
    query_anchor: int,
    q_stitch: int,
    meta: dict[str, Any],
    lam: float,
) -> tuple[float, float]:
    """Return (w, coupling) for one hit. w multiplies the score; coupling is the
    log-mean component (resonance purity, in (0,1]) surfaced to the HUD.

    Sub-factor 1 — inverted symmetric log-mean over Pendinium anchors:
        candidate p_c = int(meta['pendinium_anchor'])  (set on dream-consolidated
        identity chunks); query p_q = PENDINIUM_PRIMES[stitch % N] (the
        tadah_phase_lock idiom). Both > 1, same side of 1, so x = ln(p_c)/ln(p_q)
        is well-conditioned. lm = 2/(x + 1/x) in (0,1], = 1.0 exactly at p_c==p_q.
        Absent anchor -> lm = 1.0 (no-op).

    Sub-factor 2 — GCD-preservation:
        identity (anchor present): g = gcd(p_q, p_c) in {1, p}; denom = max(p_q, p_c).
        knowledge / anchorless: stitch-1001 GCD with +1 offset (avoids gcd(0,0)):
            g = gcd(q_stitch+1, c_stitch+1); denom = max(q_stitch+1, c_stitch+1).
        gcd_factor = 1.0 + lam * g/denom, in [1.0, 1+lam]. Always >= 1.0.
    """
    lm = 1.0
    gcd_factor = 1.0

    ubbm_sig = meta.get("ubbm_signature")
    anchor_raw = meta.get("pendinium_anchor")

    if anchor_raw is not None:
        # Identity, dream-consolidated: Pendinium-prime substrate (clean 1-or-p).
        try:
            p_c = int(anchor_raw)
        except (TypeError, ValueError):
            p_c = 0
        if p_c > 1 and query_anchor > 1:
            ln_q = math.log(query_anchor)
            ln_c = math.log(p_c)
            if abs(ln_q) > 1e-9 and abs(ln_c) > 1e-9:
                if p_c == query_anchor:
                    lm = 1.0
                else:
                    x = ln_c / ln_q
                    lm = 2.0 / (x + 1.0 / x)
                    lm = min(max(lm, 1e-6), 1.0)  # float-safety clamp only
            g = gcd_substrate(query_anchor, p_c)["gcd"]
            denom = max(query_anchor, p_c)
            gcd_factor = 1.0 + lam * (g / denom)
    elif isinstance(ubbm_sig, dict):
        # Knowledge lane / ingest-only identity: no anchor -> stitch-1001 GCD.
        try:
            c_stitch = int(ubbm_sig.get("stitch_1001", 0))
        except (TypeError, ValueError):
            c_stitch = 0
        qa = q_stitch + 1
        ca = c_stitch + 1
        g = gcd_substrate(qa, ca)["gcd"]
        denom = max(qa, ca)
        gcd_factor = 1.0 + lam * (g / denom)
    # else: no ubbm_signature at all (shouldn't occur post-Phase-E) -> w stays 1.0.

    return lm * gcd_factor, lm


def _morphic_align(
    hits: list[tuple[float, dict[str, Any]]],
    query_text: str,
    lam: float,
) -> list[tuple[float, dict[str, Any]]]:
    """Phase E.5 — apply the morphic coupling weight to each hit, attach
    'morphic_coupling' (the log-mean purity, (0,1]) to metadata, re-sort desc."""
    if not hits:
        return hits
    n = len(PENDINIUM_PRIMES)
    q_stitch = binary_1001_count(query_text)
    query_anchor = PENDINIUM_PRIMES[q_stitch % n] if n > 0 else 13
    rescored: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        w, coupling = _morphic_coupling_weight(query_anchor, q_stitch, meta, lam)
        new_meta = {**meta, "morphic_coupling": round(coupling, 4)}
        rescored.append((score * w, new_meta))
    rescored.sort(key=lambda x: -x[0])
    return rescored


def _flag_prescient(
    hits: list[tuple[float, dict[str, Any]]],
    now_ts: float,
) -> list[tuple[float, dict[str, Any]]]:
    """Phase F — mark long-buried, high-scoring chunks as prescient.

    Attaches `prescient: True` and `age_days: int` to metadata when both
    conditions hold:
      • score ≥ PRESCIENT_SCORE_FLOOR (default 0.85)
      • create_time_first (or create_time_last) is ≥ PRESCIENT_AGE_DAYS old

    Does NOT re-rank — the boost has already been applied multiplicatively
    upstream. This phase exists purely to surface the signal: the HUD and
    composer can render a 🜂 / "echo" badge on prescient hits so Lumos and
    operator see that a year-old conversation is suddenly load-bearing.
    """
    out: list[tuple[float, dict[str, Any]]] = []
    for score, meta in hits:
        ts = meta.get("create_time_first") or meta.get("create_time_last")
        try:
            ts_f = float(ts) if ts is not None else 0.0
        except (TypeError, ValueError):
            ts_f = 0.0
        if ts_f <= 0.0:
            out.append((score, meta))
            continue
        age_seconds = max(0.0, now_ts - ts_f)
        age_days = int(age_seconds // _SECONDS_PER_DAY)
        new_meta = {**meta, "age_days": age_days}
        if score >= PRESCIENT_SCORE_FLOOR and age_days >= PRESCIENT_AGE_DAYS:
            new_meta["prescient"] = True
        out.append((score, new_meta))
    return out
