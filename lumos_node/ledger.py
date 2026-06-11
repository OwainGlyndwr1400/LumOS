"""Null Ledger — memory-conservation audit.

RHC frame: 0 = 0_C + 0_V. A healthy memory corpus balances to zero — every
manifest row (0_C, the conserved/centre-anchored count) is accounted for, and
the residual (0_V, orphans + mismatches) is empty.

This makes the FAISS-vs-metadata count mismatch that ``VectorStore.load()`` only
*warns* about (the corruption left by an interrupted ingest / dream cycle) a
first-class, queryable invariant instead of a buried boot-log line.

Read-only: it inspects the live in-memory stores (the same cached singletons the
chat path uses) and the atlas cluster map; it never mutates. When the ledger is
out of balance, ``lumos repair`` (or rebuilding the affected index) is the
remedy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import Settings, get_settings
from .log import get_logger
from .vectors import VectorStore


log = get_logger(__name__)


def _lane(store: VectorStore) -> dict[str, Any]:
    """Per-lane FAISS-index vs metadata-row balance."""
    faiss_n = int(store.size)
    meta_n = len(store._metadata)  # noqa: SLF001 — internal audit, same access dream.py uses
    return {
        "faiss": faiss_n,
        "metadata": meta_n,
        "residual": abs(faiss_n - meta_n),
        "balanced": faiss_n == meta_n,
    }


def memory_ledger(settings: Settings | None = None) -> dict[str, Any]:
    """Audit memory conservation across both FAISS lanes + the atlas map.

    Returns a Null-Ledger snapshot: per-lane FAISS/metadata balance, atlas
    chunk<->cluster reconciliation, and the 0_C (conserved) / 0_V (residual)
    totals. ``balanced`` is true iff 0_V == 0. Never raises on a healthy or
    empty node — store-load failure returns ``ok=False`` with the error.
    """
    settings = settings or get_settings()
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Lazy imports — keep ledger free of import cycles with retrieval/atlas/dream.
    from .atlas import get_chunk_to_cluster
    from .dream import dream_status
    from .retrieval import get_identity_store, get_knowledge_store

    try:
        identity = get_identity_store(settings)
        knowledge = get_knowledge_store(settings)
    except Exception as e:  # noqa: BLE001
        log.info("ledger.store_load_failed", error=str(e))
        return {"ok": False, "error": str(e), "balanced": False, "checked_at": checked_at}

    lanes = {"identity": _lane(identity), "knowledge": _lane(knowledge)}

    # Atlas chunk<->cluster reconciliation. Every chunk_id present across both
    # lanes; the cluster map should reference only real chunks.
    chunk_ids: set[str] = set()
    for store in (identity, knowledge):
        for meta in store._metadata:  # noqa: SLF001
            cid = str(meta.get("chunk_id", ""))
            if cid:
                chunk_ids.add(cid)
    try:
        cluster_map = dict(get_chunk_to_cluster(settings))
    except Exception:  # noqa: BLE001
        cluster_map = {}
    mapped_keys = set(cluster_map.keys())
    mapped = len(chunk_ids & mapped_keys)                 # 0_C: chunk carries a cluster
    unmapped_unclustered = len(chunk_ids - mapped_keys)   # informational: not yet clustered
    orphan_map_entries = len(mapped_keys - chunk_ids)     # 0_V: map points at a vanished chunk

    lane_residual = sum(lane["residual"] for lane in lanes.values())
    zero_v = lane_residual + orphan_map_entries
    # 0_C = everything conserved: the aligned faiss/meta rows + mapped chunks.
    zero_c = sum(min(lane["faiss"], lane["metadata"]) for lane in lanes.values()) + mapped
    balanced = zero_v == 0

    try:
        dream = dream_status()
        dream_state = dream.get("state", {}) if isinstance(dream, dict) else {}
    except Exception:  # noqa: BLE001
        dream_state = {}

    return {
        "ok": True,
        "balanced": balanced,
        "null_ledger": {"zero_c": zero_c, "zero_v": zero_v, "identity": "0 = 0_C + 0_V"},
        "lanes": lanes,
        "atlas": {
            "chunks_total": len(chunk_ids),
            "mapped": mapped,
            "unmapped_unclustered": unmapped_unclustered,
            "orphan_map_entries": orphan_map_entries,
        },
        "dream": {
            "total_consolidated": dream_state.get("total_consolidated"),
            "last_run_at": dream_state.get("last_run_at"),
        },
        "remedy": None if balanced else "out of balance — run `lumos repair` or rebuild the affected lane index",
        "checked_at": checked_at,
    }
