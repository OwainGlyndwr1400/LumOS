"""One-time merge: lift the old AGI memory FAISS (bge-built, verified cos=1.0)
into LumOS's knowledge lane WITHOUT re-embedding.

Reuses the stored vectors directly (same embedding space), maps each bare-string
entry into the knowledge metadata schema, and appends to knowledge.faiss/.jsonl.
Backs up the existing index first. Run with LumOS OFFLINE.

NOTE: this is a direct index merge, NOT a source change. A future
`lumos ingest --rebuild --knowledge-only` rebuilds knowledge from dream_pings +
corpus and WILL drop these entries — restore from backup or re-run this then.
Do NOT run this twice on the same index (it would duplicate); restore the
*.premerge.bak first if you need to re-run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import faiss
import numpy as np
import orjson

from lumos_node.config import get_settings
from lumos_node.vectors import VectorStore

OLD = Path("path/to/old/agi/memory")  # edit to your old-memory export dir
OLD_FAISS = OLD / "private_memory_index.faiss"
OLD_JSONL = OLD / "private_entries.jsonl"
BATCH = 50_000


def _derive_subject(t: str) -> str:
    flat = " ".join((t or "").split())
    if len(flat) <= 72:
        return flat or "memory"
    cut = flat[:72]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 30 else cut) + "…"


def main() -> None:
    s = get_settings()
    cache = s.cache_dir.expanduser()
    if not cache.is_absolute():
        cache = (Path.cwd() / cache).resolve()
    kf, kj, kmf = cache / "knowledge.faiss", cache / "knowledge.jsonl", cache / "knowledge.manifest.json"

    # 1) backup
    for p in (kf, kj, kmf):
        if p.exists():
            shutil.copy2(p, p.with_name(p.name + ".premerge.bak"))
    print("backed up knowledge index -> *.premerge.bak")

    # 2) load existing knowledge
    store = VectorStore.load(kf, kj)
    before = store.size
    print("knowledge before:", before)

    # 3) load old faiss + row-aligned texts
    oi = faiss.read_index(str(OLD_FAISS))
    n_old = int(oi.ntotal)
    if oi.d != store.dim:
        raise SystemExit(f"ABORT: dim mismatch (old {oi.d} vs knowledge {store.dim})")
    texts: list[str] = []
    with OLD_JSONL.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = orjson.loads(line)
            texts.append(e if isinstance(e, str) else (e.get("content") or e.get("text") or ""))
    print("old vectors:", n_old, "| old texts:", len(texts))
    if len(texts) != n_old:
        raise SystemExit(
            f"ABORT: count mismatch (texts {len(texts)} vs vectors {n_old}) — row alignment unsafe"
        )

    # 4) batch reconstruct -> normalize (L2 source -> IP store) -> add + map metadata
    added = 0
    for start in range(0, n_old, BATCH):
        end = min(start + BATCH, n_old)
        vecs = oi.reconstruct_n(start, end - start).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1.0, norms)
        store.index.add(vecs)
        for j in range(start, end):
            t = texts[j]
            store._metadata.append(
                {
                    "chunk_id": hashlib.sha256(f"oldmem|{j}|{t}".encode("utf-8", "replace")).hexdigest()[:16],
                    "ping_id": f"oldmem-{j}",
                    "sigil": "oldmem",
                    "agent": "private_memory",
                    "urgency_score": 0,
                    "urgency_weight": 0,
                    "source": "old_agi_memory",
                    "subject": _derive_subject(t),
                    "seed": t,
                    "fragment_count": 0,
                    "text": t,
                }
            )
        added += end - start
        print(f"  merged {added}/{n_old}", flush=True)

    # 5) save + update manifest count
    store.save(kf, kj)
    if kmf.exists():
        m = json.loads(kmf.read_text(encoding="utf-8"))
        m["chunk_count"] = store.size
        kmf.write_text(json.dumps(m, indent=2), encoding="utf-8")

    print(f"DONE — merged {n_old} old-memory chunks. knowledge {before} -> {store.size}")


if __name__ == "__main__":
    main()
