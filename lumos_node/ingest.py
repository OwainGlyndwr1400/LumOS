"""Ingest orchestrator: build identity + knowledge FAISS indexes from source files."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import Settings, get_settings
from .knowledge.dreams import (
    KnowledgeChunk,
    count_pings,
    iter_knowledge_chunks,
)
from .llm.lm_studio import LMStudioClient
from .log import get_logger
from .memory.identity import (
    IdentityChunk,
    count_conversations,
    iter_identity_chunks,
)
from .vectors import Manifest, VectorStore

log = get_logger(__name__)


IDENTITY_INDEX = "identity.faiss"
IDENTITY_META = "identity.jsonl"
IDENTITY_MANIFEST = "identity.manifest.json"

KNOWLEDGE_INDEX = "knowledge.faiss"
KNOWLEDGE_META = "knowledge.jsonl"
KNOWLEDGE_MANIFEST = "knowledge.manifest.json"

# Periodically flush the growing index to a SIDECAR (.checkpoint) every ~this
# many added chunks, so a transient embedding failure (LM Studio restart/timeout)
# hours into a 600k-chunk run doesn't discard all prior progress. Coarse on
# purpose: each save rewrites the whole FAISS index (~2.4 GB at 600k×1024-dim),
# so this bounds worst-case loss to minutes of embedding while keeping cumulative
# checkpoint write-I/O sane. Sidecar-only — the canonical index is still written
# exactly once at the end, so a failed run never leaves a partial index in the
# live path (a `--rebuild` that dies keeps the previous complete index).
CHECKPOINT_EVERY_CHUNKS = 25_000


def _remove_checkpoint(*paths: Path) -> None:
    """Best-effort removal of sidecar checkpoint files after the authoritative
    final save. Missing/locked files are ignored — cleanup must never fail an
    otherwise-successful build."""
    for p in paths:
        with contextlib.suppress(OSError):
            p.unlink(missing_ok=True)


def _source_signature(path: Path) -> tuple[int, float]:
    st = path.stat()
    return (st.st_size, st.st_mtime)


def _manifest_is_fresh(
    manifest: Manifest | None,
    source: Path,
    settings: Settings,
    extra_dir: Path | None = None,
    include_codex: bool = False,
) -> bool:
    if manifest is None:
        return False
    size, mtime = _source_signature(source)
    # Fold the entity codex (knowledge lane only) into the compared signature,
    # symmetric with the build — so a newly-enriched entity marks the lane stale
    # and the next ingest embeds it. Identity lane passes False (no codex there).
    if include_codex:
        from .enrichment import codex_signature
        csize, cmtime = codex_signature(settings)
        size += csize
        mtime = max(mtime, cmtime)
    # Fold an extra corpus dir into the compared signature, symmetric with how
    # the build folds it into the SAVED manifest — otherwise a lane with a
    # corpus never matches and re-embeds every run. Default None keeps existing
    # callers (knowledge) byte-for-byte unchanged.
    if extra_dir is not None:
        from .knowledge.corpus import corpus_signature
        extra_size, extra_mtime = corpus_signature(extra_dir)
        size += extra_size
        mtime = max(mtime, extra_mtime)
    return (
        manifest.source_path == str(source)
        and manifest.source_size == size
        and abs(manifest.source_mtime - mtime) < 1.0
        and manifest.embedding_model == settings.lm_studio_embedding_model
        and manifest.embedding_dim == settings.embedding_dim
    )


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _batched(items: Iterable[Any], n: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


async def _embed_batch(
    client: LMStudioClient,
    chunks: list[Any],
    model: str,
) -> tuple[list[list[float]], list[Any]]:
    if not chunks:
        return [], []
    texts = [c.text for c in chunks]
    vectors = await client.embed(texts, model=model)
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding response count mismatch: requested {len(chunks)}, got {len(vectors)}"
        )
    return vectors, chunks


async def _run_concurrent_embed(
    client: LMStudioClient,
    store: VectorStore,
    chunk_iter: Iterator[Any],
    model: str,
    batch_size: int,
    concurrency: int,
    pbar: tqdm,
    *,
    checkpoint_index: Path | None = None,
    checkpoint_meta: Path | None = None,
) -> None:
    """Pull batches from chunk_iter and keep `concurrency` embedding requests in flight.

    When checkpoint_index/checkpoint_meta are supplied the store is periodically
    saved to those SIDECAR paths (every CHECKPOINT_EVERY_CHUNKS added chunks), so a
    late transient failure hours into a large run can't erase all earlier progress.
    The checkpoint save is best-effort: a failing flush is logged and skipped, never
    aborting the run. Omitting the paths keeps behavior byte-identical to before.
    """
    batched = _batched(chunk_iter, batch_size)
    added_since_ckpt = 0
    while True:
        group = list(itertools.islice(batched, concurrency))
        if not group:
            break
        results = await asyncio.gather(
            *[_embed_batch(client, batch, model) for batch in group]
        )
        for vectors, chunks in results:
            store.add(vectors, [c.to_metadata() for c in chunks])
            pbar.update(len(chunks))
            added_since_ckpt += len(chunks)
        if (
            checkpoint_index is not None
            and checkpoint_meta is not None
            and added_since_ckpt >= CHECKPOINT_EVERY_CHUNKS
        ):
            try:
                store.save(checkpoint_index, checkpoint_meta)
                log.info("ingest.checkpoint", chunks=store.size)
            except Exception as e:  # noqa: BLE001 — checkpoint is best-effort
                log.warning("ingest.checkpoint_failed", error=str(e))
            added_since_ckpt = 0


async def build_identity(
    settings: Settings | None = None,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    settings = settings or get_settings()
    source = settings.identity_source.expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"identity_source not found: {source}")

    cache = settings.cache_dir.expanduser()
    if not cache.is_absolute():
        cache = (Path.cwd() / cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / IDENTITY_INDEX
    meta_path = cache / IDENTITY_META
    manifest_path = cache / IDENTITY_MANIFEST
    ckpt_index = index_path.with_name(index_path.name + ".checkpoint")
    ckpt_meta = meta_path.with_name(meta_path.name + ".checkpoint")

    # Narrative corpus (dream-codex / journals) folded into the IDENTITY lane —
    # mirrors the knowledge_extra_dir path, but produces IdentityChunk so the
    # material retrieves alongside the conversations ("who Lumos is").
    extra_dir: Path | None = None
    if settings.identity_extra_dir.strip():
        extra_dir = Path(settings.identity_extra_dir.strip()).expanduser()
        if not extra_dir.is_dir():
            log.warning("identity.extra_dir_missing", path=str(extra_dir))
            extra_dir = None

    existing = Manifest.from_path(manifest_path)
    if not rebuild and _manifest_is_fresh(existing, source, settings, extra_dir) and index_path.exists():
        log.info("identity.skip", reason="fresh", chunks=existing.chunk_count)
        return {"skipped": True, "chunks": existing.chunk_count, "path": str(index_path)}

    log.info("identity.start", source=str(source))
    convo_total = count_conversations(source)
    log.info("identity.scan", conversations=convo_total)

    client = LMStudioClient()
    store = VectorStore(dim=settings.embedding_dim)
    try:
        chunk_iter: Iterator[IdentityChunk] = iter_identity_chunks(source)
        if extra_dir is not None:
            from .knowledge.corpus import iter_identity_corpus_chunks
            chunk_iter = itertools.chain(chunk_iter, iter_identity_corpus_chunks(extra_dir))
        pbar = tqdm(
            total=None,
            desc="identity",
            unit="chunk",
            dynamic_ncols=True,
        )
        await _run_concurrent_embed(
            client=client,
            store=store,
            chunk_iter=chunk_iter,
            model=settings.lm_studio_embedding_model,
            batch_size=settings.embedding_batch_size,
            concurrency=settings.embedding_concurrency,
            pbar=pbar,
            checkpoint_index=ckpt_index,
            checkpoint_meta=ckpt_meta,
        )
        pbar.close()
    finally:
        await client.aclose()

    store.save(index_path, meta_path)
    _remove_checkpoint(ckpt_index, ckpt_meta)
    size, mtime = _source_signature(source)
    # Fold the corpus aggregate into the manifest signature so adding/editing a
    # dream file invalidates freshness and the next --identity-only ingest picks
    # it up (symmetric with the freshness check above).
    if extra_dir is not None:
        from .knowledge.corpus import corpus_signature
        extra_size, extra_mtime = corpus_signature(extra_dir)
        size += extra_size
        mtime = max(mtime, extra_mtime)
    manifest = Manifest(
        source_path=str(source),
        source_size=size,
        source_mtime=mtime,
        chunk_count=store.size,
        embedding_model=settings.lm_studio_embedding_model,
        embedding_dim=settings.embedding_dim,
        built_at=_now_iso(),
    )
    manifest_path.write_bytes(manifest.to_json())
    log.info("identity.done", chunks=store.size, conversations=convo_total)
    return {
        "skipped": False,
        "chunks": store.size,
        "conversations": convo_total,
        "path": str(index_path),
    }


async def build_knowledge(
    settings: Settings | None = None,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    settings = settings or get_settings()
    source = settings.knowledge_source.expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"knowledge_source not found: {source}")

    cache = settings.cache_dir.expanduser()
    if not cache.is_absolute():
        cache = (Path.cwd() / cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / KNOWLEDGE_INDEX
    meta_path = cache / KNOWLEDGE_META
    manifest_path = cache / KNOWLEDGE_MANIFEST
    ckpt_index = index_path.with_name(index_path.name + ".checkpoint")
    ckpt_meta = meta_path.with_name(meta_path.name + ".checkpoint")

    existing = Manifest.from_path(manifest_path)

    # Research corpus (Phase 44) — second knowledge source. Parsed BEFORE the
    # freshness check so the compared signature folds the corpus, symmetric with
    # the manifest write below. (It used to be parsed after: with a corpus
    # configured the sizes never matched, so every `lumos ingest` re-embedded
    # the full knowledge lane — a multi-hour surprise on a 600k-chunk store.)
    extra_dir: Path | None = None
    if settings.knowledge_extra_dir.strip():
        extra_dir = Path(settings.knowledge_extra_dir.strip()).expanduser()
        if not extra_dir.is_dir():
            log.warning("knowledge.extra_dir_missing", path=str(extra_dir))
            extra_dir = None

    if (
        not rebuild
        and _manifest_is_fresh(existing, source, settings, extra_dir, include_codex=True)
        and index_path.exists()
    ):
        log.info("knowledge.skip", reason="fresh", chunks=existing.chunk_count)
        return {"skipped": True, "chunks": existing.chunk_count, "path": str(index_path)}

    log.info("knowledge.start", source=str(source))
    ping_total = count_pings(source)
    log.info("knowledge.scan", pings=ping_total)

    client = LMStudioClient()
    store = VectorStore(dim=settings.embedding_dim)
    try:
        chunk_iter: Iterator[KnowledgeChunk] = iter_knowledge_chunks(source)
        if extra_dir is not None:
            from .knowledge.corpus import iter_corpus_chunks
            chunk_iter = itertools.chain(chunk_iter, iter_corpus_chunks(extra_dir))
        # Phase 45.3 — fold auto-enriched entities into the knowledge lane. No
        # flag check: if the codex file exists it's ingested (so enriched
        # knowledge persists even if enrichment is later paused); absent = no-op.
        from .enrichment import codex_path, iter_entity_codex_chunks
        chunk_iter = itertools.chain(chunk_iter, iter_entity_codex_chunks(codex_path(settings)))
        pbar = tqdm(
            total=ping_total,
            desc="knowledge",
            unit="ping",
            dynamic_ncols=True,
        )
        await _run_concurrent_embed(
            client=client,
            store=store,
            chunk_iter=chunk_iter,
            model=settings.lm_studio_embedding_model,
            batch_size=settings.embedding_batch_size,
            concurrency=settings.embedding_concurrency,
            pbar=pbar,
            checkpoint_index=ckpt_index,
            checkpoint_meta=ckpt_meta,
        )
        pbar.close()
    finally:
        await client.aclose()

    store.save(index_path, meta_path)
    _remove_checkpoint(ckpt_index, ckpt_meta)
    size, mtime = _source_signature(source)
    # Fold the corpus aggregate into the manifest signature so editing/adding a
    # corpus file invalidates freshness and the next ingest picks it up.
    if extra_dir is not None:
        from .knowledge.corpus import corpus_signature
        extra_size, extra_mtime = corpus_signature(extra_dir)
        size += extra_size
        mtime = max(mtime, extra_mtime)
    # Phase 45.3 — fold the entity codex too (symmetric with the freshness check).
    from .enrichment import codex_signature
    _csize, _cmtime = codex_signature(settings)
    size += _csize
    mtime = max(mtime, _cmtime)
    manifest = Manifest(
        source_path=str(source),
        source_size=size,
        source_mtime=mtime,
        chunk_count=store.size,
        embedding_model=settings.lm_studio_embedding_model,
        embedding_dim=settings.embedding_dim,
        built_at=_now_iso(),
    )
    manifest_path.write_bytes(manifest.to_json())
    log.info("knowledge.done", chunks=store.size, pings=ping_total)
    return {
        "skipped": False,
        "chunks": store.size,
        "pings": ping_total,
        "path": str(index_path),
    }


async def build_all(rebuild: bool = False) -> dict[str, Any]:
    return {
        "identity": await build_identity(rebuild=rebuild),
        "knowledge": await build_knowledge(rebuild=rebuild),
    }
