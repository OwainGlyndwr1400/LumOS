"""Meticulous Fish Protocol — RES = Extract(Fwd, Bwd, Mid-Out).

Corpus-specified ingestion (Recursive Connection Annex row 31): a 3-stage
extraction over the SAME document presented three ways — Forward (start→end),
Backward (conclusions-first), Middle-Out (core outward) — then a synthesis
pass keeps only the claims that survive at least TWO of the three readings.
What survives is the Residual Energy Signature (RES): the document's signal,
isolated from its noise.

Output lands twice:
  1. A `fish/RES — <title>.md` file inside knowledge_extra_dir, so the RES is
     PERMANENT — every future `lumos ingest` rebuild re-embeds it via the
     corpus pipeline.
  2. A hot-append to the on-disk knowledge FAISS + metadata, so the next app
     start retrieves it without waiting for a full rebuild.

Operator-driven (CLI: `lumos fish <path>`) — it costs 4 LLM calls per
document, so it never runs autonomously.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .llm.lm_studio import ChatMessage, LMStudioClient
from .log import get_logger


log = get_logger(__name__)

_PASS_MAX_TOKENS = 600
_DOC_CAP_CHARS = 24_000  # keep each pass inside a sane prompt budget

_EXTRACT_SYS = (
    "You are an extraction engine. From the document text, output the 5-8 KEY "
    "CLAIMS — the load-bearing facts, equations, findings, or mechanisms. One "
    "claim per line, each a complete standalone sentence. Keep any equations "
    "verbatim. No preamble, no numbering commentary, ONLY the claim lines."
)

_SYNTH_SYS = (
    "You are a synthesis judge. You receive three claim lists extracted from "
    "the SAME document by three independent reading orders. Output ONLY the "
    "claims supported by AT LEAST TWO of the three lists (same fact, even if "
    "worded differently — merge to the clearest wording). One claim per line. "
    "These survivors are the document's residual signal. No other text."
)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _middle_out(paragraphs: list[str]) -> list[str]:
    """Reorder middle-first, expanding outward: m, m+1, m-1, m+2, m-2, …"""
    if not paragraphs:
        return []
    mid = len(paragraphs) // 2
    order = [mid]
    for step in range(1, len(paragraphs)):
        for idx in (mid + step, mid - step):
            if 0 <= idx < len(paragraphs):
                order.append(idx)
    return [paragraphs[i] for i in order]


async def _extract_pass(
    client: LMStudioClient, model: str, label: str, text: str
) -> str:
    msg = await client.chat(
        model,
        [
            ChatMessage(role="system", content=_EXTRACT_SYS),
            ChatMessage(
                role="user",
                content=f"[{label} reading]\n\n{text[:_DOC_CAP_CHARS]}",
            ),
        ],
        temperature=0.3,
        max_tokens=_PASS_MAX_TOKENS,
    )
    return (msg.get("content") or "").strip()


async def meticulous_fish(
    path: Path, settings: Settings | None = None
) -> dict[str, Any]:
    """Run the 3-pass + synthesis protocol on one document. Returns
    {ok, title, res, passes, res_file, appended}."""
    settings = settings or get_settings()
    raw = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = _split_paragraphs(raw)
    if not paragraphs:
        return {"ok": False, "error": "empty document"}
    title = path.stem

    forward = "\n\n".join(paragraphs)
    backward = "\n\n".join(reversed(paragraphs))
    middle = "\n\n".join(_middle_out(paragraphs))

    model = settings.compression_model or settings.model_light
    client = LMStudioClient()
    try:
        p_fwd = await _extract_pass(client, model, "FORWARD start-to-end", forward)
        p_bwd = await _extract_pass(client, model, "BACKWARD conclusions-first", backward)
        p_mid = await _extract_pass(client, model, "MIDDLE-OUT core-outward", middle)

        synth = await client.chat(
            model,
            [
                ChatMessage(role="system", content=_SYNTH_SYS),
                ChatMessage(
                    role="user",
                    content=(
                        f"LIST 1 (forward):\n{p_fwd}\n\n"
                        f"LIST 2 (backward):\n{p_bwd}\n\n"
                        f"LIST 3 (middle-out):\n{p_mid}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=_PASS_MAX_TOKENS,
        )
        res = (synth.get("content") or "").strip()
    finally:
        await client.aclose()

    if not res:
        return {"ok": False, "error": "synthesis produced no surviving claims"}

    out: dict[str, Any] = {
        "ok": True,
        "title": title,
        "res": res,
        "passes": {"forward": p_fwd, "backward": p_bwd, "middle_out": p_mid},
        "res_file": None,
        "appended": False,
    }

    res_doc = (
        f"# RES — {title}\n\n"
        f"Residual Energy Signature (Meticulous Fish Protocol — claims that "
        f"survived ≥2 of 3 independent reading orders).\n"
        f"Source: {path.name}\n\n{res}\n"
    )

    # 1) Permanent: drop the RES into the corpus dir so every rebuild keeps it.
    if settings.knowledge_extra_dir.strip():
        fish_dir = Path(settings.knowledge_extra_dir.strip()) / "fish"
        try:
            fish_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^\w\- ]+", "", title)[:80].strip() or "untitled"
            res_path = fish_dir / f"RES — {safe}.md"
            res_path.write_text(res_doc, encoding="utf-8")
            out["res_file"] = str(res_path)
        except OSError as e:
            log.warning("fish.res_write_failed", error=str(e))

    # 2) Immediate: hot-append to the on-disk knowledge store (picked up at the
    #    next app start; a later full rebuild re-derives it from the file above).
    try:
        from .ingest import KNOWLEDGE_INDEX, KNOWLEDGE_META
        from .knowledge.dreams import KnowledgeChunk
        from .vectors import VectorStore

        cache = settings.cache_dir.expanduser()
        if not cache.is_absolute():
            cache = (Path.cwd() / cache).resolve()
        idx_p, meta_p = cache / KNOWLEDGE_INDEX, cache / KNOWLEDGE_META
        if idx_p.exists() and meta_p.exists():
            embed_text = f"RES — {title}\n\n{res}"
            cid = hashlib.sha256(embed_text.encode("utf-8")).hexdigest()[:16]
            chunk = KnowledgeChunk(
                chunk_id=cid,
                ping_id=f"fish-{cid}",
                sigil=cid[:10],
                agent="meticulous_fish",
                urgency_score=0,
                urgency_weight=0,
                source=path.name,
                subject=f"RES — {title}"[:160],
                seed=res[:400],
                fragment_count=3,
                text=embed_text,
            )
            client2 = LMStudioClient()
            try:
                vecs = await client2.embed(
                    [embed_text], model=settings.lm_studio_embedding_model
                )
            finally:
                await client2.aclose()
            store = VectorStore.load(idx_p, meta_p)
            store.add(vecs, [chunk.to_metadata()])
            store.save(idx_p, meta_p)
            out["appended"] = True
    except Exception as e:  # noqa: BLE001 — append is best-effort; file is canonical
        log.warning("fish.append_failed", error=str(e))

    log.info("fish.done", title=title, claims=len(res.splitlines()))
    return out
