"""One-time OCR-correction pass over garbled dream-ping seeds.

Many dream pings are distilled from scanned-PDF sources and carry OCR noise:
missing spaces ("Thestudyofsuchrapidly"), character confusions (rn->m, cl->d),
broken line-wraps. This module distils them back to legible prose.

Design goals (all three matter — this runs against live data):
  * NON-DESTRUCTIVE — reads the real source, writes a `.cleaned.jsonl` sidecar;
    the original is never touched. Adoption (swap + re-ingest) is a separate,
    user-approved step.
  * RESUMABLE — every ping (clean OR corrected) is written to the sidecar, so a
    re-run skips ids already present. Stop/restart freely; the live sentinel can
    keep pinging while this runs.
  * STRUCTURE-SAFE — only the SEED *body* is substituted, spliced back via the
    exact regex span. Every `Agent:` / `Subject:` / `--- marker ---` stays
    byte-identical. Any anomalous correction falls back to the original, so a
    chunk can only ever improve, never corrupt.

Only the garbled minority (~26%) is sent to the model; clean chunks pass
through untouched (running the LLM on good text just risks "fixing" what wasn't
broken).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import orjson

from ..config import get_settings
from ..llm.lm_studio import ChatMessage, LMStudioClient
from .dreams import _RE_SEED, _seed, iter_dream_pings

_SYS = (
    "You are an OCR-correction engine. The text was extracted from a scanned "
    "document and has OCR errors: missing spaces between words, character "
    "confusions (rn->m, cl->d, ff/fi), and broken line-wraps. Output ONLY the "
    "corrected text with proper spacing and spelling, preserving the exact "
    "meaning, technical terms, names, numbers and equations. Do not summarize, "
    "do not add commentary, do not omit content. If a word is truly "
    "unrecoverable, leave it exactly as-is."
)


def _garble_score(text: str) -> float:
    """Fraction of run-together long 'words' — the missing-space signal."""
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) > 15) / len(words)


def needs_clean(seed: str) -> bool:
    return bool(seed) and (_garble_score(seed) >= 0.04 or seed.count("  ") >= 8)


def _sane(original: str, cleaned: str) -> bool:
    """Reject empty / runaway / degenerate output so we never worsen a chunk."""
    if not cleaned or not cleaned.strip():
        return False
    lo, lc = len(original), len(cleaned)
    if lc < lo * 0.5 or lc > lo * 2.0:
        return False
    if re.search(r"(.)\1{20,}", cleaned):  # one char/glyph spammed
        return False
    return True


async def _clean_seed(client: LMStudioClient, model: str, seed: str) -> str:
    """Return a cleaned seed, or the original unchanged on any failure."""
    try:
        resp = await client.chat(
            model,
            [ChatMessage(role="system", content=_SYS),
             ChatMessage(role="user", content=seed)],
            temperature=1.0,
            top_p=1.0,
            max_tokens=min(1500, len(seed) // 3 + 300),
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception:
        return seed
    out = (resp.get("content") or "").strip()
    return out if _sane(seed, out) else seed


def _splice_seed(content: str, cleaned: str) -> str:
    """Replace the SEED body span with `cleaned`, leaving all structure intact."""
    m = _RE_SEED.search(content)
    if not m:
        return content
    return content[: m.start(1)] + cleaned + content[m.end(1) :]


async def clean_pings(
    *, model: str | None = None, concurrency: int = 4, limit: int | None = None
) -> dict:
    settings = get_settings()
    src = settings.knowledge_source.expanduser()
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    out = src.with_name(src.stem + ".cleaned" + src.suffix)
    model = model or settings.model_light

    # Resume: anything already in the sidecar is done.
    done: set[str] = set()
    if out.exists():
        for ping in iter_dream_pings(out):
            done.add(str(ping.get("id") or ""))

    sem = asyncio.Semaphore(concurrency)
    client = LMStudioClient()
    stats = {"total": 0, "clean": 0, "cleaned": 0, "fallback": 0, "skipped_done": len(done)}
    # asyncio is single-threaded; line-atomic appends need no extra lock.
    fh = out.open("ab")

    async def handle(ping: dict) -> None:
        content = str(ping.get("content") or "")
        seed = _seed(content)
        if needs_clean(seed):
            async with sem:
                cleaned = await _clean_seed(client, model, seed)
            if cleaned != seed:
                ping["content"] = _splice_seed(content, cleaned)
                stats["cleaned"] += 1
            else:
                stats["fallback"] += 1
        else:
            stats["clean"] += 1
        fh.write(orjson.dumps(ping) + b"\n")
        done_n = stats["clean"] + stats["cleaned"] + stats["fallback"]
        if done_n % 50 == 0:
            print(f"  ...{done_n} processed "
                  f"(cleaned={stats['cleaned']} fallback={stats['fallback']})",
                  flush=True)

    try:
        tasks: list = []
        for ping in iter_dream_pings(src):
            pid = str(ping.get("id") or "")
            if pid in done:
                continue
            stats["total"] += 1
            if limit and stats["total"] > limit:
                stats["total"] -= 1
                break
            tasks.append(asyncio.create_task(handle(ping)))
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        fh.close()
        await client.aclose()

    stats["output"] = str(out)
    return stats
