"""Conflict indicators — derived-from-news, mirroring Osiris /api/gdelt.

Like Osiris, this module does NOT hit GDELT (its v2 Geo endpoint is frequently
down). Instead it scores world-news headlines against a conflict lexicon and
surfaces the hottest items, so Lumos gets a cheap, keyless war/escalation signal
off the same RSS the news layer already fetches.

Two surfaces:
  - classify_conflict(items)            — PURE TRANSFORM, no network.
  - fetch_conflict_indicators(client)   — COSMIC-style: takes a shared client,
                                          calls news.fetch_news, then classifies.

Mirrors the cosmic.py idiom: identical import header, tcache caching with the
failure/empty path cached, {"ok": bool, ...} return contract, narrow error
swallow (httpx.HTTPError, ValueError).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache
from . import news


log = get_logger(__name__)


_TIMEOUT = httpx.Timeout(8.0, connect=4.0)

# Derived-from-news cache TTL. Integration adds a "conflict" key to
# tcache.DEFAULT_TTL_SECONDS (target 300s, matching the news TTL); resolved
# defensively until then so a missing key can't KeyError out of the swallow.
_CONFLICT_TTL = 300

# Conflict lexicon → per-hit weight. A term hit adds its weight to the item's
# raw score; the item severity is the capped sum. Heavier terms (kinetic /
# escalatory) outweigh softer ones (protest / sanctions). Matched on word
# boundaries, case-insensitive, against title + source.
_CONFLICT_LEXICON: dict[str, int] = {
    "war": 3,
    "invasion": 4,
    "invade": 4,
    "airstrike": 4,
    "air strike": 4,
    "missile": 3,
    "rocket": 2,
    "drone": 2,
    "strike": 2,
    "shelling": 3,
    "artillery": 3,
    "offensive": 3,
    "troops": 2,
    "military": 2,
    "soldiers": 2,
    "killed": 2,
    "casualties": 2,
    "dead": 1,
    "wounded": 1,
    "bomb": 3,
    "bombing": 3,
    "explosion": 2,
    "blast": 2,
    "ceasefire": 2,
    "frontline": 3,
    "front line": 3,
    "siege": 3,
    "clash": 2,
    "clashes": 2,
    "fighting": 2,
    "combat": 2,
    "nuclear": 3,
    "warhead": 4,
    "incursion": 3,
    "mobilization": 2,
    "mobilisation": 2,
    "evacuation": 1,
    "refugees": 1,
    "coup": 3,
    "insurgent": 2,
    "insurgents": 2,
    "militia": 2,
    "hostage": 2,
    "hostages": 2,
    "occupied": 2,
    "annex": 3,
    "sanctions": 1,
    "escalation": 2,
    "escalate": 2,
}

# Per-item severity is capped so one keyword-stuffed headline can't dominate.
_SEVERITY_CAP = 10

# An item with severity >= this is "hot" (surfaced + drives the aggregate).
_HOT_THRESHOLD = 4

# Aggregate conflict_score is 0..100. We scale the mean of the top hot items'
# severities (0..10) and fold in breadth (how many items are hot).
_SCORE_CAP = 100


def _score_text(text: str) -> tuple[int, list[str]]:
    """Score one text blob against the lexicon. Returns (capped_score, terms)."""
    low = (text or "").lower()
    score = 0
    matched: list[str] = []
    for term, weight in _CONFLICT_LEXICON.items():
        # Word-boundary-ish match without regex: pad with spaces so 'war' won't
        # hit 'warehouse' / 'forward'. Cheap and dependency-free.
        needle = term if " " in term else term
        padded = f" {low} "
        if f" {needle} " in padded or f" {needle}, " in padded or f" {needle}. " in padded:
            score += weight
            matched.append(term)
    return min(score, _SEVERITY_CAP), matched


def classify_conflict(items: list[dict]) -> dict:
    """PURE TRANSFORM. Score each news item against the conflict lexicon.

    Args:
      items: list of {id,title,link,published,source} (news.fetch_news shape).

    Returns:
      {
        "ok": bool,                       # True when there was something to score
        "conflict_score": int,            # 0..100 aggregate
        "scored": [ {**item, "severity": int, "terms": [...]} ],  # all items
        "hot_items": [ ...subset severity >= threshold, desc by severity ],
        "summary": str,
      }
    Never raises — guards every field access; an empty/garbage list yields an
    ok:False zero-score payload.
    """
    if not isinstance(items, list) or not items:
        return {
            "ok": False,
            "conflict_score": 0,
            "scored": [],
            "hot_items": [],
            "summary": _summarize(0, []),
        }

    scored: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        source = str(item.get("source") or "")
        severity, terms = _score_text(f"{title} {source}")
        scored.append(
            {
                "id": item.get("id"),
                "title": title,
                "link": item.get("link"),
                "published": item.get("published"),
                "source": source,
                "severity": severity,
                "terms": terms,
            }
        )

    scored.sort(key=lambda x: -x["severity"])
    hot_items = [s for s in scored if s["severity"] >= _HOT_THRESHOLD]

    # Aggregate: mean severity of hot items (0..10) → 0..70, plus breadth bonus
    # (up to +30 as hot-item count climbs). Capped at 100.
    if hot_items:
        mean_sev = sum(h["severity"] for h in hot_items) / len(hot_items)
        breadth = min(len(hot_items), 10) / 10.0  # 0..1
        conflict_score = int(min(_SCORE_CAP, round(mean_sev * 7 + breadth * 30)))
    else:
        conflict_score = 0

    return {
        "ok": True,
        "conflict_score": conflict_score,
        "scored": scored,
        "hot_items": hot_items,
        "summary": _summarize(conflict_score, hot_items),
    }


def _summarize(conflict_score: int, hot_items: list[dict]) -> str:
    """One-line natural-language TLDR of the conflict picture. Pure + guarded."""
    if not hot_items:
        return "no significant conflict signal in current news"
    band = (
        "elevated"
        if conflict_score < 40
        else "high"
        if conflict_score < 70
        else "severe"
    )
    parts: list[str] = [f"conflict score {conflict_score}/100 ({band})"]
    parts.append(f"{len(hot_items)} hot item(s)")
    top = hot_items[0]
    title = (top.get("title") or "").strip()
    if title:
        snippet = title if len(title) <= 80 else title[:77] + "…"
        parts.append(f'top: "{snippet}" (sev {top.get("severity")})')
    return " · ".join(parts)


async def fetch_conflict_indicators(client: httpx.AsyncClient) -> dict:
    """COSMIC-style composite: pull world news, classify it, surface hot items.

    Takes a shared `client` (threaded from a caller's
    `async with httpx.AsyncClient()`) to match the cosmic snapshot_all idiom.
    The `client` arg is part of the contract; the news fetch manages its own
    client internally, so we accept and ignore it for now (kept for signature
    compatibility with the other cosmic-style fetchers / future direct use).

    Returns {"ok", "conflict_score", "hot_items":[...], "summary",
             "total_items", "fetched_at"}. Cached 300s; failure path cached too.
    Never raises — news errors are already swallowed upstream, and classify is pure.
    """
    cache_key = "conflict_news"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    # news.fetch_news swallows its own httpx.HTTPError/ValueError and returns an
    # ok:False payload; we still guard the call defensively per the contract.
    try:
        news_result = await news.fetch_news()
    except (httpx.HTTPError, ValueError) as e:
        log.info("telemetry.conflict.fetch_failed", error=str(e))
        result = {
            "ok": False,
            "conflict_score": 0,
            "hot_items": [],
            "total_items": 0,
            "summary": "conflict indicators unavailable (news fetch failed)",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("conflict", _CONFLICT_TTL))
        return result

    items = news_result.get("items", []) if isinstance(news_result, dict) else []
    classified = classify_conflict(items)

    result = {
        "ok": bool(classified.get("ok")),
        "conflict_score": classified.get("conflict_score", 0),
        "hot_items": classified.get("hot_items", [])[:15],
        "total_items": len(items),
        "summary": classified.get("summary", "no conflict signal"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("conflict", _CONFLICT_TTL))
    return result
