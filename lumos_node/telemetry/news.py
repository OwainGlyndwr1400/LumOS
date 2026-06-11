"""News telemetry client — RSS/Atom feeds + public Telegram channel previews.

Mirrors the feed set Osiris uses for its SIGINT-news layer
(E:/Claude code projects 2/Osiris/osiris/src/app/api/news/route.ts):
  - 4 public Telegram channels scraped via the t.me/s/<channel> web-preview HTML
    page (NOT the Bot API): OSINTtechnical, Faytuks, Liveuamap, CyberKnow.
  - RSS fallback feeds (used when Telegram is IP-blocked): BBC World, Al Jazeera,
    GDACS — extended here with Reuters World and AP Top as the spec's named
    fallbacks.

Parsing is stdlib-only: xml.etree.ElementTree for RSS 2.0 <item> and Atom
<entry>; a narrow regex pass for the t.me/s HTML preview. NO feedparser dep.

Network discipline (identical to cosmic.py / airspace.py):
  - httpx.Timeout(8.0, connect=4.0) per request, passed explicitly every call.
  - Each feed fetched concurrently inside ONE shared httpx.AsyncClient.
  - Every fetcher returns {"ok": bool, ...}; per-feed-set TTL cache via tcache.
  - Errors swallowed narrowly (httpx.HTTPError, ValueError); never raises.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx

from ..config import get_settings
from ..log import get_logger
from . import cache as tcache


log = get_logger(__name__)


_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
_TELEGRAM_BASE = "https://t.me/s"

# Per-feed-set / per-channel cache TTL. The integration step adds a "news" key
# to tcache.DEFAULT_TTL_SECONDS (target 300s); until then we resolve it
# defensively so a missing key can't KeyError out of the narrow swallow clause.
_NEWS_TTL = 300

# Spoofed Chrome UA — t.me/s/ and some RSS edges 403 a bare httpx UA. This is a
# politeness/compat header, NOT a security boundary (mirrors Osiris stealthFetch).
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Default RSS/Atom feed set. Osiris recon surfaced BBC World, Al Jazeera all,
# GDACS; the spec's named fallbacks (Reuters World, AP Top) lead the list.
_DEFAULT_FEEDS: list[str] = [
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.apnews.com/rss/apf-topnews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.gdacs.org/xml/rss.xml",
]

# Human-readable source labels keyed by host substring (best-effort; the feed's
# own <title>/<channel> wins when present, this is the fallback).
_SOURCE_LABELS: list[tuple[str, str]] = [
    ("reuters.com", "Reuters"),
    ("apnews.com", "AP"),
    ("bbci.co.uk", "BBC World"),
    ("aljazeera.com", "Al Jazeera"),
    ("gdacs.org", "GDACS"),
    ("nytimes.com", "NYT"),
]

# Telegram preview scraping is allowlisted — only these channels may be fetched,
# matching the Osiris hardcoded primary set. Anything else is rejected.
_TELEGRAM_ALLOWLIST: frozenset[str] = frozenset(
    {"OSINTtechnical", "Faytuks", "Liveuamap", "CyberKnow"}
)

# Atom namespace used by ElementTree tag lookups.
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

# t.me/s preview message block + sub-field regexes (HTML, not XML).
_TG_BLOCK_RE = re.compile(
    r'<div class="tgme_widget_message_wrap.*?</div>\s*</div>\s*</div>',
    re.DOTALL,
)
_TG_TEXT_RE = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.DOTALL
)
_TG_TIME_RE = re.compile(r'<time[^>]*datetime="([^"]+)"')
_TG_LINK_RE = re.compile(r'<a class="tgme_widget_message_date" href="([^"]+)"')
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _source_label(url: str) -> str:
    """Best-effort human source name from a feed/permalink URL."""
    low = (url or "").lower()
    for needle, label in _SOURCE_LABELS:
        if needle in low:
            return label
    return "RSS"


def _parse_date(raw: str | None) -> str | None:
    """Normalize an RSS/Atom date string to an ISO-8601 UTC string, or None.

    Accepts RFC-822 (RSS pubDate) and ISO-8601 (Atom updated/published).
    Every coercion is guarded — a malformed date just yields None.
    """
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if dt is None:
        return None
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError):
        return None


def _sort_key(item: dict) -> str:
    """Sort helper — published ISO string, empties sort last (oldest)."""
    return item.get("published") or ""


def _item_id(link: str | None, published: str | None) -> str:
    """Stable id = md5(link + published), mirroring Osiris item ids."""
    basis = f"{link or ''}{published or ''}".encode("utf-8", "ignore")
    return hashlib.md5(basis).hexdigest()


def _strip_tags(html: str) -> str:
    """Crude HTML→text for Telegram preview blocks (no parser dep)."""
    text = html.replace("<br/>", " ").replace("<br>", " ").replace("<br />", " ")
    text = _TAG_STRIP_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_feed(xml_text: str, feed_url: str) -> list[dict]:
    """Parse RSS 2.0 <item> or Atom <entry> into normalized news dicts.

    Returns [] on any parse failure (swallowed) — never raises. Each dict:
      {id, title, link, published, source}
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.info("telemetry.news.parse_failed", url=feed_url, error=str(e))
        return []

    out: list[dict] = []

    # Feed-level source label: prefer <channel><title> (RSS) or Atom <title>.
    feed_title = None
    channel = root.find("channel")
    if channel is not None:
        t = channel.find("title")
        if t is not None and t.text:
            feed_title = t.text.strip()
    if feed_title is None:
        at = root.find(f"{_ATOM_NS}title")
        if at is not None and at.text:
            feed_title = at.text.strip()
    source = feed_title or _source_label(feed_url)

    # RSS 2.0 items live under <channel>; scan both channel and root to be safe.
    items = []
    if channel is not None:
        items = channel.findall("item")
    if not items:
        items = root.findall(".//item")

    for it in items:
        title_el = it.find("title")
        link_el = it.find("link")
        date_el = it.find("pubDate")
        if date_el is None:
            # Some RSS dialects use Dublin Core <dc:date>.
            date_el = it.find("{http://purl.org/dc/elements/1.1/}date")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        published = _parse_date(date_el.text if date_el is not None else None)
        if not title and not link:
            continue
        out.append(
            {
                "id": _item_id(link, published),
                "title": title,
                "link": link,
                "published": published,
                "source": source,
            }
        )

    # Atom <entry> (namespaced). Only scan if RSS items were absent.
    if not out:
        for en in root.findall(f"{_ATOM_NS}entry"):
            title_el = en.find(f"{_ATOM_NS}title")
            link = ""
            for ln in en.findall(f"{_ATOM_NS}link"):
                href = ln.get("href")
                rel = ln.get("rel") or "alternate"
                if href and rel == "alternate":
                    link = href.strip()
                    break
                if href and not link:
                    link = href.strip()
            date_el = en.find(f"{_ATOM_NS}published")
            if date_el is None:
                date_el = en.find(f"{_ATOM_NS}updated")
            title = (title_el.text or "").strip() if title_el is not None else ""
            published = _parse_date(date_el.text if date_el is not None else None)
            if not title and not link:
                continue
            out.append(
                {
                    "id": _item_id(link, published),
                    "title": title,
                    "link": link,
                    "published": published,
                    "source": source,
                }
            )

    return out


async def _get_text(client: httpx.AsyncClient, url: str, **kwargs) -> str | None:
    """GET → response text with error swallowing. Returns None on any failure.

    The text analogue of cosmic.py:_get_json — same narrow except clause,
    same explicit per-request timeout, same structured-log on failure.
    """
    headers = {"User-Agent": _USER_AGENT}
    headers.update(kwargs.pop("headers", {}) or {})
    try:
        r = await client.get(url, timeout=_TIMEOUT, headers=headers, **kwargs)
        r.raise_for_status()
        return r.text
    except (httpx.HTTPError, ValueError) as e:
        log.info("telemetry.news.fetch_failed", url=url, error=str(e))
        return None


async def _fetch_one_feed(client: httpx.AsyncClient, feed_url: str) -> list[dict]:
    """Fetch + parse a single RSS/Atom feed. Returns [] on failure."""
    text = await _get_text(client, feed_url)
    if not text:
        return []
    return _parse_feed(text, feed_url)


async def fetch_news(feeds: list[str] | None = None, limit: int = 30) -> dict:
    """Fetch every feed concurrently, merge + sort by date desc, cap at `limit`.

    Args:
      feeds: list of RSS/Atom URLs. None → the default Osiris-aligned set.
      limit: max merged items returned.

    Returns {"ok": bool, "count": int, "items": [{id,title,link,published,source}]}.
    Cached per-feed-set (sorted feed URLs hashed into the key) for 300s. The
    failure/empty path is cached too, so a down feed-set won't retry-spam.
    """
    feed_list = list(feeds) if feeds else list(_DEFAULT_FEEDS)
    if limit < 1:
        limit = 1

    # Cache key embeds the feed-set (order-independent) + limit, with explicit
    # precision on the int. Distinct feed-sets never collide.
    set_hash = hashlib.md5(
        "|".join(sorted(feed_list)).encode("utf-8", "ignore")
    ).hexdigest()[:12]
    cache_key = f"news_{set_hash}_{limit:d}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_fetch_one_feed(client, url) for url in feed_list)
        )

    merged: list[dict] = []
    seen: set[str] = set()
    for batch in results:
        for item in batch:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            merged.append(item)

    if not merged:
        result = {"ok": False, "count": 0, "items": []}
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("news", _NEWS_TTL))
        return result

    merged.sort(key=_sort_key, reverse=True)
    merged = merged[:limit]
    result = {"ok": True, "count": len(merged), "items": merged}
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("news", _NEWS_TTL))
    return result


def _parse_telegram(html: str, channel: str, limit: int = 8) -> list[dict]:
    """Regex-parse a t.me/s/<channel> HTML preview into news dicts.

    Mirrors the Osiris tgme_widget_message_wrap extraction: text + datetime +
    t.me permalink, last `limit` messages. Returns [] on no matches.
    """
    out: list[dict] = []
    for block in _TG_BLOCK_RE.findall(html):
        text_m = _TG_TEXT_RE.search(block)
        time_m = _TG_TIME_RE.search(block)
        link_m = _TG_LINK_RE.search(block)
        title = _strip_tags(text_m.group(1)) if text_m else ""
        if not title:
            continue
        link = link_m.group(1).strip() if link_m else f"https://t.me/s/{channel}"
        published = _parse_date(time_m.group(1) if time_m else None)
        out.append(
            {
                "id": _item_id(link, published),
                "title": title,
                "link": link,
                "published": published,
                "source": f"t.me/{channel}",
            }
        )
    # Preview page lists oldest→newest; keep the most recent `limit`.
    return out[-limit:]


async def fetch_telegram_channel(channel: str) -> dict:
    """Fetch + parse a public Telegram channel's t.me/s/<channel> web preview.

    ALLOWLISTED ONLY — `channel` must be in _TELEGRAM_ALLOWLIST or this returns
    an ok:False payload without any network call.

    Returns {"ok": bool, "count": int, "items": [...]}. Cached per channel for
    300s; failure/empty path cached too.
    """
    channel = (channel or "").strip().lstrip("@")
    if channel not in _TELEGRAM_ALLOWLIST:
        log.info("telemetry.news.telegram_rejected", channel=channel)
        return {"ok": False, "count": 0, "items": [], "error": "channel not allowlisted"}

    cache_key = f"news_tg_{channel}"
    cached = tcache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{_TELEGRAM_BASE}/{channel}"
    async with httpx.AsyncClient() as client:
        html = await _get_text(client, url)

    if not html:
        result = {"ok": False, "count": 0, "items": []}
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("news", _NEWS_TTL))
        return result

    items = _parse_telegram(html, channel)
    if not items:
        result = {"ok": False, "count": 0, "items": []}
        tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("news", _NEWS_TTL))
        return result

    items.sort(key=_sort_key, reverse=True)
    result = {"ok": True, "count": len(items), "items": items}
    tcache.put(cache_key, result, ttl_seconds=tcache.DEFAULT_TTL_SECONDS.get("news", _NEWS_TTL))
    return result
