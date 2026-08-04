"""Web search and URL fetching tools.

`web_search` returns search results (title, URL, snippet). Tavily backend if
TAVILY_API_KEY is set, DuckDuckGo (via ddgs) otherwise.

`fetch_url` deep-fetches a single URL and returns text content. HTML is
stripped to plain prose for token efficiency. SSRF guarded against localhost
and RFC1918 private ranges.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

from ..config import get_settings
from ..log import get_logger
from . import register

log = get_logger(__name__)


_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_HTML_ENTITIES = {
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&mdash;": "—",
    "&ndash;": "–",
    "&hellip;": "…",
}


def _strip_html(html: str) -> str:
    text = _SCRIPT_RE.sub(" ", html)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _is_blocked_url(url: str) -> tuple[bool, str]:
    """Block localhost + RFC1918 private ranges to prevent SSRF."""
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return True, "malformed URL"
    host = (parsed.hostname or "").lower()
    if not host:
        return True, "no host"
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True, "localhost"
    if host.startswith("127.") or host.startswith("10.") or host.startswith("192.168."):
        return True, "private network"
    if host.startswith("169.254."):
        return True, "link-local"
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return True, "private network"
        except (ValueError, IndexError):
            pass
    return False, ""


def _ip_blocked(ip_str: str) -> tuple[bool, str]:
    """True if an IP address string is in a non-public range. Handles IPv4-mapped
    IPv6 (::ffff:127.0.0.1) by normalizing to the embedded v4 address first."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True, f"unparseable IP {ip_str!r}"
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True, f"non-public address {ip}"
    return False, ""


async def _check_url_target(url: str) -> tuple[bool, str]:
    """Authoritative SSRF gate: resolve the host and reject if ANY resolved
    address is non-public. This catches what the hostname-string check cannot —
    decimal/hex/octal IP literals (the OS resolver normalizes them), IPv4-mapped
    IPv6, and public hostnames that resolve to internal IPs (DNS-rebinding class).
    Called for the initial URL AND every redirect hop.

    Residual: a validate-then-connect DNS-rebind race (TOCTOU) is not fully
    closed without pinning the checked IP into the connection; acceptable for a
    single-operator local node whose tool calls are trace-visible."""
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return True, "malformed URL"
    if parsed.scheme not in ("http", "https"):
        return True, f"scheme {parsed.scheme!r} not allowed"
    host = parsed.hostname
    if not host:
        return True, "no host"
    # Cheap literal pre-check (obvious localhost/RFC1918 by name), then the
    # authoritative resolve-based check.
    blocked, reason = _is_blocked_url(url)
    if blocked:
        return True, reason
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        return True, f"DNS resolution failed: {e}"
    if not infos:
        return True, "no addresses resolved"
    for info in infos:
        ip_str = info[4][0]
        blocked, reason = _ip_blocked(ip_str)
        if blocked:
            return True, f"{reason} (host {host} resolved to a blocked address)"
    return False, ""


@register(
    name="web_search",
    description=(
        "Search the public web. Call for recent events, current state, public "
        "references not in local memory. Returns up to 10 results "
        "(title, URL, snippet). Pair with fetch_url to read full content."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (free text)."},
            "max_results": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    },
)
async def web_search(query: str, max_results: int = 5) -> dict:
    settings = get_settings()
    if not query.strip():
        return {"error": "empty query"}

    # SearXNG path (preferred when configured — self-hosted, sovereignty-first).
    # SearXNG aggregates multiple search engines client-side without exposing
    # the operator's query directly to commercial providers. Standard endpoint
    # `/search?format=json` returns {results: [{title, url, content, ...}]}.
    if settings.searxng_url:
        searxng_base = settings.searxng_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(
                    f"{searxng_base}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "safesearch": 0,
                    },
                    headers={"User-Agent": "Lumos/0.1 (local research agent)"},
                )
                r.raise_for_status()
                data = r.json()
                raw_results = (data.get("results") or [])[:max_results]
                return {
                    "provider": "searxng",
                    "query": query,
                    "results": [
                        {
                            "title": x.get("title", "") or "",
                            "url": x.get("url", "") or "",
                            "snippet": (x.get("content") or "")[:500],
                            "engine": x.get("engine"),
                        }
                        for x in raw_results
                    ],
                }
        except Exception as e:  # noqa: BLE001
            log.warning("web_search.searxng_failed", error=str(e))
            # fall through to Tavily / DDG

    # Tavily path (preferred when key configured).
    if settings.tavily_api_key:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": True,
                        "include_raw_content": False,
                    },
                )
                r.raise_for_status()
                data = r.json()
                return {
                    "provider": "tavily",
                    "query": query,
                    "answer": data.get("answer"),
                    "results": [
                        {
                            "title": x.get("title", "") or "",
                            "url": x.get("url", "") or "",
                            "snippet": (x.get("content") or "")[:500],
                            "score": x.get("score"),
                        }
                        for x in (data.get("results") or [])
                    ],
                }
        except Exception as e:  # noqa: BLE001
            log.warning("web_search.tavily_failed", error=str(e))
            # fall through to DDG

    # DuckDuckGo fallback (no API key required).
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError:
            return {
                "error": (
                    "no web search backend available. "
                    "Install `ddgs` or set TAVILY_API_KEY."
                )
            }

    try:
        results: list[dict] = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": hit.get("title", "") or "",
                        "url": hit.get("href", "") or hit.get("url", "") or "",
                        "snippet": (hit.get("body") or hit.get("snippet") or "")[:500],
                    }
                )
        return {"provider": "duckduckgo", "query": query, "results": results}
    except Exception as e:  # noqa: BLE001
        return {"error": f"web search failed: {e}"}


@register(
    name="fetch_url",
    description=(
        "Fetch the text content of a web URL. CALL THIS after web_search to read a "
        "specific result's full content, OR when Erydir gives you a URL to look at. "
        "HTML is stripped to plain prose for readability. Returns up to 80KB. "
        "Blocked: localhost, private/reserved IPs, and redirects to them — "
        "SSRF-checked by resolved IP address on every redirect hop."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "HTTP(S) URL to fetch."},
        },
        "required": ["url"],
    },
)
async def fetch_url(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    max_redirects = 5
    current = url
    try:
        # follow_redirects=False: we follow manually so EVERY hop is re-validated
        # against the SSRF gate (httpx's auto-follow would silently chase a 302
        # to 127.0.0.1 after the first check passed).
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=False,
            headers={"User-Agent": "Lumos/0.1 (local research agent)"},
        ) as client:
            for _hop in range(max_redirects + 1):
                blocked, reason = await _check_url_target(current)
                if blocked:
                    return {"error": f"blocked URL ({reason})"}
                r = await client.get(current)
                location = r.headers.get("location")
                if r.status_code in (301, 302, 303, 307, 308) and location:
                    nxt = urljoin(current, location)
                    if not nxt.startswith(("http://", "https://")):
                        return {"error": "redirect to non-HTTP(S) scheme blocked"}
                    current = nxt
                    continue  # re-validate the new target at the top of the loop
                r.raise_for_status()
                content_type = r.headers.get("content-type", "").lower()
                raw = r.text
                text = _strip_html(raw) if "html" in content_type else raw
                truncated = False
                if len(text) > 80_000:
                    text = text[:80_000]
                    truncated = True
                return {
                    "url": str(r.url),
                    "status": r.status_code,
                    "content_type": content_type,
                    "text": text,
                    "truncated": truncated,
                    "chars": len(text),
                }
            return {"error": f"too many redirects (>{max_redirects})"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {current}"}
    except httpx.TimeoutException:
        return {"error": "request timed out (20s)"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
