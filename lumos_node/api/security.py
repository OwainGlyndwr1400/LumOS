"""HTTP security perimeter for the local Lumos API.

Default posture preserves the existing developer loop: requests from loopback
work without a token. Any non-loopback caller must use LUMOS_API_TOKEN.
Unsafe browser requests also need a trusted Origin/Referer, which blocks
drive-by local POSTs from random web pages.

Loopback trust is refused for requests that carry proxy/forwarding headers
(X-Forwarded-For, Forwarded, ...). request.client.host reports only the
immediate TCP peer, so a reverse proxy or tunnel (nginx, cloudflared, ngrok,
Tailscale Funnel) in front of the loopback port would otherwise make every
forwarded internet request look like it came from 127.0.0.1. To expose the API
beyond this machine, set LUMOS_API_TOKEN — do not rely on loopback identity
once a proxy can reach the port.
"""

from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status

from ..config import Settings, get_settings
from ..log import get_logger

log = get_logger(__name__)

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# Headers that indicate the request was relayed by a proxy/tunnel rather than
# arriving directly on the loopback socket. A direct local client (the HUD, the
# same-box Discord bridge) sets none of these; a reverse proxy/tunnel sets at
# least one. Their presence voids loopback-as-identity trust.
_FORWARDING_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "forwarded",
    "cf-connecting-ip",
    "true-client-ip",
    "x-original-forwarded-for",
)


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    h = host.lower().strip("[]")
    return h in LOOPBACK_HOSTS or h.startswith("127.")


def _looks_proxied(request: Request) -> bool:
    """True if the request carries any proxy/forwarding header (Starlette header
    lookup is case-insensitive), i.e. it did not arrive directly on loopback."""
    return any(h in request.headers for h in _FORWARDING_HEADERS)


def _origin_from_referer(referer: str | None) -> str | None:
    if not referer:
        return None
    parsed = urlparse(referer)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _allowed_origins(settings: Settings) -> set[str]:
    return {
        origin.strip().rstrip("/")
        for origin in settings.api_allowed_origins.split(",")
        if origin.strip()
    }


def _extract_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    header_token = request.headers.get("x-lumos-token", "")
    if header_token:
        return header_token.strip()
    # EventSource cannot send headers, so token query support is intentionally
    # allowed for SSE and remote HUD use. Prefer headers for normal fetch calls.
    return request.query_params.get("token", "").strip()


def _extract_pele_header_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("pele "):
        return auth[5:].strip()
    header_token = request.headers.get("x-lumos-pele", "")
    return header_token.strip()


async def _extract_pele_token(
    request: Request,
    body_token: str | None = None,
) -> str:
    if body_token:
        return body_token.strip()
    header_token = _extract_pele_header_token(request)
    if header_token:
        return header_token
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return ""
    try:
        body: Any = await request.json()
    except Exception:  # noqa: BLE001 - malformed/non-object JSON means no token.
        return ""
    if isinstance(body, dict):
        token = body.get("pele_token")
        if isinstance(token, str):
            return token.strip()
    return ""


def _enforce_origin(request: Request, settings: Settings) -> None:
    if request.method.upper() in SAFE_METHODS:
        return

    origin = request.headers.get("origin")
    if not origin:
        origin = _origin_from_referer(request.headers.get("referer"))

    # Non-browser callers commonly send neither Origin nor Referer.
    if not origin:
        return

    normalized = origin.rstrip("/")
    if normalized not in _allowed_origins(settings):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="origin not allowed",
        )


def require_api_access(request: Request) -> None:
    """FastAPI dependency that keeps local dev easy and remote exposure guarded."""

    settings = get_settings()
    _enforce_origin(request, settings)

    configured_token = settings.api_token.strip()
    if configured_token:
        supplied_token = _extract_token(request)
        if secrets.compare_digest(supplied_token, configured_token):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="valid Lumos API token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    client_host = request.client.host if request.client else None
    # Trust loopback ONLY for a genuinely direct local client. A loopback peer
    # that also carries forwarding headers is a proxy/tunnel relaying remote
    # traffic — loopback-as-identity cannot be trusted there, so fail closed.
    if _is_loopback_host(client_host) and not _looks_proxied(request):
        return

    if _is_loopback_host(client_host):
        log.warning(
            "api.loopback_trust_refused_proxied",
            reason="request arrived on loopback but carried proxy/forwarding "
            "headers; refusing loopback trust. Set LUMOS_API_TOKEN to expose "
            "the API through a reverse proxy or tunnel.",
            path=request.url.path,
        )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="remote API access requires LUMOS_API_TOKEN",
    )


async def verify_pele_access(
    request: Request,
    body_token: str | None = None,
) -> None:
    """Require the PELE privileged-action token.

    This gate deliberately fails closed when LUMOS_PELE_TOKEN is unset. Localhost
    bypass applies only to normal API access, never to privileged mutation.
    """

    configured_token = get_settings().pele_token.strip()
    if not configured_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="PELE token not configured; privileged actions are locked",
        )
    supplied_token = await _extract_pele_token(request, body_token)
    if secrets.compare_digest(supplied_token, configured_token):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="valid PELE token required",
    )


async def require_pele_access(request: Request) -> None:
    await verify_pele_access(request)


def client_error_detail(public: str, exc: Exception | None = None) -> str:
    """Return public-safe error text unless debug_errors is explicitly enabled."""

    if exc is not None and get_settings().debug_errors:
        return f"{public}: {exc}"
    return public
