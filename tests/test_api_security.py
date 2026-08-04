from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from lumos_node.api.security import require_api_access, require_pele_access
from lumos_node.config import get_settings


def _request(
    *,
    method: str = "GET",
    client: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
    query: bytes = b"",
) -> Request:
    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1"))
        for k, v in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": method,
            "path": "/api/chat",
            "headers": raw_headers,
            "query_string": query,
            "client": (client, 12345),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_loopback_without_token_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "api_token", "")

    require_api_access(_request(method="POST", client="127.0.0.1"))


def test_remote_without_token_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "api_token", "")

    with pytest.raises(HTTPException) as exc:
        require_api_access(_request(method="GET", client="192.168.1.50"))

    assert exc.value.status_code == 403


def test_configured_token_allows_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "api_token", "secret-token")

    require_api_access(
        _request(
            method="GET",
            client="192.168.1.50",
            headers={"Authorization": "Bearer secret-token"},
        )
    )


def test_unsafe_bad_origin_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(
        settings,
        "api_allowed_origins",
        "http://localhost:5173,http://127.0.0.1:5173",
    )

    with pytest.raises(HTTPException) as exc:
        require_api_access(
            _request(
                method="POST",
                client="127.0.0.1",
                headers={"Origin": "https://bad.example"},
            )
        )

    assert exc.value.status_code == 403


async def test_pele_unconfigured_blocks_even_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "pele_token", "")

    with pytest.raises(HTTPException) as exc:
        await require_pele_access(_request(method="POST", client="127.0.0.1"))

    assert exc.value.status_code == 403


async def test_pele_header_allows_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "pele_token", "ring-token")

    await require_pele_access(
        _request(
            method="POST",
            client="127.0.0.1",
            headers={"X-Lumos-PELE": "ring-token"},
        )
    )


async def test_pele_wrong_header_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "pele_token", "ring-token")

    with pytest.raises(HTTPException) as exc:
        await require_pele_access(
            _request(
                method="POST",
                client="127.0.0.1",
                headers={"X-Lumos-PELE": "wrong"},
            )
        )

    assert exc.value.status_code == 403
