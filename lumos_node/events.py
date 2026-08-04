"""In-process event bus for server→client push (Phase 2 delivery).

Before this, the ONLY operator-facing channel was the request-scoped /api/chat
SSE stream. Autonomous turns (alert wakes) have no inbound request to ride on,
so they publish here and a standing GET /api/events SSE endpoint fans events
out to every connected HUD tab.

Design:
  - One EventBus lives on app.state, created at lifespan startup.
  - Each connected SSE client gets its own bounded asyncio.Queue subscriber.
  - publish() is NON-BLOCKING: a slow/full subscriber drops the event for THAT
    client rather than back-pressuring the producer — an autonomous turn must
    never block on a stalled browser tab.
  - A ring buffer of recent events is replayed to a freshly-connected client.
    Autonomous turns publish a final self-contained `message` event (full text),
    so even if the streamed `delta`s aged out of the ring, a tab connecting after
    the wake still reconstructs the whole message from that one buffered event.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from .log import get_logger

log = get_logger(__name__)

_MAX_QUEUE = 256       # per-subscriber backlog before we drop (slow-tab guard)
_REPLAY_BUFFER = 50    # recent events replayed to a fresh subscriber


class EventBus:
    """Fan-out pub/sub over per-subscriber asyncio.Queues."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=_REPLAY_BUFFER)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        """Fan an event out to all subscribers + buffer it for replay.
        Non-blocking: a full subscriber queue drops the event for that client
        only (its tab is stalled), never the producer."""
        self._recent.append(event)
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.info("eventbus.subscriber_full_dropped")

    def recent(self) -> list[dict[str, Any]]:
        """Snapshot of buffered recent events (replay to a new subscriber)."""
        return list(self._recent)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
