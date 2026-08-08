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
_DEAD_SUBSCRIBER_STRIKES = 50  # consecutive full-queue publishes before eviction


class EventBus:
    """Fan-out pub/sub over per-subscriber asyncio.Queues."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._recent: deque[dict[str, Any]] = deque(maxlen=_REPLAY_BUFFER)
        # Consecutive full-queue strikes per subscriber. A tab that never drains
        # is gone (closed/frozen/reloaded without a clean disconnect); after
        # enough strikes we evict it rather than fan out to it forever.
        self._full_streak: dict[asyncio.Queue[dict[str, Any]], int] = {}

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)
        self._full_streak.pop(q, None)

    def publish(self, event: dict[str, Any]) -> None:
        """Fan an event out to all subscribers + buffer it for replay.
        Non-blocking: a full subscriber queue drops the event for that client
        only (its tab is stalled), never the producer.

        Drops are summarised in ONE log line per publish rather than one per
        subscriber — a handful of stalled tabs used to emit hundreds of
        identical lines and bury the real log.
        """
        self._recent.append(event)
        dropped = 0
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
                if q in self._full_streak:
                    del self._full_streak[q]
            except asyncio.QueueFull:
                dropped += 1
                streak = self._full_streak.get(q, 0) + 1
                self._full_streak[q] = streak
                if streak >= _DEAD_SUBSCRIBER_STRIKES:
                    dead.append(q)
        # Mutate the set only AFTER iterating it.
        for q in dead:
            self._subscribers.discard(q)
            self._full_streak.pop(q, None)
        if dropped:
            log.info(
                "eventbus.dropped",
                subscribers=dropped,
                evicted=len(dead),
                live=len(self._subscribers),
            )

    def recent(self) -> list[dict[str, Any]]:
        """Snapshot of buffered recent events (replay to a new subscriber)."""
        return list(self._recent)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
