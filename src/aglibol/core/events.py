"""Asynchronous event bus for decoupled agent communication and telemetry."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """Event payload emitted during workflow execution."""

    event_type: str
    session_id: str = ""
    step: int = 0
    agent: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


EventHandler = Callable[[AgentEvent], Coroutine[Any, Any, None]]


class EventBus:
    """Async pub/sub event bus supporting multiple subscribers per event type."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._wildcard_subscribers: list[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        if event_type == "*":
            self._wildcard_subscribers.append(handler)
        else:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unregister a handler."""
        if event_type == "*" and handler in self._wildcard_subscribers:
            self._wildcard_subscribers.remove(handler)
        elif handler in self._subscribers.get(event_type, []):
            self._subscribers[event_type].remove(handler)

    async def emit(self, event: AgentEvent) -> None:
        """Publish an event to all matching subscribers."""
        handlers = list(self._subscribers.get(event.event_type, [])) + list(
            self._wildcard_subscribers
        )
        if not handlers:
            return

        tasks = []
        for h in handlers:
            if asyncio.iscoroutinefunction(h):
                tasks.append(asyncio.create_task(h(event)))
            else:
                tasks.append(asyncio.to_thread(h, event))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger = logging.getLogger("aglibol.events")
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Event handler failed: {r}")
