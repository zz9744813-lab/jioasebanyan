"""轻量事件总线。订阅者通过 asyncio.Queue 接收事件。

事件结构: {"type": str, "ts": float, "payload": dict}
所有事件可 JSON 序列化，直接喂给 SSE。

EventBus 是进程级单例，CLI 模式下没有订阅者时事件被丢弃，零开销。
"""
import asyncio
import time
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        evt = {"type": event_type, "ts": time.time(), "payload": payload}
        for q in self._subscribers:
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(evt)
                except Exception:
                    pass


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus


def emit(event_type: str, **payload: Any) -> None:
    _bus.emit(event_type, payload)
