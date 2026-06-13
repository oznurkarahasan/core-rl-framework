import time
import threading
import uuid
from collections import deque
from typing import Any, Dict, List, Optional


class ReviewQueue:
    """
    Thread-safe FIFO queue for human review of uncertain agent states.

    The simulation loop calls push() without blocking. When capacity is
    reached, the oldest (stalest) item is silently evicted — a fresh
    uncertain state is always more actionable than a stale one.

    A Flask server (Phase 5) calls get() and resolve() from a separate
    thread. Dependency is one-directional: Flask knows ReviewQueue,
    ReviewQueue knows nothing about Flask.
    """

    def __init__(self, capacity: int = 1000):
        self._queue: deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._queue.maxlen  # type: ignore[return-value]

    def push(self, state: Any, entropy: float, item_id: Optional[str] = None) -> str:
        """
        Enqueue an uncertain state. Returns the item id (UUID).

        Pass item_id to share the same UUID with a GenericReplayBuffer push,
        so on_resolve callbacks can call buffer.update_reward(item_id, reward).

        Non-blocking: if at capacity, the oldest item is evicted before
        the new one is inserted.
        """
        item_id = item_id or str(uuid.uuid4())
        item: Dict[str, Any] = {
            "id": item_id,
            "state": state,
            "entropy": entropy,
            "timestamp": time.time(),
        }
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                evicted = self._queue[0]
                self._index.pop(evicted["id"], None)
            self._queue.append(item)
            self._index[item_id] = item
        return item_id

    def get(self) -> List[Dict[str, Any]]:
        """Return all pending items, oldest first. Does not consume them."""
        with self._lock:
            return list(self._queue)

    def resolve(self, item_id: str, reward: float) -> Optional[Dict[str, Any]]:
        """
        Mark an item as reviewed by a human.

        Removes the item from the queue and returns it with the
        human-assigned reward attached under the key ``human_reward``.
        Returns None if the id is not found (already resolved or evicted).

        Phase 5 uses the returned dict to update the Replay Buffer reward.
        """
        with self._lock:
            item = self._index.pop(item_id, None)
            if item is None:
                return None
            try:
                self._queue.remove(item)
            except ValueError:
                pass
            return {**item, "human_reward": reward}

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
