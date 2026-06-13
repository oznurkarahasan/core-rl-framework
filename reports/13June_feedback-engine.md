# Core-RL Framework — Phase 5: Feedback Engine Report

**Date:** June 13, 2026  
**Scope:** Phase 5 — FastAPI Backend (Task 5.1) + Reward-Update Bridge (Task 5.2)  
**Status:** Completed

---

## Overview

Phase 5 closes the human-in-the-loop cycle. Phases 3 and 4 built the queue that collects uncertain states and the EWC layer that protects learned knowledge. Phase 5 exposes that queue over HTTP so a human teacher UI can review flagged states and submit corrected rewards — which then flow back into the Replay Buffer and change what the agent trains on next.

By the end of this session: **148 tests passing.**

---

## Task 5.1 — FastAPI Backend

**File:** `src/core_rl/api/app.py` (new)

### Endpoint Design

The `todo.md` spec listed `/get_uncertain_state` and `/submit_label`. These were replaced with RESTful equivalents that are easier to evolve:

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/health` | none | Liveness probe — always open, no auth required |
| `GET` | `/queue/status` | optional | Count, capacity, and fill rate |
| `GET` | `/queue/items` | optional | All pending items, oldest first |
| `POST` | `/queue/resolve/{id}` | optional | Approve item and submit human reward |

`/health` is intentionally excluded from auth so monitoring tools and container orchestrators can reach it without API credentials.

### App Factory

The API is created via a factory function, not a module-level app:

```python
app = create_app(queue, on_resolve=lambda id, r: buffer.update_reward(id, r))
```

This keeps the app fully testable — each test creates its own `create_app(queue)` instance with a fresh `ReviewQueue`. There is no shared global state between tests.

### Authentication

Optional API key, controlled by a single environment variable:

```bash
# Local development — no auth, zero friction
unset AL_API_KEY

# Remote / production — all queue endpoints require X-API-Key header
export AL_API_KEY=your-secret-here
```

When `AL_API_KEY` is set, the key is read once at `create_app()` time and injected as a FastAPI `Depends` on the queue router. When unset or empty, the dependency is not added. `/health` is never behind the dependency.

The header name is `X-API-Key` — the conventional name for custom API key authentication. Standard HTTP clients and tools like `curl` handle it without any special setup.

### State Serialisation

`ReviewQueue` items contain `state`, which may be a NumPy array, a nested dict of arrays, or any other observation format the environment returns. FastAPI's default JSON encoder does not handle NumPy. A recursive `_serialize()` helper converts arrays to nested lists at the `/queue/items` response boundary — the rest of the codebase remains NumPy-native.

### on_resolve Callback

The API has no knowledge of the Replay Buffer. The wiring is done by the caller:

```python
app = create_app(queue, on_resolve=lambda id, r: buffer.update_reward(id, r))
```

`on_resolve` is called after `queue.resolve()` succeeds. If it is `None`, the resolve still works — the item is removed from the queue, the 200 response is returned, and the buffer is simply not updated. This makes the callback opt-in and safe to omit in tests or scenarios where replay buffer feedback is not needed.

---

## Task 5.2 — Reward-Update Bridge

The bridge is the wiring between the API callback and `GenericReplayBuffer.update_reward()`. Three components changed to make this work correctly.

### ReviewQueue — `item_id` Parameter

`push()` gained an optional `item_id` parameter. If provided, that UUID is used as the item's identifier instead of generating a new one:

```python
# Before: always generates a new UUID
item_id = queue.push(state, entropy)

# After: caller can provide their own UUID to share with the buffer
exp_id = str(uuid.uuid4())
buffer.push(state, action, reward, next_state, done, experience_id=exp_id)
queue.push(state, entropy, item_id=exp_id)   # same UUID — bridge works
```

`ReviewQueue` also gained a `capacity` property (`self._queue.maxlen`) so the API's `/queue/status` endpoint can report fill rate without exposing the internal deque.

### GenericReplayBuffer — `experience_id` and `update_reward()`

`push()` gained an optional `experience_id: str` parameter. When provided, the buffer records the association needed for `update_reward()`.

`update_reward(experience_id, new_reward)` updates the reward field of the corresponding stored experience, returning `True` on success and `False` if the experience was not found or has been evicted.

#### O(1) Implementation

The first implementation used `_id_to_counter: Dict[str, int]` to map ids to push sequence numbers, then computed a deque index from that number. This had two problems identified in review:

1. **`self.buffer[idx]` is O(n)** — Python's `deque.__getitem__` traverses the linked structure for middle elements. Computing the correct `idx` via arithmetic is O(1), but the subsequent deque access erases that benefit.

2. **`_id_to_counter` was never cleaned up** — evicted items' ids accumulated in the dict indefinitely. At 100k capacity with all experiences tracked, this would grow to 100k entries and stay there forever.

The fix replaced `_id_to_counter` with two structures:

| Structure | Type | Purpose |
|-----------|------|---------|
| `_id_to_item` | `Dict[str, List]` | Direct reference to the list object inside the deque — O(1) update |
| `_id_deque` | `deque(maxlen=capacity)` | Parallel deque of ids (None for untracked) — automatic eviction |

```python
# push() — before deque evicts, clean up the outgoing item's id
if len(self.buffer) == self._capacity:
    evicted_id = self._id_deque[0]      # O(1) — left end of deque
    if evicted_id is not None:
        self._id_to_item.pop(evicted_id, None)

item = [state, action, reward, next_state, done]
self.buffer.append(item)
self._id_deque.append(experience_id)
if experience_id is not None:
    self._id_to_item[experience_id] = item  # store reference, not copy

# update_reward() — O(1) in full
item = self._id_to_item.get(experience_id)  # O(1) dict lookup
if item is None:
    return False
item[2] = new_reward                        # O(1) list index assignment
return True
```

**Why `_id_deque` solves eviction cleanup:** both `self.buffer` and `_id_deque` are `deque(maxlen=capacity)`. When a new item is appended and the deque is full, both deques evict their leftmost element simultaneously. Before `buffer.append()` runs, `_id_deque[0]` is read — it holds the id of the experience that is about to be evicted. That id is removed from `_id_to_item`. After the appends, both deques are in sync and `_id_to_item` never grows beyond the number of tracked experiences currently in the buffer.

**List vs tuple storage:** storing list objects (instead of tuples) was necessary for in-place mutation (`item[2] = new_reward`). This does not affect `deque(maxlen=N)` eviction — the deque evicts by item count, not by item type.

---

## Files Summary

### New Files Created

| File | Purpose |
|------|---------|
| `src/core_rl/api/__init__.py` | Package init exporting `create_app` |
| `src/core_rl/api/app.py` | FastAPI app factory, all endpoints, auth, state serialisation |
| `tests/test_api.py` | 19 tests covering all endpoints, auth, and reward-update bridge |

### Modified Files

| File | Changes |
|------|---------|
| `src/core_rl/buffer/buffer.py` | `experience_id` param in `push()`; `_id_to_item` + `_id_deque` for O(1) `update_reward()`; list storage instead of tuples |
| `src/core_rl/active_learning/review_queue.py` | `capacity` property; `item_id` param in `push()` |
| `pyproject.toml` | Replaced `flask>=3.0.0` with `fastapi>=0.100.0` in core deps; added `[api]` optional group (`uvicorn[standard]`); added `httpx` to dev deps for `TestClient` |

---

## Test Summary

| Test Class | Tests | What Is Covered |
|------------|-------|-----------------|
| `TestHealth` | 3 | 200 response; body; accessible without API key even when key is set |
| `TestQueueStatus` | 2 | Empty queue; partial fill rate calculation |
| `TestQueueItems` | 4 | Empty list; expected keys; numpy state serialised to list; oldest-first ordering |
| `TestResolveItem` | 5 | Known item; item removed after resolve; 404 for unknown; callback triggered; no callback is safe |
| `TestAuth` | 4 | Blocked without key; allowed with correct key; 401 for wrong key; open access when no key set |
| `TestRewardUpdateBridge` | 1 | Full round-trip: push to buffer + queue with shared UUID → resolve via API → buffer reward updated |

| Metric | Phase 4 End | This Session End |
|--------|-------------|-----------------|
| Tests passing | 129 | 148 |

---

## Usage Reference

```python
import uuid
import uvicorn
from core_rl.active_learning.review_queue import ReviewQueue
from core_rl.buffer.buffer import GenericReplayBuffer
from core_rl.api.app import create_app

queue = ReviewQueue(capacity=1000)
buffer = GenericReplayBuffer(capacity=100_000)

app = create_app(queue, on_resolve=lambda id, r: buffer.update_reward(id, r))

# In the training loop — when a state is flagged as uncertain:
exp_id = str(uuid.uuid4())
buffer.push(state, action, reward, next_state, done, experience_id=exp_id)
queue.push(state, entropy, item_id=exp_id)

# Start the server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# Check queue
curl http://localhost:8000/queue/status

# List items
curl http://localhost:8000/queue/items

# Submit human reward for item abc-123
curl -X POST http://localhost:8000/queue/resolve/abc-123 \
     -H "Content-Type: application/json" \
     -d '{"reward": 1.0}'

# With auth
curl http://localhost:8000/queue/status -H "X-API-Key: your-secret"
```

---

## Next Steps

All five phases of the framework are complete. The full pipeline:

1. **Phase 1** — `GenericReplayBuffer` collects agent experiences
2. **Phase 2** — `SACAgent` / `PPOAgent` learn policies from the buffer
3. **Phase 3** — `ActiveLearningCore` + `ReviewQueue` flag uncertain states for human review
4. **Phase 4** — `EWC` prevents catastrophic forgetting when learning new tasks
5. **Phase 5** — FastAPI server exposes the queue; human rewards flow back into the buffer

Possible next directions:
- A minimal web UI (React/plain HTML) that calls the Phase 5 API to display states and submit labels
- CARLA integration — replace Gymnasium environments with CARLA's continuous observation/action space and wire the full pipeline to a real autonomous driving scenario
- Online EWC — accumulate Fisher across multiple tasks rather than overwriting on each `consolidate()` call
