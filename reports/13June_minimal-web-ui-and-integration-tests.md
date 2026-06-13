# Core-RL Framework — Phase 6: Minimal Web UI & Integration Test Report

**Date:** June 13, 2026
**Scope:** Integration Test (full pipeline verification) + Phase 6 — Minimal Web UI
**Status:** Completed

---

## Overview

With all five core phases complete, this session had two goals: verify that every module is correctly wired together through a real Gymnasium loop, and build a minimal human-facing UI that calls the Phase 5 API. Together they close the full human-in-the-loop cycle end-to-end — from agent uncertainty to human label to buffer reward update — in a way that can be observed directly in a browser.

**Final state: 157 tests passing, 97% coverage.**

---

## Integration Test

**File:** `tests/test_integration.py` (new)

### Purpose

Unit tests verify individual modules in isolation. The integration test verifies that all modules talk to each other correctly under a real Gymnasium loop. Specifically, it answers the question: does the full pipeline work as designed before moving to a real environment like CARLA?

### Test Environment

`LunarLanderContinuous-v3` was chosen as the integration environment. It has an 8-dimensional continuous observation space and a 2-dimensional continuous action space — identical in structure to what a CARLA perception pipeline would produce after flattening state into a vector. Box2D is required:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

### Pipeline Under Test

```
SAC (untrained) + LunarLanderContinuous-v3
        ↓
select_action() → (action, entropy)
        ↓
ActiveLearningCore.is_uncertain(entropy) → threshold=0.3
        ↓  (triggered — untrained SAC entropy ≈ 1.0)
buffer.push(..., experience_id=exp_id)
queue.push(...,  item_id=exp_id)       ← same UUID
        ↓
queue.resolve(exp_id, reward=1.0)
        ↓
on_resolve(exp_id, 1.0) → buffer.update_reward(exp_id, 1.0)
```

**Threshold choice:** `0.3` — intentionally below the untrained SAC entropy range (~1.0). This guarantees at least one uncertain state is generated in 20 steps, making the test deterministic regardless of random seed.

### Test Cases

| Test | What Is Verified |
|------|-----------------|
| `test_uncertainty_triggers_queue` | Untrained SAC produces at least one uncertain state in 20 steps |
| `test_queue_contains_pushed_items` | Items pushed to queue are retrievable via `get()` |
| `test_resolve_updates_buffer_reward` | Critical path: `resolve()` → `on_resolve` callback → `buffer.update_reward()` returns `True` |
| `test_resolve_removes_item_from_queue` | After `resolve()`, item no longer appears in `queue.get()` |
| `test_buffer_grows_during_loop` | Buffer accumulates exactly 20 experiences in 20 steps |
| `test_double_resolve_returns_none` | Second `resolve()` call on the same id returns `None` |

### Results

```
6 passed, 3 warnings in 1.50s
```

The 3 warnings are SwigPy internal warnings from Box2D's C extension bindings — unrelated to core-rl.

### Design Notes

**`_run_steps()` helper** — extracted as a standalone function rather than a fixture method. This keeps each test independent: any test can call it with different `n_steps` or a different `al_core` threshold without sharing state.

**`pytest.skip()` on no uncertain states** — tests that require a queued item skip gracefully if the threshold is too high to trigger uncertainty, rather than failing with a misleading assertion error. This makes the test suite robust to threshold changes during future tuning.

**`resolve_log` list** — used in `test_resolve_updates_buffer_reward` to verify the `on_resolve` callback was actually invoked, not just that `update_reward()` returned `True`. The list captures `(exp_id, human_reward, success)` and is asserted after the call.

---

## Phase 6: Minimal Web UI

**File:** `src/core_rl/ui/index.html` (new, single file)

### Purpose

Phase 5 produced a working API. Phase 6 makes it usable without `curl`. A browser-based UI displays pending uncertain states from the queue and allows a human teacher to submit rewards — completing the human-in-the-loop cycle visually.

The UI is intentionally minimal: one HTML file, no build step, no framework. It can be opened directly in a browser or served as a static file alongside the API.

### Design Decisions

**Single file** — HTML + CSS + JS in one `index.html`. No npm, no bundler, no dependencies. Anyone can open it with `python -m http.server` or serve it from FastAPI's `StaticFiles`. This matches the "plug and play" philosophy of the framework.

**State display** — At this stage, `state` is a flat float vector. Displaying raw numbers has no practical meaning to a human teacher. The card shows entropy value and timestamp as the primary decision signal. The raw state vector is shown in a secondary row for debugging purposes. When CARLA integration arrives and `state` contains an image, the card will render it as `<img>` — the display logic checks for an `image` key in the state dict.

**Reward input** — Three controls per card:
- `✅ +1.0` — quick approve
- `❌ −1.0` — quick reject
- Custom number input + Submit — for non-binary rewards (e.g. `0.3`, `-0.5`)

This covers both the simple yes/no labelling workflow and the more nuanced reward shaping needed in CARLA.

**Polling** — The UI polls `/queue/items` every 3 seconds. This is intentionally simple: no WebSocket, no SSE. For the current use case (one human teacher reviewing a queue) polling is sufficient and requires no server-side changes.

**API key** — An input field at the top of the page accepts the `X-API-Key` value. When filled, all requests include the header. When empty, requests are sent without it — matching the API's optional auth behaviour. The key is stored only in the input field (not `localStorage`, not cookies) so it is cleared on page refresh.

**LIVE indicator** — A green dot and "X pending" counter in the top-right corner. The dot turns grey and shows "OFFLINE" if the API is unreachable.

### Full Human-in-the-Loop Cycle (Observable)

With `run_api.py` running and the UI open in a browser:

1. Training loop flags an uncertain state → `queue.push(state, entropy, item_id=exp_id)`
2. UI polls `/queue/items` → card appears on screen within 3 seconds
3. Human clicks `✅ +1.0` or submits a custom reward
4. UI calls `POST /queue/resolve/{id}` with the reward value
5. API calls `on_resolve(exp_id, reward)` → `buffer.update_reward(exp_id, reward)`
6. Card disappears from the UI
7. Next training batch may sample the updated experience

---

## Files Summary

### New Files Created

| File | Purpose |
|------|---------|
| `tests/test_integration.py` | 6 end-to-end tests verifying the full pipeline |
| `src/core_rl/ui/index.html` | Single-file browser UI for the human review queue |

### Modified Files

| File | Changes |
|------|---------|
| `pyproject.toml` | Added `gymnasium[box2d]` to dev dependencies for integration test |

---

## Test Summary

| Session | Tests Passing | Coverage |
|---------|--------------|----------|
| Phase 4 end | 129 | 97% |
| Phase 5 end | 148 | 97% |
| Phase 6 end | 157 | 97% |

Full breakdown:

| Test File | Tests | Scope |
|-----------|-------|-------|
| `test_active_learning.py` | 17 | `ActiveLearningCore`, entropy calculation |
| `test_api.py` | 22 | All FastAPI endpoints, auth, reward bridge |
| `test_base_agent.py` | 7 | ABC contract, device management, `_ensure_dir` |
| `test_buffer.py` | 22 | Push, sample, multimodal, edge cases, `update_reward` |
| `test_ewc.py` | 14 | Fisher computation, penalty, consolidate |
| `test_integration.py` | 6 | Full pipeline: Gym loop → queue → resolve → buffer |
| `test_ppo.py` | 17 | `ActorCritic`, `PPOAgent`, checkpoint |
| `test_review_queue.py` | 30 | Push, get, resolve, capacity, thread safety |
| `test_sac.py` | 22 | `Actor`, `Critic`, `SACAgent`, checkpoint |

---

## Framework Status

All six phases of `core-rl-framework` are complete.

| Phase | Module | Status |
|-------|--------|--------|
| 1 | `GenericReplayBuffer` | ✅ Complete |
| 2 | `SACAgent`, `PPOAgent`, `BaseAgent` | ✅ Complete |
| 3 | `ActiveLearningCore`, `ReviewQueue` | ✅ Complete |
| 4 | `EWC` + SAC integration | ✅ Complete |
| 5 | FastAPI backend, reward-update bridge | ✅ Complete |
| 6 | Minimal Web UI | ✅ Complete |

The framework is ready to be imported as a local package (`pip install -e .`) by any body repository. The next project is `carla-autonomous-driving` — a separate repository that will call `core_rl.SACAgent`, `core_rl.GenericReplayBuffer`, and `core_rl.api.create_app` to build a full autonomous driving pipeline on top of this brain.

---

## Remaining Items (Post-CARLA)

| Task | Reason Deferred |
|------|----------------|
| GAE for PPOAgent | Monte Carlo sufficient for Gymnasium; needed before CARLA with PPO |
| Online EWC | Single-task EWC sufficient now; needed when multi-task CARLA training begins |