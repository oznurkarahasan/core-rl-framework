# Core-RL Framework — Pre-Phase 3 Cleanup & Phase 3 Development Report

**Date:** June 13, 2026  
**Scope:** Pre-Phase 3 Structural Fixes + Cleanup (Tasks 2.4–2.9) + Phase 3 (Active Learning Layer)  
**Status:** Completed

---

## Overview

This report covers two consecutive bodies of work. The first is a set of structural fixes and cleanup tasks that were identified during Phase 2 review and completed before Phase 3 began. The second is Phase 3 itself, which adds an uncertainty-detection layer and a human review queue on top of the Phase 2 agents.

By the end of this session: **115 tests passing, 98% coverage.**

---

## Pre-Phase 3: Structural Fixes

These issues were identified during Phase 2 review. They were blocking correct use of the framework and had to be resolved before building further.

### Fix 1 — Empty `__init__.py` Files

**Files:** `src/core_rl/__init__.py`, `src/core_rl/agents/__init__.py`, `src/core_rl/buffer/__init__.py`

All three package init files were completely empty. Every import shown in the README raised an `ImportError` at runtime. The fix added explicit exports to each file and wired them together at the top-level package.

```python
# src/core_rl/__init__.py (after fix)
from core_rl.agents import BaseAgent, SACAgent, PPOAgent
from core_rl.buffer import BaseBuffer, GenericReplayBuffer, RolloutBuffer
```

---

### Fix 2 — PPO / BaseAgent Architecture Mismatch

**Files:** `src/core_rl/agents/base_agent.py`, `src/core_rl/buffer/` (new files)

`BaseAgent.update()` was typed to accept `GenericReplayBuffer` — an off-policy structure that PPO fundamentally cannot use. PPO's actual implementation received a raw `Dict` and cast it with `rollouts = replay_buffer`, which would have caused a crash the moment PPO and SAC were used polymorphically (Phase 4).

The correct fix required creating a proper buffer hierarchy:

| New File | Purpose |
|----------|---------|
| `src/core_rl/buffer/base_buffer.py` | `BaseBuffer` ABC with `__len__()` and `clear()` as the shared contract |
| `src/core_rl/buffer/rollout_buffer.py` | `RolloutBuffer` — on-policy trajectory store with `push()`, `get()`, `clear()` |

`GenericReplayBuffer` was updated to inherit from `BaseBuffer` and gained a `clear()` method. `BaseAgent.update()` was re-typed to accept `"BaseBuffer"` via `TYPE_CHECKING`, making both agents interchangeable at the call site.

---

### Fix 3 — `_ensure_dir()` Defined But Never Used

**File:** `src/core_rl/agents/sac.py`

`BaseAgent` provided `_ensure_dir()` specifically to centralise checkpoint directory creation. `SACAgent.save_checkpoint()` ignored it and called `Path.mkdir()` directly — the exact pattern the helper was meant to replace.

```python
# Before
save_dir = Path(checkpoint_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# After
save_dir = self._ensure_dir(checkpoint_dir)
```

---

### Fix 4 — Build Artifacts Tracked by Git

**File:** `.gitignore`

`.coverage`, `htmlcov/`, and `venv/` were missing from `.gitignore` and were being tracked by the repository. Coverage files are generated artifacts; including them pollutes the git history and creates spurious diffs on every test run.

---

## Pre-Phase 3: Cleanup Tasks (2.4–2.9)

### Task 2.4 — Gradient Clipping in PPO

**File:** `src/core_rl/agents/ppo.py`

PPO had no gradient clipping. Large gradient updates are a known instability source in on-policy training, particularly during the early epochs of a rollout update. Added `clip_grad_norm_` after each `loss.backward()` call.

```python
loss.mean().backward()
torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
self.optimizer.step()
```

`max_grad_norm` defaults to `0.5` (standard for PPO) and is a constructor parameter.

---

### Task 2.5 — PPO Entropy Coefficient as Hyperparameter

**File:** `src/core_rl/agents/ppo.py`

The entropy bonus coefficient `0.01` was a magic number hardcoded into the loss calculation. Phase 3 requires tuning this value per environment, so it was promoted to a constructor parameter.

```python
# Before
loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(...) - 0.01 * dist_entropy

# After
def __init__(self, ..., entropy_coef: float = 0.01, ...):
    self.entropy_coef = entropy_coef

loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(...) - self.entropy_coef * dist_entropy
```

---

### Task 2.6 — Naming Convention Fix (`MseLoss` → `mse_loss`)

**File:** `src/core_rl/agents/ppo.py`

`self.MseLoss` violated Python's `snake_case` convention for instance attributes. Renamed to `self.mse_loss`.

---

### Task 2.7 — CI Coverage Threshold

**File:** `.github/workflows/ci.yml`

The CI pipeline ran coverage but had no failure threshold — coverage could silently drop to 0% without breaking the build. Added `--cov-fail-under=80` to the pytest invocation so regressions are caught automatically.

---

### Task 2.8 — Stale Report Entries Removed

**File:** `reports/12June_foundation-and-brain-core.md`

The "Known Remaining Issues" section listed three items (empty `__init__.py`, missing `pyproject.toml`, no unit tests) that had all been resolved. The section was removed to prevent it from being misread as still-open work.

---

### Task 2.9 — End-to-End Gymnasium Validation

**File:** `examples/train_e2e.py` (new)

Written to verify that both agents can complete a full training loop before Phase 3 built on top of them.

| Agent | Environment | Validates |
|-------|------------|-----------|
| `SACAgent` | `LunarLanderContinuous-v3` | Continuous action space, off-policy loop, alpha tuning |
| `PPOAgent` | `LunarLander-v3` | Discrete action space, on-policy rollout, `RolloutBuffer` integration |

The SAC run also logged `alpha` (entropy scale) values across training. Observed: alpha starts near `1.0` at initialisation. This range informed the `ActiveLearningCore` default threshold in Phase 3.

Also resolved during this step: `pyproject.toml` gained an `examples` optional dependency group (`swig`, `gymnasium[box2d]`) so Box2D environments can be installed with `pip install -e ".[examples]"` without polluting the core or dev dependency sets.

---

## Phase 3: Active Learning Layer

### Task 3.1 — ActiveLearningCore

**File:** `src/core_rl/active_learning/core.py` (new)

#### What Was Built

A single-responsibility class that answers one question: *is this entropy value uncertain enough to warrant human review?* It receives a pre-computed scalar — it has no knowledge of SAC, PPO, or any agent internals.

```python
core = ActiveLearningCore(threshold=1.0)
is_uncertain, entropy_val = core.is_uncertain(entropy)
```

- `threshold` is a plain attribute, directly reassignable — no setter method needed.
- `is_uncertain()` accepts both `float` and `torch.Tensor` and always returns `(bool, float)`. Returning both values in one call avoids re-computing entropy for logging.
- Uncertainty is defined as *strictly greater than* the threshold, so `entropy == threshold` does not trigger review.

**Units:** nats (natural log). Both agents now expose entropy in nats:
- SAC: `entropy ≈ -log_prob` (squashed Gaussian sample estimate)
- PPO: `Categorical.entropy()` (exact closed-form)

#### Interface Change — `select_action()` Now Returns Entropy

To expose entropy without requiring a second forward pass, the return type of `select_action()` was extended for both agents.

| Agent | Before | After |
|-------|--------|-------|
| `SACAgent` | `np.ndarray` | `(np.ndarray, float)` |
| `PPOAgent` | `(action, log_prob, value)` | `(action, log_prob, value, float)` |

In evaluate mode (`evaluate=True`), entropy is `0.0` for both agents — a deterministic policy has no distributional uncertainty to report.

`ActorCritic.act()` was updated to compute and return `dist.entropy()` in the same forward pass, so no additional computation is required at the agent level. All callers were updated: `tests/test_ppo.py`, `tests/test_sac.py`, `examples/train_e2e.py`.

---

### Task 3.2 — ReviewQueue

**File:** `src/core_rl/active_learning/review_queue.py` (new)

#### What Was Built

A thread-safe FIFO queue that holds uncertain states pending human review. The simulation loop calls `push()` without blocking; a Flask server (Phase 5) calls `get()` and `resolve()` from a separate thread.

Each queued item stores:

```python
{
    "id":        str,    # UUID — needed to target a specific item for resolution
    "state":     Any,    # whatever the agent received
    "entropy":   float,  # why it was flagged
    "timestamp": float,  # when it was flagged (time.time())
}
```

The `id` field is non-optional. Without it, `resolve()` cannot identify which item a human has reviewed — the `/submit_label` endpoint in Phase 5 would have no way to target a specific queue entry.

#### Key Design Decisions

**Capacity overflow — evict oldest:** When the queue is at capacity, the oldest item is silently dropped before the new one is inserted. A fresh uncertain state is always more actionable than a stale one. The simulation loop never blocks or raises.

**`_index` dict for O(1) resolve:** A parallel `Dict[id → item]` allows `resolve()` to locate any item in constant time regardless of queue length. The deque alone would require an O(n) linear scan.

**Eviction consistency:** Before `deque.append()` displaces the oldest item, that item's `id` is removed from `_index` — the two structures stay in sync.

**`threading.Lock`** wraps all three public methods. The lock is held only for the duration of the data structure operation, so it does not impede the simulation loop.

#### Interface

```python
queue = ReviewQueue(capacity=1000)

# Simulation loop (main thread)
item_id = queue.push(state, entropy=1.8)

# Flask server (background thread)
pending = queue.get()                          # list, oldest first, non-consuming
result  = queue.resolve(item_id, reward=1.0)  # removes item, returns with human_reward
```

`resolve()` returns `None` if the `id` is not found — it was either already resolved or silently evicted when the queue was full. Phase 5 can treat this as a no-op.

#### Dependency Direction

Flask → ReviewQueue. `ReviewQueue` imports nothing from Flask and has no knowledge of HTTP. The coupling is strictly one-directional, so the queue can be unit-tested and used independently of the web layer.

---

## Files Summary

### New Files Created

| File | Purpose |
|------|---------|
| `src/core_rl/buffer/base_buffer.py` | `BaseBuffer` ABC — shared contract for off-policy and on-policy buffers |
| `src/core_rl/buffer/rollout_buffer.py` | `RolloutBuffer` — on-policy trajectory store for PPO |
| `src/core_rl/active_learning/__init__.py` | Package init exporting `ActiveLearningCore`, `ReviewQueue` |
| `src/core_rl/active_learning/core.py` | `ActiveLearningCore` — agent-agnostic entropy threshold checker |
| `src/core_rl/active_learning/review_queue.py` | `ReviewQueue` — thread-safe human review queue with FIFO eviction |
| `examples/train_e2e.py` | End-to-end training loop for SAC + PPO validation |
| `tests/test_active_learning.py` | Unit tests for `ActiveLearningCore` |
| `tests/test_review_queue.py` | Unit tests for `ReviewQueue` (including thread safety) |

### Modified Files

| File | Changes |
|------|---------|
| `src/core_rl/__init__.py` | Added exports: `BaseBuffer`, `RolloutBuffer`, `ActiveLearningCore`, `ReviewQueue` |
| `src/core_rl/agents/__init__.py` | Added exports: `BaseAgent`, `SACAgent`, `PPOAgent` |
| `src/core_rl/buffer/__init__.py` | Added exports: `BaseBuffer`, `GenericReplayBuffer`, `RolloutBuffer` |
| `src/core_rl/buffer/buffer.py` | Inherits `BaseBuffer`; gained `clear()` method |
| `src/core_rl/agents/base_agent.py` | `update()` re-typed to accept `"BaseBuffer"` for polymorphism |
| `src/core_rl/agents/sac.py` | `select_action` returns `(action, entropy_float)`; uses `_ensure_dir()` |
| `src/core_rl/agents/ppo.py` | Gradient clipping; `entropy_coef` parameter; `mse_loss` rename; `select_action` returns 4-tuple |
| `tests/test_ppo.py` | Updated for 4-tuple `select_action` return; uses `RolloutBuffer` in fixtures |
| `tests/test_sac.py` | Updated for 2-tuple `select_action` return; added entropy assertions |
| `tests/test_buffer.py` | Added `TestRolloutBuffer`; added `test_clear_empties_buffer` |
| `.github/workflows/ci.yml` | Added `--cov-fail-under=80` to pytest command |
| `pyproject.toml` | Added `[examples]` optional dependency group for Box2D environments |
| `.gitignore` | Added `.coverage`, `htmlcov/`, `venv/` |

---

## Test & Coverage Summary

| Metric | Phase 2 End | This Session End |
|--------|-------------|-----------------|
| Tests passing | 66 | 115 |
| Coverage | 96% | 98% |
| CI threshold | none | 80% (fail-under) |

---

## Next Steps

The framework is ready for **Phase 4 (Continual Learning)**:

- **Task 4.1 — EWC:** Compute and store the Fisher Information Matrix to identify which network weights are critical for previously learned tasks.
- **Task 4.2 — EWC Penalty Integration:** Add the EWC penalty term to SAC and PPO loss calculations to prevent catastrophic forgetting when the agent encounters new edge cases.
