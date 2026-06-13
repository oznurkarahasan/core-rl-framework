# Core-RL Framework — Phase 4: Continual Learning Report

**Date:** June 13, 2026  
**Scope:** Phase 4 — Elastic Weight Consolidation (Tasks 4.1–4.2)  
**Status:** Completed

---

## Overview

Phase 4 adds a continual learning layer to the framework. The goal is to prevent catastrophic forgetting: when an agent learns a new task (e.g. a new traffic sign or edge case in CARLA), it should not overwrite the weights that encode what it already knows.

The implementation uses **Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., 2017). The core insight is that not all network weights are equally important for a given task. EWC identifies the important ones by computing the Fisher Information Matrix diagonal, then penalises the loss function when future training moves those weights away from their task-optimal values.

By the end of this session: **129 tests passing.**

---

## Task 4.1 — EWC Class

**File:** `src/core_rl/continual/ewc.py` (new)

### What Was Built

A standalone, agent-agnostic `EWC` class. It has no knowledge of SAC, PPO, or any specific network architecture — it operates on any `nn.Module` via a callback.

```python
ewc = EWC(ewc_lambda=5000.0)
ewc.consolidate(model, states, log_prob_fn)  # snapshot Fisher + optimal params
ewc.penalty(model)                           # returns regularisation term to add to loss
```

### Fisher Information Matrix — Implementation Detail

The diagonal of the Fisher Information Matrix is approximated using the **empirical Fisher**:

```
F_i ≈ (∂/∂θ_i [ -E[log π(a|s)] ])²
```

In practice, this is computed by:
1. Running a forward pass over a representative batch of states to get `log_prob`
2. Computing `-log_prob.mean().backward()` to get gradients
3. Squaring those gradients element-wise: `F_i = (∂ loss / ∂θ_i)²`

This gives a per-parameter importance score. A weight with a high Fisher value is one the policy relies on heavily for the current task; changing it will hurt performance on that task.

The per-sample Fisher (squaring each sample's gradient separately before averaging) would be more statistically accurate but requires per-sample gradients, which PyTorch does not compute natively. The mean-then-square approximation is the standard in the EWC literature and works well in practice.

### EWC Penalty

Once consolidated, the penalty term added to the actor loss is:

```
L_EWC = (λ/2) * Σ_i  F_i * (θ_i - θ*_i)²
```

Where `θ*_i` are the optimal (consolidated) parameter values and `F_i` is the Fisher diagonal. High-Fisher parameters are pulled back toward their consolidated values with stronger force; low-Fisher parameters are allowed to drift freely for new task learning.

`penalty()` returns `0.0` before `consolidate()` is ever called — no guard needed at the call site.

### Training Mode Safety

`consolidate()` switches the model to `eval()` for the forward pass (to disable dropout/batchnorm stochasticity), then restores it to whatever mode it was in before returning. This means calling `consolidate()` mid-training does not silently leave the model in `eval()` mode.

---

## Task 4.2 — EWC Integration into SACAgent

**File:** `src/core_rl/agents/sac.py` (modified)

### Design Decisions

**SAC first, PPO later.** EWC is integrated into SAC because the CARLA target uses continuous action spaces. PPO can follow the identical pattern in under 30 minutes when needed — the `EWC` class is already agent-agnostic.

**Manual `consolidate()` trigger.** Automatic consolidation (e.g. detecting task boundaries from reward statistics) is premature at this stage. The user decides when a task is "done" and calls `consolidate()` explicitly. This keeps the framework deterministic and the agent dumb in the right places.

**`ewc_lambda` is adjustable after construction.** Literature uses 5000 as a standard starting value — high enough to protect, low enough to keep learning. The right value depends on the task and network size, so it is exposed as a plain property.

### Interface Changes

Three additions to `SACAgent`:

```python
# Constructor — new parameter with default
SACAgent(..., ewc_lambda=5000.0)

# After task A is done:
agent.consolidate(replay_buffer)            # default 256 samples
agent.consolidate(replay_buffer, batch_size=1024)  # larger for richer observation spaces

# Tune protection strength at any time:
agent.ewc_lambda = 10000  # tighter protection
agent.ewc_lambda = 1000   # more plasticity for new task
```

The `ewc_lambda` property delegates directly to `self._ewc.ewc_lambda`, so the EWC object is the single source of truth.

### Loss Modification

A single line change in `update()`:

```python
# Before
actor_loss = (self.alpha * log_probs - min_q_new).mean()

# After
actor_loss = (self.alpha * log_probs - min_q_new).mean() + self._ewc.penalty(self.actor)
```

The EWC penalty is only applied to the **actor**. The critic learns value estimates for the current task and should adapt freely. The actor encodes the policy — it is what catastrophic forgetting actually damages.

### Typical Usage Flow

```python
# --- Task A: Stop Signs ---
agent = SACAgent(state_dim=..., action_dim=..., ewc_lambda=5000.0)
for step in range(task_a_steps):
    agent.update(buffer_a, batch_size=256)

agent.consolidate(buffer_a)  # lock in stop-sign knowledge

# --- Task B: Yield Signs (new edge cases) ---
# actor_loss now includes EWC penalty — stop-sign weights are protected
for step in range(task_b_steps):
    agent.update(buffer_b, batch_size=256)
```

---

## Files Summary

### New Files Created

| File | Purpose |
|------|---------|
| `src/core_rl/continual/__init__.py` | Package init exporting `EWC` |
| `src/core_rl/continual/ewc.py` | `EWC` — agent-agnostic Fisher matrix + penalty computation |
| `tests/test_ewc.py` | 14 unit tests for `EWC` and `SACAgent` EWC integration |

### Modified Files

| File | Changes |
|------|---------|
| `src/core_rl/agents/sac.py` | `ewc_lambda` constructor param + property; `consolidate()` method; EWC penalty in `update()` |
| `src/core_rl/__init__.py` | Added `EWC` to top-level exports |

---

## Test Summary

| Test Class | Tests | What Is Covered |
|------------|-------|-----------------|
| `TestEWCPenalty` | 4 | Zero before consolidation; zero at optimal params; positive after perturbation; scales with lambda |
| `TestEWCConsolidate` | 4 | `_consolidated` flag; Fisher + param storage; model mode restoration (train → train, eval → eval) |
| `TestSACAgentEWC` | 6 | Default lambda; setter; `update()` before and after consolidation; `consolidate()` marks EWC; penalty changes actor loss |

| Metric | Phase 3 End | This Session End |
|--------|-------------|-----------------|
| Tests passing | 115 | 129 |

---

## Design Notes for Future Reference

**Why only the actor gets EWC, not the critic?**  
The critic estimates Q-values for the current environment. It is expected to shift as the task distribution changes. Only the actor — the policy that maps states to actions — is what the agent "learned" about a task and must retain.

**`ewc_lambda` sensitivity.**  
Too low (< 100): protection is negligible, forgetting happens normally. Too high (> 50000): the actor cannot update meaningfully on the new task. The Gymnasium tests (Phase 2.9) provide a baseline: observe how much the actor weights drift across 10k steps of normal training, then set lambda to a value that would meaningfully penalise drift of that magnitude.

**Multiple task consolidation.**  
The current implementation overwrites `_fisher` and `_optimal_params` on each `consolidate()` call. This covers the immediate use case (one prior task). Online EWC (accumulating Fisher across tasks) can be layered on later by summing instead of replacing.

---

## Next Steps

**Phase 5 — Feedback Engine (Human Interface & API):**

- **Task 5.1:** Flask/FastAPI backend with `/get_uncertain_state` and `/submit_label` endpoints to expose the Phase 3 `ReviewQueue` to a web UI.
- **Task 5.2:** Reward-update bridge — when a human label arrives, update the corresponding experience reward inside the `GenericReplayBuffer` so the agent retrains on corrected signal.
