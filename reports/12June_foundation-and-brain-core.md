# Core-RL Framework — Phase 1 & Phase 2 Development Report

**Date:** June 12, 2026  
**Scope:** Phase 1 (Foundation & Memory) + Phase 2 (Brain Core & Decision Engines)  
**Status:** Completed

---

## Overview

This report documents the implementation and review of the first two phases of the Core-RL framework. Phase 1 establishes the memory infrastructure (Replay Buffer) and Phase 2 builds the decision-making engines (SAC and PPO agents) on top of it.

---

## Phase 1: Foundation (Memory & Data Management)

### Task 1.1 & 1.2 — GenericReplayBuffer

**File:** `src/core_rl/buffer/buffer.py`  
**Status:** Implemented & Reviewed

#### What Was Built
A generic experience replay buffer that supports:
- Standard flat array observations (e.g., velocity, position vectors)
- Multimodal Dict observations (e.g., `{"camera": np.array, "speed": float}`)
- Automatic recursive conversion of nested Dict structures into PyTorch tensors
- Configurable capacity with automatic eviction of oldest experiences (`deque(maxlen=...)`)
- CPU/GPU device management

#### Bugs Found & Fixed

| Bug | Problem | Fix Applied |
|-----|---------|-------------|
| Missing batch size guard | `sample()` crashed with `ValueError` when `batch_size > len(buffer)` | Added fail-fast check with descriptive error message |
| Empty list crash | `_to_tensor()` raised `IndexError` on empty `batch_data` | Added `not batch_data` guard clause before accessing `[0]` |
| Ragged array crash | `np.array()` silently failed or raised warnings on mixed-shape data | Replaced with `np.stack()` wrapped in `try/except` with clear error message |

#### Final State
- 61 lines, clean English documentation
- All edge cases handled with descriptive error messages
- Type hints include `Optional[Any]` for nullable returns

---

## Phase 2: The Brain Core (Decision Engines)

### Task 2.1 — BaseAgent Abstract Base Class

**File:** `src/core_rl/agents/base_agent.py`  
**Status:** Implemented & Reviewed

#### What Was Built
An Abstract Base Class (ABC) that enforces a strict contract for all RL agents:
- `select_action(state, evaluate)` — action selection (exploration vs exploitation)
- `update(buffer, batch_size)` — neural network parameter updates; accepts any `BaseBuffer` subclass
- `save_checkpoint(checkpoint_dir, suffix)` — model serialization
- `load_checkpoint(checkpoint_dir, suffix)` — model deserialization

#### Bugs Found & Fixed

| Bug | Problem | Fix Applied |
|-----|---------|-------------|
| Unused `import os` | Dead import, `os` was never referenced | Removed, replaced with `pathlib.Path` |
| Weak `replay_buffer` typing | Parameter typed as `Any`, no type safety | Changed to `"GenericReplayBuffer"` via `TYPE_CHECKING` to avoid circular imports |
| No directory creation helper | Subclasses had to independently handle `mkdir`, risking inconsistency | Added `_ensure_dir()` helper method in the base class |

#### Final State
- 74 lines, clean English documentation
- Proper `TYPE_CHECKING` pattern for cross-module type hints
- Centralized `_ensure_dir()` utility for checkpoint management

---

### Task 2.2 — SACAgent (Soft Actor-Critic)

**File:** `src/core_rl/agents/sac.py`  
**Status:** Implemented & Reviewed

#### What Was Built

**Neural Networks:**
- `Critic` — Double Q-Network (two independent Q-value estimators to mitigate overestimation bias)
- `Actor` — Squashed Gaussian Policy Network (outputs continuous actions via `tanh` squashing)

**Agent (`SACAgent`):**
- Automatic entropy tuning via learnable `log_alpha` parameter
- Soft target network updates (Polyak averaging with `tau`)
- Reparameterization trick (`rsample()`) for differentiable action sampling
- Full checkpoint save/load with optimizer state preservation

#### Bugs Found & Fixed

| Bug | Problem | Fix Applied |
|-----|---------|-------------|
| Alpha optimizer breaks after load | `self.log_alpha = checkpoint[...]` replaced the tensor object, breaking the optimizer's reference | Changed to in-place `self.log_alpha.data.copy_(...)` |
| Inconsistent path handling | Used `os.makedirs` + `os.path.join` instead of modern `pathlib` | Migrated to `Path` objects throughout |
| Insecure model loading | `torch.load` without `weights_only=True` triggers PyTorch security warnings | Added `weights_only=True` parameter |
| Wrong return type hint | `select_action` annotated as `-> torch.Tensor` but actually returns `np.ndarray` | Fixed to `-> np.ndarray` |
| Weak buffer typing | `replay_buffer` typed as `Any` | Changed to `GenericReplayBuffer` with direct import |
| Inconsistent alpha serialization | `log_alpha` saved as raw tensor while everything else used `.state_dict()` | Now saves `.data` and loads via `.data.copy_()` for consistency |
| Broken `target_entropy` | `np.prod(action_dim).item()` fails because `int` has no `.item()` method | Simplified to `-float(action_dim)` |
| Unused import | `import os` no longer needed after `pathlib` migration | Removed |

#### Final State
- 237 lines (including 17-line architecture overview header)
- Header comment block explaining Actor, Critic, and Temperature (Alpha) roles
- All comments and error messages in English
- Clean, production-ready code

---

### Task 2.3 — PPOAgent (Proximal Policy Optimization)

**File:** `src/core_rl/agents/ppo.py`  
**Status:** Implemented & Reviewed

#### What Was Built

**Neural Network:**
- `ActorCritic` — Shared feature extractor with separate Actor (policy logits) and Critic (state value) heads
- Uses `Tanh` activation (better suited for PPO than `ReLU`)
- `Categorical` distribution for discrete action sampling

**Agent (`PPOAgent`):**
- Clipped surrogate objective (`eps_clip = 0.2`) to prevent destructively large policy updates
- On-policy design: uses fresh rollouts instead of a replay buffer
- Monte Carlo returns (rewards-to-go) with normalization
- Multi-epoch optimization (`k_epochs`) over collected trajectories
- Old policy / new policy separation for ratio calculation

#### Bugs Found & Fixed

| Bug | Problem | Fix Applied |
|-----|---------|-------------|
| Missing `Tuple` import | `Tuple` used in type hints but never imported — causes `NameError` at import time | Added `Tuple` to `from typing import ...` |
| `update()` signature mismatch | Parameter names didn't match `BaseAgent.update(replay_buffer, batch_size)` | Renamed to `replay_buffer` with `rollouts = replay_buffer` alias inside |
| Unused `import numpy` | `np` was imported but never used | Replaced with `from collections import deque` |
| O(n²) reward computation | `rewards.insert(0, ...)` shifts the entire list each iteration | Replaced with `deque.appendleft()` which is O(1) |

#### Final State
- 196 lines (including 24-line architecture overview header)
- Header comment block explaining PPO purpose, why it exists alongside SAC, and key differences
- BaseAgent contract fully respected
- All comments and error messages in English

---

## Files Summary

### New Files Created
| File | Purpose |
|------|---------|
| `src/core_rl/agents/base_agent.py` | Abstract Base Class defining the agent interface |
| `src/core_rl/agents/sac.py` | Soft Actor-Critic agent for continuous action spaces |
| `src/core_rl/agents/ppo.py` | Proximal Policy Optimization agent for discrete action spaces |

### Modified Files
| File | Changes |
|------|---------|
| `src/core_rl/buffer/buffer.py` | Bug fixes (3), English translation, defensive error handling |
| `todo.md` | Tasks 1.1, 1.2, 2.1, 2.2, 2.3 marked as completed |

### Unchanged Files
| File | Note |
|------|------|
| `src/core_rl/buffer/__init__.py` | Still empty — needs `GenericReplayBuffer` export |
| `src/core_rl/agents/__init__.py` | Still empty — needs `SACAgent`, `PPOAgent` exports |

---

## Next Steps

The framework is ready for **Phase 3 (Active Learning)** and **Phase 4 (Continual Learning)**:
- Phase 3: Entropy-based uncertainty detection + human review queue
- Phase 4: EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting
