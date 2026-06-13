## Phase 1: Foundation (Memory & Data Management)

[x] Task 1.1: Design the GenericReplayBuffer class. Ensure it supports multimodal observations (e.g., nested Dict structures combining image matrices and numerical telemetrics), rather than just standard flat arrays.

[x] Task 1.2: Implement push (experience storage) and sample (batch retrieval) methods, optimizing them for seamless integration with PyTorch tensors.

## Phase 2: The Brain Core (Decision Engines)

[x] Task 2.1: Create the BaseAgent Abstract Base Class (ABC). Define the standard interface template, including select_action, update, and save/load methods.

[x] Task 2.2: Implement the SACAgent (Soft Actor-Critic) for continuous action spaces. Build the Actor and Critic neural networks and integrate automatic entropy tuning (alpha).

[x] Task 2.3: Develop the basic PPOAgent skeleton, providing support for discrete action spaces (e.g., trading bots or grid-based environments).

## Pre-Phase 3: Cleanup & Validation
 
[x] Task 2.4 (Cleanup): Add gradient clipping to PPOAgent.update() to prevent training instability from large gradient updates.
 
[x] Task 2.5 (Cleanup): Make PPO entropy coefficient a constructor hyperparameter (`entropy_coef=0.01`) instead of a hardcoded magic number.
 
[x] Task 2.6 (Cleanup): Rename `self.MseLoss` → `self.mse_loss` in PPOAgent to follow Python naming conventions.
 
[x] Task 2.7 (Cleanup): Add a coverage threshold to CI (e.g., fail if coverage drops below 80%) so regressions are caught automatically.
 
[x] Task 2.8 (Cleanup): Update reports/12June_foundation-and-brain-core.md — remove stale "Known Remaining Issues" entries that have already been resolved (empty __init__.py, missing pyproject.toml, no unit tests).
 
[x] Task 2.9 (Validation): Write an end-to-end Gymnasium training loop. Use LunarLanderContinuous-v3 with SACAgent and LunarLander-v3 with PPOAgent. This validates that the brain works before building on top of it, and establishes a baseline entropy range needed for Phase 3 threshold tuning.

## Phase 3: Active Learning Layer (Human-in-the-Loop)

[x] Task 3.1: Develop the ActiveLearningCore module. Implement the mathematical functions required to calculate policy entropy (uncertainty distribution) during inference.

[x] Task 3.2: Design an asynchronous "Review Queue". Ensure that when the uncertainty threshold is exceeded, the uncertain state is queued for human review without freezing the main simulation loop.

## Phase 4: Continual Learning

[x] Task 4.1: Build the EWC (Elastic Weight Consolidation) class. Implement the logic to compute and store the Fisher Information Matrix to identify and protect critical network weights.

[x] Task 4.2: Integrate the EWC penalty function into the SAC/PPO agent's loss calculation to actively prevent catastrophic forgetting when learning new edge cases.

## Phase 5: Feedback Engine (Human Interface & API)

[x] Task 5.1: Set up the FastAPI backend. Create API endpoints (GET /health, GET /queue/status, GET /queue/items, POST /queue/resolve/{id}) to expose the active learning queue to the web-based teacher UI.

[x] Task 5.2: Implement the reward-update bridge. Ensure that once a human label is submitted, the system dynamically updates the corresponding experience reward within the Replay Buffer.

## Phase 6: Minimal Web UI

[ ] Task 6.1: Build a minimal HTML/JS interface that calls the Phase 5 API. Displays pending uncertain states from the queue, allows the human teacher to approve or reject with a reward value.

[ ] Task 6.2: Connect the UI to the /queue/items and /queue/resolve/{id} endpoints. Ensure the full human-in-the-loop cycle works end-to-end: uncertain state appears on screen → human labels it → reward updates in buffer.