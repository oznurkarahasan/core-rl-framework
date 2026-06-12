## Phase 1: Foundation (Memory & Data Management)

[ ] Task 1.1: Design the GenericReplayBuffer class. Ensure it supports multimodal observations (e.g., nested Dict structures combining image matrices and numerical telemetrics), rather than just standard flat arrays.

[ ] Task 1.2: Implement push (experience storage) and sample (batch retrieval) methods, optimizing them for seamless integration with PyTorch tensors.

## Phase 2: The Brain Core (Decision Engines)

[ ] Task 2.1: Create the BaseAgent Abstract Base Class (ABC). Define the standard interface template, including select_action, update, and save/load methods.

[ ] Task 2.2: Implement the SACAgent (Soft Actor-Critic) for continuous action spaces. Build the Actor and Critic neural networks and integrate automatic entropy tuning (alpha).

[ ] Task 2.3: Develop the basic PPOAgent skeleton, providing support for discrete action spaces (e.g., trading bots or grid-based environments).

## Phase 3: Active Learning Layer (Human-in-the-Loop)

[ ] Task 3.1: Develop the ActiveLearningCore module. Implement the mathematical functions required to calculate policy entropy (uncertainty distribution) during inference.

[ ] Task 3.2: Design an asynchronous "Review Queue". Ensure that when the uncertainty threshold is exceeded, the uncertain state is queued for human review without freezing the main simulation loop.

## Phase 4: Continual Learning

[ ] Task 4.1: Build the EWC (Elastic Weight Consolidation) class. Implement the logic to compute and store the Fisher Information Matrix to identify and protect critical network weights.

[ ] Task 4.2: Integrate the EWC penalty function into the SAC/PPO agent's loss calculation to actively prevent catastrophic forgetting when learning new edge cases.

## Phase 5: Feedback Engine (Human Interface & API)

[ ] Task 5.1: Set up the generic Flask/FastAPI backend. Create API endpoints (e.g., /get_uncertain_state, /submit_label) to expose the active learning queue to the web-based teacher UI.

[ ] Task 5.2: Implement the reward-update bridge. Ensure that once a human label is submitted, the system dynamically updates the corresponding experience reward within the Replay Buffer.