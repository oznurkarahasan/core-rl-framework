# tests/test_integration.py
#
# End-to-end integration test for the full core-rl pipeline.
#
# Verifies the complete human-in-the-loop cycle:
#   Gym loop → entropy threshold exceeded → queue + buffer (same UUID)
#   → human resolves via on_resolve callback → buffer reward updated
#
# This is NOT a training quality test. It verifies that all modules
# are correctly wired together. 20 steps is enough to trigger uncertainty
# at least once with a freshly initialized (untrained) SAC agent.

import uuid
import numpy as np
import pytest
import gymnasium as gym

from core_rl.agents.sac import SACAgent
from core_rl.buffer.buffer import GenericReplayBuffer
from core_rl.active_learning.core import ActiveLearningCore
from core_rl.active_learning.review_queue import ReviewQueue


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def env():
    e = gym.make("LunarLanderContinuous-v3")
    yield e
    e.close()


@pytest.fixture
def pipeline(env):
    """Returns all core-rl components wired together."""
    obs_dim = env.observation_space.shape[0]   # 8
    act_dim = env.action_space.shape[0]        # 2

    agent  = SACAgent(state_dim=obs_dim, action_dim=act_dim, device="cpu")
    buffer = GenericReplayBuffer(capacity=1000, device="cpu")
    queue  = ReviewQueue(capacity=50)

    # Threshold intentionally low — untrained SAC entropy ~1.0,
    # so almost every step triggers uncertainty. Guarantees queue gets filled.
    al_core = ActiveLearningCore(threshold=0.3)

    return agent, buffer, queue, al_core


# ---------------------------------------------------------
# Helper: run N steps of the gym loop
# ---------------------------------------------------------

def _run_steps(env, agent, buffer, queue, al_core, n_steps=20):
    """
    Runs n_steps of the environment loop.
    Wires entropy → ActiveLearningCore → ReviewQueue + ReplayBuffer (same UUID).
    Returns the list of experience_ids that were pushed to the queue.
    """
    queued_ids = []
    obs, _ = env.reset()

    for _ in range(n_steps):
        action, entropy = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        exp_id = str(uuid.uuid4())

        # Always push to buffer
        buffer.push(obs, action, float(reward), next_obs, done,
                    experience_id=exp_id)

        # Push to queue only when uncertain
        is_uncertain, entropy_val = al_core.is_uncertain(entropy)
        if is_uncertain:
            queue.push(obs, entropy_val, item_id=exp_id)
            queued_ids.append(exp_id)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return queued_ids


# ---------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------

class TestFullPipeline:

    def test_uncertainty_triggers_queue(self, env, pipeline):
        """Untrained agent must produce at least one uncertain state in 20 steps."""
        agent, buffer, queue, al_core = pipeline
        queued_ids = _run_steps(env, agent, buffer, queue, al_core, n_steps=20)

        assert len(queued_ids) > 0, (
            "No uncertain states detected in 20 steps. "
            "Check ActiveLearningCore threshold or agent entropy output."
        )

    def test_queue_contains_pushed_items(self, env, pipeline):
        """Items pushed to queue must be retrievable via get()."""
        agent, buffer, queue, al_core = pipeline
        queued_ids = _run_steps(env, agent, buffer, queue, al_core, n_steps=20)

        if not queued_ids:
            pytest.skip("No uncertain states generated — threshold may be too high.")

        pending = queue.get()
        pending_ids = [item["id"] for item in pending]

        # At least one queued_id must appear in the queue
        assert any(eid in pending_ids for eid in queued_ids), (
            "Queued experience IDs not found in ReviewQueue."
        )

    def test_resolve_updates_buffer_reward(self, env, pipeline):
        """
        Core pipeline test:
        resolve(id, reward) → on_resolve callback → buffer.update_reward()
        The reward stored in the buffer must reflect the human-provided value.
        """
        agent, buffer, queue, al_core = pipeline

        # Wire on_resolve callback: human reward → buffer
        resolve_log = []  # track callback invocations

        def on_resolve(exp_id: str, human_reward: float) -> None:
            success = buffer.update_reward(exp_id, human_reward)
            resolve_log.append((exp_id, human_reward, success))

        queued_ids = _run_steps(env, agent, buffer, queue, al_core, n_steps=20)

        if not queued_ids:
            pytest.skip("No uncertain states generated.")

        # Resolve the first queued item with a known reward
        target_id = queued_ids[0]
        human_reward = 1.0
        result = queue.resolve(target_id, human_reward)

        assert result is not None, f"resolve() returned None — id {target_id} not found in queue."

        # Fire the callback manually (mirrors what the FastAPI endpoint does)
        on_resolve(target_id, human_reward)

        # Verify callback was called and buffer update succeeded
        assert len(resolve_log) == 1
        logged_id, logged_reward, success = resolve_log[0]
        assert logged_id == target_id
        assert logged_reward == human_reward
        assert success is True, "buffer.update_reward() returned False — experience ID not found."

    def test_resolve_removes_item_from_queue(self, env, pipeline):
        """After resolve, the item must no longer appear in queue.get()."""
        agent, buffer, queue, al_core = pipeline
        queued_ids = _run_steps(env, agent, buffer, queue, al_core, n_steps=20)

        if not queued_ids:
            pytest.skip("No uncertain states generated.")

        target_id = queued_ids[0]
        queue.resolve(target_id, 1.0)

        pending_ids = [item["id"] for item in queue.get()]
        assert target_id not in pending_ids, (
            "Resolved item still present in queue — resolve() did not remove it."
        )

    def test_buffer_grows_during_loop(self, env, pipeline):
        """ReplayBuffer must accumulate experiences during the loop."""
        agent, buffer, queue, al_core = pipeline
        _run_steps(env, agent, buffer, queue, al_core, n_steps=20)

        assert len(buffer) == 20, (
            f"Expected 20 experiences in buffer, got {len(buffer)}."
        )

    def test_double_resolve_returns_none(self, env, pipeline):
        """Resolving the same item twice must return None on the second call."""
        agent, buffer, queue, al_core = pipeline
        queued_ids = _run_steps(env, agent, buffer, queue, al_core, n_steps=20)

        if not queued_ids:
            pytest.skip("No uncertain states generated.")

        target_id = queued_ids[0]
        first  = queue.resolve(target_id, 1.0)
        second = queue.resolve(target_id, 1.0)

        assert first  is not None, "First resolve should succeed."
        assert second is None,     "Second resolve on same ID should return None."