# tests/test_ppo.py

import pytest
import numpy as np
import torch
from core_rl.agents.ppo import PPOAgent, ActorCritic
from core_rl.buffer.rollout_buffer import RolloutBuffer

STATE_DIM = 4
ACTION_DIM = 3  # Discrete: 3 possible actions


# ---------------------------------------------------------
# Neural Network Tests
# ---------------------------------------------------------

class TestActorCriticNetwork:
    def test_forward_output_shapes(self):
        net = ActorCritic(STATE_DIM, ACTION_DIM)
        state = torch.randn(8, STATE_DIM)
        logits, value = net(state)
        assert logits.shape == (8, ACTION_DIM)
        assert value.shape == (8, 1)

    def test_act_returns_correct_shapes(self):
        net = ActorCritic(STATE_DIM, ACTION_DIM)
        state = torch.randn(1, STATE_DIM)
        action, log_prob, value, entropy = net.act(state)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)
        assert entropy.shape == (1,)

    def test_act_returns_valid_actions(self):
        net = ActorCritic(STATE_DIM, ACTION_DIM)
        state = torch.randn(100, STATE_DIM)
        actions, _, _, _ = net.act(state)
        assert (actions >= 0).all()
        assert (actions < ACTION_DIM).all()

    def test_evaluate_output_shapes(self):
        net = ActorCritic(STATE_DIM, ACTION_DIM)
        states = torch.randn(8, STATE_DIM)
        actions = torch.randint(0, ACTION_DIM, (8,))
        log_probs, values, entropy = net.evaluate(states, actions)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)
        assert entropy.shape == (8,)

    def test_entropy_is_non_negative(self):
        net = ActorCritic(STATE_DIM, ACTION_DIM)
        states = torch.randn(8, STATE_DIM)
        actions = torch.randint(0, ACTION_DIM, (8,))
        _, _, entropy = net.evaluate(states, actions)
        assert (entropy >= 0).all()


# ---------------------------------------------------------
# PPOAgent Tests
# ---------------------------------------------------------

class TestPPOAgent:
    @pytest.fixture
    def agent(self):
        return PPOAgent(STATE_DIM, ACTION_DIM, device="cpu")

    def test_initialization(self, agent):
        assert agent.gamma == 0.99
        assert agent.eps_clip == 0.2
        assert agent.k_epochs == 4
        assert agent.device == torch.device("cpu")

    def test_policies_start_synchronized(self, agent):
        for p1, p2 in zip(agent.policy.parameters(), agent.policy_old.parameters()):
            assert torch.allclose(p1, p2)


class TestPPOSelectAction:
    @pytest.fixture
    def agent(self):
        return PPOAgent(STATE_DIM, ACTION_DIM, device="cpu")

    def test_returns_valid_action(self, agent):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob, value, entropy = agent.select_action(state)
        assert isinstance(action, int)
        assert 0 <= action < ACTION_DIM

    def test_explore_returns_log_prob_and_value(self, agent):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob, value, entropy = agent.select_action(state, evaluate=False)
        assert log_prob is not None
        assert value is not None
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_evaluate_returns_none_extras(self, agent):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob, value, entropy = agent.select_action(state, evaluate=True)
        assert isinstance(action, int)
        assert log_prob is None
        assert value is None
        assert entropy == 0.0

    def test_evaluate_is_deterministic(self, agent):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        a1, _, _, _ = agent.select_action(state, evaluate=True)
        a2, _, _, _ = agent.select_action(state, evaluate=True)
        assert a1 == a2


class TestPPOUpdate:
    @pytest.fixture
    def setup(self):
        agent = PPOAgent(STATE_DIM, ACTION_DIM, device="cpu")
        buffer = RolloutBuffer(device="cpu")
        for i in range(32):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            a, lp, _, _ = agent.select_action(s, evaluate=False)
            buffer.push(s, a, lp, float(np.random.randn()), i == 31)
        return agent, buffer

    def test_update_returns_loss_dict(self, setup):
        agent, buffer = setup
        metrics = agent.update(buffer)
        assert "ppo_loss" in metrics

    def test_update_returns_finite_loss(self, setup):
        agent, buffer = setup
        metrics = agent.update(buffer)
        assert np.isfinite(metrics["ppo_loss"])

    def test_policies_sync_after_update(self, setup):
        agent, buffer = setup
        agent.update(buffer)
        for p1, p2 in zip(agent.policy.parameters(), agent.policy_old.parameters()):
            assert torch.allclose(p1, p2), "policy_old should sync with policy after update"

    def test_update_changes_weights(self, setup):
        agent, buffer = setup
        initial_params = [p.clone() for p in agent.policy.parameters()]
        agent.update(buffer)
        changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, agent.policy.parameters())
        )
        assert changed, "Policy weights should change after update"


class TestPPOCheckpoint:
    @pytest.fixture
    def agent(self):
        return PPOAgent(STATE_DIM, ACTION_DIM, device="cpu")

    def test_save_and_load(self, agent, tmp_path):
        agent.save_checkpoint(str(tmp_path), suffix="test")

        new_agent = PPOAgent(STATE_DIM, ACTION_DIM, device="cpu")
        new_agent.load_checkpoint(str(tmp_path), suffix="test")

        for p1, p2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_nonexistent_raises_error(self, agent, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            agent.load_checkpoint(str(tmp_path), suffix="nonexistent")
