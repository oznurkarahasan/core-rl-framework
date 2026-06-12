# tests/test_sac.py

import pytest
import numpy as np
import torch
from core_rl.agents.sac import SACAgent, Actor, Critic
from core_rl.buffer.buffer import GenericReplayBuffer

STATE_DIM = 4
ACTION_DIM = 2


# ---------------------------------------------------------
# Neural Network Tests
# ---------------------------------------------------------

class TestCriticNetwork:
    def test_output_shapes(self):
        critic = Critic(STATE_DIM, ACTION_DIM)
        state = torch.randn(8, STATE_DIM)
        action = torch.randn(8, ACTION_DIM)
        q1, q2 = critic(state, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_double_q_produces_different_values(self):
        critic = Critic(STATE_DIM, ACTION_DIM)
        state = torch.randn(1, STATE_DIM)
        action = torch.randn(1, ACTION_DIM)
        q1, q2 = critic(state, action)
        # After random init, Q1 and Q2 should differ
        assert not torch.allclose(q1, q2)


class TestActorNetwork:
    def test_forward_output_shapes(self):
        actor = Actor(STATE_DIM, ACTION_DIM)
        state = torch.randn(8, STATE_DIM)
        mean, log_std = actor(state)
        assert mean.shape == (8, ACTION_DIM)
        assert log_std.shape == (8, ACTION_DIM)

    def test_log_std_is_clamped(self):
        actor = Actor(STATE_DIM, ACTION_DIM)
        state = torch.randn(32, STATE_DIM)
        _, log_std = actor(state)
        assert log_std.min().item() >= -20
        assert log_std.max().item() <= 2

    def test_sample_output_shapes(self):
        actor = Actor(STATE_DIM, ACTION_DIM)
        state = torch.randn(8, STATE_DIM)
        action, log_prob, deterministic = actor.sample(state)
        assert action.shape == (8, ACTION_DIM)
        assert log_prob.shape == (8, 1)
        assert deterministic.shape == (8, ACTION_DIM)

    def test_actions_within_bounds(self):
        actor = Actor(STATE_DIM, ACTION_DIM, action_limit=1.0)
        state = torch.randn(100, STATE_DIM)
        action, _, _ = actor.sample(state)
        assert action.abs().max().item() <= 1.0 + 1e-6

    def test_custom_action_limit(self):
        actor = Actor(STATE_DIM, ACTION_DIM, action_limit=3.0)
        state = torch.randn(100, STATE_DIM)
        action, _, _ = actor.sample(state)
        assert action.abs().max().item() <= 3.0 + 1e-6


# ---------------------------------------------------------
# SACAgent Tests
# ---------------------------------------------------------

class TestSACAgent:
    @pytest.fixture
    def agent(self):
        return SACAgent(STATE_DIM, ACTION_DIM, device="cpu")

    def test_initialization(self, agent):
        assert agent.gamma == 0.99
        assert agent.tau == 0.005
        assert agent.action_dim == ACTION_DIM
        assert agent.device == torch.device("cpu")

    def test_alpha_is_positive(self, agent):
        assert agent.alpha.item() > 0

    def test_target_entropy(self, agent):
        assert agent.target_entropy == -float(ACTION_DIM)


class TestSACSelectAction:
    @pytest.fixture
    def agent(self):
        return SACAgent(STATE_DIM, ACTION_DIM, device="cpu")

    def test_returns_numpy_array(self, agent):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = agent.select_action(state)
        assert isinstance(action, np.ndarray)

    def test_action_shape(self, agent):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = agent.select_action(state)
        assert action.shape == (ACTION_DIM,)

    def test_evaluate_mode_is_deterministic(self, agent):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action1 = agent.select_action(state, evaluate=True)
        action2 = agent.select_action(state, evaluate=True)
        np.testing.assert_array_equal(action1, action2)

    def test_explore_mode_has_stochasticity(self, agent):
        """Over many samples, explore mode should produce varying actions."""
        state = np.random.randn(STATE_DIM).astype(np.float32)
        actions = [agent.select_action(state, evaluate=False) for _ in range(20)]
        unique_actions = len(set(tuple(a) for a in actions))
        assert unique_actions > 1


class TestSACUpdate:
    @pytest.fixture
    def setup(self):
        agent = SACAgent(STATE_DIM, ACTION_DIM, device="cpu")
        buffer = GenericReplayBuffer(capacity=100, device="cpu")
        for _ in range(32):
            s = np.random.randn(STATE_DIM).astype(np.float32)
            a = np.random.randn(ACTION_DIM).astype(np.float32)
            ns = np.random.randn(STATE_DIM).astype(np.float32)
            buffer.push(s, a, np.random.randn(), ns, False)
        return agent, buffer

    def test_update_returns_loss_dict(self, setup):
        agent, buffer = setup
        metrics = agent.update(buffer, batch_size=16)
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha_loss" in metrics
        assert "alpha" in metrics

    def test_update_returns_finite_values(self, setup):
        agent, buffer = setup
        metrics = agent.update(buffer, batch_size=16)
        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_multiple_updates_change_weights(self, setup):
        agent, buffer = setup
        initial_params = [p.clone() for p in agent.actor.parameters()]
        for _ in range(5):
            agent.update(buffer, batch_size=16)
        changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, agent.actor.parameters())
        )
        assert changed, "Actor weights should change after multiple updates"


class TestSACCheckpoint:
    @pytest.fixture
    def agent(self):
        return SACAgent(STATE_DIM, ACTION_DIM, device="cpu")

    def test_save_and_load(self, agent, tmp_path):
        # Modify alpha to a known value
        agent.log_alpha.data.fill_(0.5)
        agent.save_checkpoint(str(tmp_path), suffix="test")

        # Create a fresh agent and load
        new_agent = SACAgent(STATE_DIM, ACTION_DIM, device="cpu")
        new_agent.load_checkpoint(str(tmp_path), suffix="test")

        # Verify alpha was restored
        assert torch.allclose(new_agent.log_alpha, agent.log_alpha)

    def test_load_nonexistent_raises_error(self, agent, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            agent.load_checkpoint(str(tmp_path), suffix="nonexistent")

    def test_alpha_optimizer_works_after_load(self, agent, tmp_path):
        """The critical bug fix: alpha optimizer must still function after loading."""
        agent.save_checkpoint(str(tmp_path), suffix="opt")

        new_agent = SACAgent(STATE_DIM, ACTION_DIM, device="cpu")
        new_agent.load_checkpoint(str(tmp_path), suffix="opt")

        # Verify the optimizer still references the same log_alpha tensor
        opt_param = new_agent.alpha_optimizer.param_groups[0]['params'][0]
        assert opt_param is new_agent.log_alpha, \
            "Alpha optimizer lost reference to log_alpha after checkpoint load"
