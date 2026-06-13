import torch
import torch.nn as nn
import pytest

from core_rl.continual.ewc import EWC
from core_rl.agents.sac import SACAgent
from core_rl.buffer.buffer import GenericReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_model() -> nn.Linear:
    model = nn.Linear(4, 2)
    nn.init.constant_(model.weight, 0.5)
    nn.init.zeros_(model.bias)
    return model


def _linear_log_prob_fn(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    return model(states).sum(dim=-1, keepdim=True)


def _make_sac(ewc_lambda: float = 5000.0) -> SACAgent:
    return SACAgent(state_dim=4, action_dim=2, ewc_lambda=ewc_lambda)


def _make_buffer(n: int = 300) -> GenericReplayBuffer:
    buf = GenericReplayBuffer(capacity=500)
    for _ in range(n):
        buf.push(
            state=torch.zeros(4).numpy(),
            action=torch.zeros(2).numpy(),
            reward=1.0,
            next_state=torch.zeros(4).numpy(),
            done=False,
        )
    return buf


# ---------------------------------------------------------------------------
# EWC unit tests
# ---------------------------------------------------------------------------

class TestEWCPenalty:
    def test_penalty_zero_before_consolidate(self):
        ewc = EWC()
        model = _make_linear_model()
        assert ewc.penalty(model).item() == pytest.approx(0.0)

    def test_penalty_zero_at_optimal_params(self):
        ewc = EWC(ewc_lambda=5000.0)
        model = _make_linear_model()
        states = torch.randn(16, 4)
        ewc.consolidate(model, states, _linear_log_prob_fn)
        # Params unchanged → penalty must be 0
        assert ewc.penalty(model).item() == pytest.approx(0.0, abs=1e-5)

    def test_penalty_positive_after_perturbation(self):
        ewc = EWC(ewc_lambda=5000.0)
        model = _make_linear_model()
        states = torch.randn(16, 4)
        ewc.consolidate(model, states, _linear_log_prob_fn)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.ones_like(param))

        assert ewc.penalty(model).item() > 0.0

    def test_penalty_scales_with_lambda(self):
        states = torch.randn(16, 4)

        ewc_low = EWC(ewc_lambda=100.0)
        model_low = _make_linear_model()
        ewc_low.consolidate(model_low, states, _linear_log_prob_fn)
        with torch.no_grad():
            for p in model_low.parameters():
                p.add_(torch.ones_like(p))

        ewc_high = EWC(ewc_lambda=10000.0)
        model_high = _make_linear_model()
        ewc_high.consolidate(model_high, states, _linear_log_prob_fn)
        with torch.no_grad():
            for p in model_high.parameters():
                p.add_(torch.ones_like(p))

        assert ewc_high.penalty(model_high).item() > ewc_low.penalty(model_low).item()


class TestEWCConsolidate:
    def test_consolidated_flag_set(self):
        ewc = EWC()
        assert not ewc._consolidated
        model = _make_linear_model()
        ewc.consolidate(model, torch.randn(8, 4), _linear_log_prob_fn)
        assert ewc._consolidated

    def test_fisher_and_params_stored(self):
        ewc = EWC()
        model = _make_linear_model()
        ewc.consolidate(model, torch.randn(8, 4), _linear_log_prob_fn)
        assert set(ewc._fisher.keys()) == set(ewc._optimal_params.keys())
        assert len(ewc._fisher) > 0

    def test_model_back_to_train_mode(self):
        ewc = EWC()
        model = _make_linear_model()
        model.train()
        ewc.consolidate(model, torch.randn(8, 4), _linear_log_prob_fn)
        assert model.training

    def test_model_stays_eval_if_was_eval(self):
        ewc = EWC()
        model = _make_linear_model()
        model.eval()
        ewc.consolidate(model, torch.randn(8, 4), _linear_log_prob_fn)
        assert not model.training


# ---------------------------------------------------------------------------
# SACAgent integration tests
# ---------------------------------------------------------------------------

class TestSACAgentEWC:
    def test_default_ewc_lambda(self):
        agent = _make_sac()
        assert agent.ewc_lambda == pytest.approx(5000.0)

    def test_ewc_lambda_setter(self):
        agent = _make_sac()
        agent.ewc_lambda = 1000.0
        assert agent.ewc_lambda == pytest.approx(1000.0)

    def test_update_works_before_consolidate(self):
        agent = _make_sac()
        buf = _make_buffer()
        metrics = agent.update(buf, batch_size=64)
        assert "actor_loss" in metrics

    def test_consolidate_marks_ewc_consolidated(self):
        agent = _make_sac()
        buf = _make_buffer()
        assert not agent._ewc._consolidated
        agent.consolidate(buf, batch_size=64)
        assert agent._ewc._consolidated

    def test_update_works_after_consolidate(self):
        agent = _make_sac()
        buf = _make_buffer()
        agent.consolidate(buf, batch_size=64)
        metrics = agent.update(buf, batch_size=64)
        assert "actor_loss" in metrics

    def test_ewc_penalty_increases_actor_loss(self):
        """Actor loss with EWC should be >= actor loss without EWC after perturbation."""
        torch.manual_seed(42)

        agent_no_ewc = _make_sac(ewc_lambda=0.0)
        agent_ewc = _make_sac(ewc_lambda=50000.0)

        buf = _make_buffer()

        # Copy identical weights into both agents
        agent_ewc.actor.load_state_dict(agent_no_ewc.actor.state_dict())
        agent_ewc.critic.load_state_dict(agent_no_ewc.critic.state_dict())
        agent_ewc.critic_target.load_state_dict(agent_no_ewc.critic_target.state_dict())

        # Consolidate only the EWC agent, then perturb its actor
        agent_ewc.consolidate(buf, batch_size=64)
        with torch.no_grad():
            for p in agent_ewc.actor.parameters():
                p.add_(torch.ones_like(p) * 0.1)

        metrics_no_ewc = agent_no_ewc.update(buf, batch_size=64)
        metrics_ewc = agent_ewc.update(buf, batch_size=64)

        assert metrics_ewc["actor_loss"] != metrics_no_ewc["actor_loss"]
