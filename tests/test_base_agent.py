# tests/test_base_agent.py

import pytest
import torch
from core_rl.agents.base_agent import BaseAgent


# ---------------------------------------------------------
# ABC Contract Tests
# ---------------------------------------------------------

class TestBaseAgentABC:
    def test_cannot_instantiate_directly(self):
        """BaseAgent is abstract — direct instantiation must raise TypeError."""
        with pytest.raises(TypeError):
            BaseAgent()

    def test_incomplete_subclass_raises_error(self):
        """A subclass that doesn't implement all abstract methods must raise TypeError."""
        class IncompleteAgent(BaseAgent):
            def select_action(self, state, evaluate=False):
                pass
            # Missing: update, save_checkpoint, load_checkpoint

        with pytest.raises(TypeError):
            IncompleteAgent()

    def test_complete_subclass_can_instantiate(self):
        """A subclass that implements all abstract methods should work."""
        class DummyAgent(BaseAgent):
            def select_action(self, state, evaluate=False):
                return 0
            def update(self, replay_buffer, batch_size):
                return {}
            def save_checkpoint(self, checkpoint_dir, suffix=""):
                pass
            def load_checkpoint(self, checkpoint_dir, suffix=""):
                pass

        agent = DummyAgent()
        assert agent is not None


# ---------------------------------------------------------
# Device Management Tests
# ---------------------------------------------------------

class TestBaseAgentDevice:
    def _make_agent(self, device="cpu"):
        class DummyAgent(BaseAgent):
            def select_action(self, state, evaluate=False):
                return 0
            def update(self, replay_buffer, batch_size):
                return {}
            def save_checkpoint(self, checkpoint_dir, suffix=""):
                pass
            def load_checkpoint(self, checkpoint_dir, suffix=""):
                pass
        return DummyAgent(device=device)

    def test_default_device_is_cpu(self):
        agent = self._make_agent()
        assert agent.device == torch.device("cpu")

    def test_custom_device(self):
        agent = self._make_agent(device="cpu")
        assert agent.device == torch.device("cpu")


# ---------------------------------------------------------
# _ensure_dir Helper Tests
# ---------------------------------------------------------

class TestEnsureDir:
    def _make_agent(self):
        class DummyAgent(BaseAgent):
            def select_action(self, state, evaluate=False):
                return 0
            def update(self, replay_buffer, batch_size):
                return {}
            def save_checkpoint(self, checkpoint_dir, suffix=""):
                pass
            def load_checkpoint(self, checkpoint_dir, suffix=""):
                pass
        return DummyAgent()

    def test_ensure_dir_creates_directory(self, tmp_path):
        agent = self._make_agent()
        new_dir = tmp_path / "checkpoints" / "nested"
        result = agent._ensure_dir(str(new_dir))
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_on_existing_directory(self, tmp_path):
        agent = self._make_agent()
        existing = tmp_path / "existing"
        existing.mkdir()
        result = agent._ensure_dir(str(existing))
        assert result == existing
