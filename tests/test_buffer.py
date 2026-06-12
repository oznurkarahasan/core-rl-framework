# tests/test_buffer.py

import pytest
import numpy as np
import torch
from core_rl.buffer.buffer import GenericReplayBuffer


# ---------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------

class TestBufferInit:
    def test_creates_with_default_capacity(self):
        buffer = GenericReplayBuffer()
        assert len(buffer) == 0

    def test_creates_with_custom_capacity(self):
        buffer = GenericReplayBuffer(capacity=50)
        assert len(buffer) == 0

    def test_device_defaults_to_cpu(self):
        buffer = GenericReplayBuffer()
        assert buffer.device == torch.device("cpu")


# ---------------------------------------------------------
# Push Tests
# ---------------------------------------------------------

class TestBufferPush:
    def test_push_increments_length(self):
        buffer = GenericReplayBuffer(capacity=10)
        state = np.array([1.0, 2.0, 3.0])
        buffer.push(state, 0, 1.0, state, False)
        assert len(buffer) == 1

    def test_push_respects_capacity(self):
        buffer = GenericReplayBuffer(capacity=3)
        for i in range(5):
            buffer.push(np.array([float(i)]), 0, 1.0, np.array([float(i)]), False)
        assert len(buffer) == 3

    def test_push_with_dict_state(self):
        buffer = GenericReplayBuffer(capacity=10)
        state = {"camera": np.zeros((3, 64, 64)), "speed": np.array([5.0])}
        buffer.push(state, np.array([0.5, -0.3]), 1.0, state, False)
        assert len(buffer) == 1

    def test_push_with_none_next_state(self):
        buffer = GenericReplayBuffer(capacity=10)
        buffer.push(np.array([1.0]), 0, 0.0, None, True)
        assert len(buffer) == 1


# ---------------------------------------------------------
# Sample Tests
# ---------------------------------------------------------

class TestBufferSample:
    @pytest.fixture
    def filled_buffer(self):
        buffer = GenericReplayBuffer(capacity=100)
        for i in range(20):
            state = np.array([float(i), float(i + 1)], dtype=np.float32)
            action = np.array([float(i) * 0.1], dtype=np.float32)
            next_state = np.array([float(i + 1), float(i + 2)], dtype=np.float32)
            buffer.push(state, action, float(i) * 0.5, next_state, i == 19)
        return buffer

    def test_sample_returns_correct_batch_size(self, filled_buffer):
        states, actions, rewards, next_states, dones = filled_buffer.sample(8)
        assert states.shape[0] == 8
        assert actions.shape[0] == 8
        assert rewards.shape == (8, 1)
        assert next_states.shape[0] == 8
        assert dones.shape == (8, 1)

    def test_sample_returns_tensors(self, filled_buffer):
        states, actions, rewards, next_states, dones = filled_buffer.sample(4)
        assert isinstance(states, torch.Tensor)
        assert isinstance(actions, torch.Tensor)
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(dones, torch.Tensor)

    def test_sample_preserves_state_dimensions(self, filled_buffer):
        states, _, _, _, _ = filled_buffer.sample(5)
        assert states.shape == (5, 2)

    def test_sample_raises_on_insufficient_data(self):
        buffer = GenericReplayBuffer(capacity=10)
        buffer.push(np.array([1.0]), 0, 1.0, np.array([2.0]), False)
        with pytest.raises(ValueError, match="Not enough data"):
            buffer.sample(5)

    def test_sample_raises_on_empty_buffer(self):
        buffer = GenericReplayBuffer(capacity=10)
        with pytest.raises(ValueError, match="Not enough data"):
            buffer.sample(1)


# ---------------------------------------------------------
# Dict (Multimodal) Support Tests
# ---------------------------------------------------------

class TestBufferMultimodal:
    @pytest.fixture
    def dict_buffer(self):
        buffer = GenericReplayBuffer(capacity=100)
        for i in range(10):
            state = {
                "image": np.random.randn(3, 32, 32).astype(np.float32),
                "velocity": np.array([float(i)], dtype=np.float32),
            }
            next_state = {
                "image": np.random.randn(3, 32, 32).astype(np.float32),
                "velocity": np.array([float(i + 1)], dtype=np.float32),
            }
            buffer.push(state, np.array([0.5], dtype=np.float32), 1.0, next_state, False)
        return buffer

    def test_sample_dict_returns_dict(self, dict_buffer):
        states, _, _, _, _ = dict_buffer.sample(4)
        assert isinstance(states, dict)
        assert "image" in states
        assert "velocity" in states

    def test_sample_dict_has_correct_shapes(self, dict_buffer):
        states, _, _, _, _ = dict_buffer.sample(4)
        assert states["image"].shape == (4, 3, 32, 32)
        assert states["velocity"].shape == (4, 1)

    def test_sample_dict_returns_tensors(self, dict_buffer):
        states, _, _, _, _ = dict_buffer.sample(4)
        assert isinstance(states["image"], torch.Tensor)
        assert isinstance(states["velocity"], torch.Tensor)


# ---------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------

class TestBufferEdgeCases:
    def test_sample_with_none_next_state(self):
        buffer = GenericReplayBuffer(capacity=10)
        for _ in range(5):
            buffer.push(np.array([1.0, 2.0]), np.array([0.1]), 1.0, None, True)
        states, actions, rewards, next_states, dones = buffer.sample(3)
        assert next_states is None

    def test_ragged_array_raises_error(self):
        buffer = GenericReplayBuffer(capacity=10)
        buffer.push(np.array([1.0, 2.0]), 0, 1.0, np.array([1.0, 2.0]), False)
        buffer.push(np.array([1.0, 2.0, 3.0]), 0, 1.0, np.array([1.0, 2.0, 3.0]), False)
        with pytest.raises(ValueError, match="Batch dimensions do not match"):
            buffer.sample(2)

    def test_clear_empties_buffer(self):
        buffer = GenericReplayBuffer(capacity=10)
        for _ in range(5):
            buffer.push(np.array([1.0]), 0, 1.0, np.array([1.0]), False)
        buffer.clear()
        assert len(buffer) == 0


# ---------------------------------------------------------
# RolloutBuffer Tests
# ---------------------------------------------------------

class TestRolloutBuffer:
    @pytest.fixture
    def filled_buffer(self):
        from core_rl.buffer.rollout_buffer import RolloutBuffer
        import torch
        buf = RolloutBuffer(device="cpu")
        for i in range(10):
            s = np.random.randn(4).astype(np.float32)
            lp = torch.tensor(-0.5)
            buf.push(s, i % 3, lp, float(np.random.randn()), i == 9)
        return buf

    def test_len(self, filled_buffer):
        assert len(filled_buffer) == 10

    def test_get_returns_correct_keys(self, filled_buffer):
        rollout = filled_buffer.get()
        assert set(rollout.keys()) == {"states", "actions", "log_probs", "rewards", "dones"}

    def test_get_tensor_shapes(self, filled_buffer):
        rollout = filled_buffer.get()
        assert rollout["states"].shape == (10, 4)
        assert rollout["actions"].shape == (10,)
        assert rollout["log_probs"].shape == (10,)

    def test_clear_empties_buffer(self, filled_buffer):
        filled_buffer.clear()
        assert len(filled_buffer) == 0
