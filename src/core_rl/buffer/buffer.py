# src/core_rl/buffer/buffer.py

import random
from collections import deque
import numpy as np
import torch
from typing import Any, Dict, Tuple, List, Optional

from core_rl.buffer.base_buffer import BaseBuffer


class GenericReplayBuffer(BaseBuffer):
    """
    Generic Reinforcement Learning experience replay buffer that supports
    multimodal data and Dict structures.
    """

    def __init__(self, capacity: int = 100_000, device: str = "cpu"):
        self._capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        # Parallel deque of experience_ids (None for untracked experiences).
        # Same maxlen keeps it in sync with self.buffer — when buffer evicts
        # its oldest item, _id_deque evicts the corresponding id automatically.
        self._id_deque: deque = deque(maxlen=capacity)
        # Direct reference to the list object inside the deque. O(1) update.
        self._id_to_item: Dict[str, List] = {}
        self.device = torch.device(device)

    def push(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        experience_id: Optional[str] = None,
    ) -> None:
        # Before the deque evicts, clean the id mapping of the outgoing item.
        if len(self.buffer) == self._capacity:
            evicted_id = self._id_deque[0]  # O(1) — deque left end
            if evicted_id is not None:
                self._id_to_item.pop(evicted_id, None)

        item = [state, action, reward, next_state, done]
        self.buffer.append(item)
        self._id_deque.append(experience_id)
        if experience_id is not None:
            self._id_to_item[experience_id] = item  # direct reference, not a copy

    def update_reward(self, experience_id: str, new_reward: float) -> bool:
        """
        Update the stored reward for a previously pushed experience.

        O(1): dict lookup + list index assignment.
        Returns True if updated, False if the experience was not found or evicted.
        """
        item = self._id_to_item.get(experience_id)
        if item is None:
            return False
        item[2] = new_reward  # mutates the list inside the deque in-place
        return True

    def sample(self, batch_size: int) -> Tuple[Any, Any, torch.Tensor, Any, torch.Tensor]:
        # Fail-fast: raise early if buffer has insufficient data.
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough data in buffer! Requested: {batch_size}, Available: {len(self.buffer)}")

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            self._to_tensor(states),
            self._to_tensor(actions),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
            self._to_tensor(next_states),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        )

    def _to_tensor(self, batch_data: List[Any]) -> Optional[Any]:
        # Guard clause: block empty lists or None values.
        if not batch_data or batch_data[0] is None:
            return None

        # If the data is a dictionary, recursively convert each key.
        if isinstance(batch_data[0], dict):
            stacked_dict = {}
            for key in batch_data[0].keys():
                stacked_dict[key] = self._to_tensor([d[key] for d in batch_data])
            return stacked_dict
        
        # Ragged array protection: use np.stack to enforce uniform shapes.
        try:
            stacked_array = np.stack(batch_data)
            return torch.as_tensor(stacked_array, dtype=torch.float32, device=self.device)
        except ValueError as e:
            raise ValueError(
                f"Batch dimensions do not match (ragged array)! Neural networks expect "
                f"fixed-size inputs. Please check your environment outputs. Detail: {e}"
            )

    def clear(self) -> None:
        self.buffer.clear()
        self._id_deque.clear()
        self._id_to_item.clear()

    def __len__(self) -> int:
        return len(self.buffer)