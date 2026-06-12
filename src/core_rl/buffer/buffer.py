# src/core_rl/buffer/buffer.py

import random
from collections import deque
import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional

class GenericReplayBuffer:
    """
    Generic Reinforcement Learning experience replay buffer that supports
    multimodal data and Dict structures.
    """
    
    def __init__(self, capacity: int = 100_000, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

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

    def __len__(self) -> int:
        return len(self.buffer)