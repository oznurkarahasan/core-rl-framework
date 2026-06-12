from typing import Any, Dict, List

import numpy as np
import torch

from core_rl.buffer.base_buffer import BaseBuffer


class RolloutBuffer(BaseBuffer):
    """On-policy trajectory buffer for PPO. Collects one rollout then discards after update."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._states: List[Any] = []
        self._actions: List[int] = []
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []

    def push(self, state: Any, action: int, log_prob: torch.Tensor, reward: float, done: bool) -> None:
        self._states.append(state)
        self._actions.append(action)
        self._log_probs.append(log_prob.squeeze())
        self._rewards.append(float(reward))
        self._dones.append(done)

    def get(self) -> Dict[str, Any]:
        return {
            "states": torch.FloatTensor(np.array(self._states)).to(self.device),
            "actions": torch.LongTensor(self._actions).to(self.device),
            "log_probs": torch.stack(self._log_probs).to(self.device),
            "rewards": self._rewards,
            "dones": self._dones,
        }

    def clear(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._dones.clear()

    def __len__(self) -> int:
        return len(self._states)
