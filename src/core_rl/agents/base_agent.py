# src/core_rl/agents/base_agent.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from core_rl.buffer.base_buffer import BaseBuffer

class BaseAgent(ABC):
    """
    Abstract Base Class for all Reinforcement Learning agents in the Core-RL framework.
    Enforces a strict contract for action selection, model updating, and checkpoint management.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def _ensure_dir(self, path: str) -> Path:
        """
        Ensures the given directory exists, creating it if necessary.
        Returns the Path object for convenience.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @abstractmethod
    def select_action(self, state: Any, evaluate: bool = False) -> Any:
        """
        Selects an action based on the current state.
        
        Args:
            state (Any): The current environment observation (can be a Dict, array, etc.).
            evaluate (bool): If True, returns the deterministic action (no exploration noise).
                             If False, samples from the stochastic policy.
                             
        Returns:
            Any: The chosen action to be sent to the environment.
        """
        pass

    @abstractmethod
    def update(self, buffer: "BaseBuffer", batch_size: int = 0) -> Dict[str, float]:
        """
        Updates the agent's neural networks from the provided buffer.

        Off-policy agents (SAC) sample batch_size transitions randomly.
        On-policy agents (PPO) consume the full rollout and ignore batch_size.

        Returns:
            Dict[str, float]: Loss metrics for logging (e.g., actor_loss, critic_loss).
        """
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_dir: str, suffix: str = "") -> None:
        """
        Serializes and saves the agent's neural network weights and optimizer states.
        Crucial for Continual Learning and pausing/resuming training.

        Note: Use self._ensure_dir(checkpoint_dir) to guarantee the directory exists.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: str, suffix: str = "") -> None:
        """
        Loads the agent's neural network weights and optimizer states from disk.
        """
        pass