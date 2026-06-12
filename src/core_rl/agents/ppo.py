# src/core_rl/agents/ppo.py  — Proximal Policy Optimization (PPO)
#
# Why PPO?
#   SAC is designed for continuous action spaces (e.g., steering angle, throttle).
#   PPO fills the gap for discrete action spaces (e.g., buy/sell/hold in trading,
#   turn left/right/go straight in grid environments).
#
# Key Differences from SAC:
#
#   Policy Type:
#       SAC  → Off-policy  (learns from old experiences stored in a replay buffer)
#       PPO  → On-policy   (learns only from fresh trajectories collected by the current policy)
#
#   Action Space:
#       SAC  → Continuous   (Squashed Gaussian distribution)
#       PPO  → Discrete     (Categorical distribution)
#
#   Architecture:
#       SAC  → Separate Actor and Critic networks
#       PPO  → Shared feature extractor with Actor and Critic heads
#
#   Update Mechanism:
#       SAC  → Soft Q-learning with entropy regularization
#       PPO  → Clipped surrogate objective to prevent destructively large policy updates

from pathlib import Path
from collections import deque
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any, Tuple, TYPE_CHECKING

from core_rl.agents.base_agent import BaseAgent

if TYPE_CHECKING:
    from core_rl.buffer.rollout_buffer import RolloutBuffer

# ---------------------------------------------------------
# Neural Network Definitions
# ---------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared feature extraction architecture for PPO (Discrete Action Space).
    Outputs both action probabilities (Policy) and State Value (Critic).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        self.base_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        features = self.base_net(state)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_logits, state_value

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, state_value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, action_log_prob, state_value, entropy

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, state_value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_log_prob, torch.squeeze(state_value), dist_entropy


# ---------------------------------------------------------
# Main PPO Agent Implementation
# ---------------------------------------------------------

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99, eps_clip: float = 0.2,
                 k_epochs: int = 4, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, device: str = "cpu"):
        super().__init__(device)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def select_action(self, state: Any, evaluate: bool = False) -> Any:
        """
        Returns (action_int, log_prob_tensor, state_value_tensor, entropy_float).
        evaluate=True: greedy action, log_prob/state_value are None, entropy is 0.0.
        evaluate=False: sampled action with full distribution statistics.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action_logits, _ = self.policy_old(state_tensor)
                action = torch.argmax(action_logits, dim=-1)
                return action.item(), None, None, 0.0
            else:
                action, action_log_prob, state_value, entropy = self.policy_old.act(state_tensor)
                return action.item(), action_log_prob, state_value, entropy.item()

    def update(self, buffer: "RolloutBuffer", batch_size: int = 0) -> Dict[str, float]:
        rollouts = buffer.get()

        old_states = rollouts['states']
        old_actions = rollouts['actions']
        old_log_probs = rollouts['log_probs']
        
        # FIX 5: O(1) performance instead of O(n^2) for reward collection
        rewards_deque = deque()
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rollouts['rewards']), reversed(rollouts['dones'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_deque.appendleft(discounted_reward) # O(1) time complexity
            
        # Convert deque directly to list then tensor
        rewards = torch.tensor(list(rewards_deque), dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        total_loss_val = 0.0
        
        for _ in range(self.k_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss_val += loss.mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())

        return {"ppo_loss": total_loss_val / self.k_epochs}

    def save_checkpoint(self, checkpoint_dir: str, suffix: str = "") -> None:
        save_dir = self._ensure_dir(checkpoint_dir)
        filepath = save_dir / f"ppo_checkpoint_{suffix}.pth"
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, checkpoint_dir: str, suffix: str = "") -> None:
        filepath = Path(checkpoint_dir) / f"ppo_checkpoint_{suffix}.pth"
        
        if filepath.exists():
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")