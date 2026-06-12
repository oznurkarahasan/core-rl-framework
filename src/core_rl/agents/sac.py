# src/core_rl/agents/sac.py  — Soft Actor-Critic (SAC)
#
# SAC Architecture Overview:
#
#   Actor (Policy Network):
#       Given a state, produces an action and its log-probability (log_prob).
#       Uses a Squashed Gaussian distribution to output continuous actions.
#
#   Critic (Value Network):
#       Estimates how "good" a (state, action) pair is by predicting Q-values.
#       Two independent Q-networks (Double-Q) are used to mitigate
#       overestimation bias.
#
#   Temperature (Alpha):
#       An automatic entropy tuning engine that dynamically adjusts how much
#       the agent should explore (randomness) vs. exploit (greedy behavior).
#       Higher alpha → more exploration, lower alpha → more exploitation.

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Tuple, Any

# Core module imports
from core_rl.agents.base_agent import BaseAgent
from core_rl.buffer.buffer import GenericReplayBuffer

# ---------------------------------------------------------
# Neural Network Definitions
# ---------------------------------------------------------

class Critic(nn.Module):
    """
    Double Q-Network architecture for SAC.
    Outputs two Q-values to mitigate overestimation bias.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa), self.q2_net(sa)


class Actor(nn.Module):
    """
    Squashed Gaussian Policy Network.
    Outputs continuous action distributions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, action_limit: float = 1.0):
        super(Actor, self).__init__()
        self.action_limit = action_limit
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal_dist = Normal(mean, std)
        x_t = normal_dist.rsample()
        action = torch.tanh(x_t)
        
        log_prob = normal_dist.log_prob(x_t)
        log_prob -= torch.log(self.action_limit * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.action_limit, log_prob, torch.tanh(mean) * self.action_limit


# ---------------------------------------------------------
# Main SAC Agent Implementation
# ---------------------------------------------------------

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) Agent.
    """
    
    def __init__(self, state_dim: int, action_dim: int, action_limit: float = 1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, device: str = "cpu"):
        super().__init__(device)
        
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, action_limit=action_limit).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Simplified and safe target entropy calculation
        self.target_entropy = -float(action_dim)
        
        # log_alpha is managed carefully to avoid losing optimizer references
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: Any, evaluate: bool = False) -> Tuple[np.ndarray, float]:
        """
        Returns (action_np, entropy_float).
        evaluate=True: deterministic action, entropy is 0.0.
        evaluate=False: stochastic action, entropy ≈ -log_prob (nats).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                _, _, deterministic_action = self.actor.sample(state_tensor)
                return deterministic_action.cpu().numpy()[0], 0.0
            else:
                action, log_prob, _ = self.actor.sample(state_tensor)
                return action.cpu().numpy()[0], -log_prob.item()

    def update(self, replay_buffer: GenericReplayBuffer, batch_size: int) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Critic Update
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1.0 - dones) * self.gamma * min_target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha Update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Target Network Update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item()
        }

    def save_checkpoint(self, checkpoint_dir: str, suffix: str = "") -> None:
        save_dir = self._ensure_dir(checkpoint_dir)
        filepath = save_dir / f"sac_checkpoint_{suffix}.pth"
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'log_alpha_data': self.log_alpha.data, 
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, filepath)

    def load_checkpoint(self, checkpoint_dir: str, suffix: str = "") -> None:
        filepath = Path(checkpoint_dir) / f"sac_checkpoint_{suffix}.pth"
        
        if filepath.exists():
            # Added weights_only=True for secure loading
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            
            # In-place copy to prevent breaking the optimizer reference
            self.log_alpha.data.copy_(checkpoint['log_alpha_data'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")