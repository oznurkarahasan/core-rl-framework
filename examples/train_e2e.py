"""
End-to-end training validation for Phase 2 agents.

  SACAgent  → LunarLanderContinuous-v3  (continuous action space)
  PPOAgent  → LunarLander-v3            (discrete action space)

Run:
    python examples/train_e2e.py

Expected outcome: both agents improve over baseline random performance within
~200 episodes. The entropy range printed at the end informs Phase 3 threshold
tuning for uncertainty detection.
"""

import numpy as np
import torch
import gymnasium as gym

from core_rl.agents.sac import SACAgent
from core_rl.agents.ppo import PPOAgent
from core_rl.buffer.buffer import GenericReplayBuffer
from core_rl.buffer.rollout_buffer import RolloutBuffer


# ── SAC on LunarLanderContinuous-v3 ──────────────────────────────────────────

def train_sac(
    n_episodes: int = 200,
    batch_size: int = 256,
    buffer_capacity: int = 50_000,
    warmup_steps: int = 1_000,
) -> None:
    env = gym.make("LunarLanderContinuous-v3")
    state_dim = env.observation_space.shape[0]   # 8
    action_dim = env.action_space.shape[0]        # 2
    action_limit = float(env.action_space.high[0])  # 1.0

    agent = SACAgent(state_dim, action_dim, action_limit=action_limit)
    buffer = GenericReplayBuffer(capacity=buffer_capacity)

    total_steps = 0
    entropy_log: list[float] = []

    print("\n── SAC · LunarLanderContinuous-v3 ──")
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if total_steps >= warmup_steps and len(buffer) >= batch_size:
                metrics = agent.update(buffer, batch_size)
                if "alpha" in metrics:
                    entropy_log.append(metrics["alpha"])

        if ep % 50 == 0:
            print(f"  ep {ep:>3}  reward={ep_reward:>8.1f}  steps={total_steps}")

    env.close()

    if entropy_log:
        arr = np.array(entropy_log)
        print(f"\n  Alpha (entropy scale) stats over training:")
        print(f"    min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
        print(f"  → Phase 3 uncertainty threshold suggestion: alpha > {arr.mean():.4f}")


# ── PPO on LunarLander-v3 ─────────────────────────────────────────────────────

def train_ppo(
    n_episodes: int = 200,
    rollout_len: int = 512,
) -> None:
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.n             # 4

    agent = PPOAgent(state_dim, action_dim)
    buffer = RolloutBuffer()

    ep_rewards: list[float] = []
    entropy_log: list[float] = []

    print("\n── PPO · LunarLander-v3 ──")
    state, _ = env.reset()
    ep_reward = 0.0
    ep_count = 0

    for step in range(1, n_episodes * rollout_len + 1):
        action, log_prob, _ = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, log_prob, reward, done)
        state = next_state
        ep_reward += reward

        if done:
            ep_rewards.append(ep_reward)
            ep_count += 1
            ep_reward = 0.0
            state, _ = env.reset()

        if step % rollout_len == 0:
            metrics = agent.update(buffer)
            entropy_log.append(metrics["ppo_loss"])
            buffer.clear()

            if ep_count % 50 == 0 and ep_rewards:
                recent = np.mean(ep_rewards[-20:])
                print(f"  ep {ep_count:>3}  reward(20-avg)={recent:>8.1f}")

    env.close()

    if entropy_log:
        arr = np.array(entropy_log)
        print(f"\n  PPO loss stats over training:")
        print(f"    min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    train_sac()
    train_ppo()

    print("\nEnd-to-end validation complete.")
