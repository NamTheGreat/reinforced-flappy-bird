from dataclasses import dataclass, field
from typing import List


@dataclass
class DQNConfig:
    # Environment
    env_id: str = "FlappyBird-v0"
    max_episode_steps: int = 5000
    obs_dim: int = 12
    n_actions: int = 2
    reward_shaping: bool = False
    shaping_scale: float = 0.05

    # Architecture
    hidden_dim: int = 128
    dueling: bool = False

    # Optimization
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    grad_clip: float = 10.0

    # Target Network Synchronization
    tau: float = 1.0  # 1.0 = hard update every target_update_interval; < 1.0 = soft update each step
    target_update_interval: int = 1000  # used when tau == 1.0

    # Replay buffer & Multi-step returns
    buffer_type: str = "uniform"  # "uniform" or "per"
    buffer_capacity: int = 65536  # 2^16
    warmup_steps: int = 2000
    train_freq: int = 1
    n_step: int = 1  # 1 = standard 1-step Bellman; > 1 = n-step return bootstrapping

    # Exploration (step-based epsilon decay)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 50000

    # Prioritized Experience Replay
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 100000

    # Algorithm variant
    double: bool = False

    # Training & evaluation
    total_timesteps: int = 200000
    eval_freq: int = 5000
    eval_episodes: int = 20
    seed: int = 0
