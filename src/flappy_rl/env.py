"""Environment creation and utilities for Flappy Bird RL with optional reward shaping."""

from typing import Optional
import numpy as np
import gymnasium
import flappy_bird_gymnasium


class FlappyRewardShapingWrapper(gymnasium.Wrapper):
    """Potential-based reward shaping wrapper for Flappy Bird.

    Theoretical Foundation:
        In FlappyBird-v0, rewards (+1.0 for clearing pipes, -1.0 on collision) are delayed.
        This wrapper adds an alignment shaping term rewarding the bird for maintaining
        vertical alignment with the approaching pipe's open gap.

        Potential formulation:
            Phi(s) = - |y_bird - y_gap_center| / gap_height
            F(s, s') = gamma * Phi(s') - Phi(s)

        Under Ng, Harada, & Russell (1999) Potential-Based Reward Shaping (PBRS),
        shaping of the form F(s, s') = gamma * Phi(s') - Phi(s) is guaranteed to
        preserve the exact optimal policy pi* of the unshaped MDP.
    """

    def __init__(self, env: gymnasium.Env, gamma: float = 0.99, scale: float = 0.05):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale
        self.prev_phi = 0.0

    def _potential(self, obs: np.ndarray) -> float:
        # obs[1]: top pipe bottom edge y
        # obs[2]: bottom pipe top edge y
        # obs[9]: bird vertical position y
        gap_center = 0.5 * (obs[1] + obs[2])
        gap_height = max(0.1, obs[2] - obs[1])
        norm_dist = abs(obs[9] - gap_center) / gap_height
        return -float(norm_dist)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_phi = self._potential(obs)
        return obs, info

    def step(self, action):
        next_obs, reward, term, trunc, info = self.env.step(action)
        curr_phi = 0.0 if (term or trunc) else self._potential(next_obs)
        # Potential-based shaping: F = gamma * Phi(s') - Phi(s)
        f_shaping = self.scale * (self.gamma * curr_phi - self.prev_phi)
        shaped_reward = float(reward + f_shaping)
        self.prev_phi = curr_phi
        return next_obs, shaped_reward, term, trunc, info


def make_env(
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    max_episode_steps: int = 5000,
    reward_shaping: bool = False,
    shaping_scale: float = 0.05,
    gamma: float = 0.99,
) -> gymnasium.Env:
    """Create and return a configured Flappy Bird Gymnasium environment.

    Args:
        seed: Random seed for reproducibility.
        render_mode: None, "human", or "rgb_array".
        max_episode_steps: Maximum timesteps per episode to prevent infinite play.
        reward_shaping: Whether to wrap with potential-based gap alignment shaping.
        shaping_scale: Scaling factor for potential-based shaping term.
        gamma: Discount factor used in potential difference F = gamma * Phi(s') - Phi(s).

    Returns:
        Configured gymnasium environment.
    """
    env = gymnasium.make(
        "FlappyBird-v0",
        render_mode=render_mode,
        use_lidar=False,
        max_episode_steps=max_episode_steps,
    )
    if reward_shaping:
        env = FlappyRewardShapingWrapper(env, gamma=gamma, scale=shaping_scale)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
