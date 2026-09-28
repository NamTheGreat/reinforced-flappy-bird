"""Environment creation and utilities for Flappy Bird RL."""

from typing import Optional
import gymnasium
import flappy_bird_gymnasium


def make_env(
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    max_episode_steps: int = 5000,
) -> gymnasium.Env:
    """Create and return a configured Flappy Bird Gymnasium environment.

    Args:
        seed: Random seed for reproducibility.
        render_mode: None, "human", or "rgb_array".
        max_episode_steps: Maximum timesteps per episode to prevent infinite play.

    Returns:
        Configured gymnasium environment.
    """
    env = gymnasium.make(
        "FlappyBird-v0",
        render_mode=render_mode,
        use_lidar=False,
        max_episode_steps=max_episode_steps,
    )
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
