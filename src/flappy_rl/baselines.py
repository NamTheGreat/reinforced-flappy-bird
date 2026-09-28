"""Baseline agents for Flappy Bird: RandomAgent and HeuristicAgent."""

import numpy as np


class RandomAgent:
    """Agent that chooses actions uniformly at random."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, 2))


class HeuristicAgent:
    """Heuristic rule-based agent that flaps when below the next pipe gap center.

    Observation layout (12 dimensions):
        obs[0..2]: Pipe 0 [x_pos, upper_bottom_y, lower_top_y]
        obs[3..5]: Pipe 1 [x_pos, upper_bottom_y, lower_top_y]
        obs[6..8]: Pipe 2 [x_pos, upper_bottom_y, lower_top_y]
        obs[9]: Player vertical position y (0 = top, 1 = ground)
        obs[10]: Player vertical velocity vel_y
        obs[11]: Player rotation

    The player bird x position is fixed at 0.2 (normalized).
    Pipe width is ~0.18. If pipe 0 has passed behind the bird (e.g. x < 0.05),
    the next obstacle is pipe 1.
    """

    def __init__(self, player_x_norm: float = 0.2, margin: float = 0.02):
        self.player_x_norm = player_x_norm
        self.margin = margin

    def act(self, obs: np.ndarray) -> int:
        # Determine whether pipe 0 is still ahead or already passed
        # Pipe right edge is obs[0] + 0.18. If right edge < player_x, pipe 0 is passed.
        # ponytail: simple threshold check on pipe[0] x position
        if obs[0] + 0.15 < self.player_x_norm:
            # Target pipe is pipe 1
            gap_center = (obs[4] + obs[5]) / 2.0
        else:
            # Target pipe is pipe 0
            gap_center = (obs[1] + obs[2]) / 2.0

        player_y = obs[9]
        # In Pygame screen coords, y=0 is ceiling, y increases towards ground.
        # If bird is lower than gap center (y > gap_center + margin), flap (action 1).
        # We also check vertical velocity to avoid over-flapping if already climbing fast.
        vel_y = obs[10]
        if player_y > gap_center + self.margin:
            # Flap to climb if not already ascending rapidly
            if vel_y > -0.5:
                return 1
            return 0
        return 0
