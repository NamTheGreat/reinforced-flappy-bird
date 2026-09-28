"""Neural network architectures for DQN."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Deep Q-Network MLP mapping state observations to action values Q(s, a).

    Architecture: Linear(obs_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, n_actions)
    """

    def __init__(self, obs_dim: int = 12, hidden_dim: int = 128, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
