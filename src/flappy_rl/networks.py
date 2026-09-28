"""Neural network architectures for DQN and Dueling DQN."""

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


class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network decomposing Q(s, a) into state value V(s) and advantage A(s, a).

    Formulation:
        Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum_{a'} A(s, a'))

    Architecture:
        Shared: Linear(obs_dim, hidden_dim) -> ReLU
        Value stream: Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)
        Advantage stream: Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, n_actions)
    """

    def __init__(self, obs_dim: int = 12, hidden_dim: int = 128, n_actions: int = 2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.val_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        val = self.val_stream(features)
        adv = self.adv_stream(features)
        return val + (adv - adv.mean(dim=-1, keepdim=True))


def make_q_network(
    obs_dim: int = 12,
    hidden_dim: int = 128,
    n_actions: int = 2,
    dueling: bool = False,
) -> nn.Module:
    """Instantiate standard QNetwork or DuelingQNetwork."""
    if dueling:
        return DuelingQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions)
    return QNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions)
