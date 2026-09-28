"""DQN Agent implementation with Vanilla DQN and Double DQN support."""

from typing import Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from flappy_rl.networks import QNetwork


class DQNAgent:
    """Deep Q-Network agent supporting Vanilla DQN and Double DQN.

    Key algorithmic features:
    - Target network for stable Q-learning targets ("freeze the goalpost")
    - Epsilon-greedy exploration ("explore early, exploit later")
    - Double DQN mode to mitigate Q-value overestimation bias ("one picks, another judges")
    - Importance sampling weights support for Prioritized Experience Replay
    """

    def __init__(
        self,
        obs_dim: int = 12,
        hidden_dim: int = 128,
        n_actions: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        double: bool = False,
        grad_clip: float = 10.0,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.double = double
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.q_net = QNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        """Epsilon-greedy action selection."""
        if eps > 0.0 and np.random.rand() < eps:
            return int(np.random.randint(0, self.n_actions))

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(obs_t)
            return int(q_values.argmax(dim=1).item())

    def learn(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        s2: torch.Tensor,
        d: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Perform one gradient update step.

        Returns:
            Tuple of (loss_value, td_error_tensor).
        """
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        q_pred = self.q_net(s).gather(1, a)

        with torch.no_grad():
            if self.double:
                # Double DQN: online network selects action, target network evaluates it
                a2 = self.q_net(s2).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s2).gather(1, a2)
            else:
                # Standard DQN: target network selects max action value
                q_next = self.target_net(s2).max(dim=1, keepdim=True)[0]
            q_target = r + self.gamma * q_next * (1.0 - d)

        td_error = q_target - q_pred

        if weights is not None:
            weights = weights.to(self.device)
            loss = (weights * td_error.pow(2)).mean()
        else:
            loss = td_error.pow(2).mean()

        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.opt.step()

        return loss.item(), td_error.detach()

    def sync_target(self) -> None:
        """Hard copy online weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, filepath: str) -> None:
        """Save agent state dict."""
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "opt": self.opt.state_dict(),
                "double": self.double,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load agent state dict."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        if "opt" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt"])
        if "double" in checkpoint:
            self.double = checkpoint["double"]
