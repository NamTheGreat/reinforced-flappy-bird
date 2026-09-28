"""Unit tests for Prioritized Experience Replay buffer, sum-tree, and DQN agent."""

import numpy as np
import pytest
import torch
import per_rs

from flappy_rl.buffers import PERBuffer, PurePythonSumTree
from flappy_rl.agent import DQNAgent


def test_per_tree_add_and_total():
    tree = per_rs.PerTree(16, 0.6)
    assert tree.total() == 0.0

    # Add 4 elements; each gets max_p (default 1.0)
    for _ in range(4):
        tree.add()

    assert tree.len() == 4
    # Total priority should be 4 * 1.0 = 4.0
    assert pytest.approx(tree.total(), rel=1e-5) == 4.0


def test_per_tree_update_priorities():
    tree = per_rs.PerTree(8, 0.5)
    for _ in range(4):
        tree.add()

    # Update priorities with specific TD errors
    # p_i = (|td| + 1e-6)^0.5
    tds = np.array([4.0, 9.0, 16.0, 25.0], dtype=np.float64)
    idxs = np.array([0, 1, 2, 3], dtype=np.int64)
    tree.update(idxs, tds)

    expected_ps = (tds + 1e-6) ** 0.5
    expected_total = np.sum(expected_ps)

    assert pytest.approx(tree.total(), rel=1e-4) == expected_total


def test_sampling_frequency():
    """Verify empirical sampling frequency matches theoretical probability distribution."""
    cap = 8
    alpha = 1.0  # Linear priority for transparent testing
    tree = per_rs.PerTree(cap, alpha)

    for _ in range(4):
        tree.add()

    # Leaf priorities: [10.0, 20.0, 30.0, 40.0]
    # Total = 100.0, probabilities = [0.1, 0.2, 0.3, 0.4]
    tds = np.array([10.0 - 1e-6, 20.0 - 1e-6, 30.0 - 1e-6, 40.0 - 1e-6], dtype=np.float64)
    idxs = np.array([0, 1, 2, 3], dtype=np.int64)
    tree.update(idxs, tds)

    total = tree.total()
    expected_probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Draw 50,000 samples
    n_samples = 50000
    batch_size = 50
    counts = np.zeros(4)

    for _ in range(n_samples // batch_size):
        sampled_idxs, _ = tree.sample(batch_size, beta=0.4)
        for idx in sampled_idxs:
            if idx < 4:
                counts[idx] += 1

    empirical_probs = counts / counts.sum()
    # Check within 2% margin
    assert np.allclose(empirical_probs, expected_probs, atol=0.03)


def test_is_weights_properties():
    """Verify IS weights are in (0, 1] and max is 1.0."""
    tree = per_rs.PerTree(16, 0.6)
    for _ in range(8):
        tree.add()

    tds = np.array([1.0, 5.0, 2.0, 10.0, 0.5, 3.0, 8.0, 4.0], dtype=np.float64)
    idxs = np.arange(8, dtype=np.int64)
    tree.update(idxs, tds)

    idxs_sampled, ws = tree.sample(batch=8, beta=0.5)

    assert len(ws) == 8
    assert np.all(ws > 0.0)
    assert np.all(ws <= 1.00001)
    assert pytest.approx(np.max(ws), rel=1e-5) == 1.0


def test_cross_check_rust_vs_python_tree():
    """Verify Rust PerTree and PurePythonSumTree produce identical totals and structures."""
    cap = 64
    alpha = 0.6
    rust_tree = per_rs.PerTree(cap, alpha)
    py_tree = PurePythonSumTree(cap, alpha)

    np.random.seed(123)
    # Add 40 items
    for _ in range(40):
        r_idx = rust_tree.add()
        p_idx = py_tree.add()
        assert r_idx == p_idx

    assert pytest.approx(rust_tree.total(), rel=1e-6) == py_tree.total()

    # Update with random TD errors
    tds = np.random.uniform(0.1, 10.0, size=40).astype(np.float64)
    idxs = np.arange(40, dtype=np.int64)

    rust_tree.update(idxs, tds)
    py_tree.update(idxs, tds)

    assert pytest.approx(rust_tree.total(), rel=1e-6) == py_tree.total()


def test_dqn_sync_target():
    agent = DQNAgent(obs_dim=12, hidden_dim=64, n_actions=2)

    # Online and target should have identical weights initially
    for p_online, p_target in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_online, p_target)

    # Perturb online weights
    with torch.no_grad():
        for p in agent.q_net.parameters():
            p.add_(1.0)

    # Now they differ
    has_diff = any(
        not torch.allclose(p_online, p_target)
        for p_online, p_target in zip(agent.q_net.parameters(), agent.target_net.parameters())
    )
    assert has_diff

    # Sync target
    agent.sync_target()
    for p_online, p_target in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert torch.allclose(p_online, p_target)


def test_dqn_overfit_sanity_check():
    """Sanity check: Agent can overfit a tiny fixed batch to near-zero loss."""
    agent = DQNAgent(obs_dim=4, hidden_dim=64, n_actions=2, lr=1e-2)

    # Fixed batch of 8 transitions
    s = torch.randn(8, 4)
    a = torch.randint(0, 2, (8, 1))
    r = torch.randn(8, 1)
    s2 = torch.randn(8, 4)
    d = torch.zeros(8, 1)

    initial_loss, _ = agent.learn(s, a, r, s2, d)

    # Train on this batch for 150 steps
    for _ in range(150):
        loss, _ = agent.learn(s, a, r, s2, d)
        agent.sync_target()

    # Loss should have decreased dramatically
    assert loss < initial_loss * 0.1
    assert loss < 0.05
