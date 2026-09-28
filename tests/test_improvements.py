"""Unit tests for advanced improvements: Dueling DQN, Polyak target updates, N-step returns, and PBRS."""

import pytest
import numpy as np
import torch

from flappy_rl.networks import DuelingQNetwork, make_q_network, QNetwork
from flappy_rl.agent import DQNAgent
from flappy_rl.buffers import NStepCollector
from flappy_rl.env import make_env, FlappyRewardShapingWrapper


def test_dueling_network_shapes_and_centering():
    """Verify DuelingQNetwork output shapes and advantage mean centering."""
    obs_dim = 12
    hidden_dim = 64
    n_actions = 2
    batch_size = 16

    net = DuelingQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions)
    x = torch.randn(batch_size, obs_dim)
    q_vals = net(x)

    assert q_vals.shape == (batch_size, n_actions)

    # Verify manual decomposition
    features = net.shared(x)
    val = net.val_stream(features)
    adv = net.adv_stream(features)

    assert val.shape == (batch_size, 1)
    assert adv.shape == (batch_size, n_actions)

    centered_adv = adv - adv.mean(dim=-1, keepdim=True)
    # The mean along action dim of centered advantage must be identically 0
    np.testing.assert_allclose(centered_adv.mean(dim=-1).detach().numpy(), 0.0, atol=1e-6)

    expected_q = val + centered_adv
    np.testing.assert_allclose(q_vals.detach().numpy(), expected_q.detach().numpy(), atol=1e-6)


def test_make_q_network_factory():
    """Verify factory returns appropriate network classes."""
    std_net = make_q_network(12, 64, 2, dueling=False)
    assert isinstance(std_net, QNetwork)

    duel_net = make_q_network(12, 64, 2, dueling=True)
    assert isinstance(duel_net, DuelingQNetwork)


def test_dueling_agent_training_and_save_load(tmp_path):
    """Verify DQNAgent operates correctly with dueling=True and save/load."""
    agent = DQNAgent(obs_dim=12, hidden_dim=64, n_actions=2, dueling=True, double=True)
    assert agent.dueling is True
    assert isinstance(agent.q_net, DuelingQNetwork)

    # Fake mini-batch
    batch_size = 8
    s = torch.randn(batch_size, 12)
    a = torch.randint(0, 2, (batch_size, 1))
    r = torch.randn(batch_size, 1)
    s2 = torch.randn(batch_size, 12)
    d = torch.zeros(batch_size, 1)

    loss, td = agent.learn(s, a, r, s2, d)
    assert loss >= 0.0
    assert td.shape == (batch_size, 1)

    # Test save and load
    save_file = str(tmp_path / "dueling_agent.pt")
    agent.save(save_file)

    loaded_agent = DQNAgent(obs_dim=12, hidden_dim=64, n_actions=2, dueling=False)
    loaded_agent.load(save_file)
    assert loaded_agent.dueling is True
    assert isinstance(loaded_agent.q_net, DuelingQNetwork)


def test_polyak_soft_target_sync():
    """Verify target network soft updates interpolate weights accurately."""
    agent = DQNAgent(obs_dim=12, hidden_dim=64, n_actions=2)

    # Initialize target net with known values
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.fill_(0.0)
        for p in agent.q_net.parameters():
            p.fill_(10.0)

    # Soft update with tau = 0.2
    # New target = 0.2 * 10.0 + 0.8 * 0.0 = 2.0
    agent.sync_target(tau=0.2)
    for p in agent.target_net.parameters():
        np.testing.assert_allclose(p.data.numpy(), 2.0, atol=1e-5)

    # Hard update with tau = 1.0
    # New target = 10.0
    agent.sync_target(tau=1.0)
    for p in agent.target_net.parameters():
        np.testing.assert_allclose(p.data.numpy(), 10.0, atol=1e-5)


def test_n_step_collector_returns_and_flushing():
    """Verify N-step return accumulation, discounting, and terminal queue flushing."""
    collector = NStepCollector(n_step=3, gamma=0.9)

    s0 = np.array([0.0])
    s1 = np.array([1.0])
    s2 = np.array([2.0])
    s3 = np.array([3.0])
    s4 = np.array([4.0])

    # Step 0: return should be empty (queue len 1 < 3)
    out0 = collector.add(s0, 0, 1.0, s1, False)
    assert len(out0) == 0

    # Step 1: return should be empty (queue len 2 < 3)
    out1 = collector.add(s1, 1, 2.0, s2, False)
    assert len(out1) == 0

    # Step 2: queue reaches 3, yields 3-step return for s0
    # R = 1.0 + 0.9 * 2.0 + 0.9^2 * 3.0 = 1.0 + 1.8 + 2.43 = 5.23
    out2 = collector.add(s2, 0, 3.0, s3, False)
    assert len(out2) == 1
    t0 = out2[0]
    np.testing.assert_allclose(t0[0], s0)
    assert t0[1] == 0
    np.testing.assert_allclose(t0[2], 5.23, atol=1e-5)
    np.testing.assert_allclose(t0[3], s3)
    assert t0[4] is False

    # Step 3: Terminal step (done=True). Should yield s1 return and flush s2 and s3!
    # queue had: [s1 (r=2), s2 (r=3)]
    # add(s3, 1, 4.0, s4, True) pushes s3.
    # queue reaches 3 -> pops s1: R = 2.0 + 0.9*3.0 + 0.9^2*4.0 = 2 + 2.7 + 3.24 = 7.94, next state s4, done=True
    # then d=True flushes:
    # pop s2: R = 3.0 + 0.9*4.0 = 6.6, next state s4, done=True
    # pop s3: R = 4.0, next state s4, done=True
    out3 = collector.add(s3, 1, 4.0, s4, True)
    assert len(out3) == 3

    # Transition 1
    np.testing.assert_allclose(out3[0][0], s1)
    np.testing.assert_allclose(out3[0][2], 7.94, atol=1e-5)
    assert out3[0][4] is True

    # Transition 2
    np.testing.assert_allclose(out3[1][0], s2)
    np.testing.assert_allclose(out3[1][2], 6.6, atol=1e-5)
    assert out3[1][4] is True

    # Transition 3
    np.testing.assert_allclose(out3[2][0], s3)
    np.testing.assert_allclose(out3[2][2], 4.0, atol=1e-5)
    assert out3[2][4] is True

    # Queue should now be empty
    assert len(collector.queue) == 0


def test_reward_shaping_wrapper_potential():
    """Verify potential-based reward shaping wrapper modifies rewards consistently."""
    env = make_env(seed=42, reward_shaping=True, shaping_scale=0.1, gamma=0.99)
    assert isinstance(env, FlappyRewardShapingWrapper)

    obs, info = env.reset(seed=42)
    assert obs.shape == (12,)
    assert isinstance(env.prev_phi, float)

    # Perform a step and check reward is a valid float
    next_obs, reward, term, trunc, info = env.step(0)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert next_obs.shape == (12,)
    env.close()
