"""Unit tests for UniformBuffer."""

import numpy as np
import pytest
from flappy_rl.buffers import UniformBuffer


def test_uniform_buffer_push_and_length():
    buf = UniformBuffer(cap=10, obs_dim=12)
    assert len(buf) == 0

    s = np.zeros(12, dtype=np.float32)
    s2 = np.ones(12, dtype=np.float32)

    for i in range(5):
        buf.push(s, 0, 1.0, s2, False)
    assert len(buf) == 5

    # Push beyond capacity
    for i in range(10):
        buf.push(s, 1, 0.5, s2, True)
    assert len(buf) == 10  # Capped at capacity


def test_uniform_buffer_sample_shapes():
    buf = UniformBuffer(cap=100, obs_dim=12)
    for i in range(50):
        s = np.full(12, i, dtype=np.float32)
        s2 = np.full(12, i + 1, dtype=np.float32)
        buf.push(s, i % 2, float(i), s2, i == 49)

    batch_size = 16
    s, a, r, s2, d = buf.sample(batch_size)

    assert s.shape == (batch_size, 12)
    assert s2.shape == (batch_size, 12)
    assert a.shape == (batch_size, 1)
    assert r.shape == (batch_size, 1)
    assert d.shape == (batch_size, 1)


def test_uniform_buffer_circular_overwrite():
    buf = UniformBuffer(cap=4, obs_dim=2)
    for i in range(6):
        s = np.full(2, i, dtype=np.float32)
        s2 = np.full(2, i + 1, dtype=np.float32)
        buf.push(s, i % 2, float(i), s2, False)

    assert len(buf) == 4
    # The pointer should be at 6 % 4 = 2
    assert buf.ptr == 2
    # Elements 0 and 1 were overwritten with i=4 and i=5
    assert np.allclose(buf.s[0], [4.0, 4.0])
    assert np.allclose(buf.s[1], [5.0, 5.0])
    assert np.allclose(buf.s[2], [2.0, 2.0])
    assert np.allclose(buf.s[3], [3.0, 3.0])
