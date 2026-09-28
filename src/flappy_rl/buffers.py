"""Replay buffers for DQN: UniformBuffer and Prioritized Experience Replay (PER)."""

from typing import Tuple, Optional
import numpy as np

try:
    import per_rs
    HAS_RUST_PER = True
except ImportError:
    HAS_RUST_PER = False


class PurePythonSumTree:
    """Pure-Python reference SumTree for verification and benchmarking."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        assert (capacity & (capacity - 1) == 0) and capacity > 0, "Capacity must be power of 2"
        self.cap = capacity
        self.alpha = alpha
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.write = 0
        self.size = 0
        self.max_p = 1.0

    def total(self) -> float:
        return self.tree[1]

    def _set(self, leaf: int, p: float):
        i = leaf + self.cap
        delta = p - self.tree[i]
        while i >= 1:
            self.tree[i] += delta
            i //= 2

    def add(self) -> int:
        idx = self.write
        self._set(idx, self.max_p)
        self.write = (self.write + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        return idx

    def update(self, idxs: np.ndarray, td: np.ndarray):
        for i, t in zip(idxs, td):
            p = (abs(t) + 1e-6) ** self.alpha
            self.max_p = max(self.max_p, p)
            self._set(int(i), p)

    def sample(self, batch_size: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        assert self.size > 0, "Cannot sample from empty tree"
        total = self.tree[1]
        assert total > 0.0, "Total priority must be positive"

        seg = total / batch_size
        n = self.size
        idxs = np.zeros(batch_size, dtype=np.int64)
        ws = np.zeros(batch_size, dtype=np.float64)

        for k in range(batch_size):
            low = seg * k
            high = min(total, seg * (k + 1))
            s = np.random.uniform(low, high) if high > low else low
            i = 1
            while i < self.cap:
                l = 2 * i
                if s <= self.tree[l] or self.tree[l + 1] == 0.0:
                    i = l
                else:
                    s -= self.tree[l]
                    i = l + 1
            idx = i - self.cap
            idxs[k] = idx
            p_i = max(self.tree[i], 1e-12)
            p_sample = p_i / total
            ws[k] = (n * p_sample) ** (-beta)

        max_w = ws.max()
        norm_factor = max_w if (max_w > 0 and np.isfinite(max_w)) else 1.0
        ws /= norm_factor
        return idxs, ws


class UniformBuffer:
    """Experience replay buffer with uniform random sampling."""

    def __init__(self, cap: int = 65536, obs_dim: int = 12):
        self.cap = cap
        self.obs_dim = obs_dim
        self.s = np.zeros((cap, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((cap, obs_dim), dtype=np.float32)
        self.a = np.zeros((cap, 1), dtype=np.int64)
        self.r = np.zeros((cap, 1), dtype=np.float32)
        self.d = np.zeros((cap, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        d: bool,
    ) -> None:
        idx = self.ptr
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s2[idx] = s2
        self.d[idx] = float(d)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.s[idxs],
            self.a[idxs],
            self.r[idxs],
            self.s2[idxs],
            self.d[idxs],
        )


class PERBuffer:
    """Prioritized Experience Replay buffer powered by Rust sum-tree."""

    def __init__(
        self,
        cap: int = 65536,
        obs_dim: int = 12,
        alpha: float = 0.6,
        use_rust: bool = True,
    ):
        self.cap = cap
        self.obs_dim = obs_dim
        self.alpha = alpha

        if use_rust and HAS_RUST_PER:
            self.tree = per_rs.PerTree(cap, alpha)
            self.is_rust = True
        else:
            self.tree = PurePythonSumTree(cap, alpha)
            self.is_rust = False

        self.s = np.zeros((cap, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((cap, obs_dim), dtype=np.float32)
        self.a = np.zeros((cap, 1), dtype=np.int64)
        self.r = np.zeros((cap, 1), dtype=np.float32)
        self.d = np.zeros((cap, 1), dtype=np.float32)
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        d: bool,
    ) -> None:
        idx = self.tree.add()
        self.s[idx] = s
        self.a[idx] = a
        self.r[idx] = r
        self.s2[idx] = s2
        self.d[idx] = float(d)
        self.size = min(self.size + 1, self.cap)

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:
        idxs, ws = self.tree.sample(batch_size, beta)
        batch = (
            self.s[idxs],
            self.a[idxs],
            self.r[idxs],
            self.s2[idxs],
            self.d[idxs],
        )
        return batch, idxs, ws

    def update(self, idxs: np.ndarray, td_errors: np.ndarray) -> None:
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        td_arr = np.asarray(td_errors, dtype=np.float64)
        self.tree.update(idxs_arr, td_arr)


from collections import deque
from typing import NamedTuple, List


class StepTransition(NamedTuple):
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    d: bool


class NStepCollector:
    r"""Accumulates multi-step transitions for n-step return bootstrapping.

    Given horizon n >= 1 and discount factor gamma:
        R_t^{(n)} = \sum_{k=0}^{m-1} \gamma^k r_{t+k}
    where m = min(n, steps_to_terminal).
    The resulting transition is (s_t, a_t, R_t^{(n)}, s_{t+m}, done_{t+m}).
    """

    def __init__(self, n_step: int = 1, gamma: float = 0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.queue: deque = deque()

    def add(
        self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Add a 1-step transition and yield ready n-step transitions."""
        if self.n_step <= 1:
            return [(s, a, float(r), s2, bool(d))]

        self.queue.append(StepTransition(s, a, float(r), s2, bool(d)))
        ready = []

        if len(self.queue) >= self.n_step:
            ready.append(self._pop_transition())

        if d:
            while self.queue:
                ready.append(self._pop_transition())

        return ready

    def _pop_transition(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        s0 = self.queue[0].s
        a0 = self.queue[0].a
        ret = 0.0
        done = False
        s_next = self.queue[-1].s2

        for i, trans in enumerate(self.queue):
            ret += (self.gamma ** i) * trans.r
            if trans.d:
                done = True
                s_next = trans.s2
                break

        self.queue.popleft()
        return (s0, a0, float(ret), s_next, done)

    def reset(self) -> None:
        """Clear collector state on episode interruption."""
        self.queue.clear()

