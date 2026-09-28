# Architecture & Engineering Decision Log

This document records the architectural, algorithmic, and engineering improvements introduced to the `reinforced-flappy-bird` project. Each Architectural Decision Record (ADR) provides the problem context, theoretical formulation, concrete implementation details, design tradeoffs, and empirical verification.

---

## Decision Index

| ADR ID | Date | Area | Title | Status |
|:---:|:---:|:---:|:---|:---:|
| **[ADR-001](#adr-001-dueling-q-network-architecture)** | 2026-09-28 | Architecture | Dueling Q-Network Representation | **Accepted & Implemented** |
| **[ADR-002](#adr-002-polyak-soft-target-network-synchronization)** | 2026-09-28 | Stability | Polyak Soft Target Network Updates ($\tau$) | **Accepted & Implemented** |
| **[ADR-003](#adr-003-n-step-return-bootstrapping)** | 2026-09-28 | Credit Assignment | $N$-Step Return Bootstrapping | **Accepted & Implemented** |
| **[ADR-004](#adr-004-potential-based-gap-alignment-reward-shaping)** | 2026-09-28 | Reward Design | Potential-Based Gap-Alignment Reward Shaping (PBRS) | **Accepted & Implemented** |
| **[ADR-005](#adr-005-hyperparameter-scaling--checkpoint-reconstruction)** | 2026-09-28 | Optimization | Hyperparameter Scaling & Dynamic Checkpoint Reconstruction | **Accepted & Implemented** |

---

## ADR-001: Dueling Q-Network Architecture

### 1. Context & Motivation
In standard Deep Q-Networks (DQN), a single feedforward neural network estimates $Q(s, a)$ for every action $a \in \mathcal{A}$ directly from state features $s \in \mathbb{R}^{12}$. In Flappy Bird, the action space is binary:
- $a = 0$: Do nothing (gravity fall)
- $a = 1$: Flap (apply vertical impulse)

For the vast majority of states (e.g., when the bird is gliding safely through the vertical center of the approaching pipe gap), the choice of action has minimal immediate consequence on long-term survival. What dominates the expected return is the **state value** $V(s)$—how well-positioned the bird is relative to obstacles. Only in critical frames (approaching the bottom pipe threshold) does the action advantage $A(s, a)$ become decisive. Standard DQN is forced to relearn the value of the state separately for each action, leading to redundant gradient updates and slower policy convergence.

### 2. Theoretical Formulation
Following Wang et al. (2016), we decouple the Q-function into two distinct streams:
1. **Value Stream** $V(s; \theta, \beta) \in \mathbb{R}$: A scalar estimating the expected cumulative return from state $s$.
2. **Advantage Stream** $A(s, a; \theta, \alpha) \in \mathbb{R}^{|\mathcal{A}|}$: A vector measuring the relative importance of taking each action compared to the average action in that state.

Directly adding $Q(s, a) = V(s) + A(s, a)$ is mathematically unidentifiable (infinitely many combinations of $V$ and $A$ yield the exact same $Q$). To ensure uniqueness and stable optimization, we enforce an identifiability constraint by subtracting the mean advantage across all actions:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

#### Why Mean-Centering Instead of Max-Centering?
While subtracting $\max_{a'} A(s, a')$ forces the optimal action to have $Q(s, a^*) = V(s)$, it makes the gradient non-smooth whenever the argmax action switches. Mean-centering stabilizes optimization because the advantages only need to track changes relative to their mean, avoiding sharp gradient oscillations.

### 3. Implementation Details
- **Module:** [`src/flappy_rl/networks.py`](src/flappy_rl/networks.py) (`DuelingQNetwork`)
- **Shared Trunk:** $\text{Linear}(12, 128) \to \text{ReLU}$
- **Value Stream:** $\text{Linear}(128, 128) \to \text{ReLU} \to \text{Linear}(128, 1)$
- **Advantage Stream:** $\text{Linear}(128, 128) \to \text{ReLU} \to \text{Linear}(128, 2)$
- **Factory:** `make_q_network(obs_dim, hidden_dim, n_actions, dueling=True)`
- **CLI Flag:** `--dueling`

### 4. Verification
Automated unit test in [`tests/test_improvements.py::test_dueling_network_shapes_and_centering`](tests/test_improvements.py):
- Confirmed output shape $(B, 2)$ from $(B, 12)$ input.
- Verified that the centered advantage tensor $\tilde{A}(s, a)$ has an empirical mean identically equal to $0.0$ along the action dimension ($\text{atol} = 10^{-6}$).

---

## ADR-002: Polyak Soft Target Network Synchronization

### 1. Context & Motivation
Standard DQN addresses the "moving target problem" by maintaining a separate target network $\theta^-$ that is frozen for $C = 1,000$ steps and then copied abruptly:
$$\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}$$

While this stabilizes Bellman targets, it introduces a **periodic shock** to the TD-error surface:
1. Immediately before step $C$, the target network is 1,000 gradient updates stale.
2. At step $C$, the target network jumps abruptly, causing sudden spikes in TD-errors and temporary loss divergence.
3. For Prioritized Experience Replay (PER), this causes outdated tree leaf priorities, distorting sampling distributions until priorities are re-evaluated.

### 2. Theoretical Formulation
Following Lillicrap et al. (2015) and standard continuous control practices, we implement Polyak averaging (exponential moving average / soft updates) on the target network parameters:

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \quad \forall t \text{ where optimization occurs}$$

where $\tau \in (0, 1]$ is the interpolation parameter:
- When $\tau = 1.0$, it executes a standard hard copy.
- When $\tau \ll 1.0$ (e.g., $\tau = 0.005$), the target network continuously and smoothly tracks the online network parameters with an effective exponential smoothing half-life of $T_{1/2} \approx \frac{\ln 2}{\tau} \approx 138$ update steps.

### 3. Implementation Details
- **Module:** [`src/flappy_rl/agent.py`](src/flappy_rl/agent.py) (`DQNAgent.sync_target(self, tau: float = 1.0)`)
- **Execution:** When $\tau < 1.0$, `agent.sync_target(tau=cfg.tau)` is executed at every training step after `opt.step()`.
- **CLI Flag:** `--tau 0.005` (defaults to `1.0` for backward compatibility).

### 4. Verification
Automated unit test in [`tests/test_improvements.py::test_polyak_soft_target_sync`](tests/test_improvements.py):
- Verified that setting online weights to $10.0$ and target weights to $0.0$ with $\tau = 0.2$ yields exact target parameter values of $2.0 \pm 10^{-5}$.
- Verified that $\tau = 1.0$ executes an exact hard copy.

---

## ADR-003: $N$-Step Return Bootstrapping

### 1. Context & Motivation
In 1-step Q-learning, the Bellman target relies on immediate reward plus 1-step bootstrap:
$$y_t^{(1)} = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

In Flappy Bird, passing through a pipe yields $+1.0$, but takes approximately 30–50 frames of continuous flight. The per-frame survival reward is $+0.1$. Under 1-step bootstrapping, the high-value signal of successfully clearing a pipe propagates backward by only **one transition per training step**. If the network's value estimates early in the pipe approach are poor, credit assignment is severely delayed.

### 2. Theoretical Formulation
$N$-step bootstrapping (Sutton & Barto, 2018) generalizes 1-step TD and Monte Carlo methods by accumulating rewards across an $n$-step horizon before bootstrapping from the target network:

$$R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$$
$$y_t^{(n)} = R_t^{(n)} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-) (1 - d_{t+n})$$

#### Bias-Variance Tradeoff
- $n = 1$: Minimum variance, maximum bias (heavily dependent on neural network approximation errors in $Q$).
- $n = \infty$ (Monte Carlo): Zero bias, maximum variance (sensitive to stochastic environment transitions).
- $n = 3$: Optimal empirical compromise for Flappy Bird, propagating rewards 3 steps into the past without suffering from excessive trajectory variance.

### 3. Implementation Details
- **Module:** [`src/flappy_rl/buffers.py`](src/flappy_rl/buffers.py) (`NStepCollector`)
- **Decoupled Architecture:** `NStepCollector` maintains a sliding FIFO deque of size $n$.
  - When length reaches $n$, it yields the discounted $n$-step transition $(s_t, a_t, R_t^{(n)}, s_{t+n}, d_{t+n})$ to be pushed into the replay buffer.
  - Upon episode termination (`d == True`), all remaining transitions in the deque are flushed with appropriate discounted sums and the terminal flag set.
- **Replay Buffer Compatibility:** Both `UniformBuffer` and the native Rust `PERBuffer` remain completely unchanged! The transitions stored in the replay buffer already contain the accumulated $n$-step discounted reward.
- **Discount Factor Alignment:** When $n > 1$, the Bellman discount factor is scaled to $\gamma^n$:
  $$\gamma_{\text{effective}} = \gamma^n$$
- **CLI Flag:** `--n-step 3` (defaults to `1` for standard 1-step DQN).

### 4. Verification
Automated unit test in [`tests/test_improvements.py::test_n_step_collector_returns_and_flushing`](tests/test_improvements.py):
- Verified exact discounted returns for sequence $[1.0, 2.0, 3.0]$ with $\gamma = 0.9$: $R = 1.0 + 0.9(2.0) + 0.81(3.0) = 5.23$.
- Verified terminal queue flushing: upon encountering a terminal state, verified that all remaining sub-windows are flushed to completion with correct partial discounting and $d = \text{True}$.

---

## ADR-004: Potential-Based Gap-Alignment Reward Shaping

### 1. Context & Motivation
Flappy Bird's default reward structure provides $+0.1$ for surviving each frame, $+1.0$ for clearing a pipe, and $-1.0$ for collision. In early training iterations, random actions frequently cause the bird to fly directly into the ceiling or crash into the ground without ever discovering that staying vertically centered between the pipes is the viable strategy.

However, naive reward shaping (e.g., adding arbitrary bonuses for height) can cause **policy distortion** or reward hacking (e.g., hovering in a loop or oscillating vertically to collect shaping rewards without clearing pipes).

### 2. Theoretical Formulation: Potential-Based Reward Shaping (PBRS)
To eliminate reward hacking, we implement Potential-Based Reward Shaping (Ng, Harada, & Russell, 1999).

**Theorem (Ng et al., 1999):** Let $\Phi: \mathcal{S} \to \mathbb{R}$ be a real-valued potential function defined over the state space. If the reward function is augmented by an additional shaping term $F(s, a, s')$ defined strictly as the discounted difference of potentials:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
then the optimal policy $\pi^*$ in the shaped MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R} + F, \gamma)$ is **guaranteed to be identical** to the optimal policy in the original unshaped MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

#### Potential Function Formulation for Flappy Bird
We define the state potential as the normalized negative distance between the bird's vertical position and the center of the approaching pipe gap:
$$y_{\text{gap}} = \frac{obs[1] + obs[2]}{2}$$
$$h_{\text{gap}} = \max(0.1, obs[2] - obs[1])$$
$$\Phi(s) = - \frac{|obs[9] - y_{\text{gap}}|}{h_{\text{gap}}}$$

When the bird moves closer to the gap center, $\Phi(s') > \Phi(s)$, yielding $F(s, s') > 0$. When the bird drifts away, $F(s, s') < 0$. At terminal states, $\Phi(s_{\text{terminal}}) = 0$.

#### Strict Evaluation Protocol
To maintain academic integrity:
- **Training Environment:** Wrapped with `FlappyRewardShapingWrapper` (`reward_shaping=True`).
- **Evaluation Environment:** Always runs on the **raw, unshaped environment** (`reward_shaping=False`). This ensures evaluation metrics reflect genuine game score without shaping artifacts.

### 3. Implementation Details
- **Module:** [`src/flappy_rl/env.py`](src/flappy_rl/env.py) (`FlappyRewardShapingWrapper`)
- **Factory Integration:** `make_env(..., reward_shaping=True, shaping_scale=0.05, gamma=0.99)`
- **CLI Flag:** `--reward-shaping --shaping-scale 0.05`

### 4. Verification
Automated unit test in [`tests/test_improvements.py::test_reward_shaping_wrapper_potential`](tests/test_improvements.py):
- Verified wrapper preservation of observation space and types.
- Verified dynamic potential tracking and valid numeric float rewards on environment steps.

---

## ADR-005: Hyperparameter Scaling & Checkpoint Reconstruction

### 1. Context & Motivation
With the introduction of Dueling DQN, soft target updates, and $N$-step returns, saved model checkpoints (`.pt`) must maintain self-contained metadata so that evaluation scripts (`evaluate.py`) can load checkpoints seamlessly without requiring manual architecture flag specification.

### 2. Implementation Details
- **Module:** [`src/flappy_rl/agent.py`](src/flappy_rl/agent.py) (`save` and `load`)
- **Self-Describing Checkpoints:** `agent.save()` serializes `q_net`, `target_net`, `opt`, `double`, `dueling`, `obs_dim`, `hidden_dim`, and `gamma`.
- **Dynamic Reconstitution:** In `agent.load()`, if the saved checkpoint specifies `dueling=True` while the current agent instance has `dueling=False`, the agent dynamically reconstructs its internal `DuelingQNetwork`, updates `target_net`, and recreates the Adam optimizer with matching parameter groups.
- **Unified CLI:** [`src/flappy_rl/train.py`](src/flappy_rl/train.py) and [`src/flappy_rl/evaluate.py`](src/flappy_rl/evaluate.py) expose all configurable options:
  ```bash
  # Example: Training a Dueling Double DQN with Rust PER, Polyak updates, and 3-step returns
  python src/flappy_rl/train.py \
    --buffer per \
    --double \
    --dueling \
    --tau 0.005 \
    --n-step 3 \
    --reward-shaping \
    --lr 0.0003 \
    --timesteps 200000
  ```

---

## Automated Verification Suite

All improvements are covered by regression and feature tests. Executed via `pytest -v`:

```
tests/test_improvements.py::test_dueling_network_shapes_and_centering PASSED
tests/test_improvements.py::test_make_q_network_factory PASSED
tests/test_improvements.py::test_dueling_agent_training_and_save_load PASSED
tests/test_improvements.py::test_polyak_soft_target_sync PASSED
tests/test_improvements.py::test_n_step_collector_returns_and_flushing PASSED
tests/test_improvements.py::test_reward_shaping_wrapper_potential PASSED
tests/test_per_buffer.py::test_per_tree_add_and_total PASSED
tests/test_per_buffer.py::test_per_tree_update_priorities PASSED
tests/test_per_buffer.py::test_sampling_frequency PASSED
tests/test_per_buffer.py::test_is_weights_properties PASSED
tests/test_per_buffer.py::test_cross_check_rust_vs_python_tree PASSED
tests/test_per_buffer.py::test_dqn_sync_target PASSED
tests/test_per_buffer.py::test_dqn_overfit_sanity_check PASSED
tests/test_uniform_buffer.py::test_uniform_buffer_push_and_length PASSED
tests/test_uniform_buffer.py::test_uniform_buffer_sample_shapes PASSED
tests/test_uniform_buffer.py::test_uniform_buffer_circular_overwrite PASSED

============================== 16 passed in 0.79s ==============================
```
