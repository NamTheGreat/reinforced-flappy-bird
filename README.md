# Flappy Bird RL Agent: DQN with a Rust Prioritized Replay Buffer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-flappy--bird-green.svg)](https://github.com/Talendar/flappy-bird-gymnasium)
[![Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)]()

> **Course:** Reinforcement Learning (Final Year B.Tech CSE)  
> **Category:** Capstone / Mini Project  
> **Topic:** Deep Q-Network (DQN) with Prioritized Experience Replay (PER) accelerated via a native Rust Sum-Tree (`PyO3`/`maturin`).

---

## 1. Project Overview

This project implements a Deep Q-Network (DQN) agent that learns to play Flappy Bird purely from reward signals without labeled data or hand-crafted rules.

### Key Contributions
1. **From-Scratch PyTorch DQN & Double DQN**: Modular, transparent implementation of deep Q-learning with target networks and $\epsilon$-greedy exploration.
2. **Native Rust Sum-Tree (`per_rs`)**: A high-performance binary sum-tree implemented in Rust and bound to Python via **PyO3** and **Maturin**, accelerating $O(\log N)$ priority updates and stratified sampling.
3. **Rigorous Benchmarking & Ablations**:
   - **Microbenchmark:** Compares Rust `PerTree` vs pure Python sum-tree vs NumPy `random.choice`. The Rust implementation provides **~30x faster sampling** and **~117x faster priority updates**.
   - **Baselines:** Compares learned policies against a `RandomAgent` (0.0 pipes) and a hand-crafted `HeuristicAgent` (mean ~69 pipes).
   - **Replay Ablation:** Compares Uniform Replay vs Prioritized Replay vs Double DQN.

---

## 2. Core RL Formulations

### Bellman Optimality Equation
The Q-network approximates the optimal action-value function $Q^*(s, a)$:
$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \;\middle|\; s, a\right]$$

### Double DQN Target
To prevent overestimation bias caused by the $\max$ operator:
$$y_t = r_t + \gamma \, Q(s_{t+1}, \operatorname{argmax}_{a'} Q(s_{t+1}, a'; \theta); \theta^-)$$
where $\theta$ denotes the online network parameters and $\theta^-$ denotes the target network parameters.

### Prioritized Experience Replay (PER)
- **Transition Priority**: $p_i = (|\delta_i| + \epsilon)^\alpha$, where $\delta_i = y_i - Q(s_i, a_i)$ is the TD-error.
- **Sampling Probability**: $P(i) = \frac{p_i}{\sum_k p_k}$
- **Importance Sampling (IS) Weights**: To correct for non-uniform sampling bias:
  $$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta \Big/ \max_j w_j$$
  where $\beta$ is annealed linearly from $0.4 \to 1.0$ over training.

---

## 3. Environment Details (`FlappyBird-v0`)

The agent interacts with `flappy-bird-gymnasium` (`use_lidar=False`, feature mode):

- **State Space (12 Continuous Features, normalized to $[-1, 1]$):**
  - `obs[0..2]`: Pipe 0 horizontal distance, top pipe bottom edge $y$, bottom pipe top edge $y$
  - `obs[3..5]`: Pipe 1 horizontal distance, top pipe bottom edge $y$, bottom pipe top edge $y$
  - `obs[6..8]`: Pipe 2 horizontal distance, top pipe bottom edge $y$, bottom pipe top edge $y$
  - `obs[9]`: Bird vertical position $y$ ($0 = \text{top}$, $1 = \text{ground}$)
  - `obs[10]`: Bird vertical velocity $v_y$
  - `obs[11]`: Bird rotation angle $\theta$
- **Action Space (`Discrete(2)`):**
  - `0`: Do nothing (fall under gravity)
  - `1`: Flap (apply vertical impulse)
- **Reward Function:**
  - $+0.1$ for surviving each frame
  - $+1.0$ for successfully clearing a pipe
  - $-0.5$ for touching the ceiling
  - $-1.0$ for crashing (terminal state)

---

## 4. Repository Structure

```
reinforced-flappy-bird/
├── README.md                  # Project manual and viva guide
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Python packaging configuration
├── per_rs/                    # Rust sum-tree crate
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/lib.rs             # PerTree implementation with PyO3 bindings
├── src/flappy_rl/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters (DQNConfig dataclass)
│   ├── env.py                 # Environment factory make_env()
│   ├── networks.py            # QNetwork architecture (PyTorch)
│   ├── buffers.py             # UniformBuffer, PERBuffer, PurePythonSumTree
│   ├── agent.py               # DQNAgent (Vanilla & Double DQN)
│   ├── baselines.py           # RandomAgent, HeuristicAgent
│   ├── train.py               # Modular training loop with CSV logging
│   ├── evaluate.py            # Evaluation & GIF recording
│   └── utils.py               # Seeding, moving average
├── tests/
│   ├── test_uniform_buffer.py # Buffer shape, overwrite, and capacity tests
│   └── test_per_buffer.py     # Rust tree math, empirical sampling, overfit test
├── benchmarks/
│   └── bench_buffers.py       # Microbenchmark (Python vs NumPy vs Rust)
├── scripts/
│   ├── evaluate_baselines.py  # Evaluates baselines over 100 episodes
│   └── make_plots.py          # Generates training & evaluation curves
├── results/
│   ├── runs/                  # Episode and evaluation CSV logs
│   ├── plots/                 # High-resolution PNG figures
│   └── tables/                # Markdown and CSV result tables
├── models/                    # Checkpoints (.pt)
├── demo/                      # Recorded GIF gameplay demonstrations
└── report/
    ├── report.md              # 4-page academic report draft
    └── report.tex             # LaTeX source code
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Rust toolchain (`cargo`, `rustc` via [rustup.rs](https://rustup.rs))
- Maturin (`pip install maturin`)

### Quick Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd reinforced-flappy-bird

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
pip install -e .

# 4. Build and install the Rust PER extension
cd per_rs
maturin develop --release
cd ..

# 5. Run test suite
pytest -v
```

---

## 6. Running the Experiments

### 6.1 Baseline Evaluation
Evaluate `RandomAgent` and `HeuristicAgent` on 100 episodes:
```bash
python scripts/evaluate_baselines.py --episodes 100
```
Outputs saved to `results/tables/baselines.csv` and `results/tables/baselines.md`.

### 6.2 Sum-Tree Microbenchmark
Compare sampling and priority update throughput across Pure-Python, NumPy, and Rust:
```bash
python benchmarks/bench_buffers.py --samples 2000 --updates 2000
```

### 6.3 Local Smoke Run (Verification)
Verify the complete training loop locally without executing full multi-hour runs:
```bash
# Uniform DQN smoke test (2500 steps)
python src/flappy_rl/train.py --buffer uniform --timesteps 2500 --warmup 500 --eval-freq 1000

# Rust PER DQN smoke test (2500 steps)
python src/flappy_rl/train.py --buffer per --timesteps 2500 --warmup 500 --eval-freq 1000

# Double DQN with Rust PER smoke test (2500 steps)
python src/flappy_rl/train.py --buffer per --double --timesteps 2500 --warmup 500 --eval-freq 1000
```

### 6.4 Full Training (Remote / High-Compute Cluster)
For complete training runs (200k to 500k timesteps across 3 seeds per method):

```bash
# Methods to train:
# 1. Vanilla DQN + Uniform Replay (Seeds 0, 1, 2)
python src/flappy_rl/train.py --buffer uniform --seed 0 --timesteps 300000
python src/flappy_rl/train.py --buffer uniform --seed 1 --timesteps 300000
python src/flappy_rl/train.py --buffer uniform --seed 2 --timesteps 300000

# 2. Vanilla DQN + Rust PER (Seeds 0, 1, 2)
python src/flappy_rl/train.py --buffer per --seed 0 --timesteps 300000
python src/flappy_rl/train.py --buffer per --seed 1 --timesteps 300000
python src/flappy_rl/train.py --buffer per --seed 2 --timesteps 300000

# 3. Double DQN + Rust PER (Seeds 0, 1, 2)
python src/flappy_rl/train.py --buffer per --double --seed 0 --timesteps 300000
python src/flappy_rl/train.py --buffer per --double --seed 1 --timesteps 300000
python src/flappy_rl/train.py --buffer per --double --seed 2 --timesteps 300000
```

### 6.5 Plot Generation
Generate curves (mean ± std shaded error bands) and summary tables from all runs in `results/runs/`:
```bash
python scripts/make_plots.py
```
Outputs saved to `results/plots/` and `results/tables/`.

### 6.6 Evaluation & GIF Recording
Evaluate a checkpoint or baseline and record an animated gameplay GIF:
```bash
# Record Heuristic agent
python -m flappy_rl.evaluate --heuristic --record demo/heuristic.gif --record-steps 500

# Record Random agent
python -m flappy_rl.evaluate --random --record demo/random.gif --record-steps 200

# Record Trained DQN checkpoint
python -m flappy_rl.evaluate --checkpoint models/<run_id>_best.pt --record demo/trained_dqn.gif
```

---

## 7. Results Summary

### 7.1 Baseline Agents Performance (100 Episodes)

| Agent | Mean Score | Std Dev | Median | Max Score | Mean Steps | Success Rate ($\ge 20$) |
|---|---|---|---|---|---|---|
| **RandomAgent** | 0.00 | 0.00 | 0.0 | 0 | 50.0 | 0.0% |
| **HeuristicAgent** | 69.28 | 43.74 | 61.5 | 132 | 2646.2 | 84.0% |

### 7.2 Buffer Microbenchmark Results (Capacity = $65,536$, Batch Size = $64$)

| Implementation | Sampling Throughput | Priority Update Throughput | Speedup vs Python (Sample) | Speedup vs Python (Update) |
|---|---|---|---|---|
| Pure-Python SumTree | 6,585 ops/sec | 11,237 ops/sec | 1.0x (baseline) | 1.0x (baseline) |
| NumPy `random.choice` | 3,645 ops/sec | 61,482 ops/sec | 0.55x | 5.47x |
| **Rust `PerTree` (PyO3)** | **197,870 ops/sec** | **1,317,886 ops/sec** | **30.05x** | **117.28x** |

---

## 8. Viva Examination Preparation Guide

Be prepared to answer these questions directly and point to the corresponding code:

| Question | Core Concept | Code Pointer |
|---|---|---|
| **What does the Bellman equation do in your loop?** | Defines the target: $y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')$. | [`DQNAgent.learn()` in `src/flappy_rl/agent.py`](src/flappy_rl/agent.py) |
| **Why is a target network needed?** | Prevents moving-target instability where network updates change both predictions and target simultaneously ("freeze the goalpost"). | `agent.sync_target()` in [`src/flappy_rl/agent.py`](src/flappy_rl/agent.py) |
| **Why is Double DQN useful?** | Standard DQN uses $\max$ for both selection and evaluation, creating upward overestimation bias. Double DQN decouples them: online net picks the action, target net scores it. | `if self.double:` branch in [`src/flappy_rl/agent.py`](src/flappy_rl/agent.py) |
| **How does the Sum-Tree work?** | Binary heap where parents equal the sum of children. Leaves hold priorities. Sampling divides total priority into $k$ segments and traverses down in $O(\log N)$. Updates propagate deltas up to root in $O(\log N)$. | [`PerTree` in `per_rs/src/lib.rs`](per_rs/src/lib.rs) |
| **Why are Importance Sampling (IS) weights required in PER?** | Prioritizing transitions alters the empirical data distribution. Non-uniform sampling introduces gradient bias. $w_i = (N \cdot P(i))^{-\beta}$ corrects this bias, with $\beta \to 1.0$ asymptotically. | [`PERBuffer.sample()` in `src/flappy_rl/buffers.py`](src/flappy_rl/buffers.py) |
| **Why implement the buffer in Rust?** | Python loops have substantial interpreter overhead for per-step tree traversal. Rust enables zero-overhead pointer arithmetic, SIMD optimizations, and tight $O(\log N)$ updates without Python GIL contention. | `per_rs/src/lib.rs` |
| **What does `alpha` and `beta` represent?** | $\alpha$ dictates prioritization intensity ($\alpha=0$ is uniform). $\beta$ dictates bias correction intensity ($\beta=1$ is full correction). | [`src/flappy_rl/config.py`](src/flappy_rl/config.py) |

---

## 9. License & Academic Integrity

This project is developed as part of the academic curriculum for the Final Year B.Tech Computer Science & Engineering Reinforcement Learning course. All foundational papers are cited in accordance with IEEE guidelines. Plagiarism similarity index is strictly verified below 10%.
