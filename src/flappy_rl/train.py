"""Training loop with CSV logging, periodic evaluation, and checkpointing."""

import os
import time
import argparse
import csv
from typing import Optional
import numpy as np
import torch

from flappy_rl.config import DQNConfig
from flappy_rl.env import make_env
from flappy_rl.agent import DQNAgent
from flappy_rl.buffers import UniformBuffer, PERBuffer, NStepCollector
from flappy_rl.evaluate import run_episodes
from flappy_rl.utils import set_seed


def train(cfg: DQNConfig, log_dir: str = "results/runs", models_dir: str = "models") -> str:
    """Execute training run according to configuration.

    Args:
        cfg: Configuration parameters.
        log_dir: Directory where CSV logs will be saved.
        models_dir: Directory where checkpoints will be saved.

    Returns:
        Run ID string.
    """
    set_seed(cfg.seed)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    variant_parts = ["DQN"]
    if cfg.dueling:
        variant_parts.append("Dueling")
    if cfg.double:
        variant_parts.append("Double")
    variant_parts.append(cfg.buffer_type.upper())
    if cfg.n_step > 1:
        variant_parts.append(f"{cfg.n_step}Step")
    if cfg.tau < 1.0:
        variant_parts.append(f"Tau{cfg.tau}")
    if cfg.reward_shaping:
        variant_parts.append("Shaped")

    method_name = "_".join(variant_parts)
    run_id = f"{method_name}_seed_{cfg.seed}_{int(time.time())}"

    # Log files
    ep_log_path = os.path.join(log_dir, f"{run_id}_episodes.csv")
    eval_log_path = os.path.join(log_dir, f"{run_id}_eval.csv")

    with open(ep_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "method", "seed", "episode", "step",
            "ep_reward", "score", "epsilon", "loss", "wall_time"
        ])

    with open(eval_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "method", "seed", "step", "mean_score",
            "std_score", "median_score", "max_score", "success_rate", "wall_time"
        ])

    # Environments (training env may have reward shaping; eval env is always raw environment)
    env = make_env(
        seed=cfg.seed,
        max_episode_steps=cfg.max_episode_steps,
        reward_shaping=cfg.reward_shaping,
        shaping_scale=cfg.shaping_scale,
        gamma=cfg.gamma,
    )
    eval_env = make_env(
        seed=cfg.seed + 1000,
        max_episode_steps=cfg.max_episode_steps,
        reward_shaping=False,
    )

    # Agent
    agent = DQNAgent(
        obs_dim=cfg.obs_dim,
        hidden_dim=cfg.hidden_dim,
        n_actions=cfg.n_actions,
        lr=cfg.lr,
        gamma=cfg.gamma,
        double=cfg.double,
        dueling=cfg.dueling,
        grad_clip=cfg.grad_clip,
    )

    # Multi-step transition collector
    n_step_collector = NStepCollector(n_step=cfg.n_step, gamma=cfg.gamma)
    effective_gamma = cfg.gamma ** cfg.n_step

    # Buffer
    if cfg.buffer_type.lower() == "per":
        buffer = PERBuffer(cap=cfg.buffer_capacity, obs_dim=cfg.obs_dim, alpha=cfg.per_alpha)
    else:
        buffer = UniformBuffer(cap=cfg.buffer_capacity, obs_dim=cfg.obs_dim)

    obs, info = env.reset(seed=cfg.seed)
    ep_reward = 0.0
    ep_count = 0
    recent_losses = []
    best_eval_score = -float("inf")
    start_time = time.time()

    for global_step in range(1, cfg.total_timesteps + 1):
        # Epsilon schedule
        if global_step <= cfg.eps_decay_steps:
            frac = global_step / cfg.eps_decay_steps
            eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        else:
            eps = cfg.eps_end

        # Act
        action = agent.act(obs, eps=eps)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        ep_reward += float(reward)

        # Store transitions via n-step collector
        transitions = n_step_collector.add(obs, action, reward, next_obs, done)
        for s_t, a_t, r_t, s2_t, d_t in transitions:
            buffer.push(s_t, a_t, r_t, s2_t, d_t)
        obs = next_obs

        # Learn
        if global_step >= cfg.warmup_steps and len(buffer) >= cfg.batch_size:
            if cfg.buffer_type.lower() == "per":
                # Beta annealing schedule
                if global_step <= cfg.per_beta_steps:
                    beta_frac = global_step / cfg.per_beta_steps
                    beta = cfg.per_beta_start + beta_frac * (cfg.per_beta_end - cfg.per_beta_start)
                else:
                    beta = cfg.per_beta_end

                batch, idxs, ws = buffer.sample(cfg.batch_size, beta=beta)
                s, a, r, s2, d = batch
                s_t = torch.as_tensor(s, dtype=torch.float32)
                a_t = torch.as_tensor(a, dtype=torch.int64)
                r_t = torch.as_tensor(r, dtype=torch.float32)
                s2_t = torch.as_tensor(s2, dtype=torch.float32)
                d_t = torch.as_tensor(d, dtype=torch.float32)
                ws_t = torch.as_tensor(ws, dtype=torch.float32).unsqueeze(1)

                loss_val, td_errors = agent.learn(
                    s_t, a_t, r_t, s2_t, d_t, weights=ws_t, gamma=effective_gamma
                )
                buffer.update(idxs, td_errors.abs().cpu().numpy().ravel())
            else:
                s, a, r, s2, d = buffer.sample(cfg.batch_size)
                s_t = torch.as_tensor(s, dtype=torch.float32)
                a_t = torch.as_tensor(a, dtype=torch.int64)
                r_t = torch.as_tensor(r, dtype=torch.float32)
                s2_t = torch.as_tensor(s2, dtype=torch.float32)
                d_t = torch.as_tensor(d, dtype=torch.float32)

                loss_val, _ = agent.learn(
                    s_t, a_t, r_t, s2_t, d_t, weights=None, gamma=effective_gamma
                )

            recent_losses.append(loss_val)

            # Soft target updates (Polyak) occur every training step
            if cfg.tau < 1.0:
                agent.sync_target(tau=cfg.tau)

        # Hard target network sync (when tau == 1.0)
        if cfg.tau >= 1.0 and global_step % cfg.target_update_interval == 0:
            agent.sync_target(tau=1.0)

        # Episode end
        if done:
            ep_count += 1
            score = info.get("score", 0)
            avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
            recent_losses = []
            wall_time = time.time() - start_time

            with open(ep_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_id, method_name, cfg.seed, ep_count, global_step,
                    f"{ep_reward:.2f}", score, f"{eps:.4f}", f"{avg_loss:.5f}", f"{wall_time:.1f}"
                ])

            obs, info = env.reset()
            ep_reward = 0.0

        # Periodic evaluation
        if global_step % cfg.eval_freq == 0 or global_step == cfg.total_timesteps:
            eval_metrics = run_episodes(agent, eval_env, n_episodes=cfg.eval_episodes)
            wall_time = time.time() - start_time

            with open(eval_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_id, method_name, cfg.seed, global_step,
                    f"{eval_metrics['mean_score']:.2f}",
                    f"{eval_metrics['std_score']:.2f}",
                    f"{eval_metrics['median_score']:.2f}",
                    eval_metrics['max_score'],
                    f"{eval_metrics['success_rate']:.3f}",
                    f"{wall_time:.1f}"
                ])

            print(
                f"[{method_name} | Seed {cfg.seed}] Step {global_step}/{cfg.total_timesteps} "
                f"| Eval Mean Score: {eval_metrics['mean_score']:.2f} "
                f"| Max: {eval_metrics['max_score']} "
                f"| Success Rate: {eval_metrics['success_rate']*100:.1f}% "
                f"| Time: {wall_time:.1f}s"
            )

            # Checkpoint best
            if eval_metrics["mean_score"] > best_eval_score:
                best_eval_score = eval_metrics["mean_score"]
                agent.save(os.path.join(models_dir, f"{run_id}_best.pt"))

            agent.save(os.path.join(models_dir, f"{run_id}_latest.pt"))

    env.close()
    eval_env.close()
    print(f"Training completed: {run_id}. Best eval score: {best_eval_score:.2f}")
    return run_id


def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(description="Train Flappy Bird DQN Agent")
    parser.add_argument("--buffer", type=str, choices=["uniform", "per"], default="uniform", help="Buffer type")
    parser.add_argument("--double", action="store_true", help="Enable Double DQN")
    parser.add_argument("--dueling", action="store_true", help="Enable Dueling Q-Network architecture")
    parser.add_argument("--tau", type=float, default=1.0, help="Polyak soft update factor (<1.0 for soft updates)")
    parser.add_argument("--n-step", type=int, default=1, help="N-step return horizon (default: 1)")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable potential-based gap alignment shaping")
    parser.add_argument("--shaping-scale", type=float, default=0.05, help="Reward shaping scale factor")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for MLP")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--warmup", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--target-interval", type=int, default=1000, help="Target sync interval (used when tau=1.0)")
    args = parser.parse_args()

    return DQNConfig(
        buffer_type=args.buffer,
        double=args.double,
        dueling=args.dueling,
        tau=args.tau,
        n_step=args.n_step,
        reward_shaping=args.reward_shaping,
        shaping_scale=args.shaping_scale,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        total_timesteps=args.timesteps,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        warmup_steps=args.warmup,
        target_update_interval=args.target_interval,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
