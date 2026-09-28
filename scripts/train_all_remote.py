"""Batch script to run complete multi-seed experiments on a remote machine / cluster."""

import os
import time
import argparse
from flappy_rl.config import DQNConfig
from flappy_rl.train import train


def run_full_suite(
    seeds=[0, 1, 2],
    timesteps: int = 300000,
    eval_freq: int = 5000,
    eval_episodes: int = 20,
    include_double: bool = True,
):
    print("=================================================================")
    print("STARTING FULL RL EXPERIMENT SUITE")
    print(f"Seeds: {seeds} | Timesteps per run: {timesteps} | Eval freq: {eval_freq}")
    print("=================================================================")

    start_all = time.time()
    configs = []

    # 1. DQN with Uniform Replay
    for s in seeds:
        configs.append(DQNConfig(
            buffer_type="uniform",
            double=False,
            seed=s,
            total_timesteps=timesteps,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
        ))

    # 2. DQN with Rust Prioritized Experience Replay
    for s in seeds:
        configs.append(DQNConfig(
            buffer_type="per",
            double=False,
            seed=s,
            total_timesteps=timesteps,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
        ))

    # 3. Double DQN with Rust Prioritized Experience Replay
    if include_double:
        for s in seeds:
            configs.append(DQNConfig(
                buffer_type="per",
                double=True,
                seed=s,
                total_timesteps=timesteps,
                eval_freq=eval_freq,
                eval_episodes=eval_episodes,
            ))

    total_runs = len(configs)
    for idx, cfg in enumerate(configs, 1):
        method_str = f"{'Double_' if cfg.double else ''}{cfg.buffer_type.upper()}"
        print(f"\n>>> Running [{idx}/{total_runs}]: DQN_{method_str} (Seed {cfg.seed}) for {cfg.total_timesteps} steps...")
        t0 = time.time()
        train(cfg)
        elapsed = time.time() - t0
        print(f">>> Completed [{idx}/{total_runs}] in {elapsed/60:.2f} minutes.")

    total_elapsed = time.time() - start_all
    print(f"\n=================================================================")
    print(f"ALL EXPERIMENTS COMPLETED IN {total_elapsed/3600:.2f} HOURS.")
    print("Now run `python scripts/make_plots.py` to generate figures & tables.")
    print("=================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete RL experiment matrix")
    parser.add_argument("--timesteps", type=int, default=300000, help="Timesteps per run")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Eval interval")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="List of seeds")
    parser.add_argument("--no-double", action="store_true", help="Skip Double DQN")
    args = parser.parse_args()

    run_full_suite(
        seeds=args.seeds,
        timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        include_double=not args.no_double,
    )
