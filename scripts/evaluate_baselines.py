"""Evaluate baseline agents (Random and Heuristic) on 100 episodes."""

import os
import argparse
import numpy as np
import pandas as pd
from flappy_rl.env import make_env
from flappy_rl.baselines import RandomAgent, HeuristicAgent
from flappy_rl.evaluate import run_episodes


def evaluate_baselines(n_episodes: int = 100, out_dir: str = "results/tables"):
    os.makedirs(out_dir, exist_ok=True)
    env = make_env(seed=42)

    results = []

    print(f"Evaluating RandomAgent on {n_episodes} episodes...")
    random_agent = RandomAgent(seed=42)
    rand_res = run_episodes(random_agent, env, n_episodes=n_episodes)
    results.append({
        "Agent": "RandomAgent",
        "Mean Score": rand_res["mean_score"],
        "Std Score": rand_res["std_score"],
        "Median Score": rand_res["median_score"],
        "Max Score": rand_res["max_score"],
        "Min Score": rand_res["min_score"],
        "Mean Reward": rand_res["mean_reward"],
        "Mean Steps": rand_res["mean_steps"],
        "Success Rate (Score >= 20)": f"{rand_res['success_rate']*100:.1f}%",
    })

    print(f"Evaluating HeuristicAgent on {n_episodes} episodes...")
    heuristic_agent = HeuristicAgent()
    heur_res = run_episodes(heuristic_agent, env, n_episodes=n_episodes)
    results.append({
        "Agent": "HeuristicAgent",
        "Mean Score": heur_res["mean_score"],
        "Std Score": heur_res["std_score"],
        "Median Score": heur_res["median_score"],
        "Max Score": heur_res["max_score"],
        "Min Score": heur_res["min_score"],
        "Mean Reward": heur_res["mean_reward"],
        "Mean Steps": heur_res["mean_steps"],
        "Success Rate (Score >= 20)": f"{heur_res['success_rate']*100:.1f}%",
    })

    env.close()

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "baselines.csv")
    df.to_csv(csv_path, index=False)

    def _to_md(d):
        headers = list(d.columns)
        out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in d.iterrows():
            out.append("| " + " | ".join(str(val) for val in row.values) + " |")
        return "\n".join(out)

    md_table = _to_md(df)
    md_path = os.path.join(out_dir, "baselines.md")
    with open(md_path, "w") as f:
        f.write("# Baseline Agents Performance (100 Episodes)\n\n")
        f.write(md_table + "\n")

    print("\nBaseline Evaluation Results:")
    print(md_table)
    print(f"\nSaved to {csv_path} and {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="results/tables")
    args = parser.parse_args()
    evaluate_baselines(args.episodes, args.out_dir)
