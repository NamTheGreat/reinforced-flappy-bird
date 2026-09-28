"""Generate publication-ready plots and comparison tables from experiment runs."""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_experiments(runs_dir: str = "results/runs", plots_dir: str = "results/plots", tables_dir: str = "results/tables"):
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    eval_files = glob.glob(os.path.join(runs_dir, "*_eval.csv"))
    episode_files = glob.glob(os.path.join(runs_dir, "*_episodes.csv"))

    if not eval_files and not episode_files:
        print(f"No experiment CSV files found in {runs_dir}. Nothing to plot.")
        return

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    # 1. Process Eval Logs
    if eval_files:
        eval_dfs = []
        for f in eval_files:
            try:
                df = pd.read_csv(f)
                eval_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")

        if eval_dfs:
            all_eval = pd.concat(eval_dfs, ignore_index=True)
            methods = all_eval["method"].unique()

            # Plot Evaluation Mean Score vs Steps
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {"DQN_UNIFORM": "#1f77b4", "DQN_PER": "#ff7f0e", "DQN_Double_UNIFORM": "#2ca02c", "DQN_Double_PER": "#d62728"}

            summary_rows = []

            for method in methods:
                m_df = all_eval[all_eval["method"] == method]
                steps = sorted(m_df["step"].unique())
                means = []
                stds = []

                for s in steps:
                    step_vals = m_df[m_df["step"] == s]["mean_score"]
                    means.append(step_vals.mean())
                    stds.append(step_vals.std() if len(step_vals) > 1 else 0.0)

                means = np.array(means)
                stds = np.array(stds)
                color = colors.get(method, None)

                ax.plot(steps, means, label=method, color=color, linewidth=2)
                ax.fill_between(steps, means - stds, means + stds, color=color, alpha=0.2)

                # Collect final step metrics
                final_step = steps[-1] if steps else 0
                final_df = m_df[m_df["step"] == final_step]
                summary_rows.append({
                    "Method": method,
                    "Seeds Evaluated": len(m_df["seed"].unique()),
                    "Final Eval Score (Mean ± Std)": f"{final_df['mean_score'].mean():.2f} ± {final_df['mean_score'].std():.2f}" if len(final_df) > 1 else f"{final_df['mean_score'].mean():.2f}",
                    "Best Eval Score (Max)": f"{m_df['max_score'].max()}",
                    "Success Rate (Mean)": f"{final_df['success_rate'].mean()*100:.1f}%",
                })

            ax.set_title("Evaluation Score vs Environment Steps", fontsize=14, fontweight="bold")
            ax.set_xlabel("Environment Steps", fontsize=12)
            ax.set_ylabel("Greedy Evaluation Score (Pipes Passed)", fontsize=12)
            ax.legend(loc="upper left", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.7)
            fig.tight_layout()

            eval_plot_path = os.path.join(plots_dir, "eval_score_curve.png")
            fig.savefig(eval_plot_path, dpi=300)
            plt.close(fig)
            print(f"Saved evaluation plot to {eval_plot_path}")

            # Save summary table
            if summary_rows:
                sum_df = pd.DataFrame(summary_rows)
                sum_csv = os.path.join(tables_dir, "method_comparison.csv")
                sum_df.to_csv(sum_csv, index=False)
                sum_md = os.path.join(tables_dir, "method_comparison.md")

                def _to_md(d):
                    headers = list(d.columns)
                    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                    for _, row in d.iterrows():
                        out.append("| " + " | ".join(str(val) for val in row.values) + " |")
                    return "\n".join(out)

                with open(sum_md, "w") as f:
                    f.write("# Method Performance Comparison\n\n")
                    f.write(_to_md(sum_df) + "\n")
                print(f"Saved summary comparison to {sum_md}")

    # 2. Process Episode Logs
    if episode_files:
        ep_dfs = []
        for f in episode_files:
            try:
                df = pd.read_csv(f)
                ep_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")

        if ep_dfs:
            all_eps = pd.concat(ep_dfs, ignore_index=True)
            methods = all_eps["method"].unique()

            fig, ax = plt.subplots(figsize=(8, 5))
            for method in methods:
                m_df = all_eps[all_eps["method"] == method].sort_values("step")
                # Windowed rolling mean of score
                rolling_score = m_df["score"].rolling(window=50, min_periods=5).mean()
                ax.plot(m_df["step"], rolling_score, label=f"{method} (rolling 50 eps)", linewidth=1.5)

            ax.set_title("Training Episode Score (Rolling Window = 50)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Environment Steps", fontsize=12)
            ax.set_ylabel("Pipes Passed", fontsize=12)
            ax.legend(loc="upper left", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.7)
            fig.tight_layout()

            ep_plot_path = os.path.join(plots_dir, "training_score_curve.png")
            fig.savefig(ep_plot_path, dpi=300)
            plt.close(fig)
            print(f"Saved training curve plot to {ep_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="results/runs")
    parser.add_argument("--plots-dir", type=str, default="results/plots")
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    args = parser.parse_args()
    plot_experiments(args.runs_dir, args.plots_dir, args.tables_dir)
