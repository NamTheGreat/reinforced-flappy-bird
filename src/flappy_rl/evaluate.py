"""Evaluation utilities for Flappy Bird agents and video recording."""

from typing import Dict, Any, List, Optional
import numpy as np
import gymnasium
from PIL import Image


def run_episodes(
    agent: Any,
    env: gymnasium.Env,
    n_episodes: int = 100,
    success_threshold: int = 20,
    seed_offset: int = 10000,
) -> Dict[str, Any]:
    """Run greedy evaluation episodes for an agent.

    Args:
        agent: Agent instance providing `act(obs)` or `act(obs, eps=0.0)`.
        env: Gymnasium environment.
        n_episodes: Number of evaluation episodes.
        success_threshold: Minimum pipes cleared to count as a 'success'.
        seed_offset: Base seed for deterministic evaluation.

    Returns:
        Dictionary containing summary statistics and raw lists.
    """
    scores: List[int] = []
    rewards: List[float] = []
    steps_list: List[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            # Check if agent accepts epsilon
            if hasattr(agent, "act") and agent.act.__code__.co_argcount >= 3:
                action = agent.act(obs, 0.0)
            else:
                action = agent.act(obs)

            obs, reward, term, trunc, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            done = term or trunc

        score = info.get("score", 0)
        scores.append(int(score))
        rewards.append(float(ep_reward))
        steps_list.append(ep_steps)

    scores_arr = np.array(scores)
    rewards_arr = np.array(rewards)
    steps_arr = np.array(steps_list)

    success_rate = float(np.mean(scores_arr >= success_threshold))

    return {
        "n_episodes": n_episodes,
        "mean_score": float(np.mean(scores_arr)),
        "std_score": float(np.std(scores_arr)),
        "median_score": float(np.median(scores_arr)),
        "max_score": int(np.max(scores_arr)),
        "min_score": int(np.min(scores_arr)),
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_steps": float(np.mean(steps_arr)),
        "success_rate": success_rate,
        "scores": scores,
        "rewards": rewards,
        "steps": steps_list,
    }


def record_gameplay_gif(
    agent: Any,
    output_path: str = "demo/gameplay.gif",
    max_steps: int = 1500,
    seed: int = 42,
    fps: int = 30,
) -> int:
    """Record an episode of gameplay to an animated GIF using render_mode='rgb_array'.

    Returns:
        Final score achieved in the recorded episode.
    """
    import os
    from flappy_rl.env import make_env

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = make_env(seed=seed, render_mode="rgb_array", max_episode_steps=max_steps)
    obs, info = env.reset(seed=seed)

    frames: List[Image.Image] = []
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))

        if hasattr(agent, "act") and agent.act.__code__.co_argcount >= 3:
            action = agent.act(obs, 0.0)
        else:
            action = agent.act(obs)

        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        step_count += 1

    final_score = info.get("score", 0)
    env.close()

    if frames:
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

    return final_score


def main():
    import argparse
    from flappy_rl.env import make_env
    from flappy_rl.agent import DQNAgent
    from flappy_rl.baselines import RandomAgent, HeuristicAgent

    parser = argparse.ArgumentParser(description="Evaluate agents and record gameplay")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained DQN .pt checkpoint")
    parser.add_argument("--heuristic", action="store_true", help="Evaluate HeuristicAgent")
    parser.add_argument("--random", action="store_true", help="Evaluate RandomAgent")
    parser.add_argument("--double", action="store_true", help="If loading Double DQN model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--record", type=str, default=None, help="Save gameplay GIF to specified path")
    parser.add_argument("--record-steps", type=int, default=1500, help="Maximum steps for GIF recording")
    parser.add_argument("--render", action="store_true", help="Render live gameplay window on screen")
    args = parser.parse_args()

    if args.heuristic:
        agent = HeuristicAgent()
        name = "HeuristicAgent"
    elif args.random:
        agent = RandomAgent()
        name = "RandomAgent"
    elif args.checkpoint:
        agent = DQNAgent(double=args.double)
        agent.load(args.checkpoint)
        name = f"DQN ({args.checkpoint})"
    else:
        print("Please specify --checkpoint, --heuristic, or --random.")
        return

    render_mode = "human" if args.render else None
    env = make_env(seed=42, render_mode=render_mode)
    print(f"Evaluating {name} over {args.episodes} episodes (render={render_mode})...")
    metrics = run_episodes(agent, env, n_episodes=args.episodes)
    env.close()

    print(f"--- Results for {name} ---")
    print(f"Mean Score:   {metrics['mean_score']:.2f} ± {metrics['std_score']:.2f}")
    print(f"Median Score: {metrics['median_score']:.2f}")
    print(f"Max Score:    {metrics['max_score']}")
    print(f"Success Rate (>=20 pipes): {metrics['success_rate']*100:.1f}%")
    print(f"Mean Steps:   {metrics['mean_steps']:.1f}")

    if args.record:
        print(f"Recording gameplay GIF to {args.record}...")
        score = record_gameplay_gif(agent, output_path=args.record, max_steps=args.record_steps)
        print(f"Recorded GIF with score: {score} -> {args.record}")


if __name__ == "__main__":
    main()
