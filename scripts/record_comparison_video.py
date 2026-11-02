"""Generate a side-by-side comparison video of Untrained vs Trained DQN agent."""

import subprocess
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flappy_rl.env import make_env
from flappy_rl.agent import DQNAgent
from flappy_rl.baselines import RandomAgent


def create_comparison_video(
    checkpoint_path: str = "models/DQN_PER_seed_1_1790610322_best.pt",
    output_mp4: str = "demo/training_comparison.mp4",
    output_gif: str = "demo/training_comparison.gif",
    duration_frames: int = 450,  # ~15 seconds at 30 fps
    fps: int = 30,
):
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)

    # Environments
    env_untrained = make_env(seed=101, render_mode="rgb_array")
    env_trained = make_env(seed=42, render_mode="rgb_array")

    # Agents
    agent_untrained = RandomAgent()
    agent_trained = DQNAgent()
    agent_trained.load(checkpoint_path)

    obs_u, _ = env_untrained.reset(seed=101)
    obs_t, _ = env_trained.reset(seed=42)

    score_u = 0
    score_t = 0
    deaths_u = 0

    frame_w = 288
    frame_h = 512
    banner_h = 50
    total_w = frame_w * 2
    total_h = frame_h + banner_h

    # Font setup
    font = ImageFont.load_default()

    # Setup ffmpeg process
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{total_w}x{total_h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    gif_frames = []

    print(f"Rendering {duration_frames} frames to {output_mp4}...")
    for frame_idx in range(duration_frames):
        # Step untrained
        act_u = agent_untrained.act(obs_u)
        obs_u, _, term_u, trunc_u, info_u = env_untrained.step(act_u)
        score_u = info_u.get("score", 0)
        if term_u or trunc_u:
            deaths_u += 1
            obs_u, _ = env_untrained.reset()

        # Step trained
        act_t = agent_trained.act(obs_t, eps=0.0)
        obs_t, _, term_t, trunc_t, info_t = env_trained.step(act_t)
        score_t = info_t.get("score", 0)
        if term_t or trunc_t:
            obs_t, _ = env_trained.reset()

        # Render frames
        rgb_u = env_untrained.render()
        rgb_t = env_trained.render()

        # Create composite image
        img_u = Image.fromarray(rgb_u)
        img_t = Image.fromarray(rgb_t)

        composite = Image.new("RGB", (total_w, total_h), color=(20, 24, 30))
        composite.paste(img_u, (0, banner_h))
        composite.paste(img_t, (frame_w, banner_h))

        # Draw HUD banner
        draw = ImageDraw.Draw(composite)

        # Header background divider
        draw.line([(frame_w, 0), (frame_w, total_h)], fill=(60, 70, 85), width=2)
        draw.line([(0, banner_h), (total_w, banner_h)], fill=(60, 70, 85), width=2)

        # Left label
        draw.text((12, 10), "UNTRAINED (Step 0)", fill=(255, 100, 100), font=font)
        draw.text((12, 28), f"Score: {score_u}  |  Crashes: {deaths_u}", fill=(200, 200, 200), font=font)

        # Right label
        draw.text((frame_w + 12, 10), "TRAINED (DQN + Rust PER)", fill=(100, 255, 140), font=font)
        draw.text((frame_w + 12, 28), f"Score: {score_t}  |  Step: 200,000", fill=(200, 200, 200), font=font)

        # Pipe raw bytes to ffmpeg
        proc.stdin.write(composite.tobytes())

        # Collect subset of frames for lightweight GIF (every 2nd frame)
        if frame_idx % 2 == 0:
            gif_frames.append(composite.copy())

    proc.stdin.close()
    proc.wait()
    env_untrained.close()
    env_trained.close()

    print(f"Saved MP4: {output_mp4} ({os.path.getsize(output_mp4) / 1024:.1f} KB)")

    # Save GIF as well
    if gif_frames and output_gif:
        print(f"Saving GIF to {output_gif}...")
        gif_frames[0].save(
            output_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(1000 / (fps / 2)),
            loop=0,
            optimize=True,
        )
        print(f"Saved GIF: {output_gif} ({os.path.getsize(output_gif) / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    create_comparison_video()
