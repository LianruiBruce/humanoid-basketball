"""Evaluate a trained shooting policy without the RTX renderer.

Runs the policy headless (physics only), reports the hit rate over many shots, and renders env 0 as a
3D stick-figure GIF with matplotlib.

    python scripts/evaluate.py --headless --num_envs 512 --shots 2000 --gif outputs/shoot.gif
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-Shoot-Franka-Play-v0")
parser.add_argument("--checkpoint", default=None, help="model_*.pt; defaults to the latest run's last checkpoint")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--shots", type=int, default=2000, help="number of finished episodes to evaluate")
parser.add_argument("--gif", default="outputs/shoot.gif")
parser.add_argument("--gif_seconds", type=float, default=8.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
import basketball  # noqa: E402, F401

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

LINKS = [f"panda_link{i}" for i in range(8)] + ["panda_hand"]
FINGERS = ["panda_leftfinger", "panda_rightfinger"]


def latest_checkpoint() -> str:
    runs = sorted(glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "franka_shoot", "*")))
    ckpts = sorted(glob.glob(os.path.join(runs[-1], "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
    return ckpts[-1]


def main():
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
    base = env.unwrapped

    ckpt = args.checkpoint or latest_checkpoint()
    print(f"[eval] checkpoint: {ckpt}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=base.device)

    robot = base.scene["robot"]
    link_ids = [robot.body_names.index(n) for n in LINKS]
    finger_ids = [robot.body_names.index(n) for n in FINGERS]
    origin0 = base.scene.env_origins[0]
    tm = base.termination_manager

    frames = []  # env-0 snapshots for the GIF
    n_gif_steps = int(args.gif_seconds / base.step_dt)
    made = missed = timeouts = 0

    obs = env.get_observations()
    step = 0
    while made + missed + timeouts < args.shots or step < n_gif_steps:
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        step += 1
        made += int(tm.get_term("made_shot").sum())
        missed += int(tm.get_term("missed").sum())
        timeouts += int(tm.get_term("time_out").sum())
        if step <= n_gif_steps:
            frames.append(
                dict(
                    links=(robot.data.body_pos_w[0, link_ids] - origin0).cpu().numpy(),
                    fingers=(robot.data.body_pos_w[0, finger_ids] - origin0).cpu().numpy(),
                    ball=(base.scene["ball"].data.root_pos_w[0] - origin0).cpu().numpy(),
                    hoop=(base.scene["hoop"].data.root_pos_w[0] - origin0).cpu().numpy(),
                    made=bool(tm.get_term("made_shot")[0]),
                    done=bool(dones[0]),
                )
            )

    total = made + missed + timeouts
    print(f"[eval] shots={total} made={made} missed={missed} timeout={timeouts} hit_rate={made / total:.1%}")
    env.close()
    render_gif(frames, args.gif, fps=int(round(1 / base.step_dt / 2)), hit_rate=made / total)


def render_gif(frames, path, fps, hit_rate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from basketball.mdp import RIM_RADIUS

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(projection="3d")
    theta = np.linspace(0, 2 * np.pi, 40)
    shot, score, trail, flash = 0, 0, [], 0
    # render every 2nd physics step, but keep made/done events from the skipped step
    sel = []
    for i in range(0, len(frames) - 1, 2):
        f = dict(frames[i])
        f["made"] = frames[i]["made"] or frames[i + 1]["made"]
        f["done"] = frames[i]["done"] or frames[i + 1]["done"]
        sel.append(f)

    def draw(i):
        nonlocal shot, score, trail, flash
        f = sel[i]
        ax.cla()
        ax.set_xlim(-0.2, 1.9)
        ax.set_ylim(-0.8, 0.8)
        ax.set_zlim(-0.3, 1.0)
        ax.set_box_aspect((2.1, 1.6, 1.3))
        ax.view_init(elev=14, azim=-62)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
        # table top
        ax.plot_surface(*np.meshgrid([-0.2, 1.0], [-0.6, 0.6]), np.zeros((2, 2)), color="#c8b89a", alpha=0.35)
        # robot
        p = f["links"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color="#3a5a8c", lw=5, ms=4)
        for q in f["fingers"]:
            ax.plot([p[-1, 0], q[0]], [p[-1, 1], q[1]], [p[-1, 2], q[2]], color="#3a5a8c", lw=3)
        # hoop: rim + backboard
        h = f["hoop"]
        ax.plot(h[0] + RIM_RADIUS * np.cos(theta), h[1] + RIM_RADIUS * np.sin(theta), np.full_like(theta, h[2]),
                color="#e0601a", lw=3)
        by, bz = np.meshgrid([-0.3, 0.3], [-0.07, 0.35])
        ax.plot_surface(np.full_like(by, h[0] + 0.18), h[1] + by, h[2] + bz, color="#dddddd", alpha=0.6)
        # ball + trail
        b = f["ball"]
        trail.append(b)
        t = np.array(trail[-25:])
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color="#e0601a", alpha=0.4, lw=1.5)
        ax.scatter(*b, s=120, color="#f07020", edgecolors="k", depthshade=False)
        if f["made"]:
            score += 1
            flash = 12
        if f["done"]:
            shot += 1
            trail = []
        if flash > 0:
            ax.text2D(0.42, 0.85, "SWISH!", transform=ax.transAxes, fontsize=22, color="#e0601a", weight="bold")
            flash -= 1
        ax.set_title(f"Franka basketball  |  made {score}/{shot}  |  eval hit rate {hit_rate:.0%}", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(sel), interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    print(f"[eval] wrote {path}")


if __name__ == "__main__":
    main()
    app.close()
