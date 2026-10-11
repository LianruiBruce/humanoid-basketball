"""Record a short clip of a task with zero (or random) actions to check the scene before training.

    python scripts/preview.py --task Basketball-G1-Shoot-Play-v0 --headless --kit_args="--/app/vulkan=false"
"""

import argparse
import os
import sys

import h5py  # noqa: F401  (see play.py)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-Shoot-Play-v0")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--random", action="store_true", help="random actions instead of zeros")
parser.add_argument("--out", default="outputs/preview")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))
import basketball  # noqa: E402, F401

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder=args.out, step_trigger=lambda s: s == 0,
                               video_length=args.steps, disable_logger=True)
base = env.unwrapped
env.reset()
ball = base.scene["ball"]
start = ball.data.root_pos_w[0].clone()
for i in range(args.steps):
    a = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
    if args.random:
        a.normal_(0.0, 0.5)
    env.step(a)
    if i in (0, 25, 50, 100):
        p = ball.data.root_pos_w[0] - base.scene.env_origins[0]
        print(f"[preview] step {i}: ball={[round(v, 3) for v in p.tolist()]}")
print(f"[preview] video in {args.out}")
env.close()
os._exit(0)
