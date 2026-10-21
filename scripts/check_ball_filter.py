"""Does each env's ball pass through its own robot (contacts filtered) or get pushed out?

Places every ball inside its robot's torso, steps physics with the grasp disabled, and prints how far each ball
moved sideways (gravity alone only moves it down).

    python scripts/check_ball_filter.py --headless --num_envs 4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootPush-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))
import basketball  # noqa: E402, F401

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.events.grasp = None
for t in ("anchor_height", "anchor_tilt", "made", "missed", "dropped"):
    setattr(cfg.terminations, t, None)
for r in ("made", "dropped", "fell_height", "fell_tilt"):
    if hasattr(cfg.rewards, r):
        setattr(cfg.rewards, r, None)
env = gym.make(args.task, cfg=cfg)
base = env.unwrapped
env.reset()
robot, ball = base.scene["robot"], base.scene["ball"]
torso = robot.body_names.index("torso_link")
torch.inference_mode().__enter__()
pose = torch.zeros(base.num_envs, 7, device=base.device)
pose[:, :3] = robot.data.body_pos_w[:, torso]
pose[:, 3] = 1.0
ball.write_root_pose_to_sim(pose)
ball.write_root_velocity_to_sim(torch.zeros(base.num_envs, 6, device=base.device))
start = pose[:, :3].clone()
zero = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
for _ in range(3):
    env.step(zero)
d = ball.data.root_pos_w - start
for e in range(base.num_envs):
    print(f"[filter] env {e}: ball moved sideways {torch.norm(d[e, :2]):.3f} m, down {-d[e, 2]:.3f} m")
os._exit(0)
