"""Sanity checks for the basketball env: ball starts in the gripper, stays there while held, drops when released."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))
from basketball.shoot_env_cfg import ShootEnvCfg  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

cfg = ShootEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = 10.0  # long enough that timeouts don't interfere
cfg.terminations.missed = None
env = ManagerBasedRLEnv(cfg)
env.reset()

ball = env.scene["ball"]
ee = env.scene["ee_frame"]
hoop = env.scene["hoop"]
origins = env.scene.env_origins


def report(tag):
    tcp = ee.data.target_pos_w[:, 0, :] - origins
    b = ball.data.root_pos_w - origins
    print(f"[{tag}] tcp={tcp[0].tolist()} ball={b[0].tolist()} |ball-tcp|={torch.norm(b - tcp, dim=1).tolist()}")


hold = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
hold[:, -1] = -1.0  # keep gripper closed, arm at default pose
release = hold.clone()
release[:, -1] = 1.0

report("start")
print("[hoop]", (hoop.data.root_pos_w - origins).tolist())
for _ in range(50):  # 1 s holding
    env.step(hold)
report("after 1s hold")
for _ in range(25):  # 0.5 s after opening
    env.step(release)
report("after release")
print("[released]", env.action_manager.get_term("gripper_action").released.tolist())

env.close()
app.close()
