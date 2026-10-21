"""How fast can the G1 push the ball? Plays the reference with PD targets, then in the last frames before the release
offsets the shoulder pitch / elbow targets by --push rad and reports the palm-center speed at the release.

    python scripts/speed_test.py --headless --kd_scale 1.0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootTrack-Play-v0")
parser.add_argument("--kd_scale", type=float, default=1.0, help="scale on the arm joint damping")
parser.add_argument("--sp", type=float, default=0.0, help="shoulder pitch target offset (rad) during the push")
parser.add_argument("--el", type=float, default=-3.0, help="elbow target offset (rad; negative extends)")
parser.add_argument("--window", type=int, default=6)
parser.add_argument("--kp", type=float, default=None, help="override the arm joint stiffness")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))
import basketball  # noqa: E402, F401
from basketball import tracking_mdp as tm_mdp  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = 1
for t in ("anchor_height", "anchor_tilt", "end_effectors", "missed", "dropped"):
    setattr(cfg.terminations, t, None)
for r in ("fell_height", "fell_tilt", "fell_feet", "dropped"):
    setattr(cfg.rewards, r, None)
for name, act in cfg.scene.robot.actuators.items():
    if "arm" in name or "shoulder" in name or "elbow" in name:
        if isinstance(act.damping, dict):
            act.damping = {k: v * args.kd_scale for k, v in act.damping.items()}
        else:
            act.damping = act.damping * args.kd_scale
        if args.kp is not None:
            act.stiffness = args.kp
print("[speed] actuators:", list(cfg.scene.robot.actuators.keys()))
env = gym.make(args.task, cfg=cfg)
base = env.unwrapped
cmd = base.command_manager.get_term("motion")
act = base.action_manager.get_term("joints")
robot = base.scene["robot"]
names = [robot.joint_names[i] for i in act._joint_ids]
push = torch.zeros(len(names), device=base.device)
for i, n in enumerate(names):
    if "shoulder_pitch" in n:
        push[i] = args.sp
    elif "elbow" in n:
        push[i] = args.el
torch.inference_mode().__enter__()
env.reset()
rel = cmd.release_frame
best = 0.0
for k in range(rel + 5):
    target = cmd.ref_joint_pos[:, act._joint_ids].clone()
    if rel - args.window <= cmd.time_steps[0].item() <= rel:
        target = target + push
    env.step((target - act._offset) / act._scale)
    v = tm_mdp.palm_center_vel(base)[0]
    best = max(best, torch.norm(v).item())
    if cmd.time_steps[0].item() == rel:
        at_release = v.tolist()
print(f"[speed] kd {args.kd_scale} sp {args.sp} el {args.el}: max {best:.2f} m/s, at release "
      f"{[round(x, 2) for x in at_release]} |v| {torch.norm(torch.tensor(at_release)):.2f}")
os._exit(0)
