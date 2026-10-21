"""Feed the reference joint angles straight to the PD controllers (no policy) and report tracking errors and
which termination fires. Separates environment/reference problems from learning problems."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--no_gravity", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))
import basketball  # noqa: E402, F401
from basketball.tracking_env_cfg import END_EFFECTORS, G1TrackingEnvCfg  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

cfg = G1TrackingEnvCfg()
cfg.scene.num_envs = 64
cfg.events.push = None
cfg.observations.policy.enable_corruption = False
for t in ("anchor_height", "anchor_tilt", "end_effectors", "motion_end", "time_out"):
    setattr(cfg.terminations, t, None)  # observe errors without resets
if args.no_gravity:
    cfg.sim.gravity = (0.0, 0.0, 0.0)
env = ManagerBasedRLEnv(cfg)
env.reset()
cmd = env.command_manager.get_term("motion")
act = env.action_manager.get_term("joints")
ee = [cmd.body_names.index(n) for n in END_EFFECTORS]
print(f"[dbg] action joints {len(act._joint_ids)}, robot joints {cmd.robot.num_joints}")
with torch.inference_mode():
    for k in range(args.steps):
        # action that makes the PD target equal the reference joint angles
        ref = cmd.ref_joint_pos[:, act._joint_ids]
        a = (ref - act._offset) / act._scale
        env.step(a)
        pos, _ = cmd.aligned_ref_bodies()
        ee_err = (pos[:, ee, 2] - cmd.robot_body_pos_w[:, ee, 2]).abs().mean(0)
        body_err = torch.norm(pos - cmd.robot_body_pos_w, dim=-1).mean(0)
        worst = torch.argmax(body_err).item()
        print(f"[dbg] step {k:2d} t={cmd.time_steps[0].item():3d}  anchor z err {(cmd.ref_anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2]).abs().mean():.3f}"
              f"  EE z err " + " ".join(f"{n.split('_')[0][0]}{n.split('_')[1][:3]}={e:.3f}" for n, e in zip(END_EFFECTORS, ee_err.tolist()))
              + f"  worst body {cmd.body_names[worst]} {body_err[worst]:.3f}")
os._exit(0)
