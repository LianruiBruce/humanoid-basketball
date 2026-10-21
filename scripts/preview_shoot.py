"""Drive the shooting env with the reference joint angles as PD targets (no policy) and record a clip.

Checks court / hoop placement and whether the ball sits in the hands, before any training.

    python scripts/preview_shoot.py --headless --kit_args="--/app/vulkan=false" --task Basketball-G1-ShootTrack-Play-v0
"""

import argparse
import os
import sys

import h5py  # noqa: F401  (see play.py)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootTrack-Play-v0")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--out", default="outputs/preview_shoot")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))
import basketball  # noqa: E402, F401

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
for t in ("anchor_height", "anchor_tilt", "end_effectors"):
    setattr(cfg.terminations, t, None)  # PD-only playback will drift; watch the ball, not the balance
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder=args.out, step_trigger=lambda s: s == 0, video_length=args.steps,
                               disable_logger=True)
base = env.unwrapped
env.reset()
cmd = base.command_manager.get_term("motion")
act = base.action_manager.get_term("joints")
ball, tm = base.scene["ball"], base.termination_manager
o = base.scene.env_origins
torch.inference_mode().__enter__()
for k in range(args.steps):
    a = (cmd.ref_joint_pos[:, act._joint_ids] - act._offset) / act._scale
    env.step(a)
    if k % 10 == 0:
        p = (ball.data.root_pos_w[0] - o[0]).tolist()
        pc = (basketball.tracking_mdp.palm_center(base)[0] - o[0]).tolist()
        fired = [n for n in ("made", "missed", "dropped", "time_out") if tm.get_term(n)[0]]
        print(f"[preview] step {k:3d} frame {cmd.time_steps[0].item():3d} ball {[round(x, 2) for x in p]} "
              f"palms {[round(x, 2) for x in pc]} {fired}")
print(f"[preview] hoop at {(base.scene['hoop'].data.root_pos_w[0] - o[0]).tolist()}")
env.close()
os._exit(0)
