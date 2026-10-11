"""Record a trained G1 policy shooting from a fixed distance, with a camera framed for that distance.

    python scripts/showcase.py --headless --kit_args="--/app/vulkan=false" --distance 6.75 --out outputs/three
"""

import argparse
import glob
import os
import sys

import h5py  # noqa: F401  (see play.py)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootForm-Play-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--distance", type=float, default=4.19)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out", default="outputs/showcase")
parser.add_argument("--side", action="store_true", help="side view instead of behind-the-shooter view")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
import basketball  # noqa: E402, F401

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
ckpt = args.checkpoint
if ckpt is None:
    run = sorted(glob.glob(os.path.join(ROOT, "logs", "rsl_rl", agent_cfg.experiment_name, "*")))[-1]
    ckpt = max(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
print(f"[showcase] checkpoint: {ckpt}")

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env_cfg.events.reset_hoop.params["lateral"] = 0.0
d = args.distance
if args.side:
    env_cfg.viewer.eye, env_cfg.viewer.lookat = (d / 2, -(d + 3.0), 2.0), (d / 2, 0.0, 1.8)
else:
    env_cfg.viewer.eye, env_cfg.viewer.lookat = (-3.4, -1.3, 1.5), (0.55 * d, 0.0, 1.75)
env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder=args.out, step_trigger=lambda s: s == 0,
                               video_length=args.steps, disable_logger=True)
env = RslRlVecEnvWrapper(env)
base = env.unwrapped
hoop_event = base.event_manager.get_term_cfg("reset_hoop").func
hoop_event.frontier, hoop_event.frontier_prob, hoop_event.spread = d, 1.0, 0.0
hoop_event.cfg.params["promote_at"] = 2.0

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=base.device)
tm = base.termination_manager

torch.inference_mode().__enter__()
obs, _ = env.reset()
made = shots = 0
for _ in range(args.steps):
    obs, _, dones, _ = env.step(policy(obs))
    made += int(tm.get_term("made_shot").sum())
    shots += int(dones.sum())
print(f"[showcase] distance={d} m  made {made}/{shots}  video in {args.out}")
env.close()
os._exit(0)
