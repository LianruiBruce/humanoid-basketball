"""Record a whole-body shooting policy on the court, or evaluate its make rate over many shots.

    # clip (behind the shooter, or --side)
    python scripts/record_shot.py --headless --kit_args="--/app/vulkan=false" --task Basketball-G1-ShootTrack-Play-v0
    # statistics only: every env shoots once per episode, outcomes tallied by termination reason
    python scripts/record_shot.py --headless --task Basketball-G1-ShootTrack-Play-v0 --eval --num_envs 512
"""

import argparse
import glob
import os
import sys

import h5py  # noqa: F401  (see play.py)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootTrack-Play-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--out", default="outputs/wholebody_shot")
parser.add_argument("--side", action="store_true", help="side view instead of behind-the-shooter view")
parser.add_argument("--eval", action="store_true", help="no video: tally episode outcomes")
parser.add_argument("--num_envs", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.eval
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
    runs = [r for r in sorted(glob.glob(os.path.join(ROOT, "logs", "rsl_rl", agent_cfg.experiment_name, "*")))
            if not os.path.basename(r).startswith("0_")]
    ckpt = max(glob.glob(os.path.join(runs[-1], "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
print(f"[record] checkpoint: {ckpt}")

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs if args.eval else 1
d = env_cfg.commands.motion.shot_distance
if args.side:
    env_cfg.viewer.eye, env_cfg.viewer.lookat = (d / 2, -(d + 2.5), 1.9), (d / 2, 0.0, 1.6)
else:
    env_cfg.viewer.eye, env_cfg.viewer.lookat = (-2.6, -1.4, 1.6), (0.6 * d, 0.0, 1.7)
env = gym.make(args.task, cfg=env_cfg, render_mode=None if args.eval else "rgb_array")
if not args.eval:
    env = gym.wrappers.RecordVideo(env, video_folder=args.out, step_trigger=lambda s: s == 0,
                                   video_length=args.steps, disable_logger=True)
env = RslRlVecEnvWrapper(env)
base = env.unwrapped

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=base.device)
tm = base.termination_manager
outcomes = {n: 0 for n in tm.active_terms}
episodes = 0

torch.inference_mode().__enter__()
obs, _ = env.reset()
for _ in range(args.steps):
    obs, _, dones, _ = env.step(policy(obs))
    episodes += int(dones.sum())
    for n in outcomes:
        outcomes[n] += int((tm.get_term(n) & dones.bool()).sum())
total = max(episodes, 1)
print(f"[record] {args.task}: {episodes} episodes (an episode can end on several terms at once)")
for n, k in outcomes.items():
    print(f"[record]   {n:14s} {k:6d}  ({100.0 * k / total:5.1f}%)")
if not args.eval:
    print(f"[record] video in {args.out}")
env.close()
os._exit(0)
