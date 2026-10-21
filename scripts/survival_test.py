"""How long does a tracking/shooting policy stay up? Prints mean episode length and termination reasons.

    python scripts/survival_test.py --headless --task Basketball-G1-ShootTrack-Play-v0 --checkpoint <ckpt> --flat_hands
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootTrack-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--motion", default=None, help="override the reference npz")
parser.add_argument("--flat_hands", action="store_true", help="fingers straight, default hand gains")
parser.add_argument("--random_start", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
import basketball  # noqa: E402, F401
from basketball.tracking_env_cfg import G1_TRACKING_CFG  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.commands.motion.random_start = args.random_start
if args.motion:
    cfg.commands.motion.motion_file = args.motion
if args.flat_hands:
    cfg.scene.robot = cfg.scene.robot.replace(init_state=G1_TRACKING_CFG.init_state,
                                              actuators=G1_TRACKING_CFG.actuators)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
base = env.unwrapped
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=base.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=base.device)
tm = base.termination_manager
cmd = base.command_manager.get_term("motion")
reasons = {n: 0 for n in tm.active_terms}
lengths = []
age = torch.zeros(base.num_envs, device=base.device)
torch.inference_mode().__enter__()
obs, _ = env.reset()
for _ in range(args.steps):
    obs, _, dones, _ = env.step(policy(obs))
    age += 1
    d = dones.bool()
    for n in reasons:
        reasons[n] += int((tm.get_term(n) & d).sum())
    lengths += age[d].tolist()
    age[d] = 0
n = max(len(lengths), 1)
print(f"[survival] {args.task} flat_hands={args.flat_hands} motion={args.motion}")
print(f"[survival] episodes {len(lengths)}  mean length {sum(lengths) / n:.1f} steps")
print("[survival] " + "  ".join(f"{k}={v / n:.2f}" for k, v in reasons.items() if v))
os._exit(0)
